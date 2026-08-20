"""
Diffusion Evolution Module - CHARLES Implementation

This module implements candidate gene generation using the CHARLES 
(Conditional, Heuristically-Adaptive ReguLarized Evolutionary Strategy through Diffusion) algorithm.

CHARLES extends HADES by using CONDITIONAL diffusion - the neural network receives 
fitness values as additional inputs during both training and sampling. During sampling,
we condition on HIGH target fitness values to guide generation toward high-fitness regions.

Key Difference from HADES:
- HADES: Network learns distribution weighted by fitness
- CHARLES: Network learns conditional distribution P(gene | fitness), 
           then samples by conditioning on high fitness targets

References:
    Hartl et al., "Heuristically Adaptive Diffusion-Model Evolutionary Strategy"
    https://arxiv.org/abs/2411.13420
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Dict, Any, Union
import time
import gc

# CondEvo imports (assumes condevo package is in HADES/ directory)
from condevo.diffusion import DDIM
# Note: Using custom ConditionalMLP defined below; remove unused imports

# Local imports for scalers and loader
# Import your existing utilities
try:
    from GA_utils_history_loader import ChunkedGeneHistoryLoader
    from GA_utils_scalers import ChunkedLogZScoreScaler, ChunkedPerDimensionScaler
except ImportError:
    # Allow standalone testing
    ChunkedGeneHistoryLoader = None
    ChunkedLogZScoreScaler = None
    ChunkedPerDimensionScaler = None


# =============================================================================
# DEFAULT PARAMETERS
# =============================================================================

DEFAULT_CHARLES_PARAMS = {
    # Data loading
    'top_k_percentile': 25.0,           # Top % of population to train on
    
    # Network architecture
    'num_hidden': None,                  # Auto: geneLength * 2
    'num_layers': 3,                     # MLP depth
    
    # Diffusion model
    'num_diffusion_steps': 100,          # Denoising steps
    'diff_range': 1.0,                   # Parameter range (genes in [-1, 1])
    'alpha_schedule': 'cosine',          # 'linear' or 'cosine'
    
    # Training
    'diff_batch_size': 32,               # Training batch size
    'diff_max_epoch': 50,                # Training epochs
    'diff_lr': 1e-3,                     # Learning rate
    'diff_weight_decay': 1e-5,           # L2 regularization
    
    # Selection and weighting
    'selection_pressure': 3.0,           # Roulette wheel selection pressure
    'emphasis_factor': 2.0,              # Fitness weighting emphasis
    'min_weight': 0.1,                   # Minimum sample weight
    
    # Conditioning (CHARLES-specific)
    'fitness_scale': 1.0,                # Scale factor for fitness conditioning
    'greedy_sampling': False,            # If True, condition on max fitness; else mean + variance
    'condition_noise_scale': 0.5,        # Noise scale for target fitness sampling
    
    # Buffer
    'buffer_size': 4,                    # Generations in buffer (multiplier)
    
    # Mutation (post-sampling noise)
    'mutation_rate': 0.0,                # Fraction of diffusion steps as noise
    'elite_ratio': 0.1,                  # Fraction of elites to protect
    
    # Memory management
    'chunk_size': None,                  # Auto-calculated if None
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _get_device(args) -> torch.device:
    """Resolve device from args.gpu boolean."""
    if getattr(args, 'gpu', False) and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def _log(msg: str, verbose: int, level: int = 1):
    """Print message if verbose level is sufficient."""
    if verbose is not None and verbose >= level:
        print(f"[CHARLES] {msg}", flush=True)


def _free_memory(device: torch.device):
    """Free GPU/CPU memory."""
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def _clamp_to_list(tensor_or_array, min_val: float = -1.0, max_val: float = 1.0) -> List[List[float]]:
    """Convert tensor/array to clamped List[List[float]]."""
    if isinstance(tensor_or_array, torch.Tensor):
        clamped = torch.clamp(tensor_or_array, min_val, max_val)
        return clamped.detach().cpu().tolist()
    else:
        clamped = np.clip(tensor_or_array, min_val, max_val)
        return clamped.tolist()


# =============================================================================
# CONDITIONAL MLP (Extended for CHARLES)
# =============================================================================

class ConditionalMLP(nn.Module):
    """
    MLP for conditional diffusion with fitness conditioning.
    
    Input: [gene (D), time (1), fitness_condition (1)] -> Output: [gene (D)]
    
    The fitness condition allows the network to learn P(gene | fitness),
    enabling targeted sampling toward specific fitness regions.
    """
    
    def __init__(
        self, 
        num_params: int, 
        num_hidden: int = 64, 
        num_layers: int = 3,
        num_conditions: int = 1,
        activation: str = 'ReLU',
        dropout: float = 0.0
    ):
        super().__init__()
        self.num_params = num_params
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.num_conditions = num_conditions
        
        # Input: gene_dim + time (1) + conditions
        input_dim = num_params + 1 + num_conditions
        output_dim = num_params
        
        # Get activation function
        if isinstance(activation, str):
            activation_fn = getattr(nn, activation)
        else:
            activation_fn = activation
        
        # Build layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, num_hidden))
        layers.append(activation_fn())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(num_hidden, num_hidden))
            layers.append(activation_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(num_hidden, output_dim))
        
        self.layers = nn.Sequential(*layers)
    
    def _build_model(self):
        """Reinitialize model weights (required by DDIM)."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, *conditions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Gene tensor (batch, num_params)
            t: Time tensor (batch, 1)
            *conditions: Condition tensors, each (batch, 1)
            
        Returns:
            Predicted noise (batch, num_params)
        """
        # Concatenate inputs
        inputs = [x, t]
        for cond in conditions:
            if cond.dim() == 1:
                cond = cond.unsqueeze(-1)
            inputs.append(cond)
        
        combined = torch.cat(inputs, dim=-1)
        return self.layers(combined)


# =============================================================================
# FITNESS CONDITION HANDLER
# =============================================================================

class FitnessConditionHandler:
    """
    Handles fitness conditioning for CHARLES.
    
    Provides:
    - evaluate(): Compute condition values from fitness scores
    - sample(): Sample target fitness values for guided generation
    - transform(): Normalize conditions for network input
    """
    
    def __init__(
        self, 
        scale: float = 1.0, 
        greedy: bool = False,
        noise_scale: float = 0.5,
        device: torch.device = None
    ):
        self.scale = scale
        self.greedy = greedy
        self.noise_scale = noise_scale
        self.device = device or torch.device('cpu')
        
        # Statistics (updated during training data collection)
        self.fitness_mean = None
        self.fitness_std = None
        self.fitness_max = None
        self.fitness_min = None
        self.elite_fitness_mean = None
        self.elite_fitness_max = None
    
    def update_statistics(
        self, 
        fitness_values: torch.Tensor, 
        elite_ratio: float = 0.1
    ):
        """Update fitness statistics from population."""
        self.fitness_mean = fitness_values.mean()
        self.fitness_std = fitness_values.std()
        self.fitness_max = fitness_values.max()
        self.fitness_min = fitness_values.min()
        
        # Elite statistics
        num_elite = max(1, int(len(fitness_values) * elite_ratio))
        elite_indices = torch.argsort(fitness_values, descending=True)[:num_elite]
        elite_fitness = fitness_values[elite_indices]
        
        self.elite_fitness_mean = elite_fitness.mean()
        self.elite_fitness_max = elite_fitness.max()
    
    def evaluate(self, fitness: torch.Tensor) -> torch.Tensor:
        """
        Compute condition values from fitness scores.
        
        Args:
            fitness: Raw fitness values (batch,)
            
        Returns:
            Normalized condition values (batch, 1)
        """
        # Normalize by scale
        conditions = (fitness / self.scale).float()
        
        # Reshape to (batch, 1)
        if conditions.dim() == 1:
            conditions = conditions.unsqueeze(-1)
        
        return conditions
    
    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Sample target fitness values for guided generation.
        
        Based on Fisher's fundamental theorem of natural selection:
        The rate of fitness increase equals the genetic variance in fitness.
        
        Args:
            num_samples: Number of target fitness values to sample
            
        Returns:
            Target fitness conditions (num_samples, 1)
        """
        if self.fitness_std is None:
            # Fallback if statistics not set
            return torch.ones(num_samples, 1, device=self.device)
        
        # Compute variance-based fitness increment
        var_W = self.fitness_std ** 2
        
        # Sample noise based on variance
        dW = var_W * self.noise_scale * torch.randn(num_samples, 1, device=self.device)
        
        if not self.greedy:
            # Sample around elite mean + positive increment
            target_fitness = self.elite_fitness_mean + torch.abs(dW)
        else:
            # Greedy: sample around elite max
            target_fitness = self.elite_fitness_max + dW
        
        # Normalize by scale
        return target_fitness / self.scale
    
    def transform(self, conditions: torch.Tensor) -> torch.Tensor:
        """
        Transform conditions for network input (identity by default).
        
        Can be overridden for more complex transformations.
        """
        return conditions


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def createCandidateGene_CondEvo_CHARLES_Ben(
    args=None,
    EA_Class=None,
    genePopulation: Optional[List[Dict[str, Any]]] = None,
    number_of_candidate_genes: Optional[int] = None,
    build_in_params: Optional[Dict] = None,
    verbose: Optional[int] = None
) -> List[List[float]]:
    """
    Generate candidate genes using CHARLES (Conditional Diffusion Evolution).
    
    CHARLES uses conditional diffusion where fitness values are passed as inputs
    to the network. During sampling, we condition on HIGH target fitness values
    to guide generation toward high-fitness regions.
    
    Algorithm Overview:
    1. Collect high-fitness solutions and their fitness values
    2. Train conditional DDIM: learns P(gene | fitness)
    3. Sample target fitness values (above current elite mean)
    4. Generate new candidates conditioned on high target fitness
    5. Output genes clamped to [-1, 1]
    
    Key Difference from HADES:
    - HADES: Network learns distribution weighted by fitness
    - CHARLES: Network learns P(gene | fitness), then conditions on high fitness
    
    Args:
        args: Arguments object with:
            - path: Save path for disk loading
            - evolutionTarget: 1 (maximize) or -1 (minimize)
            - gpu: Whether to use GPU (boolean)
            - agent_idx: Agent ID for disk loading
            - epoch: Current epoch for filtering
            - populationSize: Population size (for datapoint calculation)
        EA_Class: Class with geneLength and geneFormat attributes
        genePopulation: Optional pre-loaded population [{'gene': [...], 'fitnessScore': float}, ...]
                       If None, loads from disk using streaming.
        number_of_candidate_genes: Number of genes to generate (REQUIRED)
        build_in_params: Dict with hyperparameters. Will be modified to include:
            - CHARLES: Dict with algorithm parameters
            - CHARLES_log: Dict with runtime statistics
            - loss: Float with the convergence loss metric
        verbose: Verbosity level:
            - None or 0: Silent
            - 1: Minimal (key results)
            - 2: Standard (progress and parameters)
            - 3: Debug (all details)
    
    Returns:
        List[List[float]]: New candidate genes [[gene1], [gene2], ...]
                          Each gene is clamped to [-1, 1]
    
    Raises:
        ValueError: If required parameters are None
    """
    start_time = time.time()
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    if args is None:
        raise ValueError("args cannot be None")
    if EA_Class is None:
        raise ValueError("EA_Class cannot be None")
    if number_of_candidate_genes is None or number_of_candidate_genes <= 0:
        raise ValueError("number_of_candidate_genes must be a positive integer")
    
    # Initialize build_in_params if needed
    if build_in_params is None:
        build_in_params = {}
    
    # =========================================================================
    # SETUP PARAMETERS
    # =========================================================================
    
    # Get or create CHARLES params dict
    charles_params = build_in_params.get('HADES', {})
    
    # Merge with defaults (user params override defaults)
    params = DEFAULT_CHARLES_PARAMS.copy()
    params.update(charles_params)
    
    # Store back to build_in_params
    build_in_params['HADES'] = params
    
    # Initialize log dict
    log_dict = {}
    build_in_params['HADES_log'] = log_dict
    
    # Extract key parameters
    gene_length = EA_Class.geneLength
    evolution_target = getattr(args, 'evolutionTarget', 1)
    maximize = (evolution_target == 1)
    
    # Auto-calculate num_hidden if not specified
    if params['num_hidden'] is None:
        params['num_hidden'] = gene_length * 2
    
    # Device setup
    device = _get_device(args)
    _log(f"Using device: {device}", verbose, 2)
    
    # =========================================================================
    # INITIALIZE SCALERS
    # =========================================================================
    
    _log("Initializing scalers...", verbose, 2)
    
    fitness_scaler = ChunkedLogZScoreScaler(
        evolution_target=evolution_target,
        emphasis_factor=params['emphasis_factor'],
        min_weight=params['min_weight'],
        device=device
    )
    
    gene_scaler = ChunkedPerDimensionScaler(
        gene_length=gene_length,
        device=device,
        weight_mode='uniform'
    )
    
    # Initialize fitness condition handler
    fitness_condition = FitnessConditionHandler(
        scale=params['fitness_scale'],
        greedy=params['greedy_sampling'],
        noise_scale=params['condition_noise_scale'],
        device=device
    )
    
    # =========================================================================
    # LOAD AND PROCESS DATA
    # =========================================================================
    
    genes_list = []
    fitness_list = []
    
    if genePopulation is not None:
        # -----------------------------------------------------------------
        # MODE A: Use provided population (already filtered)
        # -----------------------------------------------------------------
        _log(f"Using provided population: {len(genePopulation)} samples", verbose, 2)
        
        # Extract genes and fitness, fit scalers
        for record in genePopulation:
            gene = record.get('gene')
            fitness = record.get('fitnessScore')
            
            if gene is None or fitness is None:
                continue
            if not np.isfinite(fitness):
                continue
            
            genes_list.append(gene)
            fitness_list.append(fitness)
            
            # Feed to scalers (phase 1)
            fitness_scaler.collect_phase_1_statistics([fitness])
            gene_scaler.collect_phase_1_statistics([gene])
        
        # Finalize phase 1
        fitness_scaler._finalize_pass1()
        gene_scaler._finalize_pass1()
        
        # Phase 2 statistics
        for gene, fitness in zip(genes_list, fitness_list):
            fitness_scaler.collect_phase_2_statistics([fitness])
            gene_scaler.collect_phase_2_statistics([gene])
        
        # Finalize scalers
        fitness_scaler.finalize()
        gene_scaler.finalize()
        
        log_dict['data_source'] = 'provided_population'
        log_dict['population_size'] = len(genes_list)
        
    else:
        # -----------------------------------------------------------------
        # MODE B: Load from disk using streaming
        # -----------------------------------------------------------------
        _log("Loading from disk using streaming...", verbose, 2)
        
        agent_id = getattr(args, 'agent_idx', 0)
        top_k_percentile = params['top_k_percentile']
        
        # Calculate datapoint limits
        pop_size = getattr(args, 'populationSize', 1000)
        min_datapoints = max(10, min(int(pop_size * 0.05), 300))
        max_datapoints = int(getattr(args, 'populationSize', 10000))
        
        # Initialize loader
        loader = ChunkedGeneHistoryLoader(
            savePath=args.path,
            agent_id=agent_id,
            geneFormat=getattr(EA_Class, 'geneFormat', 'json'),
            shuffle=True,
            shuffle_level='both',
            required_keys_and_types={'fitnessScore': 'list', 'gene': 'list'},
        )
        
        # Calculate chunk size
        chunk_size = params.get('chunk_size')
        if chunk_size is None:
            chunk_size = loader.calculate_chunk_size(
                safety_margin=0.5,
                chunk_allocation=0.20,
                min_chunk=100,
                max_chunk=5000,
                verbose=(verbose is not None and verbose >= 3)
            )
        
        _log(f"Chunk size: {chunk_size}, Top-K: {top_k_percentile}%", verbose, 2)
        
        # Stream data with scaler fitting
        total_loaded = 0
        for chunk, is_done in loader.loadTopKPercentileGeneHistoryAsListOfDicts(
            args=args,
            top_k_percentile=top_k_percentile,
            maximize=maximize,
            chunk_size=chunk_size,
            fitness_scaler=fitness_scaler,
            gene_scaler=gene_scaler,
            use_fitness_scaler_weights=True,
            use_gene_scaler_weights=True,
            # n_feature_clusters='',
            allEpochs=True,
            min_datapoints=min_datapoints,
            max_datapoints=max_datapoints,
            fit_fitness_scaler_on_top_k=True,
            fit_gene_scaler_on_top_k=True,
        ):
            for record in chunk:
                gene = record.get('gene')
                fitness = record.get('fitnessScore')
                
                if gene is not None and fitness is not None:
                    genes_list.append(gene)
                    fitness_list.append(fitness)
                    total_loaded += 1
            
            _log(f"Loaded {total_loaded} records...", verbose, 3)
            
            if is_done:
                break
        
        log_dict['data_source'] = 'disk_streaming'
        log_dict['population_size'] = len(genes_list)
        log_dict['top_k_percentile'] = top_k_percentile
    
    # =========================================================================
    # VALIDATE DATA
    # =========================================================================
    
    if len(genes_list) < 2:
        _log("ERROR: Not enough valid samples for training", verbose, 1)
        build_in_params['loss'] = float('inf')
        # Return random samples clamped to [-1, 1]
        random_genes = np.random.uniform(-1, 1, (number_of_candidate_genes, gene_length))
        return random_genes.tolist()
    
    _log(f"Collected {len(genes_list)} samples for training", verbose, 1)
    
    # =========================================================================
    # NORMALIZE DATA
    # =========================================================================
    
    _log("Normalizing genes and preparing fitness conditions...", verbose, 2)
    
    # Convert to tensors
    genes_np = np.array(genes_list, dtype=np.float32)
    fitness_np = np.array(fitness_list, dtype=np.float32)
    
    # Clear lists to free memory
    del genes_list, fitness_list
    _free_memory(device)
    
    # Normalize genes using scaler
    genes_normalized = gene_scaler.transform(genes_np)
    
    # Convert to torch tensors
    if isinstance(genes_normalized, np.ndarray):
        genes_tensor = torch.from_numpy(genes_normalized).to(device)
    else:
        genes_tensor = genes_normalized.to(device)
    
    # Convert fitness to tensor
    fitness_tensor = torch.from_numpy(fitness_np).to(device)
    
    # Update fitness condition statistics
    fitness_condition.update_statistics(
        fitness_tensor, 
        elite_ratio=params['elite_ratio']
    )
    
    # Compute fitness conditions for training (normalized fitness values)
    fitness_conditions = fitness_condition.evaluate(fitness_tensor)
    
    # Compute sample weights using scaler utility (handles normalization and floor)
    fitness_weights = fitness_scaler.compute_sample_weights(fitness_tensor)
    
    # Reshape weights for training
    fitness_weights = fitness_weights.reshape(-1, 1)
    
    # Clear numpy arrays
    del genes_np, fitness_np, genes_normalized
    _free_memory(device)
    
    log_dict['genes_mean'] = float(genes_tensor.mean().item())
    log_dict['genes_std'] = float(genes_tensor.std().item())
    log_dict['fitness_mean'] = float(fitness_tensor.mean().item())
    log_dict['fitness_std'] = float(fitness_tensor.std().item())
    log_dict['elite_fitness_mean'] = float(fitness_condition.elite_fitness_mean.item())
    log_dict['weights_mean'] = float(fitness_weights.mean().item())
    
    # =========================================================================
    # BUILD CONDITIONAL DIFFUSION MODEL
    # =========================================================================
    
    _log("Building conditional DDIM model...", verbose, 2)
    
    # Create conditional MLP (with 1 fitness condition)
    nn_model = ConditionalMLP(
        num_params=gene_length,
        num_hidden=params['num_hidden'],
        num_layers=params['num_layers'],
        num_conditions=1,  # Fitness condition
        activation='ReLU',
        dropout=0.0
    )
    
    # Create DDIM model
    ddim_model = DDIM(
        nn=nn_model,
        num_steps=params['num_diffusion_steps'],
        diff_range=params['diff_range'],
        alpha_schedule=params['alpha_schedule'],
        skip_connection=True,
        noise_level=1.0,
        sample_uniform=False,
        autoscaling=False,
        param_mean=0.0,
        param_std=1.0,
    )
    
    # Move to device
    ddim_model = ddim_model.to(device)
    
    log_dict['model_params'] = sum(p.numel() for p in ddim_model.parameters())
    
    # =========================================================================
    # TRAIN CONDITIONAL DIFFUSION MODEL
    # =========================================================================
    
    _log(f"Training conditional DDIM for {params['diff_max_epoch']} epochs...", verbose, 1)
    
    train_start = time.time()
    
    try:
        # Train with fitness conditions
        # Note: DDIM.fit() signature is fit(x, *conditions, weights=..., ...)
        loss_history = ddim_model.fit(
            genes_tensor,
            fitness_conditions,  # Conditions passed as positional args
            weights=fitness_weights,
            optimizer='Adam',
            max_epoch=params['diff_max_epoch'],
            lr=params['diff_lr'],
            weight_decay=params['diff_weight_decay'],
            batch_size=params['diff_batch_size'],
            scheduler='cosine'
        )
        
        final_loss = loss_history[-1] if loss_history else float('inf')
        
    except Exception as e:
        _log(f"Training error: {e}", verbose, 1)
        final_loss = float('inf')
        loss_history = []
    
    train_time = time.time() - train_start
    
    # Store loss
    build_in_params['loss'] = float(final_loss)
    
    log_dict['train_time'] = train_time
    log_dict['final_loss'] = float(final_loss)
    log_dict['loss_history'] = [float(l) for l in loss_history[-10:]]  # Last 10 losses
    
    _log(f"Training complete. Final loss: {final_loss:.6f}", verbose, 1)
    
    # Clear training tensors (keep fitness_tensor for condition sampling)
    del genes_tensor, fitness_weights
    _free_memory(device)
    
    # =========================================================================
    # SAMPLE NEW CANDIDATES WITH FITNESS CONDITIONING
    # =========================================================================
    
    _log(f"Sampling {number_of_candidate_genes} new candidates with fitness conditioning...", verbose, 1)
    
    sample_start = time.time()
    
    try:
        with torch.no_grad():
            ddim_model.eval()
            
            # Sample target fitness conditions (high fitness targets)
            target_fitness_conditions = fitness_condition.sample(number_of_candidate_genes)
            
            log_dict['target_fitness_mean'] = float(target_fitness_conditions.mean().item())
            log_dict['target_fitness_std'] = float(target_fitness_conditions.std().item())
            
            _log(f"Target fitness range: [{target_fitness_conditions.min():.4f}, {target_fitness_conditions.max():.4f}]", 
                 verbose, 2)
            
            # Sample from conditional diffusion model
            candidates_normalized = ddim_model.sample(
                shape=(gene_length,),
                num=number_of_candidate_genes,
                conditions=(target_fitness_conditions,),  # Condition on high fitness
                t_start=None
            )
            
    except Exception as e:
        _log(f"Sampling error: {e}", verbose, 1)
        import traceback
        traceback.print_exc()
        # Fallback: random samples
        candidates_normalized = torch.randn(
            number_of_candidate_genes, gene_length, device=device
        )
    
    sample_time = time.time() - sample_start
    log_dict['sample_time'] = sample_time
    
    # Clear fitness tensor
    del fitness_tensor
    _free_memory(device)
    
    # =========================================================================
    # APPLY MUTATION (Optional)
    # =========================================================================
    
    mutation_rate = params['mutation_rate']
    if mutation_rate > 0:
        _log(f"Applying mutation with rate {mutation_rate}", verbose, 2)
        
        # Add Gaussian noise proportional to mutation rate
        noise_scale = mutation_rate * params.get('diff_range', 1.0)
        noise = torch.randn_like(candidates_normalized) * noise_scale
        candidates_normalized = candidates_normalized + noise
    
    # =========================================================================
    # INVERSE TRANSFORM AND CLAMP
    # =========================================================================
    
    _log("Inverse transforming and clamping...", verbose, 2)
    
    # Move to CPU for inverse transform
    if candidates_normalized.device.type == 'cuda':
        candidates_normalized = candidates_normalized.cpu()
    
    # Inverse transform to original gene space
    candidates_denormalized = gene_scaler.inverse_transform(candidates_normalized)
    
    # Convert to numpy if tensor
    if isinstance(candidates_denormalized, torch.Tensor):
        candidates_np = candidates_denormalized.numpy()
    else:
        candidates_np = candidates_denormalized
    
    # Clamp to valid gene range and convert to list
    gene_min = getattr(EA_Class, 'geneMin', -1.0)
    lower_bound = 0.0 if gene_min == 0 else -1.0
    result = _clamp_to_list(candidates_np, lower_bound, 1.0)
    
    # =========================================================================
    # CLEANUP AND RETURN
    # =========================================================================
    
    total_time = time.time() - start_time
    log_dict['total_time'] = total_time
    log_dict['candidates_generated'] = len(result)
    
    # Clear GPU memory
    del ddim_model, nn_model, candidates_normalized
    _free_memory(device)
    
    _log(f"Generated {len(result)} candidates in {total_time:.2f}s", verbose, 1)
    
    return result


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_default_charles_params(gene_length: int) -> Dict[str, Any]:
    """
    Get default CHARLES parameters with auto-calculated values.
    
    Args:
        gene_length: Length of gene vectors
        
    Returns:
        Dict with default parameters
    """
    params = DEFAULT_CHARLES_PARAMS.copy()
    params['num_hidden'] = gene_length * 2
    return params


def validate_charles_params(params: Dict[str, Any], gene_length: int) -> Dict[str, Any]:
    """
    Validate and fill missing CHARLES parameters.
    
    Args:
        params: User-provided parameters
        gene_length: Length of gene vectors
        
    Returns:
        Validated parameters dict
    """
    validated = DEFAULT_CHARLES_PARAMS.copy()
    
    # Update with user params
    for key, value in params.items():
        if key in validated:
            validated[key] = value
    
    # Auto-calculate if needed
    if validated['num_hidden'] is None:
        validated['num_hidden'] = gene_length * 2
    
    return validated


# =============================================================================
# TESTING / EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    """Simple test to verify the function works."""
    
    class MockArgs:
        path = "./test_data"
        evolutionTarget = 1
        gpu = False
        agent_idx = 0
        epoch = 1
        populationSize = 100
    
    class MockEAClass:
        geneLength = 10
        geneFormat = 'json'
    
    # Create mock population with fitness correlation
    mock_population = []
    for i in range(50):
        gene = np.random.uniform(-1, 1, 10).tolist()
        # Fitness correlates with gene values (sum of genes)
        fitness = sum(gene) * 10 + np.random.random() * 5
        mock_population.append({
            'gene': gene,
            'fitnessScore': fitness
        })
    
    # Test the function
    build_params = {}
    
    try:
        candidates = createCandidateGene_CondEvo_CHARLES_Ben(
            args=MockArgs(),
            EA_Class=MockEAClass(),
            genePopulation=mock_population,
            number_of_candidate_genes=10,
            build_in_params=build_params,
            verbose=2
        )
        
        print(f"\nGenerated {len(candidates)} candidates")
        print(f"First candidate shape: {len(candidates[0])}")
        print(f"Loss: {build_params.get('loss', 'N/A')}")
        print(f"Log: {build_params.get('CHARLES_log', {})}")
        
        # Verify conditioning worked (candidates should have higher gene sums)
        candidate_sums = [sum(c) for c in candidates]
        population_sums = [sum(p['gene']) for p in mock_population]
        print(f"\nPopulation gene sum mean: {np.mean(population_sums):.4f}")
        print(f"Candidate gene sum mean: {np.mean(candidate_sums):.4f}")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()