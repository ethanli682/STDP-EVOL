
"""
Feed-Forward Predictor for Evolutionary Algorithms
===================================================

Standalone function that trains a neural network on gene-fitness pairs
and uses reverse optimization to suggest new candidate genes.

Key Features:
- Memory-efficient heap-based streaming from disk
- GPU-enabled with proper memory management
- Stateless: Trains fresh each call
- Returns multiple candidate genes

Usage:
    from FF_predictor import createCandidateGeneUsingFFPredictor
    
    candidates = createCandidateGeneUsingFFPredictor(
        args=args,
        EA_Class=ea_instance,
        genePopulation=population,  # or None for disk loading
        number_of_candidate_genes=10,
        build_in_params=params,
        verbose=2
    )
"""

import torch
import numpy as np
import heapq
import gc
import math
from typing import List, Dict, Any, Optional, Tuple
from Alternative_Approach.ReverseAutoRegressor import *


def _get_memory_info(device: torch.device) -> Dict[str, float]:
    """Get current memory usage in GB, respecting SLURM allocations."""
    import os
    info = {}
    try:
        if device.type == 'cuda' and torch.cuda.is_available():
            info['allocated_gb'] = torch.cuda.memory_allocated(device) / 1e9
            info['reserved_gb'] = torch.cuda.memory_reserved(device) / 1e9
            info['total_gb'] = torch.cuda.get_device_properties(device).total_memory / 1e9
            info['free_gb'] = info['total_gb'] - info['allocated_gb']
        else:
            # Check for SLURM allocation first
            slurm_mem_mb = None
            if 'SLURM_MEM_PER_NODE' in os.environ:
                slurm_mem_mb = int(os.environ['SLURM_MEM_PER_NODE'])
            elif 'SLURM_MEM' in os.environ:
                slurm_mem_mb = int(os.environ['SLURM_MEM'])
            elif 'SLURM_MEM_PER_CPU' in os.environ and 'SLURM_CPUS_ON_NODE' in os.environ:
                slurm_mem_mb = int(os.environ['SLURM_MEM_PER_CPU']) * int(os.environ['SLURM_CPUS_ON_NODE'])
            
            try:
                import psutil
                process = psutil.Process()
                
                # Get current process + children memory (RSS = Resident Set Size)
                mem_info = process.memory_info()
                children = process.children(recursive=True)
                total_rss = mem_info.rss
                for child in children:
                    try:
                        total_rss += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                
                info['allocated_gb'] = total_rss / 1e9
                
                if slurm_mem_mb:
                    # Use SLURM allocation as limit
                    info['total_gb'] = slurm_mem_mb / 1024  # MB to GB
                    info['free_gb'] = info['total_gb'] - info['allocated_gb']
                    info['source'] = 'SLURM'
                else:
                    # Fallback to system memory (less accurate for shared nodes)
                    mem = psutil.virtual_memory()
                    info['total_gb'] = mem.total / 1e9
                    info['free_gb'] = mem.available / 1e9
                    info['source'] = 'System'
            except ImportError:
                info['allocated_gb'] = 0
                info['total_gb'] = slurm_mem_mb / 1024 if slurm_mem_mb else 0
                info['free_gb'] = 0
                info['source'] = 'Unavailable'
    except Exception:
        info['allocated_gb'] = 0
        info['total_gb'] = 0
        info['free_gb'] = 0
        info['source'] = 'Error'
    return info


def _log_memory(device: torch.device, label: str, verbose: int = 1):
    """Log current memory state with label."""
    if verbose >= 1:
        info = _get_memory_info(device)
        if info['total_gb'] > 0:
            usage_pct = (info['allocated_gb'] / info['total_gb']) * 100
            source_tag = f" [{info.get('source', 'Unknown')}]" if 'source' in info else ""
            print(f"[MEM-{label}] {info['allocated_gb']:.2f}GB / {info['total_gb']:.2f}GB ({usage_pct:.1f}%){source_tag} - Free: {info['free_gb']:.2f}GB", flush=True)
        else:
            print(f"[MEM-{label}] Memory info unavailable", flush=True)


def _check_memory_pressure(device: torch.device, threshold: float = 0.85) -> bool:
    """Check if memory usage exceeds threshold (0-1)."""
    info = _get_memory_info(device)
    if info['total_gb'] > 0:
        usage = info['allocated_gb'] / info['total_gb']
        return usage > threshold
    return False

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _get_default_hidden_sizes(input_dim: int, available_memory_gb: float = 8.0) -> List[int]:
    """Calculate dynamic hidden layer sizes based on input dimension AND available memory.
    
    For SLURM: assume conservative memory available (8GB default).
    Each parameter in layer roughly costs: hidden_size * input_dim * 4 bytes (float32)
    """
    # Memory budget: 200MB per model for weights + activations
    max_params = int((available_memory_gb * 1e9) / (4 * 10))  # 10x safety factor
    
    # Estimate based on typical 3-layer MLP and keep a tapered architecture.
    if input_dim <= 500:
        h1 = min(int(64 * input_dim * np.log2(input_dim + 1)), max_params // max(input_dim * 2, 1))
        h1 = min(max(h1, 64), 1000)
    elif input_dim <= 1000:
        h1 = min(512, max_params // max(input_dim * 3, 1))
        h1 = max(h1, 128)
    elif input_dim <= 5000:
        h1 = min(256, max_params // max(input_dim * 4, 1))
        h1 = max(h1, 64)
    else:
        # Huge genes (>5000): keep first layer compact and strictly tapered.
        h1 = min(192, max_params // max(input_dim * 5, 1))
        h1 = max(h1, 32)

    h2 = max(16, min(h1 // 2, 512))
    h3 = max(8, min(h2 // 2, 256))

    return [h1, h2, h3]


def _calculate_memory_based_limit(
    gene_length: int,
    device: torch.device,
    safety_factor: float = 0.70
) -> int:
    """
    Calculate maximum number of genes that fit in available memory.
    
    Args:
        gene_length: Length of each gene vector
        device: torch device (cuda or cpu)
        safety_factor: Fraction of available memory to use (default 70%)
        
    Returns:
        Maximum number of genes that fit in memory
    """
    try:
        if device.type == 'cuda' and torch.cuda.is_available():
            # GPU memory
            total_mem = torch.cuda.get_device_properties(0).total_memory
            allocated_mem = torch.cuda.memory_allocated(0)
            available_mem = total_mem - allocated_mem
        else:
            # CPU memory (use psutil if available)
            try:
                import psutil
                available_mem = psutil.virtual_memory().available
            except ImportError:
                # Fallback: assume 4GB available
                available_mem = 4 * 1024 * 1024 * 1024
        
        # Apply safety factor
        usable_mem = available_mem * safety_factor
        
        # Calculate bytes per gene record
        # gene (float32) + fitness (float32) + overhead
        bytes_per_gene = (gene_length * 4) + 4 + 100  # 100 bytes overhead for dict/heap
        
        max_genes = int(usable_mem / bytes_per_gene)
        
        # Reasonable bounds
        max_genes = max(100, min(max_genes, 1_000_000))
        
        return max_genes
        
    except Exception:
        # Fallback: conservative estimate
        return 10000


def _calculate_safe_chunk_size(
    heap_capacity: int,
    gene_length: int,
    device: torch.device,
    base_chunk_size: int = 2000  # Reduced for SLURM
) -> int:
    """
    Calculate chunk size that won't cause OOM given heap capacity.
    
    Args:
        heap_capacity: Maximum heap size (number of genes)
        gene_length: Length of each gene vector
        device: torch device
        base_chunk_size: Starting chunk size to reduce from (default: 2000 for SLURM)
        
    Returns:
        Safe chunk size
    """
    try:
        if device.type == 'cuda' and torch.cuda.is_available():
            total_mem = torch.cuda.get_device_properties(0).total_memory
            allocated_mem = torch.cuda.memory_allocated(0)
            available_mem = total_mem - allocated_mem
        else:
            try:
                import psutil
                available_mem = psutil.virtual_memory().available
            except ImportError:
                available_mem = 4 * 1024 * 1024 * 1024
        
        # Memory already reserved for heap
        bytes_per_gene = (gene_length * 4) + 4 + 100
        heap_memory = heap_capacity * bytes_per_gene
        
        # Remaining memory for chunks (use 50% of remaining)
        remaining_mem = (available_mem * 0.70) - heap_memory
        chunk_memory = remaining_mem * 0.50
        
        if chunk_memory <= 0:
            return 100  # Minimum
        
        safe_chunk_size = int(chunk_memory / bytes_per_gene)
        safe_chunk_size = max(100, min(safe_chunk_size, base_chunk_size))
        
        return safe_chunk_size
        
    except Exception:
        return min(1000, base_chunk_size)  # More conservative for SLURM


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def createCandidateGeneUsingFFPredictor_MinHeap(
    args=None,
    EA_Class=None,
    genePopulation: Optional[List[Dict[str, Any]]] = None,
    number_of_candidate_genes: Optional[int] = None,
    build_in_params: Optional[Dict] = None,
    verbose: Optional[int] = None
) -> List[List[float]]:
    """
    Generate candidate genes using a Feed-Forward Neural Network Predictor.
    
    Trains a neural network on gene-fitness pairs and uses reverse optimization
    (gradient descent on inputs) to suggest genes that should achieve high fitness.
    
    Args:
        args: Argument object with:
            - path: Save path for loading gene history
            - agent_idx: Agent identifier
            - agent_counter: Current generation counter
            - numOFgenerations: Total generations
            - evolutionTarget: 1 for maximize, -1 for minimize
            - populationSize: Max genes to load from disk
            - gpu: Whether to use GPU (optional)
        EA_Class: EA class instance with:
            - geneLength: Length of gene vectors
            - geneMin: Minimum gene value (0 or -1)
            - savePath: Path for saving metadata
            - geneFormat: Gene format string
        genePopulation: Optional list of dicts with 'gene' and 'fitnessScore' keys.
                       If None, loads from disk via ChunkedGeneHistoryLoader.
        number_of_candidate_genes: Number of candidate genes to generate (required)
        build_in_params: Configuration dict. Will create/update keys:
            - 'RevFF_predictor': Configuration parameters
        verbose: Verbosity level (0/None=silent, 1=progress, 2=detailed, 3=debug)
        
    Returns:
        List[List[float]]: List of candidate genes, each as a list of floats
        
    Raises:
        ValueError: If required parameters are None
        RuntimeError: If training fails or no genes loaded
        
    Configuration (build_in_params['RevFF_predictor']):
        - use_log_scaler (bool): Use ChunkedLogZScoreScaler vs ChunkedRankScaler. Default: True
        - top_k_percentile (float): Top percentile to load from disk. Default: 40
        - ANN_Layers_size (List[int]): NN hidden layer sizes. Default: auto-calculated
        - dropout_rate (float): NN dropout rate. Default: 0.2
        - epochs (int): Training epochs. Default: 100
        - batch_size (int): Mini-batch size. Default: 32
        - lr (float): Learning rate. Default: 0.001
    """
    
    # =========================================================================
    # STEP 1: VALIDATION & SETUP
    # =========================================================================
    
    # Validate required parameters
    if args is None:
        raise ValueError("args parameter is required")
    if EA_Class is None:
        raise ValueError("EA_Class parameter is required")
    if number_of_candidate_genes is None:
        raise ValueError("number_of_candidate_genes parameter is required")
    if build_in_params is None:
        raise ValueError("build_in_params parameter is required (can be empty dict)")
    
    # Setup verbosity
    verbose = verbose if verbose is not None else 0
    
    # Setup device
    use_gpu = getattr(args, 'gpu', None)
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    if verbose >= 1:
        print(f"\n{'='*60}", flush=True)
        print(f"[FFPredictor] Starting candidate generation", flush=True)
        print(f"[FFPredictor] Device: {device}", flush=True)
        print(f"[FFPredictor] Candidates to generate: {number_of_candidate_genes}", flush=True)
    
    _log_memory(device, "START", verbose)
    
    # Extract EA parameters
    gene_length = EA_Class.geneLength
    gene_min = getattr(EA_Class, 'geneMin', None)
    lower_bound = 0 if gene_min == 0 else -1
    gene_format = getattr(EA_Class, 'geneFormat', None)
    
    # Extract args parameters
    agent_idx = getattr(args, 'agent_idx', None)
    agent_counter = getattr(args, 'agent_counter', None) or 0
    num_generations = getattr(args, 'numOFgenerations', None) or 1000
    evolution_target = getattr(args, 'evolutionTarget', None) or 1
    population_size = getattr(args, 'populationSize', None) or 10000
    
    if verbose >= 2:
        print(f"[FFPredictor] Gene length: {gene_length}", flush=True)
        print(f"[FFPredictor] Gene bounds: [{lower_bound}, 1]", flush=True)
        print(f"[FFPredictor] Evolution target: {'maximize' if evolution_target == 1 else 'minimize'}", flush=True)
    
    # =========================================================================
    # STEP 2: INITIALIZE CONFIGURATION
    # =========================================================================
    
    if 'RevFF_predictor' not in build_in_params:
        build_in_params['RevFF_predictor'] = {}
    
    config = build_in_params['RevFF_predictor']
    build_in_params['loss'] = None  # Placeholder for training loss history
    
    # Set defaults
    defaults = {
        'use_log_scaler': True,  # Default to LogZScore like working code
        'top_k_percentile': 25,
        'ANN_Layers_size': None,  # Auto-calculate
        'dropout_rate': 0.2,
        'epochs': 100,
        'batch_size': 16,  # Reduced from 32 for SLURM
        'learning_rate': 0.001,
        'early_stop_threshold': 0.05,
        'allowed_architectures': ['ANN'],  # Architecture types to train
        'max_models_to_train': 1,  # Maximum number of models to train
    }
    
    for key, value in defaults.items():
        if key not in config:
            config[key] = value
    
    # Calculate hidden sizes if not provided
    if config['ANN_Layers_size'] is None:
        # Estimate available memory for layer sizing (SLURM-aware)
        try:
            if device.type == 'cuda':
                available_mem_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
            else:
                available_mem_gb = 8.0  # Conservative for SLURM CPU
        except:
            available_mem_gb = 8.0
        
        config['ANN_Layers_size'] = _get_default_hidden_sizes(gene_length, available_mem_gb)
    
    if verbose >= 2:
        print(f"[FFPredictor] Hidden sizes: {config['ANN_Layers_size']}", flush=True)
        print(f"[FFPredictor] Use log scaler: {config['use_log_scaler']}", flush=True)

    predictor = None
    val_loss = None

    # Detect block-wise co-evolution mode (EA_Class.coevo_groups built from YAML tags)
    coevo_groups = getattr(EA_Class, 'coevo_groups', None)
    use_block_wise = coevo_groups is not None and len(coevo_groups) > 1

    def _create_predictor_with_shrink(base_sizes):
        """Try to build predictor, shrinking layers by 5% on failure with memory checks."""
        sizes = list(base_sizes) if base_sizes else []
        if len(sizes) == 0:
            raise ValueError("ANN_Layers_size must be a non-empty list")
        
        # Sanity check: pre-shrink dangerously large sizes for very large genes.
        # Keep tapering by applying per-layer caps instead of a flat cap.
        def _tapered_caps(base_cap: int, n_layers: int) -> List[int]:
            return [max(8, base_cap // (2 ** i)) for i in range(n_layers)]

        if gene_length > 5000 and any(s > 192 for s in sizes):
            if verbose >= 1:
                print(f"[FFPredictor] WARNING: Hidden sizes {sizes} too large for gene_length={gene_length}. "
                      f"Pre-shrinking to safe values.", flush=True)
            caps = _tapered_caps(base_cap=192, n_layers=len(sizes))
            sizes = [min(s, cap) for s, cap in zip(sizes, caps)]
        elif gene_length > 1000 and any(s > 512 for s in sizes):
            if verbose >= 1:
                print(f"[FFPredictor] WARNING: Hidden sizes {sizes} may be too large for gene_length={gene_length}. "
                      f"Pre-shrinking.", flush=True)
            caps = _tapered_caps(base_cap=512, n_layers=len(sizes))
            sizes = [min(s, cap) for s, cap in zip(sizes, caps)]

        def estimate_model_memory_mb(hidden_sizes):
            """Rough estimate of model memory in MB."""
            total_params = gene_length * hidden_sizes[0]
            for i in range(len(hidden_sizes) - 1):
                total_params += hidden_sizes[i] * hidden_sizes[i+1]
            total_params += hidden_sizes[-1]
            # 4 bytes per param (float32) + activations + gradients
            return (total_params * 4 * 3) / 1e6

        while True:
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Check available memory BEFORE attempting
            mem_info = _get_memory_info(device)
            estimated_mb = estimate_model_memory_mb(sizes)
            available_mb = mem_info.get('free_gb', 0) * 1024
            
            if estimated_mb > available_mb * 0.5:  # Need at least 50% of free memory
                if verbose >= 1:
                    print(f"[FFPredictor] Insufficient memory: need ~{estimated_mb:.0f}MB, "
                          f"available {available_mb:.0f}MB. Pre-shrinking layers.", flush=True)
                next_sizes = [max(int(s * 0.80), s - 1) for s in sizes]  # Very aggressive 20% cut
                if any(n < 5 for n in next_sizes):
                    raise RuntimeError(
                        f"FFPredictor init failed: insufficient memory even with minimal layers {next_sizes}"
                    )
                sizes = next_sizes
                continue
            
            try:
                if verbose >= 2:
                    print(f"[FFPredictor] Attempting model creation: {sizes}, est. {estimated_mb:.0f}MB", flush=True)
                
                model = AutoEnsemblePredictor(
                    input_size=gene_length,
                    hidden_sizes=sizes,
                    dropout_rate=config['dropout_rate'],
                    device=str(device)
                )
                
                if verbose >= 1:
                    print(f"[FFPredictor] Model created successfully: {sizes}", flush=True)
                return model, sizes
                
            except (torch.cuda.OutOfMemoryError, RuntimeError, MemoryError) as e:
                error_msg = str(e).lower()
                # Catch various OOM messages
                is_oom = any(x in error_msg for x in [
                    'out of memory', 'oom', 'can\'t allocate', 'cannot allocate', 
                    'allocation', 'memory error'
                ]) or isinstance(e, MemoryError)
                
                if is_oom:
                    # Very aggressive shrink for OOM (10%)
                    next_sizes = [max(int(s * 0.90), s - 1) for s in sizes]
                else:
                    # Less aggressive for other errors (2%)
                    next_sizes = [max(int(s * 0.98), s - 1) for s in sizes]
                
                if any(n < 5 for n in next_sizes):
                    raise RuntimeError(
                        f"FFPredictor init failed after shrinking to {next_sizes}: {e}"
                    ) from e
                
                if verbose >= 1:
                    oom_tag = " [OOM]" if is_oom else ""
                    print(f"[FFPredictor]{oom_tag} Shrinking layers {sizes} -> {next_sizes} after error: {e}", flush=True)
                
                sizes = next_sizes
                
                # Force cleanup after OOM
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            except Exception as e:
                # Catch-all for unexpected errors
                next_sizes = [max(int(s * 0.95), s - 1) for s in sizes]
                if any(n < 5 for n in next_sizes):
                    raise RuntimeError(
                        f"FFPredictor init failed with unexpected error at {next_sizes}: {e}"
                    ) from e
                if verbose >= 1:
                    print(f"[FFPredictor] Unexpected error, shrinking {sizes} -> {next_sizes}: {e}", flush=True)
                sizes = next_sizes
    
    # =========================================================================
    # STEP 3: CREATE SCALER
    # =========================================================================
    
    from GA_utils_scalers import ChunkedRankScaler, ChunkedLogZScoreScaler, ChunkedPerDimensionScaler
    
    # Calculate adaptive emphasis multiplier based on progress
    progress = agent_counter / num_generations if num_generations > 0 else 0.0
    if progress < 0.3:
        emphasis_multiplier = 0.7
    elif progress > 0.7:
        emphasis_multiplier = 1.5
    else:
        emphasis_multiplier = 0.7 + progress
    
    emphasis_factor = 1.0 + emphasis_multiplier * torch.rand(1).item()
    
    if config['use_log_scaler']:
        scaler = ChunkedLogZScoreScaler(
            evolution_target=evolution_target,
            emphasis_factor=emphasis_factor,
            min_weight=0.01,
            device=device
        )
        if verbose >= 2:
            print(f"[FFPredictor] Using ChunkedLogZScoreScaler (emphasis={emphasis_factor:.2f})", flush=True)
    else:
        scaler = ChunkedRankScaler(
            evolution_target=evolution_target,
            emphasis_factor=emphasis_factor,
            min_weight=0.01,
            device=device
        )
        if verbose >= 2:
            print(f"[FFPredictor] Using ChunkedRankScaler (emphasis={emphasis_factor:.2f})", flush=True)

    # Initialize per-dimension gene scaler for features (z-score per dimension)
    gene_scaler = ChunkedPerDimensionScaler(
        gene_length=gene_length,
        device=device,
        weight_mode='uniform'
    )
    
    # =========================================================================
    # STEP 4: PREPARE TRAINING DATA
    # =========================================================================
    
    training_genes = None  # Will be torch.Tensor
    training_fitness = None  # Will be torch.Tensor
    
    if genePopulation is not None:
        # -----------------------------------------------------------------
        # RAM HIT: Use provided population directly
        # -----------------------------------------------------------------
        if verbose >= 1:
            print(f"\n[FFPredictor] RAM HIT: Using provided population ({len(genePopulation)} genes)", flush=True)
        
        # Count valid genes first
        valid_count = 0
        for gene_dict in genePopulation:
            if (isinstance(gene_dict, dict) 
                and 'gene' in gene_dict 
                and 'fitnessScore' in gene_dict
                and gene_dict['gene'] is not None
                and isinstance(gene_dict['fitnessScore'], (int, float))
                and not np.isnan(gene_dict['fitnessScore'])):
                valid_count += 1
        
        if valid_count == 0:
            raise RuntimeError("[FFPredictor] No valid genes in provided population")
        
        if verbose >= 2:
            print(f"[FFPredictor] Valid genes: {valid_count}/{len(genePopulation)}", flush=True)
        
        # Pre-allocate tensors to avoid memory duplication
        training_genes = torch.empty((valid_count, gene_length), dtype=torch.float32, device=device)
        training_fitness = torch.empty(valid_count, dtype=torch.float32, device=device)
        
        # Fill tensors directly without intermediate lists
        idx = 0
        for gene_dict in genePopulation:
            if (isinstance(gene_dict, dict) 
                and 'gene' in gene_dict 
                and 'fitnessScore' in gene_dict
                and gene_dict['gene'] is not None
                and isinstance(gene_dict['fitnessScore'], (int, float))
                and not np.isnan(gene_dict['fitnessScore'])):
                
                # Direct assignment to pre-allocated tensor
                gene_data = gene_dict['gene']
                if isinstance(gene_data, torch.Tensor):
                    training_genes[idx] = gene_data.to(device)
                elif isinstance(gene_data, np.ndarray):
                    training_genes[idx] = torch.from_numpy(gene_data).to(device)
                else:
                    training_genes[idx] = torch.tensor(gene_data, dtype=torch.float32, device=device)
                
                training_fitness[idx] = gene_dict['fitnessScore']
                idx += 1
        
        if verbose >= 1:
            print(f"[FFPredictor] RAM HIT: Loaded {valid_count} genes to {device}", flush=True)

        # Fit gene scaler and normalize features (RAM path)
        try:
            gene_scaler.collect_phase_1_statistics(training_genes)
            gene_scaler._finalize_pass1()
            gene_scaler.collect_phase_2_statistics(training_genes)
            gene_scaler.finalize()
            training_genes = gene_scaler.transform(training_genes)
            if verbose >= 2:
                print(f"[FFPredictor] Applied ChunkedPerDimensionScaler normalization (RAM path)", flush=True)
        except Exception as e:
            if verbose >= 1:
                print(f"[FFPredictor WARNING] Gene scaler fitting failed (RAM path): {e}", flush=True)

        if not use_block_wise:
            predictor, used_hidden_sizes = _create_predictor_with_shrink(config['ANN_Layers_size'])
            config['ANN_Layers_size'] = used_hidden_sizes

    else:
        # -----------------------------------------------------------------
        # DISK HIT: Stream from history using heap
        # -----------------------------------------------------------------
        if verbose >= 1:
            print(f"\n[FFPredictor] DISK HIT: Loading from history files via heap", flush=True)
        
        from GA_utils_history_loader import ChunkedGeneHistoryLoader
        
        # Create loader
        loader = ChunkedGeneHistoryLoader(
            savePath=args.path,
            agent_id=agent_idx if agent_idx is not None else agent_counter,
            geneFormat=gene_format,
            shuffle=True,
            shuffle_level='both',
        )
        loader.clear_cache()

        # ANN initialization before memory availability checks (full-gene mode only)
        if not use_block_wise:
            predictor, used_hidden_sizes = _create_predictor_with_shrink(config['ANN_Layers_size'])
            config['ANN_Layers_size'] = used_hidden_sizes
            if verbose >= 1:
                print(f"[FFPredictor] Architecture: {config['ANN_Layers_size']}", flush=True)
                print(f"[FFPredictor] Epochs: {config['epochs']}, Batch: {config['batch_size']}, "
                    f"LR: {config['learning_rate']}", flush=True)
        
        # Calculate heap capacity
        memory_limit = _calculate_memory_based_limit(gene_length, device, safety_factor=0.7)
        heap_capacity = min(population_size, memory_limit)
        
        if verbose >= 1:
            print(f"[FFPredictor] Heap capacity: {heap_capacity} "
                  f"(populationSize={population_size}, memoryLimit={memory_limit})", flush=True)
        
        # Calculate safe chunk size (50% of dynamic, accounting for heap)
        try:
            dynamic_chunk = loader.calculate_chunk_size(
                safety_margin=0.5,  # More conservative for SLURM
                chunk_allocation=0.10,  # Use only 10% of memory per chunk
                verbose=(verbose >= 3)
            )
            base_chunk = int(dynamic_chunk * 0.50)
        except Exception:
            base_chunk = 2000  # Reduced from 10000
        
        chunk_size = _calculate_safe_chunk_size(heap_capacity, gene_length, device, base_chunk)
        # Hard cap for SLURM stability
        chunk_size = min(chunk_size, 2000)
        
        if verbose >= 2:
            print(f"[FFPredictor] Using chunk size: {chunk_size}", flush=True)
        
        # Setup heap
        # For maximization (evolutionTarget==1): use min-heap with negated fitness
        #   → smallest -fitness = largest fitness stays in heap
        # For minimization (evolutionTarget==-1): use min-heap with positive fitness
        #   → smallest fitness stays in heap
        maximize = (evolution_target == 1)
        gene_heap = []  # (priority, gene_list, fitness)
        
        # Top K percentile for loader
        top_k_percentile = config['top_k_percentile']
        loader_percentile = 100 - top_k_percentile  # Convert: 40% → 60th percentile
        
        if verbose >= 1:
            print(f"[FFPredictor] Loading top {top_k_percentile}% "
                  f"(>= {loader_percentile}th percentile)", flush=True)
        
        _log_memory(device, "BEFORE-STREAMING", verbose)
        
        total_processed = 0
        chunk_count = 0
        heap_tiebreaker = 0
        
        for chunk, is_done in loader.loadTopKPercentileGeneHistoryAsListOfDicts(
            args=args,
            top_k_percentile=loader_percentile,
            maximize=maximize,
            chunk_size=chunk_size,
            fitness_scaler=scaler,  # Let our scaler handle it
            use_fitness_scaler_weights=False,
            gene_scaler=gene_scaler,
            use_gene_scaler_weights=True,
            n_feature_clusters='adaptive',
            allEpochs=True,
            min_datapoints=min(int(population_size * 0.05), 300),
            fit_fitness_scaler_on_top_k=False,
            fit_gene_scaler_on_top_k=True,
            #balance_clusters=True # next test
        ):
            chunk_count += 1
            
            # Emergency memory check
            if _check_memory_pressure(device, threshold=0.90):
                print(f"[FFPredictor CRITICAL] Memory pressure >90% at chunk {chunk_count}, forcing cleanup", flush=True)
                _log_memory(device, f"BEFORE-EMERGENCY-GC", verbose)
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                _log_memory(device, f"AFTER-EMERGENCY-GC", verbose)
            
            if verbose >= 3:
                _log_memory(device, f"HEAP-CHUNK-{chunk_count}", verbose)
            
            for gene_dict in chunk:
                gene_data = gene_dict.get('gene')
                fitness = gene_dict.get('fitnessScore')
                
                if gene_data is None or fitness is None or np.isnan(fitness):
                    continue
                
                total_processed += 1
                
                # Calculate priority for heap
                if maximize:
                    priority = -fitness  # Min-heap: smallest -fitness = highest fitness
                else:
                    priority = fitness   # Min-heap: smallest fitness
                
                entry = (priority, heap_tiebreaker, gene_data, fitness)
                heap_tiebreaker += 1

                if len(gene_heap) < heap_capacity:
                    # Heap not full: just push
                    heapq.heappush(gene_heap, entry)
                else:
                    # Heap full: replace if better
                    if priority < gene_heap[0][0]:
                        heapq.heapreplace(gene_heap, entry)
            
            if verbose >= 2 and total_processed % 50000 == 0:
                print(f"[FFPredictor] Processed {total_processed} genes, heap size {len(gene_heap)}", flush=True)
            
            if is_done:
                break
        
        if not gene_heap:
            raise RuntimeError("[FFPredictor] No genes loaded from disk")
        
        _log_memory(device, "HEAP-LOADED", verbose)
        
        if verbose >= 1:
            print(f"[FFPredictor] DISK HIT: Processed {total_processed} genes, "
                  f"kept top {len(gene_heap)} in heap", flush=True)
        
        # Convert heap to tensors (heap is now our training data)
        heap_size = len(gene_heap)
        training_genes = torch.empty((heap_size, gene_length), dtype=torch.float32, device=device)
        training_fitness = torch.empty(heap_size, dtype=torch.float32, device=device)
        
        for idx, (_, _, gene_data, fitness) in enumerate(gene_heap):
            if isinstance(gene_data, torch.Tensor):
                training_genes[idx] = gene_data.to(device)
            elif isinstance(gene_data, np.ndarray):
                training_genes[idx] = torch.from_numpy(gene_data).to(device)
            else:
                training_genes[idx] = torch.tensor(gene_data, dtype=torch.float32, device=device)
            training_fitness[idx] = fitness
        
        # Clear heap to free memory
        del gene_heap
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        if verbose >= 1:
            fitness_min = training_fitness.min().item()
            fitness_max = training_fitness.max().item()
            print(f"[FFPredictor] Training data: {heap_size} genes, "
                  f"fitness range [{fitness_min:.4f}, {fitness_max:.4f}]", flush=True)
    
    # =========================================================================
    # STEP 5: SCALE FITNESS & NORMALIZE (shared for all paths)
    # =========================================================================

    _log_memory(device, "BEFORE-TRAINING", verbose)

    # Scale fitness values — shared signal regardless of grouping
    training_targets = scaler.scale_chunk(training_fitness)
    # Normalize training genes per-dimension using gene_scaler (loader collected stats)
    training_genes = gene_scaler.transform(training_genes)

    if verbose >= 2:
        print(f"[FFPredictor] Scaled targets range: "
              f"[{training_targets.min().item():.4f}, {training_targets.max().item():.4f}]", flush=True)

    # Adaptive target z-score (shared across both paths)
    tmp_stage = 0.15
    if np.random.rand() < tmp_stage:
        stage = f"Exploration boost ({tmp_stage*100:.0f}%)"
        base_target_z = 2.5 + torch.rand(1).item() * 1.0
    else:
        stage = f"Greedy phase ({(1-tmp_stage)*100:.0f}%)"
        base_target_z = 4.0 + torch.rand(1).item() * 1.0

    target_name = 'minimize' if evolution_target == -1 else 'maximize'

    if verbose >= 1:
        print(f"[FFPredictor] {stage} (progress={progress:.2%}): "
              f"target_z={base_target_z:.2f} ({target_name})", flush=True)

    # =========================================================================
    # STEP 6: TRAIN & GENERATE — block-wise or full-gene
    # =========================================================================

    _log_memory(device, "BEFORE-GENERATION", verbose)

    if use_block_wise:
        # -----------------------------------------------------------------
        # BLOCK-WISE CO-EVOLUTION: K small models, one per coevo_group
        # -----------------------------------------------------------------
        group_order = sorted(coevo_groups.keys(), key=lambda g: coevo_groups[g]['start'])
        K = len(group_order)
        if verbose >= 1:
            print(f"[FFPredictor-BLOCK] {K} co-evolution groups → training {K} small models", flush=True)

        group_slices    = {}   # {gname: List[ndarray(dim_g,)]}
        val_loss_by_group = {}

        for gname in group_order:
            ginfo   = coevo_groups[gname]
            g_start = ginfo['start']
            g_end   = ginfo['end']
            dim_g   = ginfo['dim']

            # Slice normalized genes for this group
            slice_g = training_genes[:, g_start:g_end]   # [N, dim_g]

            # Create per-group predictor (smaller network for smaller dim)
            sizes_g = _get_default_hidden_sizes(dim_g)
            try:
                pred_g = AutoEnsemblePredictor(
                    input_size=dim_g,
                    hidden_sizes=sizes_g,
                    dropout_rate=config['dropout_rate'],
                    device=str(device)
                )
            except Exception as e:
                raise RuntimeError(
                    f"[FFPredictor-BLOCK] Failed to create model for group '{gname}' "
                    f"(dim={dim_g}): {e}"
                )

            if verbose >= 1:
                print(f"[FFPredictor-BLOCK] Group '{gname}' "
                      f"(dim={dim_g}, layers={sizes_g}): training...", flush=True)

            # Train on the group's gene slice with the shared fitness targets
            try:
                _, g_loss = pred_g.train_model(
                    slice_g,
                    training_targets,
                    epochs=config['epochs'],
                    batch_size=config['batch_size'],
                    allowed_architectures=config['allowed_architectures'],
                    max_models_to_train=config['max_models_to_train'],
                    early_stop_threshold=config['early_stop_threshold'],
                    learning_rate=config['learning_rate'],
                    verbose=(verbose >= 2)
                )
                g_loss_val = g_loss if isinstance(g_loss, float) else g_loss.item()
                val_loss_by_group[gname] = g_loss_val
                if verbose >= 1:
                    print(f"[FFPredictor-BLOCK] Group '{gname}': loss={g_loss_val:.6f}", flush=True)
            except Exception as e:
                del pred_g
                raise RuntimeError(
                    f"[FFPredictor-BLOCK] Training failed for group '{gname}': {e}"
                )

            # Generate slice-suggestions via reverse optimisation on dim_g
            try:
                with torch.enable_grad():
                    sug_g = pred_g.suggest_features_for_fitness(
                        target_fitness=base_target_z,
                        num_features=dim_g,
                        num_suggestions=number_of_candidate_genes,
                        max_iterations=500,
                        verbose=(verbose >= 3)
                    )
                # Defensive: pad with last entry if fewer suggestions were returned
                while len(sug_g) < number_of_candidate_genes:
                    sug_g.append(sug_g[-1].copy())
                group_slices[gname] = sug_g
            except Exception as e:
                del pred_g
                raise RuntimeError(
                    f"[FFPredictor-BLOCK] Generation failed for group '{gname}': {e}"
                )

            del pred_g, slice_g
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        # Free shared training tensors
        del training_genes, training_fitness, training_targets
        gc.collect()

        # Assemble: index-pair slice[i] from each group → full candidate[i]
        assembled = np.zeros((number_of_candidate_genes, gene_length), dtype=np.float32)
        for gname in group_order:
            g_start = coevo_groups[gname]['start']
            g_end   = coevo_groups[gname]['end']
            for i in range(number_of_candidate_genes):
                assembled[i, g_start:g_end] = group_slices[gname][i]

        candidates = gene_scaler.inverse_transform(assembled)
        candidates = np.clip(candidates, lower_bound, 1.0).tolist()

        mean_loss = float(np.mean(list(val_loss_by_group.values()))) if val_loss_by_group else 0.0
        if verbose >= 1:
            print(f"[FFPredictor-BLOCK] Assembly complete: {len(candidates)} candidates "
                  f"(mean group loss={mean_loss:.6f})", flush=True)
        build_in_params['loss'] = mean_loss

    else:
        # -----------------------------------------------------------------
        # FULL-GENE PATH: single model (existing behaviour, unchanged)
        # -----------------------------------------------------------------
        if verbose >= 1:
            print(f"\n[FFPredictor] Training neural network...", flush=True)

        try:
            _, val_loss = predictor.train_model(
                training_genes,
                training_targets,
                epochs=config['epochs'],
                batch_size=config['batch_size'],
                allowed_architectures=config['allowed_architectures'],
                max_models_to_train=config['max_models_to_train'],
                early_stop_threshold=config['early_stop_threshold'],
                learning_rate=config['learning_rate'],
                verbose=(verbose >= 2)
            )

            _log_memory(device, "AFTER-TRAINING", verbose)

            final_loss = val_loss if isinstance(val_loss, float) else val_loss.item()
            build_in_params['loss'] = final_loss
            if verbose >= 1:
                print(f"[FFPredictor] Training complete. Final loss: {final_loss:.6f}", flush=True)

            if final_loss > 0.5:
                print(f"[FFPredictor WARNING] High validation loss ({final_loss:.3f}, flush=True). "
                      f"Model may not generalize well.", flush=True)

        except Exception as e:
            _log_memory(device, "TRAINING-FAILED", verbose)
            raise RuntimeError(f"[FFPredictor] Training failed: {e}")

        # Free training data memory
        del training_genes, training_fitness, training_targets
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        if verbose >= 1:
            print(f"\n[FFPredictor] Generating {number_of_candidate_genes} candidate genes...", flush=True)

        try:
            with torch.enable_grad():
                suggested_genes = predictor.suggest_features_for_fitness(
                    target_fitness=base_target_z,
                    num_features=gene_length,
                    num_suggestions=number_of_candidate_genes,
                    max_iterations=500,
                    verbose=(verbose >= 3)
                )
            _log_memory(device, "AFTER-GENERATION", verbose)
        except Exception as e:
            print(f"[FFPredictor ERROR] Candidate generation failed: {e}", flush=True)
            _log_memory(device, "GENERATION-FAILED", verbose)
            raise

        # Convert, inverse-transform to original scale, and clip
        candidates = gene_scaler.inverse_transform(suggested_genes)
        candidates = np.clip(candidates, lower_bound, 1.0).tolist()

        if verbose >= 1:
            print(f"[FFPredictor] ✓ Generated {len(candidates)} candidate genes", flush=True)

    # =========================================================================
    # STEP 7: CLEANUP AND RETURN
    # =========================================================================

    if not use_block_wise and predictor is not None:
        del predictor
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    _log_memory(device, "END", verbose)

    if verbose >= 1:
        print(f"{'='*60}\n", flush=True)

    return candidates


# # =============================================================================
# # TESTING / EXAMPLE USAGE
# # =============================================================================

# if __name__ == "__main__":
#     """
#     Example usage and basic testing.
#     """
    
#     print("FFPredictor Module Test")
#     print("=" * 60)
    
#     # Create mock args
#     class MockArgs:
#         path = "./test_data"
#         agent_idx = 0
#         agent_counter = 50
#         numOFgenerations = 100
#         evolutionTarget = 1
#         populationSize = 1000
#         gpu = False
#         epoch = 0
    
#     # Create mock EA_Class
#     class MockEAClass:
#         geneLength = 10
#         geneMin = -1
#         savePath = "./test_output"
#         geneFormat = None
    
#     args = MockArgs()
#     ea = MockEAClass()
    
#     # Create synthetic population
#     np.random.seed(42)
#     n_samples = 500
    
#     population = []
#     for i in range(n_samples):
#         gene = np.random.uniform(-1, 1, ea.geneLength).tolist()
#         # Synthetic fitness function
#         fitness = sum(g**2 for g in gene) + 0.5 * sum(gene[:3]) + np.random.normal(0, 0.1)
#         population.append({
#             'gene': gene,
#             'fitnessScore': fitness
#         })
    
#     print(f"Created synthetic population: {len(population)} genes")
#     print(f"Gene length: {ea.geneLength}")
#     print(f"Fitness range: [{min(p['fitnessScore'] for p in population):.3f}, "
#           f"{max(p['fitnessScore'] for p in population):.3f}]")
    
#     # Test the function
#     build_in_params = {
#         'RevFF_predictor': {
#             'epochs': 50,
#             'batch_size': 32,
#             'use_log_scaler': True
#         }
#     }
    
#     try:
#         candidates = createCandidateGeneUsingFFPredictor(
#             args=args,
#             EA_Class=ea,
#             genePopulation=population,
#             number_of_candidate_genes=5,
#             build_in_params=build_in_params,
#             verbose=2
#         )
        
#         print(f"\nGenerated {len(candidates)} candidates:")
#         for i, candidate in enumerate(candidates):
#             print(f"  Candidate {i+1}: length={len(candidate)}, "
#                   f"range=[{min(candidate):.3f}, {max(candidate):.3f}]")
        
#         print("\n✓ Test passed!")
        
#     except Exception as e:
#         print(f"\n✗ Test failed: {e}")
#         import traceback
#         traceback.print_exc()

