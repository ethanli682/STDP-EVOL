"""
Production-optimized FFPredictor for candidate gene generation.

Key optimizations:
1. Minimal logging (errors/warnings only)
2. Inline tensor operations (memory-safe for 80% chunk allocation)
3. Uses loader's built-in scaler fitting (two-pass)
4. Enabled scaler weights and cluster balancing
5. No JSON serialization overhead
"""

import gc
import time
import numpy as np
import torch
from typing import List, Dict, Any, Optional
from GA_utils_misc_func import extract_chunk_to_tensors


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
            total_mem = torch.cuda.get_device_properties(0).total_memory
            allocated_mem = torch.cuda.memory_allocated(0)
            available_mem = total_mem - allocated_mem
        else:
            try:
                import psutil
                available_mem = psutil.virtual_memory().available
            except ImportError:
                available_mem = 4 * 1024 * 1024 * 1024

        usable_mem = available_mem * safety_factor
        bytes_per_gene = (gene_length * 4) + 4 + 100
        max_genes = int(usable_mem / bytes_per_gene)
        max_genes = max(100, min(max_genes, 1_000_000))
        return max_genes
    except Exception:
        return 10000


def _get_default_hidden_sizes(gene_length: int, available_memory_gb: float = 8.0, data_count: int = 0) -> List[int]:
    """Calculate default hidden layer sizes based on gene length, available memory, and data size.

    For SLURM: assume conservative memory available (8GB default).
    Each parameter in layer roughly costs: hidden_size * gene_length * 4 bytes (float32)

    When data_count is provided, scales capacity up (within memory limits) so the model
    can fit a richer dataset without underfitting.
    """
    # Memory budget: 200MB per model for weights + activations
    max_params = int((available_memory_gb * 1e9) / (4 * 10))  # 10x safety factor

    # Data-driven capacity boost: sqrt scaling — double data → ~1.4x capacity
    data_scale = 1.0
    if data_count > 500:
        data_scale = min(2.0, (data_count / 500) ** 0.25)  # caps at 2x

    # Estimate based on typical 3-layer MLP
    # Layer1: gene_length -> h1, Layer2: h1 -> h2, Layer3: h2 -> 1
    # Parameters ≈ gene_length*h1 + h1*h2 + h2*1

    if gene_length <= 50:
        h1 = min(128, max_params // max(gene_length * 2, 1))
        h1 = max(h1, 64)
    elif gene_length <= 200:
        h1 = min(256, max_params // max(gene_length * 2, 1))
        h1 = max(h1, 96)
    elif gene_length <= 500:
        h1 = min(512, max_params // max(gene_length * 2, 1))
        h1 = max(h1, 128)
    elif gene_length <= 1000:
        h1 = min(1024, max_params // max(gene_length * 3, 1))
        h1 = max(h1, 192)
    elif gene_length <= 5000:
        h1 = min(256, max_params // max(gene_length * 4, 1))
        h1 = max(h1, 64)
    else:
        # Huge genes (>5000): keep first layer compact and tapered.
        h1 = min(192, max_params // max(gene_length * 5, 1))
        h1 = max(h1, 32)

    # Apply data-driven capacity boost
    h1 = int(h1 * data_scale)
    # Re-enforce memory cap after scaling
    h1 = min(h1, max_params // max(gene_length * 2, 1))
    h1 = max(h1, 32)

    h2 = max(16, min(h1 // 2, 512))
    h3 = max(8, min(h2 // 2, 256))
    return [h1, h2, h3]


def _extract_chunk_to_tensors(
    chunk: List[Dict],
    gene_length: int,
    device: torch.device,
    extract_weights: bool = False
) -> tuple:
    """Extract genes, fitness, and weights from chunk into tensors (inline).
    
    DEPRECATED: Use extract_chunk_to_tensors from GA_utils_misc_func instead.
    """
    return extract_chunk_to_tensors(chunk, gene_length, device, extract_weights)


def createCandidateGeneUsingFFPredictor_chunk(
    args=None,
    EA_Class=None,
    genePopulation: Optional[List[Dict[str, Any]]] = None,
    number_of_candidate_genes: Optional[int] = None,
    build_in_params: Optional[Dict] = None,
    verbose: Optional[int] = None
) -> List[List[float]]:
    """
    Generate candidate genes using Feed-Forward Neural Network Predictor.
    
    Production-optimized version with:
    - Minimal logging (errors/warnings only)
    - Inline memory operations
    - Loader-managed scaler fitting
    - Cluster balancing for diversity
    
    Args:
        args: Argument object (path, agent_idx, agent_counter, evolutionTarget, populationSize, gpu)
        EA_Class: EA class (geneLength, geneMin, savePath, geneFormat)
        genePopulation: Optional gene list (None = load from disk)
        number_of_candidate_genes: Number of candidates to generate
        build_in_params: Configuration dict
        verbose: 0=silent, 1=minimal, 2=detailed (default=0)
        
    Returns:
        List[List[float]]: Candidate genes
    """
    
    from GA_utils_scalers import ChunkedRankScaler, ChunkedLogZScoreScaler, ChunkedPerDimensionScaler
    from GA_utils_history_loader import ChunkedGeneHistoryLoader
    from Alternative_Approach.ReverseAutoRegressor import AutoEnsemblePredictor
    
    # =========================================================================
    # VALIDATION & SETUP
    # =========================================================================
    
    if args is None or EA_Class is None or number_of_candidate_genes is None:
        raise ValueError("Required parameters: args, EA_Class, number_of_candidate_genes")
    
    if build_in_params is None:
        build_in_params = {}
    
    verbose = verbose if verbose is not None else 0
    
    # Device setup
    use_gpu = getattr(args, 'gpu', None)
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    
    # Extract parameters
    gene_length = EA_Class.geneLength
    gene_min = getattr(EA_Class, 'geneMin', None)
    lower_bound = 0 if gene_min == 0 else -1
    gene_format = getattr(EA_Class, 'geneFormat', None)
    
    agent_idx = getattr(args, 'agent_idx', None)
    agent_counter = getattr(args, 'agent_counter', 0)
    num_generations = getattr(args, 'numOFgenerations', 1000)
    evolution_target = getattr(args, 'evolutionTarget', 1)
    population_size = getattr(args, 'populationSize', 10000)
    
    if verbose >= 1:
        print(f"[FFPredictor] Device: {device}, Candidates: {number_of_candidate_genes}, "
              f"Gene length: {gene_length}", flush=True)
    
    _log_memory(device, "START", verbose)
    
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    
    if 'RevFF_predictor' not in build_in_params:
        build_in_params['RevFF_predictor'] = {}
    
    config = build_in_params['RevFF_predictor']
    
    # Defaults (SLURM-optimized)
    defaults = {
        'use_log_scaler': True,
        'top_k_percentile': 25,
        'ANN_Layers_size': None,
        'dropout_rate': 0.2,
        'epochs': 100,
        'batch_size': 16,  # Reduced from 32 for SLURM
        'learning_rate': 0.001,
        'early_stop_threshold': 0.05,
        'min_epochs_before_early_stop': 10,
        'validation': False,
        'validation_split': 0.2,
        'gradient_clip': 1.0,  # Gradient clipping to prevent exploding gradients
        'buffer_multiplier': 2,  # Reduced from 4 for SLURM
        'allowed_architectures': ['ANN'],  # Architecture types to train
        'max_models_to_train': 1,  # Maximum number of models to train
        'max_iterations_lookback': 200,  # Sliding window: only use last N iterations of data
    }
    
    for key, value in defaults.items():
        if key not in config:
            config[key] = value
    
    if config['ANN_Layers_size'] is None:
        # Estimate available GPU/CPU memory for layer sizing
        try:
            if device.type == 'cuda':
                available_mem_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
            else:
                # CPU: assume conservative 8GB for SLURM
                available_mem_gb = 8.0
        except:
            available_mem_gb = 8.0
        
        config['ANN_Layers_size'] = _get_default_hidden_sizes(gene_length, available_mem_gb)
    
    # =========================================================================
    # CREATE SCALER
    # =========================================================================
    
    # Adaptive emphasis
    progress = agent_counter / num_generations if num_generations > 0 else 0.0
    emphasis_multiplier = 0.7 if progress < 0.3 else (1.5 if progress > 0.7 else 0.7 + progress)
    emphasis_factor = 1.0 + emphasis_multiplier * torch.rand(1).item()
    
    if config['use_log_scaler']:
        scaler = ChunkedLogZScoreScaler(
            evolution_target=evolution_target,
            emphasis_factor=emphasis_factor,
            min_weight=0.01,
            device=device
        )
    else:
        scaler = ChunkedRankScaler(
            evolution_target=evolution_target,
            emphasis_factor=emphasis_factor,
            min_weight=0.01,
            device=device
        )

    # Per-dimension gene scaler (for z-score normalization of features)
    gene_scaler = ChunkedPerDimensionScaler(
        gene_length=gene_length,
        device=device,
        weight_mode='uniform'
    )
    
    # =========================================================================
    # TRAINING DATA PREPARATION
    # =========================================================================
    
    predictor = None
    val_loss  = None

    # Detect block-wise co-evolution mode (EA_Class.coevo_groups built from YAML tags)
    coevo_groups = getattr(EA_Class, 'coevo_groups', None)
    use_block_wise = coevo_groups is not None and len(coevo_groups) > 1

    # Holds normalized data for block-wise generate section (set by RAM or DISK path)
    _bw_training_genes   = None   # normalized gene tensor [N, gene_length]
    _bw_training_targets = None   # scaled fitness tensor  [N]

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
    
    if genePopulation is not None:
        # =====================================================================
        # RAM HIT: Use provided population
        # =====================================================================
        
        if verbose >= 1:
            print(f"[FFPredictor] RAM HIT: {len(genePopulation)} genes", flush=True)
        
        # Extract valid genes
        valid_genes = [g for g in genePopulation 
                      if isinstance(g, dict) and 'gene' in g and 'fitnessScore' in g
                      and g['gene'] is not None 
                      and isinstance(g['fitnessScore'], (int, float))
                      and not np.isnan(g['fitnessScore'])]
        
        if len(valid_genes) == 0:
            raise RuntimeError("No valid genes in provided population")
        
        # Pre-allocate tensors
        with torch.no_grad():
            training_genes = torch.empty((len(valid_genes), gene_length), dtype=torch.float32, device=device)
            training_fitness = torch.empty(len(valid_genes), dtype=torch.float32, device=device)
            
            for idx, gene_dict in enumerate(valid_genes):
                gene_data = gene_dict['gene']
                if isinstance(gene_data, torch.Tensor):
                    training_genes[idx] = gene_data.to(device)
                elif isinstance(gene_data, np.ndarray):
                    training_genes[idx] = torch.from_numpy(gene_data).to(device)
                else:
                    training_genes[idx] = torch.tensor(gene_data, dtype=torch.float32, device=device)
                
                training_fitness[idx] = gene_dict['fitnessScore']
        
        _log_memory(device, "RAM-HIT-LOADED", verbose)
        
        # Fit scaler (loader does this for DISK HIT, so we do it here for consistency)
        scaler.collect_phase_1_statistics(training_fitness)
        scaler.collect_phase_2_statistics(training_fitness)
        scaler.finalize()
        
        # Scale targets
        training_targets = scaler.scale_chunk(training_fitness)

        # Fit gene scaler on provided population and normalize features
        try:
            genes_np = training_genes.detach().cpu().numpy()
            gene_scaler.collect_phase_1_statistics(genes_np)
            gene_scaler.collect_phase_2_statistics(genes_np)
            gene_scaler.finalize()
            # Use transform() for consistency with DISK path
            training_genes = gene_scaler.transform(training_genes)
        except Exception as e:
            if verbose >= 1:
                print(f"[FFPredictor WARNING] Gene scaler fitting failed (RAM path): {e}", flush=True)
        
        if use_block_wise:
            # Block-wise: save normalized tensors for the generate section
            _bw_training_genes   = training_genes
            _bw_training_targets = training_targets
            del training_fitness, valid_genes
        else:
            # Create and train model
            predictor, used_hidden_sizes = _create_predictor_with_shrink(config['ANN_Layers_size'])
            config['ANN_Layers_size'] = used_hidden_sizes

            if verbose >= 1:
                print(f"[FFPredictor] Training (epochs={config['epochs']}, batch={config['batch_size']})", flush=True)

            _, val_loss = predictor.train_model(
                training_genes,
                training_targets,
                epochs=config['epochs'],
                batch_size=config['batch_size'],
                allowed_architectures=config['allowed_architectures'],
                max_models_to_train=config['max_models_to_train'],
                early_stop_threshold=config['early_stop_threshold'],
                learning_rate=config['learning_rate'],
                validation_split=config['validation_split'],
                verbose=False
            )

            # Cleanup
            del training_genes, training_fitness, training_targets, valid_genes
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    else:
        # =====================================================================
        # DISK HIT: Stream from history with chunked training
        # =====================================================================
        
        if verbose >= 1:
            print(f"[FFPredictor] DISK HIT: Streaming from history", flush=True)
        
        if use_block_wise:
            # -----------------------------------------------------------------
            # BLOCK-WISE DISK PATH: heap-accumulate then per-group train+generate
            # -----------------------------------------------------------------
            import heapq as _heapq

            loader_bw = ChunkedGeneHistoryLoader(
                savePath=args.path,
                agent_id=agent_idx if agent_idx is not None else agent_counter,
                geneFormat=gene_format,
                shuffle=True,
                shuffle_level='both',
            )

            maximize_bw      = (evolution_target == 1)
            top_k_bw         = config['top_k_percentile']
            loader_pct_bw    = 100 - top_k_bw
            mem_limit_bw     = _calculate_memory_based_limit(gene_length, device, safety_factor=0.7)
            heap_capacity_bw = min(population_size, mem_limit_bw)
            chunk_size_bw    = min(2000, max(100, heap_capacity_bw // 10))

            if verbose >= 1:
                print(f"[FFPredictor-BLOCK] Heap capacity: {heap_capacity_bw}, "
                      f"chunk: {chunk_size_bw}", flush=True)

            gene_heap_bw  = []
            tiebreaker_bw = 0
            total_proc_bw = 0

            for _bw_chunk, _bw_done in loader_bw.loadTopKPercentileGeneHistoryAsListOfDicts(
                args=args,
                top_k_percentile=loader_pct_bw,
                maximize=maximize_bw,
                chunk_size=chunk_size_bw,
                fitness_scaler=scaler,
                use_fitness_scaler_weights=False,
                gene_scaler=gene_scaler,
                use_gene_scaler_weights=True,
                n_feature_clusters='adaptive',
                allEpochs=True,
                min_datapoints=min(int(population_size * 0.05), 300),
                fit_fitness_scaler_on_top_k=False,
                fit_gene_scaler_on_top_k=True,
            ):
                for _gd in _bw_chunk:
                    _gdata = _gd.get('gene')
                    _gfit  = _gd.get('fitnessScore')
                    if _gdata is None or _gfit is None:
                        continue
                    try:
                        _gfit = float(_gfit)
                    except (TypeError, ValueError):
                        continue
                    if np.isnan(_gfit):
                        continue
                    # Sliding window: skip records from old iterations
                    _bw_max_lookback = config.get('max_iterations_lookback', 200)
                    _bw_min_iter = max(0, agent_counter - _bw_max_lookback)
                    if _gd.get('iteration', 0) < _bw_min_iter:
                        continue
                    total_proc_bw += 1
                    _prio  = -_gfit if maximize_bw else _gfit
                    _entry = (_prio, tiebreaker_bw, _gdata, _gfit)
                    tiebreaker_bw += 1
                    if len(gene_heap_bw) < heap_capacity_bw:
                        _heapq.heappush(gene_heap_bw, _entry)
                    elif _prio < gene_heap_bw[0][0]:
                        _heapq.heapreplace(gene_heap_bw, _entry)
                if _bw_done:
                    break

            if not gene_heap_bw:
                raise RuntimeError("[FFPredictor-BLOCK] No genes loaded from disk")

            if verbose >= 1:
                print(f"[FFPredictor-BLOCK] Loaded {len(gene_heap_bw)} genes "
                      f"(processed {total_proc_bw})", flush=True)

            # Convert heap → tensors
            _hs = len(gene_heap_bw)
            _bw_raw_genes   = torch.empty((_hs, gene_length), dtype=torch.float32, device=device)
            _bw_raw_fitness = torch.empty(_hs, dtype=torch.float32, device=device)
            for _i, (_, _, _gd, _gf) in enumerate(gene_heap_bw):
                if isinstance(_gd, torch.Tensor):
                    _bw_raw_genes[_i] = _gd.to(device)
                elif isinstance(_gd, np.ndarray):
                    _bw_raw_genes[_i] = torch.from_numpy(_gd).to(device)
                else:
                    _bw_raw_genes[_i] = torch.tensor(_gd, dtype=torch.float32, device=device)
                _bw_raw_fitness[_i] = _gf
            del gene_heap_bw
            gc.collect()

            # Normalize and scale (gene_scaler/fitness scaler fitted by loader)
            _bw_training_genes   = gene_scaler.transform(_bw_raw_genes)
            _bw_training_targets = scaler.scale_chunk(_bw_raw_fitness)
            del _bw_raw_genes, _bw_raw_fitness, loader_bw
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        else:
            # -----------------------------------------------------------------
            # FULL-GENE DISK PATH: streaming chunked training (unchanged)
            # -----------------------------------------------------------------
            # Create predictor FIRST (for accurate memory calculation)
            predictor, used_hidden_sizes = _create_predictor_with_shrink(config['ANN_Layers_size'])
            config['ANN_Layers_size'] = used_hidden_sizes

            # Create loader
            loader = ChunkedGeneHistoryLoader(
                savePath=args.path,
                agent_id=agent_idx if agent_idx is not None else agent_counter,
                geneFormat=gene_format,
                shuffle=True,
                shuffle_level='both',
            )

            # Loader settings
            top_k_percentile = config['top_k_percentile']
            loader_percentile = 100 - top_k_percentile
            maximize = (evolution_target == 1)
            max_iterations_lookback = config.get('max_iterations_lookback', 200)
            min_iteration_for_window = max(0, agent_counter - max_iterations_lookback)
            cache_key = f"agent_{agent_idx}_FFPredictor_top{top_k_percentile}_win{max_iterations_lookback}"

            if verbose >= 1:
                print(f"[FFPredictor] Top {top_k_percentile}% | Sliding window: iterations [{min_iteration_for_window}, {agent_counter}]", flush=True)

            # Initialize chunked training with inspect-based parameter checking
            import inspect
            start_chunk_params = inspect.signature(predictor.start_chunk_training).parameters

            start_chunk_kwargs = {
                'savePath': args.path,
                'agent_idx': agent_idx if agent_idx is not None else 0,
                'allowed_architectures': config['allowed_architectures'],
                'epochs': config['epochs'],
                'max_models_to_train': config['max_models_to_train'],
                'validation_split': config['validation_split'],
                'batch_size': config['batch_size'],
                'learning_rate': config['learning_rate'],
                'early_stop_threshold': config['early_stop_threshold'],
                'reinitialize_models': True,
                'verbose': False,
            }

            # Add new parameters only if supported
            if 'use_validation' in start_chunk_params:
                start_chunk_kwargs['use_validation'] = config['validation']
            if 'gradient_clip' in start_chunk_params:
                start_chunk_kwargs['gradient_clip'] = config['gradient_clip']
            if 'patience' in start_chunk_params:
                start_chunk_kwargs['patience'] = config.get('patience', 20)
            if 'max_epochs' in start_chunk_params:
                start_chunk_kwargs['max_epochs'] = config.get('max_epochs', 500)

            predictor.start_chunk_training(**start_chunk_kwargs)

            # Training loop
            training_complete = False
            current_epoch = 0
            total_genes_processed = 0

            # Inline buffer (no concatenation - just override)
            buffer_size = config['batch_size'] * config['buffer_multiplier']
            buffer_genes = None
            buffer_fitness = None
            buffer_used = 0

            while not training_complete:
                current_epoch += 1
                genes_this_epoch = 0

                # Dynamic chunk size (memory-aware) - more conservative for SLURM
                _log_memory(device, f"DISK-EPOCH-{current_epoch}-START", verbose)

                try:
                    # Check memory pressure and adapt
                    if _check_memory_pressure(device, threshold=0.80):
                        print(f"[FFPredictor WARNING] High memory pressure detected (>80%), reducing chunk size", flush=True)
                        chunk_size = 500  # Emergency small chunks
                    else:
                        # Request much smaller chunks for SLURM stability
                        chunk_size = loader.calculate_chunk_size(
                            safety_margin=0.5,  # Even more conservative
                            chunk_allocation=0.10,  # Use only 10% of memory per chunk
                            verbose=False
                        )
                        # Cap chunk size aggressively for SLURM (max 2000 genes per chunk)
                        chunk_size = min(chunk_size, 2000)
                except Exception as e:
                    print(f"[FFPredictor WARNING] Chunk size calculation failed: {e}, using fallback", flush=True)
                    chunk_size = 1000  # Conservative fallback

                if verbose >= 1 and current_epoch == 1:
                    print(f"[FFPredictor] Chunk size: {chunk_size}", flush=True)

                # Stream chunks (exactly like original)
                # For SLURM: limit total loaded data to prevent memory issues
                max_load = min(int(population_size * 0.3), 3000)  # Max 3000 genes total

                if verbose >= 2:
                    print(f"[FFPredictor] Epoch {current_epoch}: max_load={max_load}, chunk_size={chunk_size}", flush=True)

                chunk_count = 0
                for chunk, is_done in loader.loadTopKPercentileGeneHistoryAsListOfDicts(
                    args=args,
                    top_k_percentile=loader_percentile,
                    maximize=maximize,
                    chunk_size=chunk_size,
                    fitness_scaler=scaler,
                    use_fitness_scaler_weights=True,
                    gene_scaler=gene_scaler,
                    use_gene_scaler_weights=True,
                    n_feature_clusters='adaptive',
                    allEpochs=True,
                    cache_key=cache_key,
                    use_cache=True,
                    min_datapoints=min(int(population_size * 0.05), 300),
                    # max_datapoints=max_load,
                    # balance_clusters=True,    # Next test
                    fit_fitness_scaler_on_top_k=False,
                    fit_gene_scaler_on_top_k=True,
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
                        _log_memory(device, f"CHUNK-{chunk_count}-START", verbose)

                    # --- SLIDING WINDOW FILTER ---
                    # Only use records from recent iterations to prevent model from fitting stale data
                    if max_iterations_lookback > 0:
                        chunk = [rec for rec in chunk if rec.get('iteration', 0) >= min_iteration_for_window]
                        if len(chunk) == 0:
                            if is_done:
                                break
                            continue

                    # Extract chunk (inline) - extract_weights=False to match original
                    try:
                        chunk_genes, chunk_fitness_raw, _, valid_count = _extract_chunk_to_tensors(
                            chunk, gene_length, device, extract_weights=False
                        )
                    except Exception as e:
                        print(f"[FFPredictor ERROR] Chunk extraction failed at chunk {chunk_count}: {e}", flush=True)
                        _log_memory(device, f"CHUNK-{chunk_count}-FAILED", verbose)
                        raise

                    if chunk_genes is None or valid_count == 0:
                        if is_done:
                            break
                        continue

                    # Scale fitness (scaler already fitted by loader)
                    chunk_fitness_scaled = scaler.scale_chunk(chunk_fitness_raw)
                    del chunk_fitness_raw  # Free immediately
                    chunk_genes = gene_scaler.transform(chunk_genes)

                    # Buffer management (inline - no concatenation)
                    if buffer_genes is not None and buffer_used > 0:
                        # Interleave buffer with chunk (memory-efficient)
                        total_size = buffer_used + valid_count

                        # Create combined tensors (single allocation)
                        combined_genes = torch.empty((total_size, gene_length), dtype=torch.float32, device=device)
                        combined_fitness = torch.empty(total_size, dtype=torch.float32, device=device)

                        # Copy buffer
                        combined_genes[:buffer_used] = buffer_genes[:buffer_used]
                        combined_fitness[:buffer_used] = buffer_fitness[:buffer_used]

                        # Copy chunk
                        combined_genes[buffer_used:] = chunk_genes
                        combined_fitness[buffer_used:] = chunk_fitness_scaled

                        # Shuffle in-place
                        perm = torch.randperm(total_size, device=device)
                        combined_genes = combined_genes[perm]
                        combined_fitness = combined_fitness[perm]

                        training_genes = combined_genes
                        training_fitness = combined_fitness
                    else:
                        training_genes = chunk_genes
                        training_fitness = chunk_fitness_scaled

                    # Update buffer (random sampling from chunk)
                    samples_for_buffer = min(buffer_size, valid_count)
                    if samples_for_buffer > 0:
                        if buffer_genes is None:
                            # Initialize buffer
                            buffer_genes = torch.empty((buffer_size, gene_length), dtype=torch.float32, device=device)
                            buffer_fitness = torch.empty(buffer_size, dtype=torch.float32, device=device)

                        # Random sample into buffer (use clone like original)
                        indices = torch.randperm(valid_count, device=device)[:samples_for_buffer]
                        buffer_genes[:samples_for_buffer] = chunk_genes[indices].clone()
                        buffer_fitness[:samples_for_buffer] = chunk_fitness_scaled[indices].clone()
                        buffer_used = samples_for_buffer

                    # Process chunk WITHOUT weights (match original)
                    try:
                        predictor.process_chunk(training_genes, training_fitness, sample_weights=None)
                        genes_this_epoch += valid_count

                        if verbose >= 3:
                            _log_memory(device, f"CHUNK-{chunk_count}-PROCESSED", verbose)
                    except Exception as e:
                        print(f"[FFPredictor ERROR] Chunk processing failed at chunk {chunk_count}: {e}", flush=True)
                        _log_memory(device, f"CHUNK-{chunk_count}-PROCESS-FAILED", verbose)
                        raise

                    # Cleanup (inline ops - free immediately)
                    del chunk_genes, chunk_fitness_scaled
                    del training_genes, training_fitness

                    if is_done:
                        break

                if genes_this_epoch == 0:
                    raise RuntimeError(f"No genes loaded in epoch {current_epoch}")

                total_genes_processed += genes_this_epoch

                # Finish epoch
                training_complete, info = predictor.finish_chunk_epoch()

                # Override early stop if below minimum
                min_epochs = config['min_epochs_before_early_stop']
                if training_complete and current_epoch < min_epochs:
                    if info.get('early_stopped', False):
                        if verbose >= 1:
                            print(f"[FFPredictor] Early stop overridden (epoch {current_epoch} < {min_epochs})", flush=True)
                        training_complete = False

                        # Resume training without resetting state (preserves
                        # optimizer momentum, adaptive tracking, best model states)
                        predictor._chunk_state['training_active'] = True
                        predictor._chunk_state['early_stopped'] = False
                        remaining = max(config['epochs'] - current_epoch, min_epochs - current_epoch)
                        predictor._chunk_state['epochs'] = current_epoch + remaining
                        predictor._chunk_state['initial_epochs'] = predictor._chunk_state['epochs']
                        predictor._chunk_state['current_epoch'] = current_epoch + 1
                        # Set models back to train mode
                        for name in predictor._chunk_state['model_names']:
                            predictor.models[name].train()

                _log_memory(device, f"EPOCH-{current_epoch}-END", verbose)

                if verbose >= 1:
                    best_loss = info.get('best_val_loss_so_far', 0)
                    print(f"[FFPredictor] Epoch {current_epoch}: {genes_this_epoch} genes, loss={best_loss:.6f}", flush=True)

                if not training_complete:
                    # Cleanup between epochs
                    gc.collect()
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()

                    # Reset buffer for next epoch
                    buffer_used = 0
                    loader.reset()

            # Training complete
            val_loss = info.get('best_loss', info.get('best_val_loss_so_far'))

            if verbose >= 1:
                print(f"[FFPredictor] Training complete: {current_epoch} epochs, "
                    f"{total_genes_processed} genes, loss={val_loss:.6f}", flush=True)

            if val_loss > 0.5:
                print(f"[FFPredictor WARNING] High validation loss ({val_loss:.3f})", flush=True)

            # Cleanup
            del loader
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    # =========================================================================
    # GENERATE CANDIDATES — block-wise or full-gene
    # =========================================================================

    # Adaptive target z-score
    if np.random.rand() < 0.15:
        base_target_z = 2.5 + torch.rand(1).item() * 1.0  # Exploration
    else:
        base_target_z = 4.0 + torch.rand(1).item() * 1.0  # Greedy

    _log_memory(device, "BEFORE-GENERATION", verbose)

    if use_block_wise:
        # -----------------------------------------------------------------
        # BLOCK-WISE: K small models, one per coevo_group
        # -----------------------------------------------------------------
        group_order = sorted(coevo_groups.keys(), key=lambda g: coevo_groups[g]['start'])
        K = len(group_order)
        if verbose >= 1:
            print(f"[FFPredictor-BLOCK] {K} co-evolution groups → training {K} small models "
                  f"(target_z={base_target_z:.2f})", flush=True)

        group_slices      = {}
        val_loss_by_group = {}

        for gname in group_order:
            ginfo   = coevo_groups[gname]
            g_start = ginfo['start']
            g_end   = ginfo['end']
            dim_g   = ginfo['dim']

            # Slice the pre-computed normalized tensor for this group
            slice_g = _bw_training_genes[:, g_start:g_end]   # [N, dim_g]

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

            # Train on this group's slice with the shared fitness targets
            try:
                _, g_loss = pred_g.train_model(
                    slice_g,
                    _bw_training_targets,
                    epochs=config['epochs'],
                    batch_size=config['batch_size'],
                    allowed_architectures=config['allowed_architectures'],
                    max_models_to_train=config['max_models_to_train'],
                    early_stop_threshold=config['early_stop_threshold'],
                    learning_rate=config['learning_rate'],
                    verbose=False
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
                        verbose=False
                    )
                # Defensive pad if fewer than requested were returned
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
        del _bw_training_genes, _bw_training_targets
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
        config['loss'] = mean_loss

    else:
        # -----------------------------------------------------------------
        # FULL-GENE: single predictor (existing behaviour, unchanged)
        # -----------------------------------------------------------------
        if verbose >= 1:
            print(f"[FFPredictor] Generating {number_of_candidate_genes} candidates "
                  f"(target_z={base_target_z:.2f})", flush=True)

        # Generate via reverse optimization
        try:
            with torch.enable_grad():
                suggested_genes = predictor.suggest_features_for_fitness(
                    target_fitness=base_target_z,
                    num_features=gene_length,
                    num_suggestions=number_of_candidate_genes,
                    max_iterations=500,
                    verbose=False
                )
        except Exception as e:
            print(f"[FFPredictor ERROR] Candidate generation failed: {e}", flush=True)
            _log_memory(device, "GENERATION-FAILED", verbose)
            raise

        _log_memory(device, "AFTER-GENERATION", verbose)

        # Convert and clip
        candidates = gene_scaler.inverse_transform(suggested_genes)
        candidates = np.clip(candidates, lower_bound, 1.0).tolist()

        # Log loss
        if val_loss is not None:
            if isinstance(val_loss, torch.Tensor):
                val_loss = val_loss.item()
            build_in_params['loss'] = float(val_loss)
            config['loss'] = float(val_loss)

        # Cleanup predictor
        if predictor is not None:
            del predictor
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    _log_memory(device, "END", verbose)

    if verbose >= 1:
        print(f"[FFPredictor] Complete: {len(candidates)} candidates generated", flush=True)

    return candidates