import numpy as np
import torch
import math
import heapq
from GA_utils_history_loader import *
from GA_utils_scalers import *
import time

# ============================================================================
# Genetic Algorithm Candidate Gene Generation
# ============================================================================

def createCandidateGene_GeneticAlgorithm(
    args=None,
    EA_Class=None,
    genePopulation: Optional[List[Dict[str, Any]]] = None,
    number_of_candidate_genes: Optional[int] = None,
    build_in_params: Optional[Dict] = None,
    verbose: Optional[int] = None
) -> List[List[float]]:
    """
    Generate candidate genes using Genetic Algorithm.
    
    Memory-efficient: Works directly with genePopulation list without 
    duplicating to numpy/torch arrays. Only converts individual genes
    during breeding.
    
    Args:
        args: Arguments object with:
            - path: Save path for disk loading
            - evolutionTarget: 1 (maximize) or -1 (minimize)
            - gpu: Whether to use GPU
            - numOFgenerations: Total generations (for adaptive rates)
            - agent_counter: Current generation
            - agent_idx: Agent ID for disk loading
        EA_Class: Class with geneLength and geneFormat attributes
        genePopulation: List of dicts [{'gene': [...], 'fitnessScore': float}, ...]
                       If provided, use as mating pool directly.
                       If None, load from disk using streaming.
        number_of_candidate_genes: Number of genes to generate (REQUIRED)
        build_in_params: Dict with GA hyperparameters. Will be modified to include:
            - GA: Dict with mutation_rate, sigma, crossover_prob
            - GA-log: Dict with adaptive parameters used during this call
        verbose: Verbosity level:
            - None or 0: Silent (no output)
            - 1: Minimal (key results only)
            - 2: Standard (progress and parameters)
            - 3: Debug (all details including internal computations)
            
    Returns:
        List[List[float]]: New candidate genes [[gene1], [gene2], ...]
        
    Raises:
        ValueError: If required parameters are None
    """
    # ========================================================================
    # VALIDATION
    # ========================================================================
    if args is None:
        raise ValueError("args must be provided")
    if EA_Class is None:
        raise ValueError("EA_Class must be provided")
    if number_of_candidate_genes is None:
        raise ValueError("number_of_candidate_genes must be specified")
    if build_in_params is None:
        raise ValueError("build_in_params must be provided (can be empty dict)")
    
    # Normalize verbose level
    v = verbose if verbose is not None else 0
    
    def vprint(level: int, msg: str):
        """Print message if verbosity level is high enough."""
        if v >= level:
            print(msg, flush=True)
    
    # ========================================================================
    # SETUP HYPERPARAMETERS
    # ========================================================================
    # Ensure GA key exists with defaults
    if 'GA' not in build_in_params:
        build_in_params['GA'] = {}
    
    ga_params = build_in_params['GA']
    
    # Set defaults if not provided
    if 'mutation_rate' not in ga_params:
        ga_params['mutation_rate'] = 0.1
    if 'sigma' not in ga_params:
        ga_params['sigma'] = 0.1
    if 'crossover_prob' not in ga_params:
        ga_params['crossover_prob'] = 0.5
    
    base_mutation_rate = ga_params['mutation_rate']
    base_sigma = ga_params['sigma']
    crossover_prob = ga_params['crossover_prob']
    
    # Initialize GA-log
    build_in_params['GA-log'] = {}
    ga_log = build_in_params['GA-log']
    
    # Device setup
    device = torch.device(
        'cuda' if getattr(args, 'gpu', False) and torch.cuda.is_available() else 'cpu'
    )
    
    evolution_target = getattr(args, 'evolutionTarget', 1)
    gene_length = EA_Class.geneLength

    gene_min = getattr(EA_Class, 'geneMin', -1.0)
    lower_bound = 0.0 if gene_min == 0 else -1.0
    
    # Co-evolution groups (block-wise mode)
    coevo_groups = getattr(EA_Class, 'coevo_groups', None)
    use_block_wise = coevo_groups is not None and len(coevo_groups) > 1

    # Compute adaptive mutation rate (decreases with progress)
    mutation_rate = _compute_adaptive_mutation_rate(args, base_mutation_rate)
    ga_log['mutation_rate'] = mutation_rate
    
    # ========================================================================
    # LOAD MATING POOL
    # ========================================================================
    if genePopulation is not None and len(genePopulation) > 0:
        pool = genePopulation
        vprint(1, f"[GA] Using provided population: {len(pool)} genes")
    else:
        pool = _load_mating_pool_from_disk(args, EA_Class, device, v)
        vprint(1, f"[GA] Loaded from disk: {len(pool)} genes")
    
    # Fallback: return random genes if pool too small
    if not pool or len(pool) < 2:
        vprint(1, "[GA] Pool too small, returning random genes")
        ga_log['sigma'] = base_sigma
        ga_log['quality_factor'] = 1.0
        ga_log['diversity_factor'] = 1.0
        ga_log['pool_fitness_min'] = None
        ga_log['pool_fitness_max'] = None
        ga_log['pool_fitness_mean'] = None
        gene_min = getattr(EA_Class, 'geneMin', -1.0)
        return _generate_random_genes(number_of_candidate_genes, gene_length, device, gene_min)
    
    # ========================================================================
    # LOG POOL STATISTICS
    # ========================================================================
    fitness_values = [g['fitnessScore'] for g in pool]
    ga_log['pool_fitness_min'] = min(fitness_values)
    ga_log['pool_fitness_max'] = max(fitness_values)
    ga_log['pool_fitness_mean'] = sum(fitness_values) / len(fitness_values)
    
    # ========================================================================
    # COMPUTE WEIGHTS & ADAPTIVE SIGMA
    # ========================================================================
    weights, sigma, quality_factor, diversity_factor = _compute_breeding_params(
        pool, evolution_target, base_sigma, device
    )
    
    ga_log['sigma'] = sigma
    ga_log['quality_factor'] = quality_factor
    ga_log['diversity_factor'] = diversity_factor
    ga_log['use_block_wise'] = use_block_wise
    ga_log['num_groups'] = len(coevo_groups) if use_block_wise else 1

    vprint(2, f"[GA] Adaptive params: mutation_rate={mutation_rate:.4f}, sigma={sigma:.4f}")
    vprint(2, f"[GA] Pool fitness: [{ga_log['pool_fitness_min']:.4f}, {ga_log['pool_fitness_max']:.4f}]")
    vprint(3, f"[GA] Quality factor: {quality_factor:.4f}, Diversity factor: {diversity_factor:.4f}")
    
    # ========================================================================
    # BREED CANDIDATE GENES
    # ========================================================================
    if use_block_wise:
        vprint(1, f"[GA-BLOCK] {len(coevo_groups)} groups | mutation_rate={mutation_rate:.4f} sigma={sigma:.4f}")
        if v >= 2:
            group_order_log = sorted(coevo_groups.keys(), key=lambda g: coevo_groups[g]['start'])
            group_dims = [f'{g}(dim={coevo_groups[g]["dim"]})' for g in group_order_log]
            vprint(2, f"[GA-BLOCK] Groups: {group_dims}")
        candidates = _breed_candidates_blockwise(
            pool=pool,
            weights=weights,
            num_candidates=number_of_candidate_genes,
            crossover_prob=crossover_prob,
            mutation_rate=mutation_rate,
            sigma=sigma,
            device=device,
            coevo_groups=coevo_groups,
            gene_length=gene_length,
            lower_bound=lower_bound
        )
        vprint(1, f"[GA-BLOCK] Assembly complete: {len(candidates)} candidates")
    else:
        candidates = _breed_candidates(
            pool=pool,
            weights=weights,
            num_candidates=number_of_candidate_genes,
            crossover_prob=crossover_prob,
            mutation_rate=mutation_rate,
            sigma=sigma,
            device=device,
            lower_bound=lower_bound
        )
    
    # Cleanup GPU memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return candidates

# ----------------------------------------------------------------------------
# PRIVATE HELPER FUNCTIONS
# ----------------------------------------------------------------------------

def _compute_adaptive_mutation_rate(args, base_rate: float) -> float:
    """Compute mutation rate that decreases with progress (annealing)."""
    num_generations = getattr(args, 'numOFgenerations', 0)
    if num_generations > 0:
        progress = getattr(args, 'agent_counter', 0) / num_generations
        return base_rate * max(0.3, 1.0 - progress * 0.7)
    return base_rate

def _generate_random_genes(
    count: int, 
    gene_length: int, 
    device: torch.device,
    gene_min: float = -1.0
) -> List[List[float]]:
    """Generate random genes as fallback."""
    lower_bound = 0.0 if gene_min == 0 else -1.0
    return [
        torch.randn(gene_length, device=device).clamp(lower_bound, 1.0).tolist()
        for _ in range(count)
    ]

def _compute_breeding_params(
    pool: List[Dict],
    evolution_target: int,
    base_sigma: float,
    device: torch.device
) -> tuple:
    """
    Compute fitness-proportionate weights and adaptive sigma.
    Memory efficient: only extracts fitness scores for weights, samples for diversity.
    
    Returns:
        (weights_tensor, sigma_float, quality_factor, diversity_factor)
    """
    pool_size = len(pool)
    
    # Extract fitness scores only (small memory footprint)
    fitness_list = [g['fitnessScore'] for g in pool]
    fitness = torch.tensor(fitness_list, device=device, dtype=torch.float32)
    
    # Fitness-proportionate weights based on evolution direction
    if evolution_target == -1:
        # Minimization: lower fitness = higher weight
        weights = 1.0 / (fitness + 1e-10)
    else:
        # Maximization: shift to positive, higher = better
        weights = fitness - fitness.min() + 1e-10
    
    weights = weights / weights.sum()
    
    # Compute adaptive sigma using sampled genes (memory efficient)
    sigma, quality_factor, diversity_factor = _compute_adaptive_sigma(
        pool, weights, base_sigma, evolution_target, device
    )
    
    return weights, sigma, quality_factor, diversity_factor

def _compute_adaptive_sigma(
    pool: List[Dict],
    weights: torch.Tensor,
    base_sigma: float,
    evolution_target: int,
    device: torch.device
) -> tuple:
    """
    Compute adaptive sigma based on pool quality and diversity.
    Samples genes to avoid loading entire pool into memory.
    
    Returns:
        (sigma, quality_factor, diversity_factor)
    """
    # Sample for diversity calculation (max 100 genes)
    sample_size = min(100, len(pool))
    sample_indices = torch.randperm(len(pool))[:sample_size].tolist()
    
    # Load only sampled genes
    sample_genes = torch.tensor(
        [pool[i]['gene'] for i in sample_indices],
        device=device,
        dtype=torch.float32
    )
    
    # Diversity factor from gene variance
    gene_variance = torch.var(sample_genes, dim=0).mean().item()
    diversity_factor = min(1.5, 0.8 + gene_variance * 2.0)
    
    # Quality factor from weight distribution
    mean_w = weights.mean().item()
    max_w = weights.max().item()
    min_w = weights.min().item()
    
    if max_w > min_w + 1e-8:
        quality_ratio = (mean_w - min_w) / (max_w - min_w)
    else:
        quality_ratio = 0.5
    
    # Higher quality = lower sigma (converging), lower quality = higher sigma (exploring)
    quality_factor = 1.0 + (1.0 - max(0, min(1, quality_ratio))) * 0.8
    
    # Direction factor (minimization needs slightly more exploration)
    direction_factor = 1.15 if evolution_target == -1 else 1.0
    
    sigma = base_sigma * quality_factor * diversity_factor * direction_factor
    sigma = max(0.01, min(0.5, sigma))
    
    return sigma, quality_factor, diversity_factor

def _breed_candidates(
    pool: List[Dict],
    weights: torch.Tensor,
    num_candidates: int,
    crossover_prob: float,
    mutation_rate: float,
    sigma: float,
    device: torch.device,
    lower_bound: float = -1.0
) -> List[List[float]]:
    """
    Breed new candidate genes using crossover and mutation.
    Memory efficient: loads parents on-demand, one pair at a time.
    """
    candidates = []
    
    for _ in range(num_candidates):
        # Select two parents via fitness-proportionate selection
        parent_indices = torch.multinomial(weights, 2, replacement=False)
        idx1, idx2 = parent_indices[0].item(), parent_indices[1].item()
        
        # Load parents on-demand (not all at once)
        parent1 = torch.tensor(pool[idx1]['gene'], device=device, dtype=torch.float32)
        parent2 = torch.tensor(pool[idx2]['gene'], device=device, dtype=torch.float32)
        
        # Crossover
        if torch.rand(1, device=device).item() < crossover_prob:
            # Blend crossover (BLX-alpha style)
            alpha = torch.rand(1, device=device)
            child = alpha * parent1 + (1.0 - alpha) * parent2
        else:
            # Uniform crossover
            mask = torch.rand_like(parent1) > 0.5
            child = torch.where(mask, parent1, parent2)
        
        # Mutation (sparse, only mutate selected positions)
        mutation_mask = torch.rand_like(child) < mutation_rate
        noise = torch.randn_like(child) * sigma
        child = child + noise * mutation_mask.float()
        
        # Clamp to valid range and convert to list
        child = torch.clamp(child, lower_bound, 1.0)
        candidates.append(child.tolist())
    
    return candidates


def _breed_candidates_blockwise(
    pool: List[Dict],
    weights: torch.Tensor,
    num_candidates: int,
    crossover_prob: float,
    mutation_rate: float,
    sigma: float,
    device: torch.device,
    coevo_groups: Dict,
    gene_length: int,
    lower_bound: float = -1.0
) -> List[List[float]]:
    """
    Cooperative Co-Evolutionary GA: each parameter group breeds independently.

    For every candidate, each co-evolution group selects its own parent pair
    from the shared fitness-proportionate weight vector, performs crossover and
    mutation on its slice [g_start:g_end], and all group slices are concatenated
    into the full gene. Groups share the fitness signal but NOT the parent pair,
    which is the standard CCGA formulation (Potter & De Jong 1994).
    """
    group_order = sorted(coevo_groups.keys(), key=lambda g: coevo_groups[g]['start'])
    candidates = []

    for _ in range(num_candidates):
        child = torch.zeros(gene_length, device=device, dtype=torch.float32)

        for gname in group_order:
            ginfo   = coevo_groups[gname]
            g_start = ginfo['start']
            g_end   = ginfo['end']

            # Independent parent selection per group
            parent_indices = torch.multinomial(weights, 2, replacement=False)
            idx1, idx2 = parent_indices[0].item(), parent_indices[1].item()

            # Load only the group slice from each parent (memory-efficient)
            p1 = torch.tensor(pool[idx1]['gene'][g_start:g_end], device=device, dtype=torch.float32)
            p2 = torch.tensor(pool[idx2]['gene'][g_start:g_end], device=device, dtype=torch.float32)

            # Crossover on group slice
            if torch.rand(1, device=device).item() < crossover_prob:
                # Blend crossover (BLX-alpha style)
                alpha   = torch.rand(1, device=device)
                child_g = alpha * p1 + (1.0 - alpha) * p2
            else:
                # Uniform crossover
                mask    = torch.rand_like(p1) > 0.5
                child_g = torch.where(mask, p1, p2)

            # Mutation on group slice
            mutation_mask = torch.rand_like(child_g) < mutation_rate
            noise         = torch.randn_like(child_g) * sigma
            child_g       = child_g + noise * mutation_mask.float()

            child[g_start:g_end] = child_g

        child = torch.clamp(child, lower_bound, 1.0)
        candidates.append(child.tolist())

    return candidates


def _load_mating_pool_from_disk(
    args,
    EA_Class,
    device: torch.device,
    verbose: int = 0
) -> List[Dict[str, Any]]:
    """
    Load elite genes from disk using streaming min-heap.
    
    Uses ChunkedGeneHistoryLoader for memory-efficient streaming
    with SLURM-aware dynamic chunk sizing.
    """
    def vprint(level: int, msg: str):
        if verbose >= level:
            print(msg, flush=True)
    
    evolution_target = getattr(args, 'evolutionTarget', 1)
    
    # Initialize loader
    loader = ChunkedGeneHistoryLoader(
        savePath=args.path,
        agent_id=getattr(args, 'agent_idx', getattr(args, 'agent_counter', 0)),
        geneFormat=getattr(EA_Class, 'geneFormat', 'json'),
        shuffle=True,
        shuffle_level='both',
        required_keys_and_types=['fitnessScore', 'gene'],
    )
    loader.clear_cache()
    
    # Calculate optimal sizes using loader's SLURM-aware methods
    # Get dynamic chunk size (SLURM-aware, process-based memory tracking)
    chunk_size = loader.calculate_chunk_size(
        safety_margin=0.5,
        chunk_allocation=0.20,
        min_chunk=100,
        max_chunk=10000,
        verbose=(verbose >= 3)
    )
    
    # Calculate pool size conservatively for GA (5000 cap)
    try:
        import psutil
        import os
        
        # Get SLURM-aware memory
        slurm_mem_bytes = None
        if 'SLURM_MEM_PER_NODE' in os.environ:
            slurm_mem_bytes = int(os.environ['SLURM_MEM_PER_NODE']) * 1024 * 1024
        elif 'SLURM_MEM' in os.environ:
            slurm_mem_bytes = int(os.environ['SLURM_MEM']) * 1024 * 1024
        
        if slurm_mem_bytes:
            process = psutil.Process()
            used_memory = process.memory_info().rss
            available_memory = slurm_mem_bytes - used_memory
        else:
            available_memory = psutil.virtual_memory().available
        
        bytes_per_record = (EA_Class.geneLength * 4) + 1024
        raw_capacity = int((available_memory * 0.5) / bytes_per_record)
        pool_size = max(256, min(5000, raw_capacity))
    except:
        pool_size = 5000
    
    # 80/20 exploration/exploitation strategy
    top_k_percentile = 25 if np.random.rand() < 0.20 else 10
    
    vprint(2, f"[GA] Loading top {100 - top_k_percentile}% genes (pool_size={pool_size})")
    vprint(3, f"[GA] Chunk size: {chunk_size}")
    
    # Stream and collect top genes using min-heap
    import itertools
    top_k_heap = []
    _heap_counter = itertools.count()  # unique tiebreaker to avoid dict/array comparison
    total_processed = 0

    for chunk, is_done in loader.loadTopKPercentileGeneHistoryAsListOfDicts(
        args=args,
        top_k_percentile=top_k_percentile,
        maximize=(evolution_target == 1),
        chunk_size=chunk_size,
        fitness_scaler=None,
        use_fitness_scaler_weights=False,
        n_feature_clusters=1,
        allEpochs=True,
        min_datapoints=min(int(getattr(args, 'populationSize', 10000) * 0.05), 300),
    ):
        for gene in chunk:
            try:
                fitness = float(gene['fitnessScore'])
            except (KeyError, TypeError, ValueError):
                continue

            # Skip invalid fitness values
            if math.isnan(fitness) or math.isinf(fitness):
                continue
            
            # Min-heap: use fitness directly, heap maintains smallest at top
            heap_key = fitness if evolution_target == 1 else -fitness
            
            if len(top_k_heap) < pool_size:
                heapq.heappush(top_k_heap, (heap_key, next(_heap_counter), gene))
            elif heap_key > top_k_heap[0][0]:
                heapq.heapreplace(top_k_heap, (heap_key, next(_heap_counter), gene))
        
        total_processed += len(chunk)
        vprint(3, f"[GA] Processed {total_processed} genes, heap size: {len(top_k_heap)}")
        
        if is_done:
            break
    
    # Extract genes from heap (sorted by fitness, best first)
    pool = []
    while top_k_heap:
        _, _, gene = heapq.heappop(top_k_heap)
        pool.append(gene)
    pool.reverse()  # Best first
    
    if pool:
        fitness_vals = [g['fitnessScore'] for g in pool]
        vprint(2, f"[GA] Loaded {len(pool)} genes, fitness: [{min(fitness_vals):.4f}, {max(fitness_vals):.4f}]")
    
    return pool

# ----------------------------------------------------------------------------
# USAGE EXAMPLE
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # Mock classes for testing
    class MockGAClass:
        geneLength = 100
        geneFormat = 'json'
    
    class MockArgs:
        path = '/path/to/data'
        evolutionTarget = 1
        gpu = False
        numOFgenerations = 100
        agent_counter = 50
        agent_idx = 0
    
    # Create mock population
    mock_population = [
        {'gene': [0.5] * 100, 'fitnessScore': 0.9},
        {'gene': [0.3] * 100, 'fitnessScore': 0.7},
        {'gene': [-0.2] * 100, 'fitnessScore': 0.5},
        {'gene': [0.1] * 100, 'fitnessScore': 0.3},
    ]
    
    # Initialize build_in_params (can be empty, defaults will be set)
    build_in_params = {}
    
    # Generate candidates with different verbosity levels
    # verbose=0 or None: Silent
    # verbose=1: Minimal output
    # verbose=2: Standard output
    # verbose=3: Debug output
    candidates = createCandidateGene_GeneticAlgorithm(
        args=MockArgs(),
        EA_Class=MockGAClass(),
        genePopulation=mock_population,
        number_of_candidate_genes=10,
        build_in_params=build_in_params,
        verbose=2  # Standard output
    )
    
    print(f"\nGenerated {len(candidates)} candidates", flush=True)
    print(f"Each candidate has {len(candidates[0])} genes", flush=True)
    print(f"Sample gene values: {candidates[0][:5]}...", flush=True)
    
    print(f"\nGA parameters used: {build_in_params['GA']}", flush=True)
    print(f"GA log: {build_in_params['GA-log']}", flush=True)


