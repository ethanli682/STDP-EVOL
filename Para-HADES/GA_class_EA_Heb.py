# Hebbian-inspired Evolution Strategy (Natural ES with centered ranks)
# Adapted from HebbianMetaLearning/evolution_strategy_hebb.py (Salimans et al., 2017)
# with Cooperative Co-Evolution (coevo_group) support.

import os
import pickle
import numpy as np
import torch
from typing import List, Dict, Optional, Any

# --- LOCAL IMPORTS ---
try:
    from GA_utils_history_loader import ChunkedGeneHistoryLoader
    from GA_utils_scalers import ChunkedRankScaler, ChunkedLogZScoreScaler, ChunkedPerDimensionScaler
except ImportError:
    print("[EA_Heb WARNING] Could not import Loader/Scaler classes. Disk streaming will fail.", flush=True)


def _get_coevo_groups(EA_Class, n):
    """
    Return an ordered dict of co-evolution groups from EA_Class.coevo_groups.
    Falls back to a single group covering the full gene when the attribute is absent.
    """
    if hasattr(EA_Class, 'coevo_groups') and EA_Class.coevo_groups:
        return EA_Class.coevo_groups
    return {'_all': {'start': 0, 'end': n, 'dim': n, 'params': ['_all']}}


def _init_group_state(n_g, sigma=0.5, learning_rate=1.0):
    """Create fresh ES state for a single coevo group of dimension n_g."""
    return {
        'iteration': 0,
        'm': np.zeros(n_g, dtype=np.float32),
        'sigma': float(sigma),
        'learning_rate': float(learning_rate),
    }


def _compute_centered_ranks(x):
    """
    Maps fitness values to centered ranks in [-0.5, 0.5].
    Equivalent to evolution_strategy_hebb.compute_centered_ranks.
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    y = ranks.astype(np.float32)
    y /= (x.size - 1)
    y -= 0.5
    return y


def _update_group_es(gs, elite_slice_np, fitness_scores, evolution_target,
                     lr_decay, sigma_decay, sigma_min, lr_min,
                     sigma_init, stagnation_threshold, sigma_increase_factor,
                     min_improvement):
    """
    Run one ES update step on a single group.
    elite_slice_np: (eff_lambda, n_g) numpy array of elite genes (unbounded) for this group.
    fitness_scores: (eff_lambda,) numpy array of raw fitness scores.

    Uses a fitness-weighted centroid update (CMA-ES mean update style) rather than
    the NES noise-vector gradient. The NES gradient requires symmetric noise vectors
    from the sampled population; when we only have elite historical genes (all
    above-average, biased in one direction), dividing deviations by sigma yields a
    systematically biased gradient. Instead we move m toward the fitness-weighted
    centroid of the elites, which is unbiased with respect to sample asymmetry.

    Adaptive sigma: sigma decays only when fitness improves.  If fitness stagnates
    for `stagnation_threshold` consecutive iterations, sigma is increased by
    `sigma_increase_factor` (capped at `sigma_init`) so the search can escape
    local optima.
    """
    m = gs['m']
    sigma = gs['sigma']
    lr = gs['learning_rate']
    iteration = gs['iteration']

    # ---- Stagnation detection ------------------------------------------------
    best_fitness_prev = gs.get('best_fitness', None)
    stagnant_count = gs.get('stagnant_count', 0)

    current_best = (float(np.max(fitness_scores)) if evolution_target == 1
                    else float(np.min(fitness_scores)))

    if best_fitness_prev is None:
        improved = True
    elif evolution_target == 1:
        improved = current_best > best_fitness_prev + abs(best_fitness_prev) * min_improvement + 1e-8
    else:
        improved = current_best < best_fitness_prev - abs(best_fitness_prev) * min_improvement - 1e-8

    if improved:
        stagnant_count = 0
        new_best_fitness = current_best
    else:
        stagnant_count += 1
        new_best_fitness = best_fitness_prev

    # ---- Sigma adaptation ----------------------------------------------------
    if improved:
        # Good iteration: apply normal decay
        sigma_new = sigma * sigma_decay if sigma > sigma_min else sigma
    elif stagnant_count >= stagnation_threshold:
        # Stuck: push sigma up to escape local optimum.
        # Cap at 3× sigma_init so boosts can exceed the starting value.
        sigma_new = min(sigma * sigma_increase_factor, sigma_init * 3.0)
        stagnant_count = 0  # reset counter after restart
    else:
        # Stagnating but not yet threshold: hold sigma constant
        sigma_new = sigma

    # For minimization, negate fitness so that lower = better maps to higher rank
    scores_for_ranking = fitness_scores if evolution_target == 1 else -fitness_scores

    # Centered rank fitness shaping → non-negative weights that sum to 1
    ranks = _compute_centered_ranks(scores_for_ranking)  # in [-0.5, 0.5]
    # Shift so all weights are positive, then normalise
    weights = ranks - ranks.min() + 1e-8
    weights = weights / weights.sum()

    # Fitness-weighted centroid of the elites
    weighted_centroid = elite_slice_np.T @ weights  # shape (n_g,)

    # Move mean toward weighted centroid
    m_new = m + lr * (weighted_centroid - m)

    # Decay learning rate
    lr_new = lr * lr_decay if lr > lr_min else lr

    gs.update({
        'm': m_new.astype(np.float32),
        'sigma': float(sigma_new),
        'learning_rate': float(lr_new),
        'iteration': iteration + 1,
        'best_fitness': new_best_fitness,
        'stagnant_count': stagnant_count,
    })
    return gs


def createCandidateGene_EA_Heb(
    args: Any = None,
    EA_Class: Any = None,
    genePopulation: Optional[List[Dict[str, Any]]] = None,
    number_of_candidate_genes: Optional[int] = None,
    build_in_params: Optional[Dict] = None,
    verbose: Optional[int] = None
) -> List[List[float]]:
    """
    Hebbian-inspired Natural Evolution Strategy with Cooperative Co-Evolution support.

    Each coevo_group gets its own independent ES state (mean, sigma, learning_rate).
    Uses centered-rank fitness shaping and antithetic (mirrored) sampling.
    """

    # ════════════════════════════════════════════════════════════════════════
    # 1. SETUP & VALIDATION
    # ════════════════════════════════════════════════════════════════════════
    if args is None: raise ValueError("args cannot be None")
    if EA_Class is None: raise ValueError("EA_Class cannot be None")
    if number_of_candidate_genes is None: raise ValueError("number_of_candidate_genes cannot be None")
    if build_in_params is None: raise ValueError("build_in_params cannot be None")

    if 'EA_Heb' not in build_in_params: build_in_params['EA_Heb'] = {}
    if 'EA_Heb-log' not in build_in_params: build_in_params['EA_Heb-log'] = {}

    config = build_in_params['EA_Heb']
    sigma_init = float(config.get('sigma', 0.5))
    learning_rate_init = float(config.get('learning_rate', 1.0))
    lr_decay = float(config.get('lr_decay', 0.995))
    sigma_decay = float(config.get('sigma_decay', 0.999))
    sigma_min = float(config.get('sigma_min', 0.01))
    lr_min = float(config.get('lr_min', 0.001))
    use_mirroring = config.get('use_mirroring', True)
    scaler_type = config.get('scaler_type', 'rank')
    evolution_target = getattr(args, 'evolutionTarget', 1)
    # Adaptive sigma / stagnation config
    stagnation_threshold = int(config.get('stagnation_threshold', 5))
    sigma_increase_factor = float(config.get('sigma_increase_factor', 1.2))
    min_improvement = float(config.get('min_improvement', 0.005))

    use_gpu = getattr(args, 'gpu', False) and torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')

    n = int(EA_Class.geneLength)
    gene_min = getattr(EA_Class, 'geneMin', -1.0)

    # ════════════════════════════════════════════════════════════════════════
    # 2. COEVO GROUPS & STATE MANAGEMENT
    # ════════════════════════════════════════════════════════════════════════
    coevo_groups = _get_coevo_groups(EA_Class, n)
    group_names = sorted(coevo_groups.keys(), key=lambda g: coevo_groups[g]['start'])

    state_path = os.path.join(EA_Class.savePath, 'running', f'{str(args.agent_idx)}_EA_Heb_metadata_.pkl')
    es_state = {}

    if os.path.exists(state_path):
        try:
            with open(state_path, 'rb') as f:
                es_state = pickle.load(f)
        except Exception as e:
            print(f"[EA_Heb ERROR] Corrupt state file: {e}. Starting fresh.", flush=True)

    # Validate state matches current group structure
    state_valid = (
        'groups' in es_state and
        set(es_state['groups'].keys()) == set(group_names)
    )
    if state_valid:
        for gname in group_names:
            gs = es_state['groups'][gname]
            expected_dim = coevo_groups[gname]['dim']
            if len(gs.get('m', [])) != expected_dim:
                state_valid = False
                break

    if not state_valid:
        if verbose and verbose > 0:
            print(f"[EA_Heb] Initializing new state with {len(group_names)} coevo groups", flush=True)
        # Warm-start mean from population centroid (arctanh space) to skip the
        # early iterations where m=0 is far from the actual gene distribution.
        init_m_full = None
        if genePopulation is not None and len(genePopulation) > 0:
            try:
                eps = 1e-6
                genes_raw = []
                for p in genePopulation:
                    if p.get('fitnessScore') is not None and p.get('gene') is not None:
                        g = np.array(p['gene'], dtype=np.float32)
                        if len(g) == n:
                            if gene_min == 0.0:
                                g = g * 2.0 - 1.0
                            genes_raw.append(np.arctanh(np.clip(g, -1.0 + eps, 1.0 - eps)))
                if genes_raw:
                    init_m_full = np.mean(genes_raw, axis=0).astype(np.float32)
            except Exception:
                init_m_full = None

        es_state = {'groups': {}}
        for gname in group_names:
            n_g = coevo_groups[gname]['dim']
            gs = _init_group_state(n_g, sigma_init, learning_rate_init)
            if init_m_full is not None:
                g_start = coevo_groups[gname]['start']
                g_end = coevo_groups[gname]['end']
                gs['m'] = init_m_full[g_start:g_end]
            es_state['groups'][gname] = gs

    if verbose and verbose > 0:
        for gname in group_names:
            gs = es_state['groups'][gname]
            n_g = coevo_groups[gname]['dim']
            print(f"[EA_Heb]   Group '{gname}': dim={n_g} sigma={gs['sigma']:.4f} "
                  f"lr={gs['learning_rate']:.4f} iter={gs['iteration']}", flush=True)

    # ════════════════════════════════════════════════════════════════════════
    # 3. DATA LOADING (IN-MEMORY OR STREAMING)
    # ════════════════════════════════════════════════════════════════════════
    elite_genes = None
    elite_fitness = None
    data_available = False
    feature_scaler = None

    # Target number of elites for the update
    pop_size = int(getattr(args, 'populationSize', 100))
    target_elites = max(8, min(pop_size, 4 + int(3 * np.log(n))))

    # --- PATH A: In-Memory Provided ---
    if genePopulation is not None and len(genePopulation) > 0:
        try:
            meta = [(i, p['fitnessScore']) for i, p in enumerate(genePopulation)
                    if p.get('fitnessScore') is not None]

            if len(meta) >= target_elites:
                meta.sort(key=lambda x: x[1], reverse=(evolution_target == 1))
                target_lambda = min(len(meta), target_elites)
                indices = [x[0] for x in meta[:target_lambda]]

                eps = 1e-6
                genes_list = []
                scores_list = []
                for idx in indices:
                    g_val = np.array(genePopulation[idx]['gene'])
                    if gene_min == 0.0:
                        g_val = g_val * 2.0 - 1.0
                    g_val = np.clip(g_val, -1.0+eps, 1.0-eps)
                    genes_list.append(np.arctanh(g_val))
                    scores_list.append(genePopulation[idx]['fitnessScore'])

                if len(genes_list[0]) == n:
                    elite_genes = np.array(genes_list, dtype=np.float32)
                    elite_fitness = np.array(scores_list, dtype=np.float32)
                    data_available = True
        except Exception as e:
            print(f"[EA_Heb ERROR] Processing genePopulation failed: {e}", flush=True)

    # --- PATH B: Disk Streaming (OOM Safe) ---
    elif genePopulation is None:
        try:
            loader = ChunkedGeneHistoryLoader(
                savePath=EA_Class.savePath, agent_id=args.agent_idx, geneFormat=getattr(EA_Class, 'geneFormat', 'json'), shuffle=True
            )
            loader.clear_cache()

            if scaler_type == 'zscore':
                fitness_scaler = ChunkedLogZScoreScaler(
                    evolution_target=evolution_target, device=device, emphasis_factor=2.0)
            else:
                fitness_scaler = ChunkedRankScaler(evolution_target=evolution_target, device=device)

            feature_scaler = ChunkedPerDimensionScaler(
                gene_length=EA_Class.geneLength, device=device, weight_mode='uniform')

            try:
                chunk_size = loader.calculate_chunk_size(
                    safety_margin=0.5, chunk_allocation=0.20,
                    verbose=(verbose and verbose > 0))
            except Exception as e:
                if verbose and verbose > 0:
                    print(f"[EA_Heb WARNING] Chunk size calculation failed: {e}. Using default 1000", flush=True)
                chunk_size = 1000

            import heapq
            import itertools
            elite_heap = []
            heap_size = chunk_size
            _heap_counter = itertools.count()  # unique tiebreaker to avoid numpy array comparison

            min_datapoints_value = max(
                min(int(pop_size * 0.05), 300),
                int(min(50, chunk_size // 3)),
            )

            for chunk, is_done in loader.loadTopKPercentileGeneHistoryAsListOfDicts(
                args=args, top_k_percentile=7.0, maximize=(evolution_target == 1),
                chunk_size=chunk_size, fitness_scaler=fitness_scaler,
                use_fitness_scaler_weights=True, allEpochs=True,
                min_datapoints=min_datapoints_value,
                fit_fitness_scaler_on_top_k=True,
                fit_gene_scaler_on_top_k=True,
                max_datapoints=int(pop_size),
                gene_scaler=feature_scaler,
            ):
                for rec in chunk:
                    if len(rec['gene']) == n:
                        eps = 1e-6
                        g_arr = np.array(rec['gene'])
                        if gene_min == 0.0:
                            g_arr = g_arr * 2.0 - 1.0
                        g_arr = np.clip(g_arr, -1.0+eps, 1.0-eps)
                        gene_unbounded = np.arctanh(g_arr)
                        score = float(rec['fitnessScore'])

                        heap_key = -score if evolution_target == 1 else score
                        if len(elite_heap) < heap_size:
                            heapq.heappush(elite_heap, (heap_key, next(_heap_counter), gene_unbounded, score))
                        elif heap_key < elite_heap[0][0]:
                            heapq.heapreplace(elite_heap, (heap_key, next(_heap_counter), gene_unbounded, score))

            collected_genes = []
            collected_scores = []
            while elite_heap:
                _, _, gene, score = heapq.heappop(elite_heap)
                collected_genes.append(gene)
                collected_scores.append(score)

            if collected_genes:
                genes_np = np.array(collected_genes)
                scores_np = np.array(collected_scores)

                if evolution_target == 1:
                    indices = np.argsort(scores_np)[::-1]
                else:
                    indices = np.argsort(scores_np)

                target_lambda = min(len(indices), target_elites)
                best_indices = indices[:target_lambda]

                genes_selected = genes_np[best_indices]
                if feature_scaler is not None and getattr(feature_scaler, 'scaling_ready', False):
                    genes_tensor = torch.tensor(genes_selected, dtype=torch.float32, device=device)
                    genes_normalized = feature_scaler.transform(genes_tensor)
                    elite_genes = genes_normalized.cpu().numpy().astype(np.float32)
                    es_state['use_gene_scaler'] = True
                    if verbose and verbose > 1:
                        print(f"[EA_Heb] Applied gene normalization via feature_scaler", flush=True)
                else:
                    elite_genes = genes_selected.astype(np.float32)
                    es_state['use_gene_scaler'] = False

                elite_fitness = scores_np[best_indices].astype(np.float32)
                data_available = True
        except Exception as e:
            print(f"[EA_Heb ERROR] Streaming failed: {e}", flush=True)

    # ════════════════════════════════════════════════════════════════════════
    # 4. PER-GROUP UPDATE (ALGORITHM CORE)
    # ════════════════════════════════════════════════════════════════════════
    log_sigmas = {}
    log_lrs = {}

    if data_available and elite_genes is not None:
        for gname in group_names:
            ginfo = coevo_groups[gname]
            g_start = ginfo['start']
            g_end = ginfo['end']
            gs = es_state['groups'][gname]

            # Slice this group's columns from elite genes
            elite_slice = elite_genes[:, g_start:g_end]

            try:
                updated_gs = _update_group_es(
                    gs, elite_slice, elite_fitness, evolution_target,
                    lr_decay, sigma_decay, sigma_min, lr_min,
                    sigma_init, stagnation_threshold, sigma_increase_factor,
                    min_improvement)

                es_state['groups'][gname] = updated_gs
                log_sigmas[gname] = updated_gs['sigma']
                log_lrs[gname] = updated_gs['learning_rate']

                if verbose and verbose > 0:
                    sc = updated_gs.get('stagnant_count', 0)
                    bf = updated_gs.get('best_fitness', None)
                    bf_str = f"{bf:.4f}" if bf is not None else "N/A"
                    print(f"[EA_Heb] Group '{gname}': sigma={updated_gs['sigma']:.4f} "
                          f"lr={updated_gs['learning_rate']:.4f} "
                          f"stagnant={sc}/{stagnation_threshold} best={bf_str}", flush=True)

            except Exception as e:
                if verbose and verbose > 0:
                    print(f"[EA_Heb ERROR] Update failed for group '{gname}': {e}", flush=True)

        # Logs
        if log_sigmas:
            avg_sigma = sum(log_sigmas.values()) / len(log_sigmas)
            avg_lr = sum(log_lrs.values()) / len(log_lrs)
            avg_stagnant = sum(
                es_state['groups'][g].get('stagnant_count', 0) for g in group_names
            ) / len(group_names)
            build_in_params['EA_Heb-log']['sigma'] = avg_sigma
            build_in_params['EA_Heb-log']['sigma_per_group'] = log_sigmas
            build_in_params['EA_Heb-log']['learning_rate'] = avg_lr
            build_in_params['EA_Heb-log']['lr_per_group'] = log_lrs
            build_in_params['EA_Heb-log']['avg_stagnant_count'] = avg_stagnant
            build_in_params['loss'] = avg_sigma

    # ════════════════════════════════════════════════════════════════════════
    # 5. PER-GROUP CANDIDATE GENERATION & ASSEMBLY
    # ════════════════════════════════════════════════════════════════════════
    # Save state
    try:
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        with open(state_path, 'wb') as f:
            pickle.dump(es_state, f)
    except Exception as e:
        if verbose and verbose > 0:
            print(f"[EA_Heb WARNING] State save failed: {e}", flush=True)

    # Generate per-group noise with antithetic (mirrored) sampling
    pop = number_of_candidate_genes
    gene_parts = []

    for gname in group_names:
        gs = es_state['groups'][gname]
        n_g = coevo_groups[gname]['dim']
        m_g = gs['m']
        sigma_g = gs['sigma']

        if use_mirroring:
            half = pop // 2
            z_half = np.random.randn(half, n_g).astype(np.float32)
            z_g = np.concatenate([z_half, -z_half], axis=0)
            if z_g.shape[0] < pop:
                z_g = np.concatenate([z_g, np.random.randn(1, n_g).astype(np.float32)], axis=0)
        else:
            z_g = np.random.randn(pop, n_g).astype(np.float32)

        # x = m + sigma * noise
        x_g = m_g[np.newaxis, :] + sigma_g * z_g
        gene_parts.append(x_g)

    x_unbounded = np.concatenate(gene_parts, axis=1)

    # Inverse transform if gene scaler was used
    use_gene_scaler = es_state.get('use_gene_scaler', False)
    if use_gene_scaler and feature_scaler is not None and getattr(feature_scaler, 'scaling_ready', False):
        x_tensor = torch.tensor(x_unbounded, dtype=torch.float32, device=device)
        x_denormalized = feature_scaler.inverse_transform(x_tensor).cpu().numpy()
        if verbose and verbose > 1:
            print(f"[EA_Heb] Applied inverse gene normalization", flush=True)
    else:
        x_denormalized = x_unbounded

    # Bound to [-1, 1] (or [0, 1] if geneMin=0) via tanh
    x_bounded = np.tanh(x_denormalized)

    if gene_min == 0.0:
        x_bounded = (x_bounded + 1.0) / 2.0

    lower_bound = 0.0 if gene_min == 0 else -1.0
    x_bounded = np.clip(x_bounded, lower_bound, 1.0)

    return x_bounded.tolist()
