# Generic SNES (Separable Natural Evolution Strategies) implementation for Genetic Algorithms

import os
import pickle
import numpy as np
import torch
import math
from typing import List, Dict, Optional, Any

# --- LOCAL IMPORTS ---
try:
    from GA_utils_history_loader import ChunkedGeneHistoryLoader
    from GA_utils_scalers import ChunkedRankScaler, ChunkedLogZScoreScaler,ChunkedPerDimensionScaler
except ImportError:
    print("[SNES WARNING] Could not import Loader/Scaler classes. Disk streaming will fail.", flush=True)


def _quality_diversity_rerank(genes_np, scores_np, evolution_target, target_count, alpha=0.3, k_neighbors=5):
    """
    Re-rank candidates using quality-diversity: blend fitness rank with novelty rank.

    Args:
        genes_np: (N, D) array of gene vectors
        scores_np: (N,) array of fitness scores
        evolution_target: 1=maximize, -1=minimize
        target_count: how many to select
        alpha: weight for novelty (0=pure fitness, 1=pure novelty)
        k_neighbors: number of neighbors for novelty score

    Returns:
        indices into genes_np/scores_np for the selected subset
    """
    n = len(scores_np)
    if n <= target_count:
        return np.arange(n)

    # Fitness ranks (0 = best)
    if evolution_target == 1:
        fitness_order = np.argsort(-scores_np)
    else:
        fitness_order = np.argsort(scores_np)
    fitness_rank = np.empty(n, dtype=np.float64)
    fitness_rank[fitness_order] = np.arange(n, dtype=np.float64)

    # Novelty: average distance to k nearest neighbors (higher = more novel)
    try:
        from sklearn.neighbors import NearestNeighbors
        k = min(k_neighbors, n - 1)
        if k > 0:
            nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean', algorithm='auto')
            nn.fit(genes_np)
            distances, _ = nn.kneighbors(genes_np)
            novelty_scores = np.mean(distances[:, 1:], axis=1)  # exclude self
        else:
            novelty_scores = np.zeros(n)
    except Exception:
        novelty_scores = np.zeros(n)

    # Novelty ranks (0 = most novel)
    novelty_order = np.argsort(-novelty_scores)
    novelty_rank = np.empty(n, dtype=np.float64)
    novelty_rank[novelty_order] = np.arange(n, dtype=np.float64)

    # Combined score (lower = better)
    combined_rank = (1.0 - alpha) * fitness_rank + alpha * novelty_rank
    selected = np.argsort(combined_rank)[:target_count]
    return selected


def _get_coevo_groups(EA_Class, n):
    """
    Return an ordered dict of co-evolution groups from EA_Class.coevo_groups.
    Falls back to a single group covering the full gene when the attribute is absent.
    """
    if hasattr(EA_Class, 'coevo_groups') and EA_Class.coevo_groups:
        return EA_Class.coevo_groups
    return {'_all': {'start': 0, 'end': n, 'dim': n, 'params': ['_all']}}


def createCandidateGene_SNES(
    args: Any = None,
    EA_Class: Any = None,
    genePopulation: Optional[List[Dict[str, Any]]] = None,
    number_of_candidate_genes: Optional[int] = None,
    build_in_params: Optional[Dict] = None,
    verbose: Optional[int] = None
) -> List[List[float]]:
    """
    State-of-the-Art SNES (Separable Natural Evolution Strategies) Generator.

    Why use this over CMA-ES?
    1. Linear Complexity O(N): Scales to millions of parameters (vs CMA-ES which fails > 5k params).
    2. Natural Gradients: Uses Fisher Information for mathematically optimal updates.
    3. GPU Native: Fully vectorized PyTorch implementation.
    4. Streaming Ready: Compatible with ChunkedLoader for memory-safe big-data training.

    Co-evolution:
    Each parameter group defined in EA_Class.coevo_groups (derived from the YAML coevo_group tags)
    maintains its own independent (mu, sigma) state and is updated using only its own noise slice.
    All groups share the same fitness signal but do NOT share gradient information.
    """

    # ════════════════════════════════════════════════════════════════════════
    # 1. SETUP & VALIDATION
    # ════════════════════════════════════════════════════════════════════════
    if args is None: raise ValueError("args cannot be None")
    if EA_Class is None: raise ValueError("EA_Class cannot be None")
    if number_of_candidate_genes is None: raise ValueError("number_of_candidate_genes cannot be None")
    if build_in_params is None: raise ValueError("build_in_params cannot be None")

    # Ensure dictionaries exist
    if 'SNES' not in build_in_params: build_in_params['SNES'] = {}
    if 'SNES-log' not in build_in_params: build_in_params['SNES-log'] = {}

    # Configuration
    config = build_in_params['SNES']
    scaler_type = config.get('scaler_type', 'rank') # 'rank' is usually more stable for SNES
    learning_rate_mu = config.get('lr_mu', 1.0)     # Global multiplier for Mean learning rate
    learning_rate_sigma = config.get('lr_sigma', 1.0) # Global multiplier for Sigma learning rate
    use_momentum = config.get('use_momentum', True)
    momentum_decay = config.get('momentum_decay', 0.9)
    use_mirroring = config.get('use_mirroring', True)
    restart_on_stagnation = config.get('restart_on_stagnation', True)
    sigma_min_restart = float(config.get('sigma_min_restart', 1e-4))
    sigma_max_restart = float(config.get('sigma_max_restart', 4.0))
    grad_stagnation_threshold = float(config.get('grad_stagnation_threshold', 1e-6))
    random_injection_rate = float(config.get('random_injection_rate', 0.15))  # fraction of candidates that are pure random
    stagnation_window = int(config.get('stagnation_window', 10))  # iterations to check for improvement
    sigma_boost_factor = float(config.get('sigma_boost_factor', 2.0))  # sigma multiplier on stagnation
    evolution_target = getattr(args, 'evolutionTarget', 1)

    # Hardware
    use_gpu = getattr(args, 'gpu', False) and torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')

    # Co-evolution groups
    n = int(EA_Class.geneLength)
    coevo_groups = _get_coevo_groups(EA_Class, n)
    group_names = list(coevo_groups.keys())

    if verbose and verbose > 0:
        print(f"[SNES] Device: {device} | Scaler: {scaler_type.upper()} | Groups: {len(group_names)}", flush=True)

    # ════════════════════════════════════════════════════════════════════════
    # 2. STATE MANAGEMENT (PERSISTENCE)
    # ════════════════════════════════════════════════════════════════════════
    state_path = os.path.join(EA_Class.savePath, 'running', f'{str(args.agent_idx)}_SNES_metadata_.pkl')
    snes_state = {}

    # Load State
    if os.path.exists(state_path):
        try:
            with open(state_path, 'rb') as f: snes_state = pickle.load(f)
        except Exception as e:
            print(f"[SNES ERROR] Corrupt state file: {e}. Starting fresh.", flush=True)

    default_lambda = 4 + int(3 * np.log(n))

    # Validate state: must have per-group entries with matching group names
    state_valid = (
        'groups' in snes_state and
        set(snes_state['groups'].keys()) == set(group_names)
    )

    if not state_valid:
        if verbose and verbose > 0:
            print(f"[SNES] Initializing new per-group state ({len(group_names)} groups, n={n})...", flush=True)
        snes_state['iteration'] = 0
        snes_state['lambda'] = default_lambda
        snes_state['groups'] = {}
        for gname, ginfo in coevo_groups.items():
            d = ginfo['dim']
            snes_state['groups'][gname] = {
                'mu':      np.zeros(d, dtype=np.float32),
                'sigma':   np.ones(d,  dtype=np.float32) * 0.5,
                'v_mu':    np.zeros(d, dtype=np.float32),
                'v_sigma': np.zeros(d, dtype=np.float32),
            }

    # ════════════════════════════════════════════════════════════════════════
    # 3. DATA LOADING (IN-MEMORY OR STREAMING)
    # ════════════════════════════════════════════════════════════════════════
    elite_genes = None
    elite_weights = None
    data_available = False
    gene_scaler = None  # Initialize at top level to avoid scope issues
    best_current_fitness = None  # tracked for stagnation detection

    # --- PATH A: In-Memory Provided ---
    if genePopulation is not None and len(genePopulation) > 0:
        try:
            meta = [(i, p['fitnessScore']) for i, p in enumerate(genePopulation) if p.get('fitnessScore') is not None]

            if len(meta) >= default_lambda:
                # Sort (Best First)
                meta.sort(key=lambda x: x[1], reverse=(evolution_target == 1))

                target_lambda = min(len(meta), default_lambda * 2)
                indices = [x[0] for x in meta[:target_lambda]]

                eps = 1e-6
                genes_list = []
                for idx in indices:
                    g_val = np.array(genePopulation[idx]['gene'])
                    if getattr(EA_Class, 'geneMin', -1.0) == 0.0:
                        g_val = g_val * 2.0 - 1.0
                    g_val = np.clip(g_val, -1.0+eps, 1.0-eps)
                    genes_list.append(np.arctanh(g_val))

                if len(genes_list[0]) == n:
                    elite_genes = torch.tensor(np.array(genes_list), dtype=torch.float32, device=device)
                    snes_state['eff_lambda'] = target_lambda
                    best_current_fitness = meta[0][1]  # best-first after sort
                    data_available = True
        except Exception as e:
            print(f"[SNES ERROR] Processing genePopulation failed: {e}", flush=True)

    # --- PATH B: Disk Streaming (OOM Safe) ---
    elif genePopulation is None:
        try:
            loader = ChunkedGeneHistoryLoader(
                savePath=EA_Class.savePath, agent_id=args.agent_idx, geneFormat=getattr(EA_Class, 'geneFormat', 'json'), shuffle=True
            )
            loader.clear_cache()

            if scaler_type == 'zscore':
                fit_scaler = ChunkedLogZScoreScaler(evolution_target=evolution_target, device=device, emphasis_factor=2.0)
            else:
                fit_scaler = ChunkedRankScaler(evolution_target=evolution_target, device=device)

            gene_scaler = ChunkedPerDimensionScaler(gene_length=EA_Class.geneLength, device=device)

            # Get SLURM-aware chunk size
            try:
                chunk_size = loader.calculate_chunk_size(safety_margin=0.5, chunk_allocation=0.20, verbose=(verbose and verbose > 0))
            except Exception as e:
                if verbose and verbose > 0:
                    print(f"[SNES WARNING] Chunk size calculation failed: {e}. Using default 10000", flush=True)
                chunk_size = 10000

            # MinHeap to keep only top chunk_size candidates
            import heapq
            import itertools
            elite_heap = []  # Min-heap: (heap_key, counter, gene, score)
            heap_size = chunk_size
            _heap_counter = itertools.count()  # unique tiebreaker to avoid numpy array comparison

            # Load top 7% to get a good gradient approximation
            for chunk, is_done in loader.loadTopKPercentileGeneHistoryAsListOfDicts(
                args=args, top_k_percentile=7.0, maximize=(evolution_target == 1),
                chunk_size=chunk_size, fitness_scaler=fit_scaler, use_fitness_scaler_weights=True, allEpochs=True,
                min_datapoints=min(int(getattr(args, 'populationSize', 1000) * 0.05), 300),
                max_datapoints=int(getattr(args, 'populationSize', 1000)),
                gene_scaler=gene_scaler,
                fit_fitness_scaler_on_top_k=True,
                fit_gene_scaler_on_top_k=True,
            ):
                for rec in chunk:
                    try:
                        gene = rec['gene']
                        score = float(rec['fitnessScore'])
                    except (KeyError, TypeError):
                        continue
                    if len(gene) == n:
                        eps = 1e-6
                        g_arr = np.array(gene)
                        if getattr(EA_Class, 'geneMin', -1.0) == 0.0:
                            g_arr = g_arr * 2.0 - 1.0
                        g_arr = np.clip(g_arr, -1.0+eps, 1.0-eps)
                        gene_unbounded = np.arctanh(g_arr)

                        heap_key = -score if evolution_target == 1 else score

                        if len(elite_heap) < heap_size:
                            heapq.heappush(elite_heap, (heap_key, next(_heap_counter), gene_unbounded, score))
                        elif heap_key < elite_heap[0][0]:
                            heapq.heapreplace(elite_heap, (heap_key, next(_heap_counter), gene_unbounded, score))

            # Extract from heap
            collected_genes = []
            collected_scores = []
            while elite_heap:
                _, _, gene, score = heapq.heappop(elite_heap)
                collected_genes.append(gene)
                collected_scores.append(score)

            if collected_genes:
                genes_np = np.array(collected_genes)
                scores_np = np.array(collected_scores)

                target_lambda = min(len(scores_np), default_lambda * 3)
                # Quality-diversity selection: blend fitness rank with novelty
                qd_alpha = float(config.get('qd_alpha', 0.2))  # 20% novelty weight
                # Per-agent diversity: different agents use different k_neighbors to favor different pockets
                agent_k = max(3, 5 + int(args.agent_idx or 0) * 2)
                best_indices = _quality_diversity_rerank(
                    genes_np, scores_np, evolution_target, target_lambda, alpha=qd_alpha, k_neighbors=agent_k
                )
                if verbose and verbose > 0:
                    print(f"[SNES] Quality-diversity selection: {len(best_indices)}/{len(scores_np)} (alpha={qd_alpha})", flush=True)

                genes_selected = genes_np[best_indices]
                if gene_scaler is not None and getattr(gene_scaler, 'scaling_ready', False):
                    genes_tensor = torch.tensor(genes_selected, dtype=torch.float32, device=device)
                    genes_normalized = gene_scaler.transform(genes_tensor)
                    elite_genes = genes_normalized.to(dtype=torch.float32, device=device)
                    if verbose and verbose > 1:
                        print(f"[SNES] Applied gene normalization via gene_scaler", flush=True)
                else:
                    elite_genes = torch.tensor(genes_selected, dtype=torch.float32, device=device)
                    if verbose and verbose > 1:
                        print(f"[SNES] Gene scaler not ready; using raw genes", flush=True)

                if scaler_type == 'zscore':
                    scores_selected = torch.tensor(scores_np[best_indices], dtype=torch.float32, device=device)
                    raw_weights = fit_scaler.compute_sample_weights(scores_selected)
                    elite_weights = raw_weights - torch.mean(raw_weights)
                    elite_weights = elite_weights / (torch.sum(torch.abs(elite_weights)) + 1e-8)
                else:
                    elite_weights = None

                snes_state['eff_lambda'] = target_lambda
                snes_state['use_gene_scaler'] = (gene_scaler is not None and getattr(gene_scaler, 'scaling_ready', False))
                scores_selected = scores_np[best_indices]
                best_current_fitness = (float(np.max(scores_selected)) if evolution_target == 1
                                        else float(np.min(scores_selected)))
                data_available = True
        except Exception as e:
             print(f"[SNES ERROR] Streaming failed: {e}", flush=True)

    # ════════════════════════════════════════════════════════════════════════
    # 4. THE UPDATE STEP — PER-GROUP CO-EVOLUTION
    # Each group is updated independently using only its own gene slice.
    # All groups share the same fitness/utility weights.
    # ════════════════════════════════════════════════════════════════════════
    if data_available and elite_genes is not None:
        try:
            eff_lambda = snes_state['eff_lambda']

            # --- UTILITIES (shared across all groups) ---
            if elite_weights is not None:
                utilities = elite_weights
            else:
                ranks = torch.arange(1, eff_lambda + 1, device=device, dtype=torch.float32)
                mu_rank = eff_lambda / 2
                raw_utilities = torch.clamp(
                    torch.log(torch.tensor(mu_rank + 1.0, device=device)) - torch.log(ranks), min=0.0
                )
                utilities = raw_utilities - torch.mean(raw_utilities)
                utilities = utilities / (torch.sum(torch.abs(utilities)) + 1e-8)

            do_restart_any = False

            # --- PER-GROUP UPDATE ---
            for gname, ginfo in coevo_groups.items():
                g_start = ginfo['start']
                g_end   = ginfo['end']
                n_g     = ginfo['dim']

                gs = snes_state['groups'][gname]
                mu    = torch.tensor(gs['mu'],    device=device, dtype=torch.float32)
                sigma = torch.tensor(gs['sigma'], device=device, dtype=torch.float32)

                # Slice: only this group's columns from the elite gene matrix
                z_samples = (elite_genes[:, g_start:g_end] - mu.unsqueeze(0)) / (sigma.unsqueeze(0) + 1e-8)

                # Natural gradients
                grad_mu    = torch.mv(z_samples.T, utilities)
                grad_sigma = torch.mv((z_samples**2 - 1.0).T, utilities)

                # Per-group learning rates (Wierstra et al. 2014, scaled to group dim)
                eta_mu    = learning_rate_mu
                eta_sigma = learning_rate_sigma * (3 + np.log(n_g)) / (5 * np.sqrt(n_g))

                if use_momentum:
                    v_mu    = torch.tensor(gs['v_mu'],    device=device, dtype=torch.float32)
                    v_sigma = torch.tensor(gs['v_sigma'], device=device, dtype=torch.float32)

                    v_mu    = momentum_decay * v_mu    + (1 - momentum_decay) * grad_mu
                    v_sigma = momentum_decay * v_sigma + (1 - momentum_decay) * grad_sigma

                    mu_new    = mu    + eta_mu * sigma * v_mu
                    sigma_new = sigma * torch.exp((eta_sigma / 2.0) * v_sigma)

                    gs['v_mu']    = v_mu.cpu().numpy()
                    gs['v_sigma'] = v_sigma.cpu().numpy()
                else:
                    mu_new    = mu    + eta_mu * sigma * grad_mu
                    sigma_new = sigma * torch.exp((eta_sigma / 2.0) * grad_sigma)

                sigma_new = torch.clamp(sigma_new, 1e-3, 5.0)

                # Restart check per group
                mean_sigma = torch.mean(sigma_new).item()
                grad_norm  = torch.norm(grad_mu).item()
                do_restart = restart_on_stagnation and (
                    mean_sigma < sigma_min_restart or
                    mean_sigma > sigma_max_restart or
                    grad_norm < grad_stagnation_threshold
                )

                if do_restart:
                    if verbose and verbose > 0:
                        reason = "sigma_too_small" if mean_sigma < sigma_min_restart else ("sigma_too_large" if mean_sigma > sigma_max_restart else "grad_vanished")
                        print(f"[SNES] Restart group '{gname}' ({reason}): sigma={mean_sigma:.3f}, grad_norm={grad_norm:.2e}", flush=True)
                    gs['mu']      = np.random.randn(n_g).astype(np.float32) * 0.1
                    gs['sigma']   = np.ones(n_g, dtype=np.float32) * 0.5
                    gs['v_mu']    = np.zeros(n_g, dtype=np.float32)
                    gs['v_sigma'] = np.zeros(n_g, dtype=np.float32)
                    do_restart_any = True
                else:
                    gs['mu']    = mu_new.cpu().numpy()
                    gs['sigma'] = sigma_new.cpu().numpy()

            snes_state['iteration'] = snes_state.get('iteration', 0) + 1

            # --- FITNESS STAGNATION DETECTION & SIGMA BOOST ---
            if 'best_fitness_history' not in snes_state:
                snes_state['best_fitness_history'] = []

            if best_current_fitness is not None:
                snes_state['best_fitness_history'].append(best_current_fitness)
                if len(snes_state['best_fitness_history']) > stagnation_window * 2:
                    snes_state['best_fitness_history'] = snes_state['best_fitness_history'][-stagnation_window * 2:]

            fitness_history = snes_state['best_fitness_history']
            if len(fitness_history) >= stagnation_window and not do_restart_any:
                window = fitness_history[-stagnation_window:]
                best_in_window = max(window) if evolution_target == 1 else min(window)
                worst_in_window = min(window) if evolution_target == 1 else max(window)
                relative_improvement = abs(best_in_window - worst_in_window) / (abs(worst_in_window) + 1e-8)

                if relative_improvement < 0.01:  # < 1% improvement over window
                    if verbose and verbose > 0:
                        print(f"[SNES] Stagnation detected (improvement={relative_improvement:.4f} over {stagnation_window} iters). Boosting sigma.", flush=True)
                    for gname in group_names:
                        gs = snes_state['groups'][gname]
                        gs['sigma'] = np.clip(gs['sigma'] * sigma_boost_factor, 1e-3, sigma_max_restart).astype(np.float32)
                    snes_state['best_fitness_history'] = []  # reset window after boost

            # Aggregate logs across groups
            all_sigmas = np.concatenate([snes_state['groups'][g]['sigma'] for g in group_names])
            build_in_params['SNES-log']['mean_sigma']   = float(np.mean(all_sigmas))
            build_in_params['SNES-log']['max_sigma']    = float(np.max(all_sigmas))
            build_in_params['SNES-log']['restart']      = do_restart_any
            build_in_params['SNES-log']['num_groups']   = len(group_names)

        except Exception as e:
            if verbose and verbose > 0: print(f"[SNES ERROR] Update failed: {e}", flush=True)

    # ════════════════════════════════════════════════════════════════════════
    # 5. GENERATE CANDIDATES & SAVE
    # ════════════════════════════════════════════════════════════════════════
    # Save state to disk
    try:
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        with open(state_path, 'wb') as f: pickle.dump(snes_state, f)
    except Exception as e:
        if verbose and verbose > 0:
            print(f"[SNES WARNING] State save failed: {e}", flush=True)

    # Generate candidates: sample noise per group, assemble full genes
    # Mirroring is applied per-group so antithetic pairs are exact across the whole gene.
    if use_mirroring:
        half = number_of_candidate_genes // 2
        z_base = {gname: torch.randn(half, coevo_groups[gname]['dim'], device=device)
                  for gname in group_names}
        z_all  = {gname: torch.cat([z_base[gname], -z_base[gname]], dim=0)
                  for gname in group_names}
        if half * 2 < number_of_candidate_genes:  # odd count
            for gname in group_names:
                z_all[gname] = torch.cat([z_all[gname],
                                          torch.randn(1, coevo_groups[gname]['dim'], device=device)], dim=0)
    else:
        z_all = {gname: torch.randn(number_of_candidate_genes, coevo_groups[gname]['dim'], device=device)
                 for gname in group_names}

    # Assemble full unbounded gene by concatenating group samples in order
    gene_parts = []
    for gname in group_names:
        gs = snes_state['groups'][gname]
        mu_g    = torch.tensor(gs['mu'],    device=device, dtype=torch.float32)
        sigma_g = torch.tensor(gs['sigma'], device=device, dtype=torch.float32)
        x_g = mu_g.unsqueeze(0) + sigma_g.unsqueeze(0) * z_all[gname]
        gene_parts.append(x_g)

    x_unbounded = torch.cat(gene_parts, dim=1)

    # Inverse transform if gene scaler was used (denormalize)
    use_gene_scaler = snes_state.get('use_gene_scaler', False)
    if use_gene_scaler and gene_scaler is not None and getattr(gene_scaler, 'scaling_ready', False):
        x_denormalized = gene_scaler.inverse_transform(x_unbounded)
        if verbose and verbose > 1:
            print(f"[SNES] Applied inverse gene normalization", flush=True)
    else:
        x_denormalized = x_unbounded

    # Bound to [-1, 1] using tanh (or [0, 1] if geneMin=0)
    x_bounded = torch.tanh(x_denormalized)

    gene_min = getattr(EA_Class, 'geneMin', -1.0)
    if gene_min == 0.0:
        x_bounded = (x_bounded + 1.0) / 2.0
        
    lower_bound = 0.0 if gene_min == 0 else -1.0
    candidates = torch.clamp(x_bounded, lower_bound, 1.0).cpu().tolist()

    # --- RANDOM INJECTION ---
    # Replace a fraction of candidates with pure random genes to maintain exploration
    num_random = max(1, int(len(candidates) * random_injection_rate))
    for i in range(num_random):
        idx = len(candidates) - 1 - i  # replace from the end (worst mirrored candidates)
        if idx >= 0:
            if gene_min == 0.0:
                candidates[idx] = np.random.uniform(0.0, 1.0, n).tolist()
            else:
                candidates[idx] = np.random.uniform(-1.0, 1.0, n).tolist()
    if verbose and verbose > 0:
        print(f"[SNES] Injected {num_random}/{len(candidates)} random candidates", flush=True)

    return candidates
