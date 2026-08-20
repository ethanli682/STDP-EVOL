# OpenAI Evolution Strategies (OpenAI-ES) implementation for Genetic Algorithms
# Reference: "Evolution Strategies as a Scalable Alternative to Reinforcement Learning" (Salimans et al., 2017)

import os
import pickle
import numpy as np
import torch
import heapq
from typing import List, Dict, Optional, Any

# --- LOCAL IMPORTS ---
try:
    from GA_utils_history_loader import ChunkedGeneHistoryLoader
    from GA_utils_scalers import ChunkedRankScaler, ChunkedLogZScoreScaler, ChunkedPerDimensionScaler
except ImportError:
    print("[OPENAI-ES WARNING] Could not import Loader/Scaler classes. Disk streaming will fail.", flush=True)


def _quality_diversity_rerank(genes_np, scores_np, evolution_target, target_count, alpha=0.3, k_neighbors=5):
    """
    Re-rank candidates using quality-diversity: blend fitness rank with novelty rank.
    """
    n = len(scores_np)
    if n <= target_count:
        return np.arange(n)

    if evolution_target == 1:
        fitness_order = np.argsort(-scores_np)
    else:
        fitness_order = np.argsort(scores_np)
    fitness_rank = np.empty(n, dtype=np.float64)
    fitness_rank[fitness_order] = np.arange(n, dtype=np.float64)

    try:
        from sklearn.neighbors import NearestNeighbors
        k = min(k_neighbors, n - 1)
        if k > 0:
            nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean', algorithm='auto')
            nn.fit(genes_np)
            distances, _ = nn.kneighbors(genes_np)
            novelty_scores = np.mean(distances[:, 1:], axis=1)
        else:
            novelty_scores = np.zeros(n)
    except Exception:
        novelty_scores = np.zeros(n)

    novelty_order = np.argsort(-novelty_scores)
    novelty_rank = np.empty(n, dtype=np.float64)
    novelty_rank[novelty_order] = np.arange(n, dtype=np.float64)

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


def createCandidateGene_OpenAI_ES(
    args: Any = None,
    EA_Class: Any = None,
    genePopulation: Optional[List[Dict[str, Any]]] = None,
    number_of_candidate_genes: Optional[int] = None,
    build_in_params: Optional[Dict] = None,
    verbose: Optional[int] = None
) -> List[List[float]]:
    """
    OpenAI Evolution Strategies Generator.

    Why use OpenAI-ES for High Dimensions:
    1. O(N) Time and Space Complexity: Easily scales to millions of parameters (unlike CMA-ES).
    2. Virtual Gradient Descent: Approximates gradients using finite differences, perfect for continuous search spaces.
    3. Adam Optimizer: Uses Adam to update the population mean, handling noisy approximations gracefully.
    4. Memory Safe: Highly parallelizable and uses footprint proportional to model parameters.

    Co-evolution:
    Each parameter group defined in EA_Class.coevo_groups (derived from the YAML coevo_group tags)
    maintains its own independent Adam state (theta, m, v, t) and sigma.
    All groups share the same fitness/utility signal but do NOT share gradient information.
    The per-group 'auto' lr and sigma formulas scale with each group's own dimension, so scalar
    beta parameters and large LoRA matrices naturally converge at appropriate rates.
    """

    # ════════════════════════════════════════════════════════════════════════
    # 1. SETUP & VALIDATION
    # ════════════════════════════════════════════════════════════════════════
    if args is None: raise ValueError("args cannot be None")
    if EA_Class is None: raise ValueError("EA_Class cannot be None")
    if number_of_candidate_genes is None: raise ValueError("number_of_candidate_genes cannot be None")
    if build_in_params is None: raise ValueError("build_in_params cannot be None")

    if 'OpenAI_ES' not in build_in_params: build_in_params['OpenAI_ES'] = {}
    if 'OpenAI_ES-log' not in build_in_params: build_in_params['OpenAI_ES-log'] = {}

    # Configuration
    config = build_in_params['OpenAI_ES']
    scaler_type      = config.get('scaler_type', 'rank')
    learning_rate_cfg = config.get('learning_rate', 'auto')
    sigma_cfg         = config.get('sigma', 'auto')
    adam_beta1 = config.get('adam_beta1', 0.9)
    adam_beta2 = config.get('adam_beta2', 0.999)
    adam_eps   = float(config.get('adam_eps', 1e-8))
    use_mirroring   = config.get('use_mirroring', True)
    random_injection_rate = float(config.get('random_injection_rate', 0.15))
    sigma_decay = float(config.get('sigma_decay', 0.999))  # per-iteration multiplicative decay
    stagnation_window = int(config.get('stagnation_window', 15))
    sigma_boost_factor = float(config.get('sigma_boost_factor', 2.0))
    evolution_target = getattr(args, 'evolutionTarget', 1)

    # Hardware
    use_gpu = getattr(args, 'gpu', False) and torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')

    # Co-evolution groups
    n = int(EA_Class.geneLength)
    coevo_groups = _get_coevo_groups(EA_Class, n)
    group_names  = list(coevo_groups.keys())

    effective_pop_for_tuning = max(2, int(number_of_candidate_genes or (4 + int(3 * np.log(n)))))

    def _per_group_lr(n_g):
        if isinstance(learning_rate_cfg, str) and learning_rate_cfg.strip().lower() == 'auto':
            lr_auto = 0.05 * np.sqrt(64.0 / float(effective_pop_for_tuning)) * np.sqrt(10000.0 / float(max(1, n_g)))
            return float(np.clip(lr_auto, 0.003, 0.08))
        return float(learning_rate_cfg)

    def _per_group_sigma(n_g):
        if isinstance(sigma_cfg, str) and sigma_cfg.strip().lower() == 'auto':
            sigma_auto = 0.12 * (64.0 / float(effective_pop_for_tuning))**0.25 * (10000.0 / float(max(1, n_g)))**0.15
            return float(np.clip(sigma_auto, 0.02, 0.25))
        return float(sigma_cfg)

    if verbose and verbose > 0:
        print(
            f"[OPENAI-ES] Device: {device} | Scaler: {scaler_type.upper()} | "
            f"Groups: {len(group_names)} | lr_cfg={learning_rate_cfg} | sigma_cfg={sigma_cfg}",
            flush=True
        )

    # ════════════════════════════════════════════════════════════════════════
    # 2. STATE MANAGEMENT (PERSISTENCE)
    # ════════════════════════════════════════════════════════════════════════
    state_path = os.path.join(EA_Class.savePath, 'running', f'{str(args.agent_idx)}_OpenAI_ES_metadata_.pkl')
    es_state = {}

    # Load State
    if os.path.exists(state_path):
        try:
            with open(state_path, 'rb') as f: es_state = pickle.load(f)
        except Exception as e:
            print(f"[OPENAI-ES ERROR] Corrupt state file: {e}. Starting fresh.", flush=True)

    default_lambda = 4 + int(3 * np.log(n))

    # Validate state: must have per-group entries with matching group names
    state_valid = (
        'groups' in es_state and
        set(es_state['groups'].keys()) == set(group_names)
    )

    if not state_valid:
        if verbose and verbose > 0:
            print(f"[OPENAI-ES] Initializing new per-group state ({len(group_names)} groups, n={n})...", flush=True)
        # Compute initial theta from population mean (in arctanh space) if data is available.
        # This prevents the slow warm-up caused by starting theta at zeros.
        init_theta_full = None
        if genePopulation is not None and len(genePopulation) > 0:
            try:
                eps = 1e-6
                genes_raw = []
                for p in genePopulation:
                    if p.get('fitnessScore') is not None and p.get('gene') is not None:
                        g = np.array(p['gene'], dtype=np.float32)
                        if len(g) == n:
                            if getattr(EA_Class, 'geneMin', -1.0) == 0.0:
                                g = g * 2.0 - 1.0
                            genes_raw.append(np.arctanh(np.clip(g, -1.0 + eps, 1.0 - eps)))
                if genes_raw:
                    init_theta_full = np.mean(genes_raw, axis=0).astype(np.float32)
            except Exception:
                init_theta_full = None

        es_state['groups'] = {}
        for gname, ginfo in coevo_groups.items():
            d = ginfo['dim']
            g_start = ginfo['start']
            g_end = ginfo['end']
            theta_init = init_theta_full[g_start:g_end] if init_theta_full is not None else np.zeros(d, dtype=np.float32)
            es_state['groups'][gname] = {
                'theta': theta_init,
                'm':     np.zeros(d, dtype=np.float32),
                'v':     np.zeros(d, dtype=np.float32),
                't':     0,
            }

    # ════════════════════════════════════════════════════════════════════════
    # 3. DATA LOADING (IN-MEMORY OR STREAMING)
    # ════════════════════════════════════════════════════════════════════════
    elite_genes = None
    elite_weights = None
    data_available = False
    gene_scaler = None
    current_elite_fitness = None  # tracked for self-tuning sigma

    if genePopulation is not None and len(genePopulation) > 0:
        try:
            meta = [(i, p['fitnessScore']) for i, p in enumerate(genePopulation) if p.get('fitnessScore') is not None]
            if len(meta) >= 2:
                meta.sort(key=lambda x: x[1], reverse=(evolution_target == 1))
                target_lambda = len(meta)
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
                    es_state['eff_lambda'] = target_lambda
                    data_available = True
                    current_elite_fitness = float(np.mean([x[1] for x in meta]))

                    ranks = torch.arange(target_lambda, device=device, dtype=torch.float32)
                    elite_weights = (ranks / (target_lambda - 1.0)) - 0.5
                    if evolution_target == 1:
                        elite_weights = -elite_weights
        except Exception as e:
            print(f"[OPENAI-ES ERROR] Processing genePopulation failed: {e}", flush=True)

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

            try:
                chunk_size = loader.calculate_chunk_size(safety_margin=0.5, chunk_allocation=0.20, verbose=(verbose and verbose > 0))
            except Exception as e:
                chunk_size = 10000

            import itertools
            elite_heap = []
            heap_size = chunk_size
            _heap_counter = itertools.count()  # unique tiebreaker to avoid numpy array comparison

            for chunk, is_done in loader.loadTopKPercentileGeneHistoryAsListOfDicts(
                args=args, top_k_percentile=15.0, maximize=(evolution_target == 1),
                chunk_size=chunk_size, fitness_scaler=fit_scaler, use_fitness_scaler_weights=True, allEpochs=True,
                min_datapoints=min(int(getattr(args, 'populationSize', 1000) * 0.1), 300),
                max_datapoints=int(getattr(args, 'populationSize', 1000) * 2),
                gene_scaler=gene_scaler,
                fit_fitness_scaler_on_top_k=True,
                fit_gene_scaler_on_top_k=True,
            ):
                for rec in chunk:
                    try:
                        score = float(rec['fitnessScore'])
                        g_val = rec['gene']
                    except (KeyError, TypeError):
                        continue
                    heap_key = score if evolution_target == 1 else -score
                    if len(elite_heap) < heap_size:
                        heapq.heappush(elite_heap, (heap_key, next(_heap_counter), score, g_val))
                    else:
                        if heap_key > elite_heap[0][0]:
                            heapq.heappushpop(elite_heap, (heap_key, next(_heap_counter), score, g_val))

            collected_genes = []
            collected_scores = []
            while elite_heap:
                _, _, score, gene = heapq.heappop(elite_heap)
                collected_genes.append(gene)
                collected_scores.append(score)

            if collected_genes:
                genes_np  = np.array(collected_genes)
                scores_np = np.array(collected_scores)

                target_lambda = min(len(scores_np), default_lambda * 3)
                # Quality-diversity selection: blend fitness rank with novelty
                qd_alpha = float(config.get('qd_alpha', 0.2))
                agent_k = max(3, 5 + int(args.agent_idx or 0) * 2)
                best_indices = _quality_diversity_rerank(
                    genes_np, scores_np, evolution_target, target_lambda, alpha=qd_alpha, k_neighbors=agent_k
                )
                if verbose and verbose > 0:
                    print(f"[OPENAI-ES] Quality-diversity selection: {len(best_indices)}/{len(scores_np)} (alpha={qd_alpha})", flush=True)

                genes_selected = genes_np[best_indices]
                if gene_scaler is not None and getattr(gene_scaler, 'scaling_ready', False):
                    genes_t    = torch.tensor(genes_selected, dtype=torch.float32, device=device)
                    elite_genes = gene_scaler.transform(genes_t)
                else:
                    eps = 1e-6
                    g_arr = genes_selected.copy()
                    if getattr(EA_Class, 'geneMin', -1.0) == 0.0:
                        g_arr = g_arr * 2.0 - 1.0
                    elite_genes = torch.tensor(
                        np.arctanh(np.clip(g_arr, -1.0+eps, 1.0-eps)),
                        dtype=torch.float32, device=device
                    )

                scores_selected = scores_np[best_indices]
                scores_t = torch.tensor(scores_selected, dtype=torch.float32, device=device)
                current_elite_fitness = float(np.mean(scores_selected))

                if scaler_type == 'zscore':
                    raw_weights = fit_scaler.compute_sample_weights(scores_t)
                    elite_weights = raw_weights - torch.mean(raw_weights)
                    elite_weights = elite_weights / (torch.sum(torch.abs(elite_weights)) + 1e-8)
                else:
                    ranks = torch.arange(target_lambda, device=device, dtype=torch.float32)
                    elite_weights = (ranks / (target_lambda - 1.0)) - 0.5
                    if evolution_target == 1:
                        elite_weights = -elite_weights

                es_state['eff_lambda']      = target_lambda
                es_state['use_gene_scaler'] = (gene_scaler is not None and getattr(gene_scaler, 'scaling_ready', False))
                data_available = True
        except Exception as e:
             print(f"[OPENAI-ES ERROR] Streaming failed: {e}", flush=True)

    # ════════════════════════════════════════════════════════════════════════
    # 4. THE UPDATE STEP — PER-GROUP CO-EVOLUTION
    # Each group is updated independently using only its own gene slice + Adam.
    # All groups share the same fitness/utility weights.
    # ════════════════════════════════════════════════════════════════════════
    if data_available and elite_genes is not None:
        try:
            eff_lambda = es_state['eff_lambda']
            utilities  = elite_weights

            # --- PER-GROUP ADAM UPDATE ---
            for gname, ginfo in coevo_groups.items():
                g_start = ginfo['start']
                g_end   = ginfo['end']
                n_g     = ginfo['dim']

                lr_g    = _per_group_lr(n_g)
                sigma_g = _per_group_sigma(n_g)

                gs    = es_state['groups'][gname]
                theta = torch.tensor(gs['theta'], device=device, dtype=torch.float32)

                # Reconstruct noise for this group: epsilon = (gene_slice - theta) / sigma_g
                epsilon_samples = (elite_genes[:, g_start:g_end] - theta.unsqueeze(0)) / sigma_g

                # Gradient approximation
                grad = torch.mv(epsilon_samples.T, utilities) / (eff_lambda * sigma_g)

                # Adam update
                m = torch.tensor(gs['m'], device=device, dtype=torch.float32)
                v = torch.tensor(gs['v'], device=device, dtype=torch.float32)
                t = gs['t'] + 1

                m = adam_beta1 * m + (1.0 - adam_beta1) * grad
                v = adam_beta2 * v + (1.0 - adam_beta2) * (grad * grad)

                m_hat = m / (1.0 - adam_beta1**t)
                v_hat = v / (1.0 - adam_beta2**t)

                theta_new = theta + lr_g * m_hat / (torch.sqrt(v_hat) + adam_eps)

                gs['theta'] = theta_new.cpu().numpy()
                gs['m']     = m.cpu().numpy()
                gs['v']     = v.cpu().numpy()
                gs['t']     = t

            # --- ITERATION TRACKING ---
            es_state['iteration'] = es_state.get('iteration', 0) + 1
            iteration_count = es_state['iteration']

            # --- SELF-TUNING: update rolling fitness history ---
            if current_elite_fitness is not None:
                fh = es_state.setdefault('fitness_history', [])
                fh.append(current_elite_fitness)
                if len(fh) > 20:
                    fh[:] = fh[-20:]

            # --- SELF-TUNING: update rolling diversity proxy ---
            # Use mean per-dimension std of elite genes in latent space (O(n), cheap)
            diversity_proxy = float(elite_genes.std(dim=0).mean().item()) if elite_genes.shape[0] > 1 else None
            if diversity_proxy is not None:
                dh = es_state.setdefault('diversity_history', [])
                dh.append(diversity_proxy)
                if len(dh) > 20:
                    dh[:] = dh[-20:]

            # --- SELF-TUNING: fitness-driven sigma adaptation ---
            # Replaces the fixed-period boost: react as soon as improvement stalls.
            fh = es_state.get('fitness_history', [])
            sigma_action = 'decay'
            improvement = None
            if len(fh) >= 5:
                window = fh[-5:]
                delta = window[-1] - window[0]
                # Relative improvement (positive = getting better for maximize)
                if evolution_target == 1:
                    improvement = delta / (abs(window[0]) + 1e-8)
                else:
                    improvement = -delta / (abs(window[0]) + 1e-8)

                if improvement < 0.01:      # < 1% gain over 5 iters → stagnating
                    sigma_action = 'boost'
                elif improvement > 0.05:    # > 5% gain → converging fast → decay faster
                    sigma_action = 'fast_decay'
                # else: normal decay

            current_sigma_mult = es_state.get('sigma_multiplier', 1.0)
            if sigma_action == 'boost':
                es_state['sigma_multiplier'] = min(current_sigma_mult * sigma_boost_factor, 5.0)
            elif sigma_action == 'fast_decay':
                es_state['sigma_multiplier'] = current_sigma_mult * (sigma_decay ** 2)
            else:
                es_state['sigma_multiplier'] = current_sigma_mult * sigma_decay

            if verbose and verbose > 0 and sigma_action != 'decay':
                print(
                    f"[OPENAI-ES] Sigma {sigma_action.upper()} | improvement={improvement:.3f} | "
                    f"sigma_mult: {current_sigma_mult:.4f} -> {es_state['sigma_multiplier']:.4f}",
                    flush=True
                )

            build_in_params['OpenAI_ES-log']['num_groups'] = len(group_names)
            build_in_params['OpenAI_ES-log']['sigma_multiplier'] = es_state.get('sigma_multiplier', 1.0)
            build_in_params['OpenAI_ES-log']['sigma_action'] = sigma_action
            build_in_params['OpenAI_ES-log']['improvement_rate'] = round(improvement, 4) if improvement is not None else None
            build_in_params['OpenAI_ES-log']['diversity_proxy'] = round(diversity_proxy, 4) if diversity_proxy is not None else None
            build_in_params['OpenAI_ES-log']['iteration'] = iteration_count

        except Exception as e:
            if verbose and verbose > 0:
                print(f"[OPENAI-ES ERROR] Update failed: {e}", flush=True)

    # ════════════════════════════════════════════════════════════════════════
    # 5. GENERATE CANDIDATES & SAVE
    # ════════════════════════════════════════════════════════════════════════
    # Save state to disk
    try:
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        with open(state_path, 'wb') as f: pickle.dump(es_state, f)
    except Exception as e:
        if verbose and verbose > 0:
            print(f"[OPENAI-ES WARNING] State save failed: {e}", flush=True)

    # Generate candidates: sample noise per group, assemble full genes
    if use_mirroring:
        half   = number_of_candidate_genes // 2
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
    sigma_mult = es_state.get('sigma_multiplier', 1.0)
    gene_parts = []
    for gname in group_names:
        gs      = es_state['groups'][gname]
        n_g     = coevo_groups[gname]['dim']
        sigma_g = _per_group_sigma(n_g) * sigma_mult  # apply adaptive sigma multiplier
        theta_g = torch.tensor(gs['theta'], device=device, dtype=torch.float32)
        x_g = theta_g.unsqueeze(0) + sigma_g * z_all[gname]
        gene_parts.append(x_g)

    x_unbounded = torch.cat(gene_parts, dim=1)

    # Inverse transform if gene scaler was used (denormalize)
    use_gene_scaler = es_state.get('use_gene_scaler', False)
    if use_gene_scaler and gene_scaler is not None and getattr(gene_scaler, 'scaling_ready', False):
        x_denormalized = gene_scaler.inverse_transform(x_unbounded)
        if verbose and verbose > 1:
            print(f"[OPENAI-ES] Applied inverse gene normalization", flush=True)
    else:
        x_denormalized = x_unbounded

    # Bound to [-1, 1] using tanh (or [0, 1] if geneMin=0)
    x_bounded = torch.tanh(x_denormalized)

    gene_min = getattr(EA_Class, 'geneMin', -1.0)
    if gene_min == 0.0:
        x_bounded = (x_bounded + 1.0) / 2.0
        
    lower_bound = 0.0 if gene_min == 0 else -1.0
    candidates = torch.clamp(x_bounded, lower_bound, 1.0).cpu().tolist()

    # --- SELF-TUNING: diversity-driven random injection rate ---
    dh = es_state.get('diversity_history', [])
    effective_injection_rate = random_injection_rate
    diversity_action = 'base'
    if len(dh) >= 5:
        diversity_slope = (dh[-1] - dh[-5]) / (abs(dh[-5]) + 1e-8)
        if diversity_slope < 0.005:    # diversity plateaued or shrinking → inject more
            effective_injection_rate = min(random_injection_rate * 2.0, 0.4)
            diversity_action = 'increase'
        elif diversity_slope > 0.02:   # diversity growing well → inject less
            effective_injection_rate = max(random_injection_rate * 0.5, 0.05)
            diversity_action = 'decrease'

    # --- RANDOM INJECTION ---
    num_random = max(1, int(len(candidates) * effective_injection_rate))
    for i in range(num_random):
        idx = len(candidates) - 1 - i
        if idx >= 0:
            if gene_min == 0.0:
                candidates[idx] = np.random.uniform(0.0, 1.0, n).tolist()
            else:
                candidates[idx] = np.random.uniform(-1.0, 1.0, n).tolist()
    if verbose and verbose > 0:
        print(
            f"[OPENAI-ES] Injected {num_random}/{len(candidates)} random candidates "
            f"(rate={effective_injection_rate:.2f} [{diversity_action}], sigma_mult={sigma_mult:.4f})",
            flush=True
        )
    build_in_params['OpenAI_ES-log']['injection_rate'] = round(effective_injection_rate, 3)
    build_in_params['OpenAI_ES-log']['diversity_action'] = diversity_action

    return candidates
