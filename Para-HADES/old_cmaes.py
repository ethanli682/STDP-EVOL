# Active CMA-ES with Cooperative Co-Evolution (coevo_group) Support

import os
import pickle
import numpy as np
import torch
import math
from typing import List, Dict, Optional, Any

# --- LOCAL IMPORTS ---
try:
    from GA_utils_history_loader import ChunkedGeneHistoryLoader
    from GA_utils_scalers import ChunkedRankScaler, ChunkedLogZScoreScaler, ChunkedPerDimensionScaler
except ImportError:
    print("[CMA-ES WARNING] Could not import Loader/Scaler classes. Disk streaming will fail.", flush=True)


def _get_coevo_groups(EA_Class, n):
    """
    Return an ordered dict of co-evolution groups from EA_Class.coevo_groups.
    Falls back to a single group covering the full gene when the attribute is absent.
    """
    if hasattr(EA_Class, 'coevo_groups') and EA_Class.coevo_groups:
        return EA_Class.coevo_groups
    return {'_all': {'start': 0, 'end': n, 'dim': n, 'params': ['_all']}}


def _decide_diag_mode(n_g, diag_mode_cfg, diag_mode_threshold, max_full_cov_gib):
    """Decide whether to use diagonal covariance for a group of dimension n_g."""
    if isinstance(diag_mode_cfg, str):
        cfg_l = diag_mode_cfg.strip().lower()
        if cfg_l == 'auto':
            use_diag = n_g > diag_mode_threshold
        else:
            use_diag = cfg_l in ['true', '1', 'yes', 'y']
    else:
        use_diag = bool(diag_mode_cfg)

    if (not use_diag) and max_full_cov_gib > 0:
        full_cov_gib = (n_g * n_g * 4) / (1024 ** 3)
        if full_cov_gib > max_full_cov_gib:
            use_diag = True
    return use_diag


def _init_group_state(n_g, diag_mode, initial_sigma=0.5):
    """Create fresh CMA-ES state for a single coevo group of dimension n_g."""
    default_lambda = max(8, 4 + int(3 * np.log(n_g))) if n_g > 1 else 8
    default_mu = int(default_lambda / 2)
    gs = {
        'iteration': 0,
        'lambda': default_lambda,
        'mu': default_mu,
        'diag_mode': diag_mode,
        'sigma': float(initial_sigma),
        'm': np.zeros(n_g, dtype=np.float32),
        'p_c': np.zeros(n_g, dtype=np.float32),
        'p_sigma': np.zeros(n_g, dtype=np.float32),
        'D': np.ones(n_g, dtype=np.float32),
        'chi_n': n_g**0.5 * (1 - 1/(4*max(n_g, 1)) + 1/(21*max(n_g, 1)**2)),
    }
    if diag_mode:
        gs['C_diag'] = np.ones(n_g, dtype=np.float32)
    else:
        gs['C'] = np.eye(n_g, dtype=np.float32)
        gs['B'] = np.eye(n_g, dtype=np.float32)
    return gs


def _compute_weights(eff_lambda, mu, scaler_type, elite_weights_ext, device):
    """
    Compute CMA-ES weights (shared across groups).
    Returns (weights, weights_pos, weights_neg, mueff).
    """
    if scaler_type == 'zscore' and elite_weights_ext is not None:
        weights = elite_weights_ext / torch.sum(elite_weights_ext)
        mueff = (1.0 / torch.sum(weights**2)).item()
        return weights, weights, None, mueff

    # Active CMA rank-based weights
    raw_weights = torch.log(torch.tensor(mu + 0.5, device=device)) - \
                  torch.log(torch.arange(1, eff_lambda + 1, dtype=torch.float32, device=device))

    pos_mask = raw_weights > 0
    neg_mask = raw_weights < 0

    weights_pos = torch.zeros_like(raw_weights)
    denom_pos = torch.sum(raw_weights[pos_mask])
    if denom_pos != 0:
        weights_pos[pos_mask] = raw_weights[pos_mask] / denom_pos
    else:
        n_pos = torch.sum(pos_mask).item()
        if n_pos > 0:
            weights_pos[pos_mask] = 1.0 / n_pos
        else:
            weights_pos[:] = 1.0 / eff_lambda

    weights_neg = torch.zeros_like(raw_weights)
    if torch.any(neg_mask):
        sum_neg = torch.sum(torch.abs(raw_weights[neg_mask]))
        if sum_neg != 0:
            weights_neg[neg_mask] = raw_weights[neg_mask] / sum_neg * 0.5

    weights = weights_pos + weights_neg
    mueff = (1.0 / torch.sum(weights_pos**2)).item()
    return weights, weights_pos, weights_neg, mueff


def _update_group_cma(gs, elite_slice, weights, weights_pos, weights_neg, mueff,
                      sigma_cfg_new, device, condition_threshold, sigma_min_restart,
                      restart_on_condition, sigma_max_restart=5.0):
    """
    Run one CMA-ES update step on a single group.
    elite_slice: (eff_lambda, n_g) tensor of elite genes for this group.
    Returns (updated_gs, did_restart, restart_info).
    """
    n_g = elite_slice.shape[1]
    diag_mode = bool(gs.get('diag_mode', True))

    m_curr = torch.tensor(gs['m'], device=device, dtype=torch.float32)
    sigma = float(gs['sigma'])
    iteration = gs['iteration']

    # Mean update (full step, no smoothing — correct for CSA)
    m_diff = (elite_slice.T @ weights_pos) - m_curr
    m_new = m_curr + m_diff  # lr=1.0, standard CMA-ES

    # CMA-ES constants scaled to group dimension
    cc = 4.0 / (n_g + 4.0)
    cs = (mueff + 2.0) / (n_g + mueff + 5.0)
    c1 = 2.0 / ((n_g + 1.3)**2 + mueff)
    alpha_cov = 2.0
    cmu = min(1.0 - c1, alpha_cov * (mueff - 2.0 + 1.0/mueff) / ((n_g + 2.0)**2 + mueff))
    damps = 1.0 + 2.0 * max(0, math.sqrt((mueff - 1.0) / (n_g + 1.0)) - 1.0) + cs
    chi_n = gs['chi_n']

    ps = torch.tensor(gs['p_sigma'], device=device, dtype=torch.float32)
    pc = torch.tensor(gs['p_c'], device=device, dtype=torch.float32)
    D = torch.tensor(gs['D'], device=device, dtype=torch.float32)

    if diag_mode:
        C_diag_old = torch.tensor(gs['C_diag'], device=device, dtype=torch.float32)
        invsqrt_diag = 1.0 / (D + 1e-10)

        ps_new = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * (invsqrt_diag * m_diff) / sigma
        norm_ps = torch.norm(ps_new).item()
        hsig_val = norm_ps / math.sqrt(1 - (1 - cs)**(2 * (iteration + 1))) / chi_n
        hsig = 1.0 if hsig_val < 1.4 + 2.0/(n_g+1) else 0.0

        pc_new = (1 - cc) * pc + (hsig * math.sqrt(cc * (2 - cc) * mueff)) * m_diff / sigma

        rank_one = pc_new * pc_new
        dh_adjustment = (1 - hsig) * cc * (2 - cc) * C_diag_old

        y_s = (elite_slice - m_curr.unsqueeze(0)) / sigma
        rank_mu = torch.sum(weights.unsqueeze(1) * (y_s * y_s), dim=0)

        weight_sum = torch.sum(weights_pos).item() if weights_neg is None else torch.sum(weights[weights > 0]).item()
        C_diag_new = (1 - c1 - cmu * weight_sum) * C_diag_old + \
                     c1 * (rank_one + dh_adjustment) + cmu * rank_mu
        C_diag_new = torch.clamp(C_diag_new, 1e-10, None)

        sigma_new = sigma * math.exp((cs / damps) * (norm_ps / chi_n - 1))
        sigma_new = max(1e-3, min(sigma_new, 10.0))

        D_new = torch.sqrt(C_diag_new)
        cond_number = (torch.max(D_new) / (torch.min(D_new) + 1e-20)).item()

        do_restart = restart_on_condition and (
            cond_number > condition_threshold or
            sigma_new < sigma_min_restart or
            sigma_new > sigma_max_restart
        )

        if do_restart:
            new_gs = _init_group_state(n_g, diag_mode=True, initial_sigma=initial_sigma)
            # Seed mean from best-found position instead of zero
            new_gs['m'] = m_new.cpu().numpy()
            restart_info = {'cond_number': cond_number, 'sigma': sigma_new}
            return new_gs, True, restart_info

        gs.update({
            'm': m_new.cpu().numpy(),
            'C_diag': C_diag_new.cpu().numpy(),
            'sigma': float(sigma_new),
            'p_sigma': ps_new.cpu().numpy(),
            'p_c': pc_new.cpu().numpy(),
            'D': D_new.cpu().numpy(),
            'iteration': iteration + 1,
            'diag_mode': True,
        })
        return gs, False, {'sigma': float(sigma_new), 'cond_number': cond_number}

    else:
        # Full covariance mode
        B = torch.tensor(gs['B'], device=device, dtype=torch.float32)

        invsqrtC = B @ torch.diag(1.0 / (D + 1e-10)) @ B.T
        ps_new = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ m_diff) / sigma

        norm_ps = torch.norm(ps_new).item()
        hsig_val = norm_ps / math.sqrt(1 - (1 - cs)**(2 * (iteration + 1))) / chi_n
        hsig = 1.0 if hsig_val < 1.4 + 2.0/(n_g+1) else 0.0

        pc_new = (1 - cc) * pc + (hsig * math.sqrt(cc * (2 - cc) * mueff)) * m_diff / sigma

        C_old = torch.tensor(gs['C'], device=device, dtype=torch.float32)
        rank_one = torch.outer(pc_new, pc_new)
        dh_adjustment = (1 - hsig) * cc * (2 - cc) * C_old

        y_s = (elite_slice - m_curr.unsqueeze(0)).T / sigma
        rank_mu = (y_s * weights) @ y_s.T

        weight_sum = torch.sum(weights_pos).item() if weights_neg is None else torch.sum(weights[weights > 0]).item()
        C_new = (1 - c1 - cmu * weight_sum) * C_old + c1 * (rank_one + dh_adjustment) + cmu * rank_mu

        sigma_new = sigma * math.exp((cs / damps) * (norm_ps / chi_n - 1))
        sigma_new = max(1e-3, min(sigma_new, 10.0))

        if iteration % max(1, int(1.0/(10*c1*cmu + 1e-10))) == 0:
            C_new = torch.triu(C_new) + torch.triu(C_new, 1).T
            vals, vecs = torch.linalg.eigh(C_new)
            vals = torch.clamp(vals, 1e-10, None)
            D_new = torch.sqrt(vals)
            B_new = vecs
        else:
            B_new = B
            D_new = D

        cond_number = (torch.max(D_new) / (torch.min(D_new) + 1e-20)).item()
        do_restart = restart_on_condition and (
            cond_number > condition_threshold or
            sigma_new < sigma_min_restart or
            sigma_new > sigma_max_restart
        )

        if do_restart:
            new_gs = _init_group_state(n_g, diag_mode=False, initial_sigma=initial_sigma)
            new_gs['m'] = m_new.cpu().numpy()
            restart_info = {'cond_number': cond_number, 'sigma': sigma_new}
            return new_gs, True, restart_info

        gs.update({
            'm': m_new.cpu().numpy(),
            'C': C_new.cpu().numpy(),
            'sigma': float(sigma_new),
            'p_sigma': ps_new.cpu().numpy(),
            'p_c': pc_new.cpu().numpy(),
            'B': B_new.cpu().numpy(),
            'D': D_new.cpu().numpy(),
            'iteration': iteration + 1,
            'diag_mode': False,
        })
        return gs, False, {'sigma': float(sigma_new), 'cond_number': cond_number}


def createCandidateGene_CMA_ES(
    args: Any = None,
    EA_Class: Any = None,
    genePopulation: Optional[List[Dict[str, Any]]] = None,
    number_of_candidate_genes: Optional[int] = None,
    build_in_params: Optional[Dict] = None,
    verbose: Optional[int] = None
) -> List[List[float]]:
    """
    Active CMA-ES with Cooperative Co-Evolution (coevo_group) support.

    Each coevo_group gets its own independent CMA-ES state (mean, covariance,
    sigma, evolution paths).  All groups share the same fitness-derived weights.
    """

    # ════════════════════════════════════════════════════════════════════════
    # 1. SETUP & VALIDATION
    # ════════════════════════════════════════════════════════════════════════
    if args is None: raise ValueError("args cannot be None")
    if EA_Class is None: raise ValueError("EA_Class cannot be None")
    if number_of_candidate_genes is None: raise ValueError("number_of_candidate_genes cannot be None")
    if build_in_params is None: raise ValueError("build_in_params cannot be None")

    if 'CMA_ES' not in build_in_params: build_in_params['CMA_ES'] = {}
    if 'CMA_ES-log' not in build_in_params: build_in_params['CMA_ES-log'] = {}

    config = build_in_params['CMA_ES']
    scaler_type = config.get('scaler_type', 'rank')
    use_mirroring = config.get('use_mirroring', True)
    use_orthogonal_sampling = config.get('use_orthogonal_sampling', True)
    restart_on_condition = config.get('restart_on_condition', True)
    condition_threshold = float(config.get('condition_threshold', 1e4))
    sigma_min_restart = float(config.get('sigma_min_restart', 1e-6))
    sigma_max_restart = float(config.get('sigma_max_restart', 5.0))
    initial_sigma = float(config.get('initial_sigma', 0.5))
    # Optional: seed CMA-ES mean from a founder pickle (supports zstd-compressed).
    # Expected content: {'gene': 1-D array in [-1, 1] space} or {'param': {...}} with
    # per-group arrays. When set, overrides the population-centroid warm-start.
    initial_mean_path = config.get('initial_mean_path', None)
    evolution_target = getattr(args, 'evolutionTarget', 1)
    diag_mode_cfg = config.get('diag_mode', 'auto')
    diag_mode_threshold = int(config.get('diag_mode_threshold', 4096))
    max_full_cov_gib = float(config.get('max_full_cov_gib', 32.0))
    # Anti-stagnation parameters (aligned with SNES / EA_Heb)
    stagnation_window = int(config.get('stagnation_window', 10))
    sigma_boost_factor = float(config.get('sigma_boost_factor', 2.0))
    random_injection_rate = float(config.get('random_injection_rate', 0.15))

    use_gpu = getattr(args, 'gpu', False) and torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')

    n = int(EA_Class.geneLength)
    gene_min = getattr(EA_Class, 'geneMin', -1.0)

    # ════════════════════════════════════════════════════════════════════════
    # 2. COEVO GROUPS & STATE MANAGEMENT
    # ════════════════════════════════════════════════════════════════════════
    coevo_groups = _get_coevo_groups(EA_Class, n)
    group_names = sorted(coevo_groups.keys(), key=lambda g: coevo_groups[g]['start'])

    state_path = os.path.join(EA_Class.savePath, 'running', f'{str(args.agent_idx)}_CMA_ES_metadata_.pkl')
    cma_state = {}

    if os.path.exists(state_path):
        try:
            with open(state_path, 'rb') as f:
                cma_state = pickle.load(f)
        except Exception as e:
            print(f"[CMA-ES ERROR] Corrupt state file: {e}. Starting fresh.", flush=True)

    # Validate state matches current group structure
    state_valid = (
        'groups' in cma_state and
        set(cma_state['groups'].keys()) == set(group_names)
    )
    if state_valid:
        # Check each group dimension still matches
        for gname in group_names:
            gs = cma_state['groups'][gname]
            expected_dim = coevo_groups[gname]['dim']
            if len(gs.get('m', [])) != expected_dim:
                state_valid = False
                break

    if not state_valid:
        if verbose and verbose > 0:
            print(f"[CMA-ES] Initializing new state with {len(group_names)} coevo groups", flush=True)
        # Warm-start mean from population centroid (arctanh space) to skip the
        # early iterations where m=0 is far from the actual gene distribution.
        init_m_full = None

        # Priority 1: explicit founder path from config
        if initial_mean_path:
            try:
                import zstandard as _zstd
                from GA_utils_misc_func import pickle_loads_compat
                with open(initial_mean_path, 'rb') as f:
                    head = f.read(2); f.seek(0)
                    if head == b'\x28\xb5':
                        with _zstd.open(f, 'rb') as zf:
                            obj = pickle_loads_compat(zf.read())
                    else:
                        obj = pickle_loads_compat(f.read())
                founder_gene = None
                if isinstance(obj, dict):
                    if 'gene' in obj:
                        founder_gene = np.array(obj['gene'], dtype=np.float32)
                if founder_gene is not None and len(founder_gene) == n:
                    eps = 1e-6
                    g = founder_gene
                    if gene_min == 0.0:
                        g = g * 2.0 - 1.0
                    init_m_full = np.arctanh(np.clip(g, -1.0 + eps, 1.0 - eps)).astype(np.float32)
                    if verbose and verbose > 0:
                        print(f"[CMA-ES] Seeded mean from founder: {initial_mean_path}", flush=True)
                else:
                    print(f"[CMA-ES WARN] founder at {initial_mean_path} has no usable 'gene' of length {n}; falling back to population centroid", flush=True)
            except Exception as e:
                print(f"[CMA-ES WARN] failed to load initial_mean_path={initial_mean_path}: {e}", flush=True)

        # Priority 2: population centroid warm-start
        if init_m_full is None and genePopulation is not None and len(genePopulation) > 0:
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

        cma_state = {'groups': {}}
        for gname in group_names:
            n_g = coevo_groups[gname]['dim']
            diag = _decide_diag_mode(n_g, diag_mode_cfg, diag_mode_threshold, max_full_cov_gib)
            gs = _init_group_state(n_g, diag, initial_sigma=initial_sigma)
            if init_m_full is not None:
                g_start = coevo_groups[gname]['start']
                g_end = coevo_groups[gname]['end']
                gs['m'] = init_m_full[g_start:g_end]
            cma_state['groups'][gname] = gs

    if verbose and verbose > 0:
        for gname in group_names:
            gs = cma_state['groups'][gname]
            n_g = coevo_groups[gname]['dim']
            mode = "DIAG" if gs['diag_mode'] else "FULL"
            lam = gs['lambda']
            print(f"[CMA-ES]   Group '{gname}': dim={n_g} cov={mode} lambda={lam} iter={gs['iteration']}", flush=True)

    # ════════════════════════════════════════════════════════════════════════
    # 3. DATA LOADING (IN-MEMORY OR STREAMING)
    # ════════════════════════════════════════════════════════════════════════
    elite_genes = None
    elite_weights = None
    data_available = False
    feature_scaler = None
    best_current_fitness = None  # tracked for stagnation detection

    # Effective lambda: use the max group lambda for data loading
    max_lambda = max(gs['lambda'] for gs in cma_state['groups'].values())
    default_mu = int(max_lambda / 2)

    # --- PATH A: In-Memory Provided ---
    if genePopulation is not None and len(genePopulation) > 0:
        try:
            meta = [(i, p['fitnessScore']) for i, p in enumerate(genePopulation)
                    if p.get('fitnessScore') is not None]

            if len(meta) >= max_lambda:
                meta.sort(key=lambda x: x[1], reverse=(evolution_target == 1))
                target_lambda = min(len(meta), max_lambda)
                indices = [x[0] for x in meta[:target_lambda]]

                eps = 1e-6
                genes_list = []
                for idx in indices:
                    g_val = np.array(genePopulation[idx]['gene'])
                    if gene_min == 0.0:
                        g_val = g_val * 2.0 - 1.0
                    g_val = np.clip(g_val, -1.0+eps, 1.0-eps)
                    genes_list.append(np.arctanh(g_val))

                if len(genes_list[0]) == n:
                    elite_genes = torch.tensor(np.array(genes_list), dtype=torch.float32, device=device)
                    cma_state['eff_lambda'] = target_lambda
                    best_current_fitness = meta[0][1]  # best-first after sort
                    data_available = True
        except Exception as e:
            print(f"[CMA-ES ERROR] Processing genePopulation failed: {e}", flush=True)

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
                    print(f"[CMA-ES WARNING] Chunk size calculation failed: {e}. Using default 1000", flush=True)
                chunk_size = 1000

            import heapq
            import itertools
            elite_heap = []
            heap_size = chunk_size
            _heap_counter = itertools.count()  # unique tiebreaker to avoid numpy array comparison

            min_datapoints_value = max(
                min(int(getattr(args, 'populationSize', 100) * 0.05), 300),
                int(min(50, chunk_size // 3)),
            )

            for chunk, is_done in loader.loadTopKPercentileGeneHistoryAsListOfDicts(
                args=args, top_k_percentile=7.0, maximize=(evolution_target == 1),
                chunk_size=chunk_size, fitness_scaler=fitness_scaler,
                use_fitness_scaler_weights=True, allEpochs=True,
                min_datapoints=min_datapoints_value,
                fit_fitness_scaler_on_top_k=True,
                fit_gene_scaler_on_top_k=True,
                max_datapoints=int(getattr(args, 'populationSize', 1000)),
                gene_scaler=feature_scaler,
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
                        if gene_min == 0.0:
                            g_arr = g_arr * 2.0 - 1.0
                        g_arr = np.clip(g_arr, -1.0+eps, 1.0-eps)
                        gene_unbounded = np.arctanh(g_arr)

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

                target_lambda = min(len(indices), max_lambda)
                best_indices = indices[:target_lambda]

                genes_selected = genes_np[best_indices]
                if feature_scaler is not None and getattr(feature_scaler, 'scaling_ready', False):
                    genes_tensor = torch.tensor(genes_selected, dtype=torch.float32, device=device)
                    genes_normalized = feature_scaler.transform(genes_tensor)
                    elite_genes = genes_normalized.to(dtype=torch.float32, device=device)
                    if verbose and verbose > 1:
                        print(f"[CMA-ES] Applied gene normalization via feature_scaler", flush=True)
                else:
                    elite_genes = torch.tensor(genes_selected, dtype=torch.float32, device=device)

                if scaler_type == 'zscore':
                    scores_selected = torch.tensor(scores_np[best_indices], dtype=torch.float32, device=device)
                    elite_weights = fitness_scaler.compute_sample_weights(scores_selected)

                cma_state['eff_lambda'] = target_lambda
                cma_state['use_gene_scaler'] = (
                    feature_scaler is not None and getattr(feature_scaler, 'scaling_ready', False))
                scores_selected = scores_np[best_indices]
                best_current_fitness = (float(np.max(scores_selected)) if evolution_target == 1
                                        else float(np.min(scores_selected)))
                data_available = True
        except Exception as e:
            print(f"[CMA-ES ERROR] Streaming failed: {e}", flush=True)

    # ════════════════════════════════════════════════════════════════════════
    # 4. PER-GROUP UPDATE (ALGORITHM CORE)
    # ════════════════════════════════════════════════════════════════════════
    log_sigmas = {}
    log_conds = {}
    any_restart = False

    if data_available and elite_genes is not None:
        eff_lambda = cma_state['eff_lambda']

        # Compute weights ONCE (shared across all groups)
        weights, weights_pos, weights_neg, mueff = _compute_weights(
            eff_lambda, default_mu, scaler_type, elite_weights, device)

        for gname in group_names:
            ginfo = coevo_groups[gname]
            g_start = ginfo['start']
            g_end = ginfo['end']
            gs = cma_state['groups'][gname]

            # Slice this group's columns from elite genes
            elite_slice = elite_genes[:, g_start:g_end]

            try:
                updated_gs, did_restart, info = _update_group_cma(
                    gs, elite_slice, weights, weights_pos, weights_neg, mueff,
                    None, device, condition_threshold, sigma_min_restart,
                    restart_on_condition, sigma_max_restart)

                cma_state['groups'][gname] = updated_gs
                log_sigmas[gname] = info.get('sigma', 0.0)
                log_conds[gname] = info.get('cond_number', 0.0)

                if did_restart:
                    any_restart = True
                    if verbose and verbose > 0:
                        print(f"[CMA-ES] Restart triggered for group '{gname}' "
                              f"(cond={info.get('cond_number', '?'):.1f}, "
                              f"sigma={info.get('sigma', '?'):.3f})", flush=True)

            except Exception as e:
                if verbose and verbose > 0:
                    print(f"[CMA-ES ERROR] Update failed for group '{gname}': {e}", flush=True)

        # ── Stagnation detection & sigma boost ──────────────────────────────
        # Track actual best fitness (not mean-norm proxy) over a rolling window.
        # If no improvement for `stagnation_window` iterations, boost sigma
        # and reset evolution paths so CSA restarts cleanly.
        if 'best_fitness_history' not in cma_state:
            cma_state['best_fitness_history'] = []

        if best_current_fitness is not None:
            cma_state['best_fitness_history'].append(best_current_fitness)
            if len(cma_state['best_fitness_history']) > stagnation_window * 2:
                cma_state['best_fitness_history'] = cma_state['best_fitness_history'][-stagnation_window * 2:]

        history = cma_state['best_fitness_history']
        did_stagnation_boost = False
        # Only run the boost branch if sigma_boost_factor actually changes sigma;
        # when factor==1.0 it's a no-op and we should not log "stagnation_boost: True".
        if len(history) >= stagnation_window and sigma_boost_factor != 1.0:
            window = history[-stagnation_window:]
            best_in_window = max(window) if evolution_target == 1 else min(window)
            worst_in_window = min(window) if evolution_target == 1 else max(window)
            relative_improvement = abs(best_in_window - worst_in_window) / (abs(worst_in_window) + 1e-10)

            if relative_improvement < 0.01:  # < 1% fitness improvement over window
                did_stagnation_boost = True
                cma_state['best_fitness_history'] = []  # reset window after boost
                if verbose and verbose > 0:
                    print(f"[CMA-ES] Stagnation detected (improvement={relative_improvement:.4f} over "
                          f"{stagnation_window} iters). Boosting sigma by {sigma_boost_factor}x.", flush=True)

                for gname in group_names:
                    gs = cma_state['groups'][gname]
                    n_g = coevo_groups[gname]['dim']
                    gs['sigma'] = min(float(gs['sigma']) * sigma_boost_factor, sigma_max_restart)
                    # Reset evolution paths — accumulated from historical elites
                    gs['p_sigma'] = np.zeros(n_g, dtype=np.float32)
                    gs['p_c'] = np.zeros(n_g, dtype=np.float32)
                    log_sigmas[gname] = float(gs['sigma'])

        # Logs
        if log_sigmas:
            avg_sigma = sum(log_sigmas.values()) / len(log_sigmas)
            build_in_params['CMA_ES-log']['sigma'] = avg_sigma
            build_in_params['CMA_ES-log']['sigma_per_group'] = log_sigmas
            build_in_params['loss'] = avg_sigma
        if log_conds:
            build_in_params['CMA_ES-log']['condition_number'] = log_conds
        if any_restart:
            build_in_params['CMA_ES-log']['restart'] = True
        if did_stagnation_boost:
            build_in_params['CMA_ES-log']['stagnation_boost'] = True
        build_in_params['CMA_ES-log']['stagnation_count'] = cma_state.get('stagnation_count', 0)

    # ════════════════════════════════════════════════════════════════════════
    # 5. PER-GROUP CANDIDATE GENERATION & ASSEMBLY
    # ════════════════════════════════════════════════════════════════════════
    # Save state
    try:
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        with open(state_path, 'wb') as f:
            pickle.dump(cma_state, f)
    except Exception as e:
        if verbose and verbose > 0:
            print(f"[CMA-ES WARNING] State save failed: {e}", flush=True)

    # Generate per-group noise with mirroring
    pop = number_of_candidate_genes
    if use_mirroring:
        half = pop // 2
        z_all = {}
        for gname in group_names:
            n_g = coevo_groups[gname]['dim']
            z_half = torch.randn(half, n_g, device=device)
            z_g = torch.cat([z_half, -z_half], dim=0)
            if z_g.shape[0] < pop:
                z_g = torch.cat([z_g, torch.randn(1, n_g, device=device)], dim=0)
            z_all[gname] = z_g
    else:
        z_all = {gname: torch.randn(pop, coevo_groups[gname]['dim'], device=device)
                 for gname in group_names}

    # Orthogonal sampling per group
    if use_orthogonal_sampling:
        for gname in group_names:
            z = z_all[gname]
            p, d = z.shape
            if d < 2:
                continue
            if p <= d:
                try:
                    q, _ = torch.linalg.qr(z.T, mode='reduced')
                    z_all[gname] = q.T * torch.norm(z, dim=1, keepdim=True)
                except Exception:
                    pass  # keep original z on QR failure
            else:
                blocks = []
                norms = torch.norm(z, dim=1, keepdim=True)
                for start in range(0, p, d):
                    block = z[start:start+d]
                    block_size = block.shape[0]
                    try:
                        q, _ = torch.linalg.qr(block.T, mode='reduced')
                        blocks.append(q.T * norms[start:start+block_size])
                    except Exception:
                        blocks.append(block)
                z_all[gname] = torch.cat(blocks, dim=0)

    # Sample x = m + sigma * B * D * z per group, then concatenate
    gene_parts = []
    for gname in group_names:
        gs = cma_state['groups'][gname]
        n_g = coevo_groups[gname]['dim']
        m_g = torch.tensor(gs['m'], device=device, dtype=torch.float32)
        sigma_g = float(gs['sigma'])
        D_g = torch.tensor(gs['D'], device=device, dtype=torch.float32)
        z_g = z_all[gname]

        if bool(gs.get('diag_mode', True)):
            y_g = z_g * D_g.unsqueeze(0)
        else:
            B_g = torch.tensor(gs['B'], device=device, dtype=torch.float32)
            y_g = (B_g @ (D_g.unsqueeze(1) * z_g.T)).T

        x_g = m_g.unsqueeze(0) + sigma_g * y_g
        gene_parts.append(x_g)

    x_unbounded = torch.cat(gene_parts, dim=1)

    # Inverse transform if gene scaler was used
    use_gene_scaler = cma_state.get('use_gene_scaler', False)
    if use_gene_scaler and feature_scaler is not None and getattr(feature_scaler, 'scaling_ready', False):
        x_denormalized = feature_scaler.inverse_transform(x_unbounded)
        if verbose and verbose > 1:
            print(f"[CMA-ES] Applied inverse gene normalization", flush=True)
    else:
        x_denormalized = x_unbounded

    # Bound to [-1, 1] (or [0, 1] if geneMin=0)
    x_bounded = torch.tanh(x_denormalized)

    if gene_min == 0.0:
        x_bounded = (x_bounded + 1.0) / 2.0

    lower_bound = 0.0 if gene_min == 0 else -1.0
    candidates = torch.clamp(x_bounded, lower_bound, 1.0).cpu().tolist()

    # ── Random injection ──────────────────────────────────────────────────
    # Replace a fraction of candidates with pure random genes to maintain
    # exploration and prevent premature convergence (same as SNES).
    num_random = max(1, int(len(candidates) * random_injection_rate))
    for i in range(num_random):
        idx = len(candidates) - 1 - i  # replace from the end (worst mirrored)
        if idx >= 0:
            if gene_min == 0.0:
                candidates[idx] = np.random.uniform(0.0, 1.0, n).tolist()
            else:
                candidates[idx] = np.random.uniform(-1.0, 1.0, n).tolist()
    if verbose and verbose > 0:
        print(f"[CMA-ES] Injected {num_random}/{len(candidates)} random candidates", flush=True)

    return candidates
