import torch
import numpy as np
import math
from typing import List, Dict, Any, Optional, Tuple
import time

# Import your custom utilities
from GA_utils_history_loader import ChunkedGeneHistoryLoader
from GA_utils_scalers import ChunkedPerDimensionScaler, ChunkedLogZScoreScaler

def createCandidateGene_Diffusion(
    args=None,
    EA_Class=None,
    genePopulation: Optional[List[Dict[str, Any]]] = None,
    number_of_candidate_genes: Optional[int] = None,
    build_in_params: Optional[Dict] = None,
    verbose: Optional[int] = None
) -> Tuple[List[List[float]], float]:
    """
    Generate candidate genes using Diffusion Evolution (DiffEvo) with Anti-Collapse mechanisms.
    
    IMPROVEMENTS:
    1. Returns (candidates, loss) tuple to fix "Loss: 0.0" logging.
    2. Implements Weight Temperature to prevent collapsing to the single best gene.
    3. Adds Softmax Temperature to broaden attention during diffusion.
    4. Auto-detects variance collapse and injects emergency jitter.
    """
    if args is None or EA_Class is None or number_of_candidate_genes is None:
        raise ValueError("Missing required arguments for DiffEvo.")
    
    # 1. Setup & Hyperparameters
    device = torch.device('cuda' if getattr(args, 'gpu', False) and torch.cuda.is_available() else 'cpu')
    gene_length = EA_Class.geneLength
    evolution_target = getattr(args, 'evolutionTarget', 1)
    
    # Defaults
    if 'DiffEvo' not in build_in_params:
        build_in_params['DiffEvo'] = {}
    params = build_in_params['DiffEvo']
        
    num_steps = params.get('steps', 30)
    top_k_percentile = params.get('top_k', 15)  # Increased default for more diversity
    latent_threshold = params.get('latent_threshold', 512)
    noise_scale = params.get('noise_scale', 1.0) 
    
    # NEW: Anti-Collapse Hyperparameters
    # Higher = flatter weights (more diversity), Lower = sharper (greedy optimization)
    weight_temperature = params.get('weight_temperature', 0.5) 
    # Higher = fuzzier attention (more diversity)
    attn_temperature = params.get('attn_temperature', 1.2) 
    
    def vprint(level, msg):
        if (verbose or 0) >= level: print(f"[DiffEvo] {msg}", flush=True)

    vprint(1, f"Init DiffEvo (Target: {evolution_target}, Steps: {num_steps}, Device: {device})")

    # 2. Initialize Scalers
    fitness_scaler = ChunkedLogZScoreScaler(
        evolution_target=evolution_target,
        emphasis_factor=2.0, 
        device=device
    )
    
    feature_scaler = ChunkedPerDimensionScaler(
        gene_length=gene_length,
        device=device,
        weight_mode='uniform' 
    )
    
    # 3. Reference Data Acquisition
    ref_genes = None
    ref_weights = None
    
    # --- PATH A: Use In-Memory Population ---
    if genePopulation is not None and len(genePopulation) > 0:
        vprint(2, f"Using provided in-memory population ({len(genePopulation)} individuals).")
        genes_raw = [g['gene'] for g in genePopulation]
        fitness_raw = [g['fitnessScore'] for g in genePopulation]
        
        genes_tensor = torch.tensor(genes_raw, dtype=torch.float32, device=device)
        fitness_tensor = torch.tensor(fitness_raw, dtype=torch.float32, device=device)
        
        # Fit Scalers
        feature_scaler.collect_phase_1_statistics(genes_tensor)
        feature_scaler._finalize_pass1() 
        feature_scaler.collect_phase_2_statistics(genes_tensor)
        feature_scaler.finalize()
        
        fitness_scaler.collect_phase_1_statistics(fitness_tensor)
        fitness_scaler._finalize_pass1()
        fitness_scaler.collect_phase_2_statistics(fitness_tensor)
        fitness_scaler.finalize()
        
        ref_genes = feature_scaler.transform(genes_tensor)
        ref_weights = fitness_scaler.get_global_weights(fitness_tensor)
        
    # --- PATH B: Load from Disk ---
    else:
        vprint(2, "No in-memory population. Streaming history from disk...")
        loader = ChunkedGeneHistoryLoader(
            savePath=args.path,
            agent_id=getattr(args, 'agent_idx', getattr(args, 'agent_counter', 0)),
            geneFormat=getattr(EA_Class, 'geneFormat', 'json'),
            required_keys_and_types=['fitnessScore', 'gene']
        )
        
        # Safe memory management
        raw_chunk_size = loader.calculate_chunk_size(verbose=(verbose or 0)>=3)
        safe_chunk_size = int(raw_chunk_size * 0.6)
        
        # Buffer limit based on device
        try:
            if device.type == 'cuda':
                mem_free, _ = torch.cuda.mem_get_info(device)
                bytes_per_item = (gene_length * 4) + 4
                max_buffer_items = int((mem_free * 0.4) / bytes_per_item)
            else:
                max_buffer_items = 10000 
        except:
            max_buffer_items = 5000

        generator = loader.loadTopKPercentileGeneHistoryAsListOfDicts(
            args=args,
            top_k_percentile=top_k_percentile,
            maximize=(evolution_target == 1),
            chunk_size=safe_chunk_size,
            fitness_scaler=fitness_scaler, 
            gene_scaler=feature_scaler,    
            use_fitness_scaler_weights=False, 
            use_gene_scaler_weights=False,
            allEpochs=True,
            # CRITICAL: We normally want to fit scalers on the whole history to define bounds,
            # but if the loader logic limits us, we fit on what we load.
            # Assuming standard behavior here.
            min_datapoints=min(int(getattr(args, 'populationSize', 10000)*0.05), 300),
            max_datapoints=int(getattr(args, 'populationSize', 10000)),
        )
        
        reference_genes_list = []
        reference_weights_list = []
        total_loaded = 0
        
        for chunk, _ in generator:
            if not chunk: continue
            
            g_raw = [g['gene'] for g in chunk]
            f_raw = [g['fitnessScore'] for g in chunk]
            
            g_tensor = torch.tensor(g_raw, dtype=torch.float32, device=device)
            f_tensor = torch.tensor(f_raw, dtype=torch.float32, device=device)
            
            g_norm = feature_scaler.transform(g_tensor)
            w_norm = fitness_scaler.get_global_weights(f_tensor)
            
            reference_genes_list.append(g_norm)
            reference_weights_list.append(w_norm)
            
            total_loaded += len(chunk)
            if total_loaded >= max_buffer_items:
                break
        
        if reference_genes_list:
            ref_genes = torch.cat(reference_genes_list, dim=0)
            ref_weights = torch.cat(reference_weights_list, dim=0)

    # 4. Data Validation & Anti-Collapse Prep
    if ref_genes is None or len(ref_genes) == 0:
        vprint(0, "Warning: No reference data. Returning random.")
        # Return dummy loss 1.0 for random
        gene_min = getattr(EA_Class, 'geneMin', -1.0)
        return _generate_random_genes(number_of_candidate_genes, gene_length, device, gene_min), 1.0

    # Variance Check: If data collapsed, add noise immediately
    std_check = torch.std(ref_genes, dim=0).mean()
    if std_check < 1e-4:
        vprint(1, f"Warning: Variance collapse detected (std={std_check:.6f}). Injecting pre-diffusion perturbation.")
        ref_genes = ref_genes + torch.randn_like(ref_genes) * 0.1

    # Apply Weight Temperature (Flattening)
    # Allows the model to consider "good" genes, not just the "best" gene.
    ref_weights = torch.pow(ref_weights, weight_temperature)
    ref_weights = ref_weights / (ref_weights.sum() + 1e-8)
    
    # 5. Latent Projection Setup
    use_latent = (gene_length > latent_threshold)
    projection_matrix = None
    
    if use_latent:
        latent_dim = 64
        projection_matrix = torch.randn(gene_length, latent_dim, device=device) / math.sqrt(gene_length)
        ref_genes_metric = torch.matmul(ref_genes, projection_matrix)
        vprint(2, f"Using Latent Diffusion (Dim: {latent_dim})")
    else:
        ref_genes_metric = ref_genes

    # 6. Diffusion Process (DDIM Reverse Sampling)
    
    # Candidates x_T ~ N(0, I)
    x_t = torch.randn(number_of_candidate_genes, gene_length, device=device)
    
    timesteps = torch.linspace(0, 1, num_steps + 1, device=device)
    alphas = _cosine_schedule(timesteps)
    
    accumulated_loss = 0.0
    
    for t_idx in range(num_steps, 0, -1):
        alpha_t = alphas[t_idx]      
        alpha_prev = alphas[t_idx - 1] 
        
        # --- A. Metric Calculation ---
        if use_latent:
            x_t_metric = torch.matmul(x_t, projection_matrix)
        else:
            x_t_metric = x_t
            
        ref_target_metric = torch.sqrt(alpha_t) * ref_genes_metric
        
        # Squared Euclidean Distance
        x_norm_sq = (x_t_metric ** 2).sum(dim=1, keepdim=True)
        y_norm_sq = (ref_target_metric ** 2).sum(dim=1, keepdim=True).t()
        xy_dot = torch.matmul(x_t_metric, ref_target_metric.t())
        
        dist_sq = x_norm_sq + y_norm_sq - 2 * xy_dot
        
        # ACCUMULATE LOSS (Mean distance at this step)
        # This acts as our "Guidance Loss" to report back
        step_loss = dist_sq.min(dim=1)[0].mean().item() # Mean distance to nearest neighbor
        accumulated_loss += step_loss

        # Probability Kernel
        denominator = 2 * (1.0 - alpha_t) + 1e-8
        log_probs = -dist_sq / denominator
        
        # Combine with Fitness Weights
        log_Q = torch.log(ref_weights + 1e-10).unsqueeze(0)
        combined_logits = log_probs + log_Q
        
        # Softmax with Temperature (Broadens attention)
        attn_weights = torch.softmax(combined_logits / attn_temperature, dim=1) 
        
        # Estimate x0 (Denoised Data)
        x0_hat = torch.matmul(attn_weights, ref_genes)
        
        # --- B. Update Step ---
        alpha_t_safe = torch.clamp(alpha_t, 0.0, 0.9999)
        eps_pred = (x_t - torch.sqrt(alpha_t_safe) * x0_hat) / torch.sqrt(1.0 - alpha_t_safe + 1e-8)
        
        # Sigma (Mutation Noise)
        alpha_ratio = torch.clamp(alpha_t / (alpha_prev + 1e-10), 0.0, 1.0)
        sigma_t = torch.sqrt((1 - alpha_prev) / (1 - alpha_t + 1e-10) * (1 - alpha_ratio))
        
        # Explicit Noise Boost: If std is low, keep noise higher for longer
        if std_check < 0.1: 
            sigma_t = sigma_t * (noise_scale * 1.5)
        else:
            sigma_t = torch.clamp(sigma_t * noise_scale, 0.0, 1.0)
        
        # Direction to x_t (Deterministic)
        dir_variance = torch.clamp(1 - alpha_prev - sigma_t**2, 0.0, 1.0)
        pred_dir_xt = torch.sqrt(dir_variance + 1e-8) * eps_pred
        
        # x_{t-1}
        x_prev = torch.sqrt(alpha_prev) * x0_hat + pred_dir_xt
        
        # Stochastic Mutation
        if t_idx > 0:
            noise = torch.randn_like(x_t)
            x_prev = x_prev + sigma_t * noise
            
        x_t = x_prev

    # 7. Finalization
    vprint(2, "Inverse transforming...")
    final_genes = feature_scaler.inverse_transform(x_t)
    
    # Check for final collapse
    final_std = torch.std(final_genes, dim=0).mean().item()
    if final_std < 1e-4:
        vprint(1, "Warning: Final population collapsed. Adding jitter.")
        final_genes = final_genes + torch.randn_like(final_genes) * 0.05

    gene_min = getattr(EA_Class, 'geneMin', -1.0)
    lower_bound = 0.0 if gene_min == 0 else -1.0
    final_genes = torch.clamp(final_genes, lower_bound, 1.0)
    candidates = final_genes.tolist()
    
    # Calculate final average loss
    final_loss = accumulated_loss / num_steps
    
    if device.type == 'cuda':
        del ref_genes, ref_weights, ref_genes_metric, x_t
        torch.cuda.empty_cache()
    
    build_in_params['loss'] = float(final_loss)

    return candidates

# ----------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------

def _cosine_schedule(t):
    """
    Cosine schedule.
    Returns alpha_bar (cumulative product of alphas).
    Range: 1.0 (t=0, Data) -> 0.0 (t=1, Noise)
    """
    s = 0.008
    steps = t * math.pi / 2
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    return f / f[0]

def _generate_random_genes(count, length, device, gene_min=-1.0):
    lower_bound = 0.0 if gene_min == 0 else -1.0
    return torch.randn(count, length, device=device).clamp(lower_bound, 1.0).tolist()