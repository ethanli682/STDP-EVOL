"""
Latent Space Codex Generation Wrapper.
FIXED: 
1. Fitness Scaler reverted to ChunkedRankScaler (Stability).
2. Gene Scaler uses 'Safe Zoom' clamp (Sensitivity + Safety).
3. Hybrid Seeding (Elites + History + Noise) active.
"""

import gc
import numpy as np
import torch
from typing import List, Dict, Any, Optional

def createCandidateGeneUsingMultiHeadPredictor_chunk(
    args=None,
    EA_Class=None,
    genePopulation: Optional[List[Dict[str, Any]]] = None,
    number_of_candidate_genes: Optional[int] = None,
    build_in_params: Optional[Dict] = None,
    verbose: Optional[int] = None
) -> List[List[float]]:

    # REVERTED: Switched back to ChunkedRankScaler for fitness
    from GA_utils_scalers import ChunkedRankScaler, ChunkedPerDimensionScaler
    from GA_utils_history_loader import ChunkedGeneHistoryLoader
    from Alternative_Approach.BidirectionalMultiHeadPredictor import EnsembleBidirectionalPredictor
    
    # Defaults
    if build_in_params is None: build_in_params = {}
    verbose = verbose if verbose is not None else 0
    config = build_in_params.setdefault('MultiHeadPredictor', {})
    
    # --- CONFIGURATION ---
    config.setdefault('epochs', 40)
    config.setdefault('hidden_sizes', [256, 128, 64]) 
    config.setdefault('loss_weights', (1.0, 0.5)) 
    
    # Gradient Ascent Settings
    config.setdefault('gradient_steps', 20)           
    config.setdefault('gradient_step_size', 0.05)     
    config.setdefault('latent_noise', 0.25)           
    config.setdefault('regularization_strength', 0.1) 
    config.setdefault('novelty_strength', 1.0)        
    
    # Safe Zoom Setting
    config.setdefault('gene_min_std', 0.01)  
    
    # Data Loading
    config.setdefault('top_k_percentile', 30)         
    config.setdefault('chunk_allocation', 0.3)        
    config.setdefault('chunk_safety_margin', 0.2)     
    config.setdefault('min_chunk_size', 500)          
    
    config.setdefault('elite_sampling_percent', 0.05) 
    config.setdefault('max_seed_buffer', 20)          
    # ------------------------------
    
    device = torch.device('cuda' if getattr(args, 'gpu', False) and torch.cuda.is_available() else 'cpu')
    gene_length = EA_Class.geneLength
    
    # Gene bounds
    gene_min = getattr(EA_Class, 'geneMin', -1.0)
    gene_max = getattr(EA_Class, 'geneMax', 1.0)
    
    evolution_target = getattr(args, 'evolutionTarget', 1)
    maximize = (evolution_target == 1)

    # 1. Setup Scalers
    # FITNESS: RankScaler (Stable gradients)
    scaler = ChunkedRankScaler(
        evolution_target=evolution_target, 
        device=device
    )
    
    # GENES: PerDimension (Sensitivity normalization)
    gene_scaler = ChunkedPerDimensionScaler(
        gene_length=gene_length, 
        device=device
    )
    
    # 2. Initialize Model
    predictor = EnsembleBidirectionalPredictor(
        gene_length=gene_length,
        hidden_sizes=config['hidden_sizes'],
        device=str(device)
    )
    
    # 3. Training Loop
    predictor.start_chunk_training(
        savePath=args.path,
        agent_idx=getattr(args, 'agent_idx', 0),
        epochs=config['epochs'],
        loss_weights=config['loss_weights']
    )
    
    loader = ChunkedGeneHistoryLoader(
        savePath=args.path,
        agent_id=getattr(args, 'agent_idx', 0),
        geneFormat=getattr(EA_Class, 'geneFormat', None),
        shuffle=True
    )
    
    try:
        chunk_size = loader.calculate_chunk_size(
            safety_margin=config['chunk_safety_margin'], 
            chunk_allocation=config['chunk_allocation'], 
            verbose=(verbose >= 1)
        )
        chunk_size = max(int(chunk_size), config['min_chunk_size'])
        if verbose >= 1: print(f"[LatentCodex] Chunk Size: {chunk_size}", flush=True)
    except Exception:
        chunk_size = 2000

    elites_buffer = []
    random_history_buffer = [] 
    final_val_loss = 0.0
    
    # Clamp value (float)
    min_std_val = config['gene_min_std']
    
    training_complete = False
    while not training_complete:
        
        for chunk, is_done in loader.loadTopKPercentileGeneHistoryAsListOfDicts(
            args=args, 
            top_k_percentile=config['top_k_percentile'], 
            maximize=maximize,
            chunk_size=chunk_size, 
            fitness_scaler=scaler, 
            gene_scaler=gene_scaler, 
            allEpochs=True,
            use_fitness_scaler_weights=True,
            fit_fitness_scaler_on_top_k=False, # Global fitness scale
            fit_gene_scaler_on_top_k=True      # Gene scale focused on high performers
        ):
            # === EXTRACT DATA ===
            valid_genes = []
            valid_fitness = []
            valid_weights = []

            for record in chunk:
                g = record.get('gene')
                f = record.get('fitnessScore')
                w = record.get('_es_weight', 1.0)

                if g is None or f is None: continue
                if hasattr(g, '__len__') and len(g) != gene_length: continue

                try:
                    g_arr = np.array(g, dtype=np.float32)
                    f_val = float(f)
                    w_val = float(w)
                    if not np.isfinite(g_arr).all() or not np.isfinite(f_val) or not np.isfinite(w_val): continue
                    
                    valid_genes.append(g_arr)
                    valid_fitness.append(f_val)
                    valid_weights.append(w_val)
                except: continue
            
            if not valid_genes:
                if is_done: break
                continue

            genes_np = np.stack(valid_genes)
            fitness_np = np.array(valid_fitness, dtype=np.float32)
            weights_np = np.array(valid_weights, dtype=np.float32)
            
            # === SCALING WITH TYPE-SAFE CLAMP ===
            genes_t = torch.from_numpy(genes_np).to(device)
            fitness_t = torch.from_numpy(fitness_np).to(device)
            
            # SAFE ZOOM: Clamp minimum variance
            if hasattr(gene_scaler, 'std'):
                if isinstance(gene_scaler.std, torch.Tensor):
                    limit = torch.tensor(min_std_val, device=gene_scaler.std.device, dtype=gene_scaler.std.dtype)
                    gene_scaler.std = torch.max(gene_scaler.std, limit)
                elif isinstance(gene_scaler.std, np.ndarray):
                    gene_scaler.std = np.maximum(gene_scaler.std, min_std_val)
                    
            elif hasattr(gene_scaler, 'scale_'): 
                if isinstance(gene_scaler.scale_, torch.Tensor):
                    limit = torch.tensor(min_std_val, device=gene_scaler.scale_.device, dtype=gene_scaler.scale_.dtype)
                    gene_scaler.scale_ = torch.max(gene_scaler.scale_, limit)
                elif isinstance(gene_scaler.scale_, np.ndarray):
                    gene_scaler.scale_ = np.maximum(gene_scaler.scale_, min_std_val)
            
            genes_scaled = gene_scaler.transform(genes_t)
            fitness_scaled = scaler.scale_chunk(fitness_t)
            
            # === BUFFER COLLECTION ===
            valid_count = len(genes_np)
            k_elite = max(1, int(valid_count * config['elite_sampling_percent']))
            top_indices = torch.topk(fitness_scaled, k_elite).indices
            
            elites_buffer.append(genes_scaled[top_indices].cpu().numpy())
            if len(elites_buffer) > config['max_seed_buffer']: elites_buffer.pop(0) 
            
            k_random = max(1, int(valid_count * 0.05))
            rand_indices = torch.randperm(valid_count)[:k_random]
            random_history_buffer.append(genes_scaled[rand_indices].cpu().numpy())
            if len(random_history_buffer) > config['max_seed_buffer']: random_history_buffer.pop(0)

            # === TRAIN ===
            predictor.process_chunk(
                genes=genes_scaled.cpu().numpy(), 
                fitness_scores=fitness_scaled.cpu().numpy(),
                sample_weights=weights_np 
            )
            
            if is_done: break
            
        training_complete, info = predictor.finish_chunk_epoch()
        if 'best_val_loss' in info: final_val_loss = info['best_val_loss']
        loader.reset()
        
    if isinstance(final_val_loss, torch.Tensor): final_val_loss = final_val_loss.item()
    build_in_params['loss'] = float(final_val_loss)

    # 4. GENERATION
    n_nn_target = int(number_of_candidate_genes * 0.6) 
    n_raw_random = number_of_candidate_genes - n_nn_target 
    
    nn_candidates = []
    
    # --- A. Prepare NN Seeds ---
    seeds_list = []
    
    if elites_buffer:
        all_elites = np.vstack(elites_buffer)
        n_elites = int(n_nn_target * 0.5)
        indices = np.random.choice(len(all_elites), n_elites, replace=True)
        seeds_list.append(all_elites[indices])
        
    if random_history_buffer:
        all_history = np.vstack(random_history_buffer)
        n_history = n_nn_target - (len(seeds_list[0]) if seeds_list else 0)
        if n_history > 0:
            indices = np.random.choice(len(all_history), n_history, replace=True)
            seeds_list.append(all_history[indices])
            
    if not seeds_list:
        seeds_list.append(np.random.uniform(-1, 1, (n_nn_target, gene_length)))
        
    seeds_array = np.vstack(seeds_list)
    
    # --- B. Evolve NN Candidates ---
    if len(seeds_array) > 0:
        candidates_scaled = predictor.evolve_population(
            seed_genes=seeds_array, 
            num_candidates=len(seeds_array),
            steps=config['gradient_steps'],
            step_size=config['gradient_step_size'],
            noise_scale=config['latent_noise'],
            regularization_strength=config['regularization_strength'],
            novelty_strength=config['novelty_strength'] 
        )
        nn_candidates = gene_scaler.inverse_transform(candidates_scaled)
    
    # --- C. Generate Raw Random Immigrants ---
    raw_random_candidates = np.random.uniform(gene_min, gene_max, (n_raw_random, gene_length))
    
    if verbose: 
        print(f"Generation: {len(nn_candidates)} via Latent Codex (Exploitation), "
              f"{len(raw_random_candidates)} via Random Injection (Exploration)", flush=True)
    
    # Combine
    if len(nn_candidates) > 0:
        final_population = np.vstack([nn_candidates, raw_random_candidates])
    else:
        final_population = raw_random_candidates
        
    del predictor, loader, scaler, gene_scaler
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    np.random.shuffle(final_population)
    
    return np.clip(final_population, gene_min, gene_max).tolist()