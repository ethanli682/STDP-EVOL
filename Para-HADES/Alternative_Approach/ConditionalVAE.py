import gc
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Dict, Any, Optional

# ==============================================================================
#  Internal Model Class: Latent Codex CVAE (Conditional VAE)
# ==============================================================================
class LatentCodexCVAE(nn.Module):
    def __init__(self, gene_length, latent_dim=32, hidden_dim=256):
        super().__init__()
        self.gene_length = gene_length
        self.latent_dim = latent_dim

        # --- 1. Encoder (Gene -> Latent Distribution) ---
        self.encoder_shared = nn.Sequential(
            nn.Linear(gene_length, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2)
        )
        self.enc_mu = nn.Linear(hidden_dim, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)

        # --- 2. Decoder (Latent -> Gene) ---
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, gene_length) 
        )

        # --- 3. Forward Critic (Latent -> Fitness) ---
        self.critic = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        )

        # --- 4. Reverse "Dreamer" (Fitness + Noise -> Latent) ---
        # Input: Fitness (1 dim) + Noise (latent_dim)
        self.dreamer = nn.Sequential(
            nn.Linear(1 + latent_dim, 128), 
            nn.LeakyReLU(0.2),
            nn.Linear(128, latent_dim)
        )

    def encode(self, x):
        h = self.encoder_shared(x)
        return self.enc_mu(h), self.enc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, fitness, noise=None):
        # 1. VAE Pass
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)

        # 2. Critic Pass (Train latent space to map to fitness)
        pred_fitness = self.critic(z)

        # 3. Dreamer Pass (Train reverse mapping)
        if noise is None:
            noise = torch.randn_like(z)
        
        # Condition: Fitness + Noise -> Predict z
        dream_z = self.dreamer(torch.cat([fitness, noise], dim=1))

        return recon_x, pred_fitness, dream_z, mu, logvar, z

# ==============================================================================
#  Main Wrapper Function
# ==============================================================================
def createCandidateGeneUsingMultiHeadPredictor_chunk(
    args=None,
    EA_Class=None,
    genePopulation: Optional[List[Dict[str, Any]]] = None,
    number_of_candidate_genes: Optional[int] = None,
    build_in_params: Optional[Dict] = None,
    verbose: Optional[int] = None
) -> List[List[float]]:

    # --- Dependencies ---
    from GA_utils_scalers import ChunkedRankScaler, ChunkedPerDimensionScaler
    from GA_utils_history_loader import ChunkedGeneHistoryLoader

    # --- Configuration ---
    if build_in_params is None: build_in_params = {}
    verbose = verbose if verbose is not None else 0
    config = build_in_params.setdefault('LatentCodex', {})

    # Hyperparameters
    config.setdefault('epochs', 30)
    config.setdefault('hidden_dim', 256)
    config.setdefault('latent_dim', 32)
    config.setdefault('lr', 0.001)
    
    # Loss Weights
    config.setdefault('w_recon', 1.0)   
    config.setdefault('w_kl', 0.01)     
    config.setdefault('w_critic', 0.5)  
    config.setdefault('w_dream', 1.0)   

    # Generation Settings
    config.setdefault('gradient_steps', 30)       
    config.setdefault('gradient_lr', 0.1)         
    config.setdefault('dream_noise_scale', 1.0)   
    
    # Data Loading
    config.setdefault('top_k_percentile', 30)
    config.setdefault('chunk_allocation', 0.3)
    config.setdefault('min_chunk_size', 500)
    config.setdefault('max_seed_buffer', 20)

    # Automatic Reset Settings
    # If loss doesn't improve for X generations, we delete the model file and retrain.
    config.setdefault('reset_patience', 10) 
    
    # Setup Device
    device = torch.device('cuda' if getattr(args, 'gpu', False) and torch.cuda.is_available() else 'cpu')
    gene_length = EA_Class.geneLength
    gene_min = getattr(EA_Class, 'geneMin', -1.0)
    gene_max = getattr(EA_Class, 'geneMax', 1.0)
    evolution_target = getattr(args, 'evolutionTarget', 1)
    maximize = (evolution_target == 1)
    agent_idx = getattr(args, 'agent_idx', 0)

    # -----------------------------------------------------------
    # 1. Model Persistence & Automatic Reset Logic
    # -----------------------------------------------------------
    # Ensure running directory exists
    running_dir = os.path.join(args.path, 'running')
    os.makedirs(running_dir, exist_ok=True)
    model_path = os.path.join(running_dir, f'{str(agent_idx)}_CondVAE.pt')

    # Retrieve history of losses to check for stagnation
    loss_history = build_in_params.setdefault('loss_history', [])
    should_reset = False

    # Check automatic reset condition (Stagnation)
    if len(loss_history) > config['reset_patience']:
        recent_losses = loss_history[-config['reset_patience']:]
        # If variance of recent losses is tiny, we might be stuck in a local minimum
        if np.var(recent_losses) < 1e-5:
            should_reset = True
            if verbose: print("[LatentCodex] Stagnation detected. Resetting model.", flush=True)

    # Initialize Model
    model = LatentCodexCVAE(
        gene_length=gene_length,
        latent_dim=config['latent_dim'],
        hidden_dim=config['hidden_dim']
    ).to(device)

    # Load Weights (if they exist and we aren't resetting)
    if os.path.exists(model_path) and not should_reset:
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            if verbose: print(f"[LatentCodex] Loaded model from {model_path}", flush=True)
        except Exception as e:
            print(f"[LatentCodex] Warning: Could not load model ({e}). Starting fresh.", flush=True)
    else:
        if should_reset:
            # Clear history if we forced a reset
            build_in_params['loss_history'] = []

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    # -----------------------------------------------------------
    # 2. Setup Scalers & Loader
    # -----------------------------------------------------------
    fitness_scaler = ChunkedRankScaler(evolution_target=evolution_target, device=device)
    gene_scaler = ChunkedPerDimensionScaler(gene_length=gene_length, device=device)

    loader = ChunkedGeneHistoryLoader(
        savePath=args.path,
        agent_id=agent_idx,
        geneFormat=getattr(EA_Class, 'geneFormat', None),
        shuffle=True
    )

    try:
        chunk_size = loader.calculate_chunk_size(
            chunk_allocation=config['chunk_allocation'], 
            verbose=(verbose >= 1)
        )
        chunk_size = max(int(chunk_size), config['min_chunk_size'])
    except Exception:
        chunk_size = 2000

    # -----------------------------------------------------------
    # 3. Training Loop
    # -----------------------------------------------------------
    elites_buffer = [] 
    epoch_losses = []
    final_val_loss = 0.0

    model.train()
    
    for _ in range(config['epochs']):
        chunk_iter = loader.loadTopKPercentileGeneHistoryAsListOfDicts(
            args=args,
            top_k_percentile=config['top_k_percentile'],
            maximize=maximize,
            chunk_size=chunk_size,
            fitness_scaler=fitness_scaler,
            gene_scaler=gene_scaler,
            allEpochs=True, 
            fit_fitness_scaler_on_top_k=False, # Global Fitness Scale
            fit_gene_scaler_on_top_k=True      # Elite Gene Scale
        )

        for chunk, is_done in chunk_iter:
            # --- Extract Data ---
            valid_genes = []
            valid_fitness = []
            
            for record in chunk:
                g = record.get('gene')
                f = record.get('fitnessScore')
                if g is not None and f is not None and len(g) == gene_length:
                    valid_genes.append(g)
                    valid_fitness.append(f)

            if not valid_genes:
                if is_done: break
                continue

            # --- Convert & Scale ---
            genes_t = torch.tensor(valid_genes, dtype=torch.float32, device=device)
            fitness_t = torch.tensor(valid_fitness, dtype=torch.float32, device=device)

            genes_scaled = gene_scaler.transform(genes_t)
            # Unsqueeze to [Batch, 1] for conditioning
            fitness_scaled = fitness_scaler.scale_chunk(fitness_t).unsqueeze(1)

            # --- Populate Elites Buffer ---
            k_elite = max(1, int(len(genes_scaled) * 0.05))
            top_indices = torch.topk(fitness_scaled.flatten(), k_elite).indices
            elites_buffer.append(genes_scaled[top_indices].detach())
            if len(elites_buffer) > config['max_seed_buffer']: elites_buffer.pop(0)

            # --- Train Step ---
            optimizer.zero_grad()
            batch_noise = torch.randn(genes_scaled.size(0), config['latent_dim'], device=device)

            recon_x, pred_fitness, dream_z, mu, logvar, z_actual = model(
                genes_scaled, fitness_scaled, batch_noise
            )

            # Losses
            loss_recon = F.mse_loss(recon_x, genes_scaled)
            loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss_kl /= genes_scaled.size(0) 
            loss_critic = F.mse_loss(pred_fitness, fitness_scaled)
            loss_dream = F.mse_loss(dream_z, z_actual.detach()) 

            total_loss = (config['w_recon'] * loss_recon) + \
                         (config['w_kl'] * loss_kl) + \
                         (config['w_critic'] * loss_critic) + \
                         (config['w_dream'] * loss_dream)

            total_loss.backward()
            optimizer.step()
            
            final_val_loss = total_loss.item()
            epoch_losses.append(final_val_loss)

            if is_done: break
        
        loader.reset()

    # --- Metrics & Saving ---
    current_avg_loss = float(np.mean(epoch_losses[-10:])) if epoch_losses else 0.0
    build_in_params['loss'] = current_avg_loss
    
    # Update loss history for automatic reset logic next time
    loss_history.append(current_avg_loss)
    # Keep history bounded to avoid unlimited growth in the JSON file
    if len(loss_history) > 50: 
        loss_history = loss_history[-50:]
    build_in_params['loss_history'] = loss_history

    if verbose:
        print(f"[LatentCodex] Training Complete. Loss: {current_avg_loss:.5f}", flush=True)

    # -----------------------------------------------------------
    # 4. Generation Phase
    # -----------------------------------------------------------
    model.eval()
    
    n_nn_target = int(number_of_candidate_genes * 0.7) 
    n_raw_random = number_of_candidate_genes - n_nn_target
    
    generated_population = []

    # --- Strategy A: Dream (Reverse Head) ---
    n_dreams = int(n_nn_target * 0.5)
    
    # Target 3.0 because RankScaler maps Best Rank (1.0) to Z-score ~3.0
    target_fitness = torch.full((n_dreams, 1), 3.0, device=device) 
    
    dream_noise = torch.randn(n_dreams, config['latent_dim'], device=device) * config['dream_noise_scale']
    
    with torch.no_grad():
        z_dreamt = model.dreamer(torch.cat([target_fitness, dream_noise], dim=1))

    # --- Strategy B: Latent Perturbation of Elites ---
    n_elites = n_nn_target - n_dreams
    z_elites = []
    
    if elites_buffer:
        all_elites = torch.cat(elites_buffer, dim=0)
        # Random sample with replacement
        indices = torch.randint(0, len(all_elites), (n_elites,), device=device)
        selected_elites = all_elites[indices]
        with torch.no_grad():
            mu, logvar = model.encode(selected_elites)
            z_elites = model.reparameterize(mu, logvar)
    else:
        z_elites = torch.randn(n_elites, config['latent_dim'], device=device)

    # Combine seeds
    if not isinstance(z_elites, torch.Tensor): z_elites = torch.tensor([], device=device)
    z_seeds = torch.cat([z_dreamt, z_elites], dim=0)

    # --- Latent Optimization (Gradient Ascent) ---
    z_opt = z_seeds.clone().detach().requires_grad_(True)
    optimizer_z = optim.Adam([z_opt], lr=config['gradient_lr'])

    for _ in range(config['gradient_steps']):
        pred_fitness = model.critic(z_opt)
        # We want to Maximize fitness, so Minimize negative fitness
        loss = -pred_fitness.mean() 
        
        optimizer_z.zero_grad()
        loss.backward()
        optimizer_z.step()

    # --- Decode ---
    with torch.no_grad():
        final_genes_scaled = model.decoder(z_opt)
        final_genes = gene_scaler.inverse_transform(final_genes_scaled)
        generated_population = final_genes.cpu().numpy().tolist()

    # --- Strategy C: Random Immigrants ---
    if n_raw_random > 0:
        random_candidates = np.random.uniform(gene_min, gene_max, (n_raw_random, gene_length))
        generated_population.extend(random_candidates.tolist())

    # -----------------------------------------------------------
    # 5. Cleanup & Save
    # -----------------------------------------------------------
    # Save Model Weights to Disk (Critical for JSON safety)
    try:
        torch.save(model.state_dict(), model_path)
    except Exception as e:
        print(f"[LatentCodex] Error saving model: {e}", flush=True)

    # Ensure no PyTorch objects remain in build_in_params
    if 'model_state' in config: del config['model_state'] 

    # Cleanup memory
    del model, loader, fitness_scaler, gene_scaler, optimizer, z_opt
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Shuffle and Clip final output
    final_pop_np = np.array(generated_population, dtype=np.float32)
    np.random.shuffle(final_pop_np)
    final_pop_np = np.clip(final_pop_np, gene_min, gene_max)

    return final_pop_np.tolist()