import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler
import numpy as np
import math
import os
from typing import List, Dict, Any, Optional

# Imports
from GA_utils_history_loader import ChunkedGeneHistoryLoader
from GA_utils_scalers import ChunkedLogZScoreScaler, ChunkedPerDimensionScaler

# ===========================================================================
# 1. THE MODEL (No Changes needed here, kept for context)
# ===========================================================================

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int, scale: float = 1.0):
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, x: torch.Tensor):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        x = x.unsqueeze(1) * self.scale
        emb = x * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1, 0, 0))
        return emb

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 512):
        super().__init__()
        # Encoder: 15k -> 2048 -> 512
        self.enc_net = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.LayerNorm(2048),
            nn.SiLU(),
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
        )
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_var = nn.Linear(1024, latent_dim)

        # Decoder: 512 -> 2048 -> 15k
        self.dec_net = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Linear(1024, 2048),
            nn.LayerNorm(2048),
            nn.SiLU(),
            nn.Linear(2048, input_dim)
        )

    def encode(self, x):
        h = self.enc_net(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.dec_net(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z)
        return recon_x, mu, log_var

# ===========================================================================
# 2. EXISTING COMPONENT: ResidualMLP (No changes, just context)
# ===========================================================================

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * 2)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(dropout)
        self.cond_proj = nn.Linear(dim, dim * 2) 
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

    def forward(self, x, condition_emb):
        resid = x
        emb = self.cond_proj(condition_emb)
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm1(x)
        x = x * (1 + scale) + shift 
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x + resid

class ResidualMLP(nn.Module):
    def __init__(self, gene_length: int, config: Dict):
        super().__init__()
        hidden_dims_config = config.get('hidden_dims', 512)
        if hidden_dims_config is None or hidden_dims_config in ['None', 'none']:
            hidden_dim = 512
        elif isinstance(hidden_dims_config, (list, tuple)):
            hidden_dim = int(hidden_dims_config[0]) if hidden_dims_config else 512
        else:
            hidden_dim = int(hidden_dims_config)
        
        hidden_dim = int(hidden_dim)
        gene_length = int(gene_length)
        
        num_layers_raw = config.get('layers', 6)
        num_layers = int(num_layers_raw if num_layers_raw is not None else 6)
        dropout = float(config.get('dropout', 0.1) or 0.1)
        emb_scale = float(config.get('fitness_embedding_scale', 1.0) or 1.0)
        
        self.input_proj = nn.Linear(gene_length, hidden_dim)
        self.time_emb = SinusoidalEmbedding(hidden_dim, scale=1.0)
        self.fit_emb = SinusoidalEmbedding(hidden_dim, scale=emb_scale) 
        self.null_fitness = nn.Parameter(torch.randn(hidden_dim))
        
        self.mlp_emb = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, gene_length)

    def forward(self, x, timestep, fitness=None):
        x = self.input_proj(x)
        t_emb = self.time_emb(timestep)
        if fitness is not None:
            fitness = torch.clamp(fitness, -5.0, 5.0)
            f_emb = self.fit_emb(fitness)
        else:
            f_emb = self.null_fitness.expand(x.shape[0], -1)
            
        cond = self.mlp_emb(t_emb + f_emb)
        for block in self.blocks:
            x = block(x, cond)
        return self.output_proj(x)

# ===========================================================================
# 3. UPDATED CLASS: EvolutionaryDiffusion (With Toggle)
# ===========================================================================

class EvolutionaryDiffusion:
    def __init__(self, gene_length: int, device: str = 'cpu', config: Dict = None):
        self.gene_length = gene_length
        self.device = torch.device(device)
        self.config = config or {}
        
        # --- NEW: Toggle between 'MLP' (Standard) and 'VAE' (Latent) ---
        self.model_type = self.config.get('model_type', 'MLP') 
        self.latent_dim = int(self.config.get('latent_dim', 512))

        # 1. Initialize VAE if selected
        self.vae = None
        if self.model_type == 'VAE':
            print(f"[HADES] Model Type: VAE (Latent Diffusion). Compressing {gene_length} -> {self.latent_dim}", flush=True)
            self.vae = VariationalAutoencoder(gene_length, self.latent_dim).to(self.device)
            self.vae_optimizer = torch.optim.AdamW(self.vae.parameters(), lr=1e-4)
            # When using VAE, the diffusion model sees 'latent_dim' inputs, not 'gene_length'
            diffusion_input_dim = self.latent_dim
            # Adjust hidden dims for diffusion model since input is smaller
            if self.config.get('hidden_dims') is None:
                self.config['hidden_dims'] = [512, 512] 
        else:
            print(f"[HADES] Model Type: Standard ResidualMLP", flush=True)
            diffusion_input_dim = gene_length
            if self.config.get('hidden_dims') is None:
                if gene_length <= 200: self.config['hidden_dims'] = [512, 256]
                else: self.config['hidden_dims'] = [1024, 512, 256]

        # 2. Initialize Diffusion Model
        self.model = ResidualMLP(diffusion_input_dim, self.config).to(self.device)
        
        # Setup Scheduler
        diffusion_steps = int(self.config.get('diffusion_steps', 1000) or 1000)
        self.scheduler = DDIMScheduler(
            num_train_timesteps=diffusion_steps,
            beta_schedule="squaredcos_cap_v2", 
            prediction_type="epsilon"
        )
        
        # Setup Diffusion Optimizer
        lr = float(self.config.get('learning_rate', 1e-5) or 1e-5)
        wd = float(self.config.get('weight_decay', 1e-4) or 1e-4)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
    
    def save_checkpoint(self, save_path: str) -> None:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            state = {
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'config': self.config
            }
            if self.vae is not None:
                state['vae_state'] = self.vae.state_dict()
            torch.save(state, save_path)
        except Exception as e:
            print(f"[Diffusion WARNING] Failed to save checkpoint: {e}", flush=True)
    
    def load_checkpoint(self, load_path: str) -> bool:
        if not os.path.exists(load_path): return False
        try:
            checkpoint = torch.load(load_path, map_location=self.device)
            # Basic check to prevent loading VAE weights into MLP or vice versa
            saved_type = checkpoint.get('config', {}).get('model_type', 'MLP')
            if saved_type != self.model_type:
                print(f"[Diffusion WARNING] Checkpoint type ({saved_type}) mismatch with current ({self.model_type}). Ignoring.", flush=True)
                return False

            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            if self.vae is not None and 'vae_state' in checkpoint:
                self.vae.load_state_dict(checkpoint['vae_state'])
            
            print(f"[Diffusion] Loaded pre-trained checkpoint from {load_path}", flush=True)
            return True
        except Exception as e:
            print(f"[Diffusion WARNING] Failed to load checkpoint: {e}", flush=True)
            return False

    def train(self, loader, args, loader_kwargs: Dict, epochs: int, batch_size: int, log_dict: Dict, is_finetuning: bool = False):
        self.model.train()
        if self.vae: self.vae.train()
        torch.set_grad_enabled(True)
        
        # --- Pre-calculate VAE epochs if needed ---
        # If we are using VAE, we spend the first 30% of epochs (or min 5) training ONLY the VAE
        # to ensure the latent space is stable before we try to diffuse in it.
        vae_epochs = 0
        if self.vae is not None:
            vae_epochs = max(5, int(epochs * 0.3))
            diffusion_epochs = epochs - vae_epochs
            print(f"[HADES] Strategy: {vae_epochs} VAE Pre-training epochs -> {diffusion_epochs} Latent Diffusion epochs", flush=True)
        else:
            diffusion_epochs = epochs

        fitness_scaler = loader_kwargs.get('fitness_scaler')
        gene_scaler = loader_kwargs.get('gene_scaler')

        # --- Helper for VAE Loss ---
        def vae_loss_fn(recon_x, x, mu, logvar):
            MSE = F.mse_loss(recon_x, x, reduction='mean')
            # KL Divergence: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            return MSE + 0.0001 * KLD # Small weight on KLD to prevent posterior collapse

        # ==========================
        # MAIN LOOP
        # ==========================
        last_epoch_loss = 0.0

        for epoch in range(epochs):
            # Check if we are in VAE-only training phase
            training_vae_phase = (self.vae is not None and epoch < vae_epochs)
            
            epoch_loss_accum = 0.0
            num_batches = 0
            
            # Reset Gradients
            self.optimizer.zero_grad()
            if self.vae: self.vae_optimizer.zero_grad()
            
            data_stream = loader.loadTopKPercentileGeneHistoryAsListOfDicts(args=args, **loader_kwargs)
            
            for chunk_data, is_done in data_stream:
                if not chunk_data: 
                    if is_done: break
                    continue

                # 1. Parse Chunk
                chunk_genes, chunk_fits, chunk_weights = [], [], []
                for record in chunk_data:
                    g = record.get('gene')
                    f = record.get('fitnessScore', 0.0)
                    w = record.get('_fitness_scaler_weight', 1.0) * record.get('_gene_scaler_weight', 1.0)
                    if g is None or len(g) != self.gene_length: continue
                    chunk_genes.append(g)
                    chunk_fits.append(f)
                    chunk_weights.append(w)
                
                if not chunk_genes: continue

                tensor_genes = torch.tensor(chunk_genes, dtype=torch.float32)
                tensor_fits = torch.tensor(chunk_fits, dtype=torch.float32)
                tensor_weights = torch.tensor(chunk_weights, dtype=torch.float32)

                # Batch Loop
                dataset_size = len(tensor_genes)
                indices = torch.randperm(dataset_size)
                
                for start_idx in range(0, dataset_size, batch_size):
                    batch_idx = indices[start_idx : min(start_idx + batch_size, dataset_size)]
                    
                    b_genes = tensor_genes[batch_idx].to(self.device)
                    b_fits = tensor_fits[batch_idx].to(self.device)
                    b_weights = tensor_weights[batch_idx].to(self.device)

                    # Normalize Inputs
                    with torch.no_grad():
                        norm_fitness = fitness_scaler.scale_chunk(b_fits)
                        norm_fitness = torch.clamp(norm_fitness, -5.0, 5.0)
                        if getattr(gene_scaler, 'scaling_ready', False):
                            b_genes.sub_(gene_scaler.mean_tensor.to(self.device)).div_(gene_scaler.std_tensor.to(self.device) + 1e-8)

                    # ==========================
                    # MODE A: VAE TRAINING
                    # ==========================
                    if training_vae_phase:
                        recon_x, mu, logvar = self.vae(b_genes)
                        loss = vae_loss_fn(recon_x, b_genes, mu, logvar)
                        
                        loss.backward()
                        self.vae_optimizer.step()
                        self.vae_optimizer.zero_grad()
                        epoch_loss_accum += loss.item()
                    
                    # ==========================
                    # MODE B: DIFFUSION TRAINING
                    # ==========================
                    else:
                        # 1. Prepare Target
                        if self.vae:
                            # If VAE, we encode genes to latents first
                            with torch.no_grad():
                                mu, logvar = self.vae.encode(b_genes)
                                # Sample from distribution to get latent
                                target_data = self.vae.reparameterize(mu, logvar) 
                        else:
                            # Standard: Target is the gene itself
                            target_data = b_genes

                        # 2. Add Noise
                        noise = torch.randn_like(target_data)
                        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, 
                                                (target_data.shape[0],), device=self.device).long()
                        noisy_input = self.scheduler.add_noise(target_data, noise, timesteps)
                        
                        # 3. Predict Noise
                        # CFG: Drop conditioning 10% of time
                        use_cond = np.random.random() > 0.1
                        model_fitness = norm_fitness if use_cond else None
                        
                        noise_pred = self.model(noisy_input, timesteps, fitness=model_fitness)
                        
                        # 4. Loss
                        loss_elementwise = F.mse_loss(noise_pred, noise, reduction='none')
                        loss_per_sample = loss_elementwise.mean(dim=1)
                        loss_per_sample = torch.clamp(loss_per_sample, max=5.0)
                        
                        avg_w = b_weights.mean() + 1e-8
                        norm_w = torch.clamp(b_weights / avg_w, 0.1, 5.0)
                        weighted_loss = (loss_per_sample * norm_w).mean()
                        
                        weighted_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        epoch_loss_accum += weighted_loss.item()
                    
                    num_batches += 1

            if num_batches > 0:
                last_epoch_loss = epoch_loss_accum / num_batches
                phase_name = "VAE-Pretrain" if training_vae_phase else "Diffusion"
                if epoch % 1 == 0:
                    print(f" Epoch {epoch+1}/{epochs} [{phase_name}]: Loss={last_epoch_loss:.6f}", flush=True)

        return last_epoch_loss

    @torch.no_grad()
    def generate(self, n_samples: int, guidance_scale: float, steps: int = 50, verbose: bool = False):
        self.model.eval()
        if self.vae: self.vae.eval()
        
        # Dimensions: If VAE, we generate Latents (e.g., 512). If MLP, we generate Genes (e.g., 15k)
        shape_dim = self.latent_dim if self.model_type == 'VAE' else self.gene_length
        
        # Start from random noise
        current_sample = torch.randn((n_samples, shape_dim), device=self.device)
        self.scheduler.set_timesteps(steps)
        
        # High fitness target
        fit_cond = torch.normal(mean=3.5, std=1.0, size=(n_samples,), device=self.device)
        fit_cond = torch.clamp(fit_cond, 2.0, 5.0)

        # Diffusion Loop
        for t in self.scheduler.timesteps:
            t_tensor = torch.full((n_samples,), t.item(), device=self.device, dtype=torch.long)
            
            # Scale input for scheduler
            sample_scaled = self.scheduler.scale_model_input(current_sample, t)
            
            # Predict noise (Uncond + Cond)
            noise_uncond = self.model(sample_scaled, t_tensor, fitness=None)
            noise_cond = self.model(sample_scaled, t_tensor, fitness=fit_cond)
            
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            
            # Step
            current_sample = self.scheduler.step(noise_pred, t, current_sample).prev_sample

        # Post-Processing
        if self.model_type == 'VAE':
            # If we generated latents, decode them to genes
            if verbose: print(f"[HADES] Decoding latents to genes...", flush=True)
            final_genes = self.vae.decode(current_sample)
        else:
            final_genes = current_sample

        return final_genes.cpu().numpy().tolist()

# ===========================================================================
# 3. ENTRY POINT
# ===========================================================================

def createCandidateGeneUsingHADES(
    args=None,
    EA_Class=None,
    genePopulation: Optional[List[Dict[str, Any]]] = None,
    number_of_candidate_genes: Optional[int] = None,
    build_in_params: Optional[Dict] = None,
    verbose: Optional[int] = None
) -> List[List[float]]:
    
    if args is None or EA_Class is None or number_of_candidate_genes is None:
        raise ValueError("args, EA_Class, and number_of_candidate_genes are required.")
    
    if build_in_params is None: build_in_params = {}
    if 'HADES' not in build_in_params: build_in_params['HADES'] = {}
    
    hades_params = build_in_params['HADES']
    
    default_config = {
        'hidden_dims': 512,
        'layers': 6,
        'dropout': 0.1,
        'learning_rate': 1e-5,
        'weight_decay': 1e-4,
        'batch_size': 64,
        'max_epochs': 10,
        'top_k_percentile': 20, 
        'diffusion_steps': 1000,
        'inference_steps': 50,
        'guidance_scale': 4.0,
        'fitness_embedding_scale': 1.0,
        'use_finetuning': True,
        'finetune_lr_scale': 0.5,
        'checkpoint_dir': 'running',
    }
    
    for k, v in default_config.items():
        if k not in hades_params:
            hades_params[k] = v
            
    device = 'cuda' if getattr(args, 'gpu', False) and torch.cuda.is_available() else 'cpu'
    
    evodiff = EvolutionaryDiffusion(EA_Class.geneLength, device, hades_params)
    
    checkpoint_path = os.path.join(EA_Class.savePath, 'running', f'{args.agent_idx}_HADES_diffusion_model.pt')
    is_finetuning = evodiff.load_checkpoint(checkpoint_path)
    
    if is_finetuning:
        print(f"[DiffusionGA] Using fine-tuning mode (pre-trained model loaded)", flush=True)
        max_epochs_raw = hades_params['max_epochs']
        max_epochs = int(max_epochs_raw if max_epochs_raw is not None else 10)
        hades_params['max_epochs'] = max(3, max_epochs // 2)
    else:
        print(f"[DiffusionGA] Training fresh model from scratch", flush=True)
    
    fitness_scaler = ChunkedLogZScoreScaler(
        evolution_target=getattr(args, 'evolutionTarget', 1),
        emphasis_factor=1.0, 
        device=device
    )
    
    gene_scaler = ChunkedPerDimensionScaler(
        gene_length=EA_Class.geneLength,
        device=device,
        weight_mode='uniform'
    )
    
    loader = ChunkedGeneHistoryLoader(
        savePath=args.path, agent_id=args.agent_idx, 
        geneFormat=EA_Class.geneFormat, device=device
    )
    
    try:
        chunk_size = loader.calculate_chunk_size(
            safety_margin=0.5,
            chunk_allocation=0.20,
            min_chunk=100,
            max_chunk=10000,
            verbose=False
        )
    except Exception as e:
        print(f"[DiffusionGA WARNING] Chunk size calculation failed: {e}, using fallback", flush=True)
        chunk_size = 5000
        
    user_top_k_raw = hades_params['top_k_percentile']
    user_top_k = float(user_top_k_raw if user_top_k_raw is not None else 20)
    loader_percentile = 100 - user_top_k if args.evolutionTarget == 1 else user_top_k
    
    loader_kwargs = {
        'top_k_percentile': loader_percentile,
        'maximize': (args.evolutionTarget == 1),
        'chunk_size': chunk_size,
        'fitness_scaler': fitness_scaler,
        'gene_scaler': gene_scaler,
        'use_fitness_scaler_weights': True,
        'use_gene_scaler_weights': True,
        'allEpochs': True,
        'cache_key': f"hades_ep{args.epoch}_top{loader_percentile}",
        'use_cache': True,
        'fit_fitness_scaler_on_top_k': True,
        'fit_gene_scaler_on_top_k': True,
        'min_datapoints': min(int(getattr(args, 'populationSize', 10000)*0.05), 300),
        'max_datapoints': int(getattr(args, 'populationSize', 10000)),
    }

    final_loss = evodiff.train(
        loader=loader,
        args=args,
        loader_kwargs=loader_kwargs,
        epochs=int(hades_params['max_epochs']),
        batch_size=int(hades_params['batch_size']),
        log_dict={},
        is_finetuning=is_finetuning
    )
    
    evodiff.save_checkpoint(checkpoint_path)
    build_in_params['loss'] = final_loss
    
    inference_steps = int(hades_params.get('inference_steps', 50))
    guidance_scale = float(hades_params.get('guidance_scale', 4.0))
    
    raw_candidates = evodiff.generate(
        n_samples=number_of_candidate_genes,
        guidance_scale=guidance_scale,
        steps=inference_steps,
        verbose=(verbose > 0)
    )
    
    try:
        cand_tensor = torch.tensor(raw_candidates, dtype=torch.float32, device=device)
        
        if getattr(gene_scaler, 'scaling_ready', False):
            unscaled = gene_scaler.inverse_transform(cand_tensor)
            if verbose >= 2:
                print(f"[HADES] Applied inverse_transform", flush=True)
        else:
            unscaled = cand_tensor
            if verbose >= 1:
                print(f"[HADES WARNING] gene_scaler not ready - skipping inverse_transform", flush=True)
        
        gene_min = getattr(EA_Class, 'geneMin', -1.0)
        lower_bound = 0.0 if gene_min == 0 else -1.0
        clipped = torch.clamp(unscaled, lower_bound, 1.0)
        final_candidates = clipped.cpu().numpy().tolist()
            
    except Exception as e:
        if verbose >= 0:
            print(f"[HADES ERROR] inverse_transform failed: {e}, using fallback", flush=True)
        candidates_array = np.array(raw_candidates, dtype=np.float32)
        final_candidates = np.clip(candidates_array, -1.0, 1.0).tolist()
    
    return final_candidates