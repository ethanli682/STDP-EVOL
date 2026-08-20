"""
FIXED Standalone Diffusion Evolution - Addresses Training Issues

Key Fixes:
1. Proper time embedding (sinusoidal positional encoding)
2. Data normalization (standardize genes)
3. Learning rate scheduling
4. Gradient clipping
5. Better noise schedule
6. EMA (Exponential Moving Average)
7. Validation monitoring
8. Diagnostic logging
"""

import gc
import os
import pickle
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple, Callable
from abc import ABC, abstractmethod
from GA_utils_misc_func import extract_chunk_to_tensors


# ============================================================================
# IMPROVED TIME EMBEDDING (Sinusoidal Positional Encoding)
# ============================================================================

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
    
    def forward(self, t):
        device = t.device
        half_dim = self.embed_dim // 2
        
        # Ensure t is 2D: (batch, 1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half_dim, dtype=torch.float32, device=device) / half_dim
        )
        
        # Scale t to larger range for better embedding spread
        args = t * 1000.0 * freqs.unsqueeze(0)  # ADD SCALING
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        return embedding

# ============================================================================
# VARIATIONAL AUTOENCODER FOR LATENT DIFFUSION
# ============================================================================

class VariationalAutoencoder(nn.Module):
    """VAE for compressing high-dimensional genes into latent space."""
    
    def __init__(self, input_dim: int, latent_dim: int = 512):
        super().__init__()
        # Encoder: input_dim -> 2048 -> latent_dim
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

        # Decoder: latent_dim -> 2048 -> input_dim
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

# ============================================================================
# IMPROVED NETWORK WITH PROPER TIME EMBEDDING
# ============================================================================

class ImprovedDiffusionNetwork(nn.Module):
    """
    Improved neural network with:
    - Sinusoidal time embedding
    - Layer normalization
    - Residual connections
    """
    
    def __init__(self, num_params, num_conditions, hidden_dims, dropout=0.1):
        super().__init__()
        
        self.num_params = num_params
        self.num_conditions = num_conditions
        
        # Time embedding (sinusoidal)
        time_embed_dim = 128
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),  # Swish activation (better than ReLU for diffusion)
        )
        
        # Main network
        input_dim = num_params + time_embed_dim + num_conditions
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))  # Layer norm for stability
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_params))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Proper weight initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x, t, *conditions):
        """
        Args:
            x: (batch, num_params) - noisy genes
            t: (batch, 1) - timestep [0, 1]
            conditions: tuple of (batch, 1) tensors
        
        Returns:
            predicted_noise: (batch, num_params)
        """
        batch_size = x.shape[0]
        
        # Time embedding
        t_embed = self.time_embed(t)
        
        # Concatenate conditions
        if conditions:
            cond_tensor = torch.cat(conditions, dim=-1)
        else:
            cond_tensor = torch.zeros(batch_size, self.num_conditions, device=x.device)
        
        # Concatenate all inputs
        net_input = torch.cat([x, t_embed, cond_tensor], dim=-1)
        
        # Predict noise
        noise = self.network(net_input)
        
        return noise


# ============================================================================
# IMPROVED DDIM WITH FIXES
# ============================================================================

class ImprovedDDIM:
    """
    Fixed DDIM implementation with:
    - Data normalization
    - Better noise schedule
    - Learning rate scheduling
    - Gradient clipping
    - EMA
    - Validation monitoring
    - Optional VAE for latent diffusion
    """
    
    def __init__(
        self,
        num_params: int,
        num_conditions: int = 1,
        hidden_dims: List[int] = None,
        num_steps: int = 100,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: str = 'cpu',
        use_vae: bool = False,
        latent_dim: int = 512
    ):
        self.original_num_params = num_params  # Store original dimension
        self.num_conditions = num_conditions
        self.num_steps = num_steps
        self.device = torch.device(device)
        self.use_vae = use_vae
        self.latent_dim = latent_dim
        
        # Initialize VAE if requested
        self.vae = None
        self.vae_optimizer = None
        if use_vae:
            print(f"[HADES] Using VAE: Compressing {num_params} -> {latent_dim}", flush=True)
            self.vae = VariationalAutoencoder(num_params, latent_dim).to(self.device)
            self.vae_optimizer = torch.optim.AdamW(self.vae.parameters(), lr=1e-4)
            # When using VAE, diffusion works in latent space
            diffusion_input_dim = latent_dim
            # Adjust default hidden dims for smaller latent space
            if hidden_dims is None:
                hidden_dims = [512, 512]
        else:
            diffusion_input_dim = num_params
            # Default architecture based on gene dimension
            if hidden_dims is None:
                if num_params <= 50:
                    hidden_dims = [256, 128]
                elif num_params <= 200:
                    hidden_dims = [512, 256]
                else:
                    hidden_dims = [1024, 512, 256]
        
        self.num_params = diffusion_input_dim  # Set to either latent_dim or num_params
        
        # Create improved network
        self.network = ImprovedDiffusionNetwork(
            num_params=diffusion_input_dim,
            num_conditions=num_conditions,
            hidden_dims=hidden_dims,
            dropout=0.1
        ).to(self.device)
        
        # Improved noise schedule (cosine schedule - better than linear)
        self.betas = self._cosine_beta_schedule(num_steps, beta_start, beta_end)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Data normalization stats (can be set externally from ChunkedPerDimensionScaler)
        self.data_mean = None
        self.data_std = None
        self.use_external_normalization = False  # Flag to indicate pre-normalized data
        
        # EMA model for stable sampling
        self.ema_network = None
        self.ema_decay = 0.9999
    
    def _cosine_beta_schedule(self, timesteps, beta_start, beta_end):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, device=self.device)
        alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        alphas_cumprod = alphas_cumprod.clamp(min=1e-6, max=1.0 - 1e-6)  # ADD CLAMPING
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, beta_start, beta_end)
        
    def _normalize_data(self, x):
        # If using external normalization (ChunkedPerDimensionScaler), data is already normalized
        if self.use_external_normalization:
            return x
        if self.data_mean is None or self.data_std is None:
            raise RuntimeError("Data normalization stats not computed.")
        std = self.data_std.clamp(min=1e-6)  # Prevent zero variance
        return (x - self.data_mean) / std

    
    def _denormalize_data(self, x):
        """Denormalize data back to original scale."""
        # If using external normalization, denormalization is handled externally
        if self.use_external_normalization:
            return x
        if self.data_mean is None or self.data_std is None:
            return x
        return x * (self.data_std + 1e-8) + self.data_mean
    
    def q_sample(self, x0, t, noise=None):
        """Forward diffusion: add noise to x0 at timestep t."""
        if noise is None:
            noise = torch.randn_like(x0)
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        return sqrt_alpha * x0 + sqrt_one_minus_alpha * noise, noise
    
    def p_sample(self, xt, t, *conditions, use_ema=True):
        """Reverse diffusion: denoise xt at timestep t."""
        # Use EMA model if available
        network = self.ema_network if (use_ema and self.ema_network is not None) else self.network
        
        # Predict noise
        t_input = torch.full((xt.shape[0], 1), t / self.num_steps, device=self.device)
        predicted_noise = network(xt, t_input, *conditions)
        
        # DDIM update
        alpha_t = self.alphas_cumprod[t]
        alpha_t_prev = self.alphas_cumprod_prev[t] if t > 0 else torch.tensor(1.0, device=self.device)
        
        # Predict x0
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
        x0_pred = (xt - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t
        
        # Clip for stability
        x0_pred = torch.clamp(x0_pred, -3.0, 3.0)
        
        if t == 0:
            return x0_pred
        
        # Direction pointing to xt
        sqrt_alpha_t_prev = torch.sqrt(alpha_t_prev)
        sqrt_one_minus_alpha_t_prev = torch.sqrt(1.0 - alpha_t_prev)
        
        xt_prev = sqrt_alpha_t_prev * x0_pred + sqrt_one_minus_alpha_t_prev * predicted_noise
        
        return xt_prev
    
    def sample(self, num_samples: int, conditions: Tuple[torch.Tensor, ...] = None, use_ema=True, verbose: bool = False):
        """Generate samples from the diffusion model."""
        # Start from random noise in appropriate space
        xt = torch.randn(num_samples, self.num_params, device=self.device)
        
        # Default conditions
        if conditions is None:
            conditions = tuple([torch.zeros(num_samples, 1, device=self.device) 
                               for _ in range(self.num_conditions)])
        
        # Reverse diffusion
        self.network.eval()
        if self.ema_network is not None:
            self.ema_network.eval()
        if self.vae is not None:
            self.vae.eval()
        
        with torch.no_grad():
            # Diffusion process (in latent space if using VAE)
            for t in reversed(range(self.num_steps)):
                xt = self.p_sample(xt, t, *conditions, use_ema=use_ema)
            
            # If VAE, decode from latent space to gene space
            if self.vae is not None:
                if verbose:
                    print(f"[HADES] Decoding latents to genes...", flush=True)
                xt = self.vae.decode(xt)
        
        # Denormalize
        xt = self._denormalize_data(xt)
        
        return xt
    
    def _update_ema(self):
        """Update EMA model."""
        if self.ema_network is None:
            # Initialize EMA by copying the entire network
            import copy
            self.ema_network = copy.deepcopy(self.network)
            self.ema_network.eval()
        
        # Update EMA weights
        with torch.no_grad():
            for ema_param, param in zip(self.ema_network.parameters(), self.network.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
    
    def train_step(self, x0_batch, conditions_batch, optimizer, clip_grad=1.0):
        batch_size = x0_batch.shape[0]
        if batch_size == 0:
            return 0.0  # Skip empty batches

        # Safety: if using VAE but data is still in gene space, encode to latent
        if self.vae is not None and x0_batch.shape[1] != self.num_params:
            with torch.no_grad():
                mu, logvar = self.vae.encode(x0_batch)
                x0_batch = self.vae.reparameterize(mu, logvar)
        
        t = torch.randint(0, self.num_steps, (batch_size,), device=self.device)
        noise = torch.randn_like(x0_batch)
        xt, _ = self.q_sample(x0_batch, t, noise)
        
        t_input = (t.float() / self.num_steps).view(-1, 1)
        predicted_noise = self.network(xt, t_input, *conditions_batch)
        
        loss = F.mse_loss(predicted_noise, noise)
        
        # Debug: catch zero/nan loss
        if loss.item() == 0.0 or torch.isnan(loss):
            print(f"[WARNING] Abnormal loss: {loss.item()}, noise_std: {noise.std().item()}, pred_std: {predicted_noise.std().item()}")
            return loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), clip_grad)
        
        optimizer.step()
        self._update_ema()
        
        return loss.item()

    
    def fit(
        self,
        x_data: torch.Tensor,
        conditions_data: Tuple[torch.Tensor, ...],
        weights: Optional[torch.Tensor] = None,
        batch_size: int = 32,
        epochs: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        verbose: bool = False,
        validation_split: float = 0.1,
        vae_epochs_ratio: float = 0.3
    ) -> List[float]:
        """
        Train the diffusion model with improvements.
        If using VAE, pre-trains VAE first, then trains diffusion in latent space.
        """
        # Calculate VAE pre-training epochs if using VAE
        vae_epochs = 0
        diffusion_start_epoch = 0
        if self.vae is not None:
            vae_epochs = max(5, int(epochs * vae_epochs_ratio))
            diffusion_start_epoch = vae_epochs
            if verbose:
                print(f"[HADES] Strategy: {vae_epochs} VAE pre-training -> {epochs - vae_epochs} latent diffusion", flush=True)
        
        # If data is pre-normalized (from ChunkedPerDimensionScaler), skip internal normalization
        if self.use_external_normalization:
            if verbose:
                print(f"[Diffusion] Using pre-normalized data (external scaler)")
            x_data_norm = x_data  # Already normalized
        else:
            # Compute normalization stats internally
            self.data_mean = x_data.mean(dim=0, keepdim=True)
            self.data_std = x_data.std(dim=0, keepdim=True)
            
            if verbose:
                print(f"[Diffusion] Data stats: mean={self.data_mean.mean().item():.4f}, "
                      f"std={self.data_std.mean().item():.4f}")
            
            # Normalize data
            x_data_norm = self._normalize_data(x_data)
        
        # Train/val split
        N = x_data.shape[0]
        if validation_split > 0:
            n_val = int(N * validation_split)
            indices = torch.randperm(N, device=self.device)
            train_indices = indices[n_val:]
            val_indices = indices[:n_val]
            
            x_train = x_data_norm[train_indices]
            x_val = x_data_norm[val_indices]
            cond_train = tuple([c[train_indices] for c in conditions_data])
            cond_val = tuple([c[val_indices] for c in conditions_data])
            weights_train = weights[train_indices] if weights is not None else None
        else:
            x_train = x_data_norm
            x_val = None
            cond_train = conditions_data
            weights_train = weights
        
        # Optimizer with learning rate scheduling
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler (cosine annealing with less aggressive decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.001  # FIX #4: Less aggressive
        )        
        self.network.train()
        if self.vae is not None:
            self.vae.train()
        
        # Helper for VAE loss
        def vae_loss_fn(recon_x, x, mu, logvar):
            MSE = F.mse_loss(recon_x, x, reduction='mean')
            # KL Divergence: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            return MSE + 0.0001 * KLD  # Small weight on KLD to prevent posterior collapse
        
        N_train = x_train.shape[0]
        losses = []
        best_val_loss = float('inf')

        if verbose >= 1:
            print(f"[DEBUG] x_train: shape={x_train.shape}, mean={x_train.mean():.4f}, std={x_train.std():.4f}")
            print(f"[DEBUG] x_train range: [{x_train.min():.4f}, {x_train.max():.4f}]")
            for i, c in enumerate(cond_train):
                print(f"[DEBUG] cond[{i}]: mean={c.mean():.4f}, std={c.std():.4f}")

        
        for epoch in range(epochs):
            training_vae_phase = (self.vae is not None and epoch < diffusion_start_epoch)
            epoch_loss = 0.0
            num_batches = 0
            
            # Shuffle
            indices = torch.randperm(N_train, device=self.device)
            
            for i in range(0, N_train, batch_size):
                batch_indices = indices[i:i+batch_size]
                
                x_batch = x_train[batch_indices]
                cond_batch = tuple([c[batch_indices] for c in cond_train])
                
                # Weight handling
                if weights_train is not None:
                    batch_weights = weights_train[batch_indices]
                    # Weighted sampling
                    probs = batch_weights / batch_weights.sum()
                    resample_indices = torch.multinomial(
                        probs, 
                        len(batch_indices), 
                        replacement=True
                    )
                    x_batch = x_batch[resample_indices]
                    cond_batch = tuple([c[resample_indices] for c in cond_batch])
                
                # MODE A: VAE TRAINING
                if training_vae_phase:
                    recon_x, mu, logvar = self.vae(x_batch)
                    loss = vae_loss_fn(recon_x, x_batch, mu, logvar)
                    
                    self.vae_optimizer.zero_grad()
                    loss.backward()
                    self.vae_optimizer.step()
                    epoch_loss += loss.item()
                    num_batches += 1
                # MODE B: DIFFUSION TRAINING
                else:
                    # If VAE, encode to latent space first
                    if self.vae is not None:
                        with torch.no_grad():
                            mu, logvar = self.vae.encode(x_batch)
                            x_batch = self.vae.reparameterize(mu, logvar)
                    
                    loss = self.train_step(x_batch, cond_batch, optimizer, clip_grad=1.0)
                    epoch_loss += loss
                    num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)
            
            # Validation
            val_loss = None
            if x_val is not None:
                self.network.eval()
                with torch.no_grad():
                    val_losses = []
                    for i in range(0, len(x_val), batch_size):
                        x_batch = x_val[i:i+batch_size]
                        cond_batch = tuple([c[i:i+batch_size] for c in cond_val])

                        # If using VAE, evaluate in latent space
                        if self.vae is not None:
                            mu, logvar = self.vae.encode(x_batch)
                            x_batch = self.vae.reparameterize(mu, logvar)
                        
                        t = torch.randint(0, self.num_steps, (x_batch.shape[0],), device=self.device)
                        noise = torch.randn_like(x_batch)
                        xt, _ = self.q_sample(x_batch, t, noise)
                        
                        t_input = (t.float() / self.num_steps).view(-1, 1)
                        predicted_noise = self.network(xt, t_input, *cond_batch)
                        
                        loss = F.mse_loss(predicted_noise, noise)
                        val_losses.append(loss.item())
                    
                    val_loss = np.mean(val_losses)
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                
                self.network.train()
            
            # Step scheduler
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                phase_name = "VAE-Pretrain" if training_vae_phase else "Diffusion"
                msg = f"Epoch {epoch}/{epochs} [{phase_name}]: train_loss={avg_loss:.6f}, lr={current_lr:.6f}"
                if val_loss is not None:
                    msg += f", val_loss={val_loss:.6f}"
                print(msg)
        
        return losses
    
    def save(self, path: str):
        """Save model state."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            'network_state': self.network.state_dict(),
            'ema_network_state': self.ema_network.state_dict() if self.ema_network else None,
            'num_params': self.num_params,
            'original_num_params': self.original_num_params,
            'num_conditions': self.num_conditions,
            'num_steps': self.num_steps,
            'data_mean': self.data_mean,
            'data_std': self.data_std,
            'use_vae': self.use_vae,
            'latent_dim': self.latent_dim,
        }
        if self.vae is not None:
            state['vae_state'] = self.vae.state_dict()
        torch.save(state, path)
    
    def load(self, path: str):
        """Load model state."""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            
            # Check VAE compatibility
            saved_use_vae = checkpoint.get('use_vae', False)
            if saved_use_vae != self.use_vae:
                print(f"[HADES WARNING] Checkpoint VAE mode ({saved_use_vae}) != current ({self.use_vae}). Ignoring.", flush=True)
                return False
            
            self.network.load_state_dict(checkpoint['network_state'])
            if checkpoint.get('ema_network_state') is not None:
                if self.ema_network is None:
                    self._update_ema()
                self.ema_network.load_state_dict(checkpoint['ema_network_state'])
            
            # Load VAE state if present
            if self.vae is not None and 'vae_state' in checkpoint:
                self.vae.load_state_dict(checkpoint['vae_state'])
            
            self.data_mean = checkpoint.get('data_mean')
            self.data_std = checkpoint.get('data_std')
            self.network.eval()
            if self.ema_network is not None:
                self.ema_network.eval()
            if self.vae is not None:
                self.vae.eval()
            return True
        return False


# ============================================================================
# CONDITIONS (Same as before)
# ============================================================================

class Condition(ABC):
    @abstractmethod
    def evaluate(self, genes: torch.Tensor, fitness: torch.Tensor) -> torch.Tensor:
        pass
    
    @abstractmethod
    def sample(self, num_samples: int, **kwargs) -> torch.Tensor:
        pass


class FitnessCondition(Condition):
    def __init__(self, target_percentile: float = 95.0):
        self.target_percentile = target_percentile
        self._fitness_stats = None
    
    def evaluate(self, genes: torch.Tensor, fitness: torch.Tensor) -> torch.Tensor:
        self._fitness_stats = {
            'mean': fitness.mean().item(),
            'std': fitness.std().item(),
            'max': fitness.max().item(),
            'min': fitness.min().item(),
        }
        return fitness.view(-1, 1)
    
    def sample(self, num_samples: int, **kwargs) -> torch.Tensor:
        device = kwargs.get('device', 'cpu')
        
        if self._fitness_stats is None:
            return torch.ones(num_samples, 1, device=device) * 3.0
        
        mean = self._fitness_stats['mean']
        std = self._fitness_stats['std']
        max_val = self._fitness_stats['max']
        
        target = mean + std * 2.0
        target = min(target, max_val)
        
        targets = torch.ones(num_samples, 1, device=device) * target
        targets += torch.randn(num_samples, 1, device=device) * std * 0.1
        
        return targets


class NoveltyCondition(Condition):
    def __init__(self, k_nearest: int = 10):
        self.k_nearest = k_nearest
        self._reference_genes = None
    
    def evaluate(self, genes: torch.Tensor, fitness: torch.Tensor) -> torch.Tensor:
        self._reference_genes = genes.clone()
        dists = torch.cdist(genes, genes)
        k = min(self.k_nearest, genes.shape[0] - 1)
        knn_dists, _ = torch.topk(dists, k + 1, largest=False, dim=1)
        novelty = knn_dists[:, 1:].mean(dim=1)
        return novelty.view(-1, 1)
    
    def sample(self, num_samples: int, **kwargs) -> torch.Tensor:
        device = kwargs.get('device', 'cpu')
        if self._reference_genes is not None:
            dists = torch.cdist(self._reference_genes, self._reference_genes)
            max_novelty = dists.max().item()
            target = max_novelty * 0.8
        else:
            target = 1.0
        targets = torch.ones(num_samples, 1, device=device) * target
        return targets


class CustomCondition(Condition):
    def __init__(self, evaluate_fn: Callable, sample_fn: Callable):
        self.evaluate_fn = evaluate_fn
        self.sample_fn = sample_fn
    
    def evaluate(self, genes: torch.Tensor, fitness: torch.Tensor) -> torch.Tensor:
        return self.evaluate_fn(genes, fitness)
    
    def sample(self, num_samples: int, **kwargs) -> torch.Tensor:
        return self.sample_fn(num_samples, **kwargs)


# ============================================================================
# HELPER FUNCTIONS (Same as before)
# ============================================================================

def _extract_chunk_to_tensors(
    chunk: List[Dict],
    gene_length: int,
    device: torch.device,
    extract_weights: bool = False
) -> Tuple:
    """Extract genes, fitness, and weights from chunk into tensors.
    
    DEPRECATED: Use extract_chunk_to_tensors from GA_utils_misc_func instead.
    """
    return extract_chunk_to_tensors(chunk, gene_length, device, extract_weights)


# ============================================================================
# FIX: SCALED HYPERPARAMETERS FOR HIGH-DIMENSIONAL PROBLEMS
# ============================================================================

def get_scaled_hyperparameters(gene_length: int, evolution_target: int) -> dict:
    """
    Calculate scaled hyperparameters based on gene_length.
    Fixes plateau issues for high-dimensional problems (15K+ genes).
    """
    
    # FIX #1: SCALE HIDDEN DIMS PROPORTIONALLY
    if gene_length <= 100:
        hidden_dims = [256, 128]
    elif gene_length <= 500:
        hidden_dims = [512, 256]
    elif gene_length <= 1000:
        hidden_dims = [1024, 512, 256]
    elif gene_length <= 5000:
        hidden_dims = [
            max(2048, gene_length // 3),
            max(1024, gene_length // 6),
            max(512, gene_length // 12)
        ]
    elif gene_length <= 10000:
        hidden_dims = [
            max(4096, gene_length // 2.5),
            max(2048, gene_length // 5),
            max(1024, gene_length // 10),
            max(512, gene_length // 20)
        ]
    else:
        # 15K+ genes
        hidden_dims = [
            max(8192, gene_length // 2),
            max(4096, gene_length // 4),
            max(2048, gene_length // 8),
            max(1024, gene_length // 16),
            max(512, gene_length // 32)
        ]
    
    # FIX #2: SCALE DIFFUSION STEPS WITH DIMENSIONALITY
    if gene_length <= 100:
        diffusion_steps = 50
    elif gene_length <= 1000:
        diffusion_steps = 100
    elif gene_length <= 5000:
        diffusion_steps = min(500, 100 + gene_length // 15)
    elif gene_length <= 10000:
        diffusion_steps = min(800, 100 + gene_length // 12)
    else:
        diffusion_steps = min(1500, 100 + gene_length // 10)
    
    # FIX #3: SCALE BATCH SIZE APPROPRIATELY
    if gene_length <= 100:
        batch_size = 16
    elif gene_length <= 500:
        batch_size = 32
    elif gene_length <= 1000:
        batch_size = 64
    elif gene_length <= 5000:
        batch_size = max(64, min(256, gene_length // 40))
    elif gene_length <= 10000:
        batch_size = max(128, min(512, gene_length // 50))
    else:
        batch_size = max(256, min(768, gene_length // 40))
    
    # FIX #4: LEARNING RATE CONFIG
    learning_rate = 1e-3
    
    # FIX #5: SCALE MAX EPOCHS FOR HIGH DIMENSIONS
    if gene_length <= 1000:
        max_epochs = 777
    elif gene_length <= 5000:
        max_epochs = 1000
    elif gene_length <= 10000:
        max_epochs = 1500
    else:
        max_epochs = 2000
    
    # FIX #6: SCALE VALIDATION SPLIT
    if gene_length <= 1000:
        validation_split = 0.1
    elif gene_length <= 5000:
        validation_split = 0.15
    else:
        validation_split = 0.2
    
    return {
        'hidden_dims': hidden_dims,
        'diffusion_steps': diffusion_steps,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'max_epochs': max_epochs,
        'validation_split': validation_split,
    }

# ============================================================================
# MAIN FUNCTION (Updated to use ImprovedDDIM)
# ============================================================================

def createCandidateGeneUsingHADESlution(
    args=None,
    EA_Class=None,
    genePopulation: Optional[List[Dict[str, Any]]] = None,
    number_of_candidate_genes: Optional[int] = None,
    build_in_params: Optional[Dict] = None,
    verbose: Optional[int] = None
) -> List[List[float]]:
    """
    FIXED version with proper training and continuous fine-tuning.
    
    Key fixes:
    - Sinusoidal time embedding
    - Data normalization
    - Learning rate scheduling
    - Gradient clipping
    - EMA (Exponential Moving Average)
    - Validation monitoring
    - Continuous fine-tuning across generations (NEW!)
    
    Configuration Parameters (build_in_params['HADES']):
        # Training
        'max_epochs': int (default=777) - Initial training epochs
        'learning_rate': float (default=1e-3) - Initial learning rate
        'batch_size': int (default=32) - Batch size
        
        # Fine-Tuning
        'use_finetuning': bool (default=True) - Enable continuous training
        'finetune_lr_scale': float (default=0.5) - Learning rate scale for fine-tuning
        'finetune_epoch_scale': float (default=0.5) - Epoch reduction factor for fine-tuning
        'finetune_every_n_gens': int (default=1) - Fine-tune every N generations
        'finetune_decay': float (default=0.95) - Epoch decay factor
        'finetune_min_epochs': int (default=10) - Minimum fine-tune epochs
        
        # Model
        'diffusion_steps': int (default=100) - Diffusion timesteps
        'hidden_dims': List[int] (default=auto) - Network architecture
        
        # Conditions
        'conditions': List[str] (default=['fitness']) - Condition types
        
        # Exploration
        'exploration_mode': str (default='adaptive') - Exploration strategy
        
        # Caching
        'use_model_cache': bool (default=True) - Save/load model
        'force_retrain': bool (default=False) - Ignore cache
        
        # Validation
        'validation_split': float (default=0.1) - Validation data fraction
    
    Returns:
        List[List[float]]: Generated candidate genes
    
    Example:
        build_in_params = {
            'HADES': {
                # Enable continuous learning
                'use_finetuning': True,
                'finetune_epoch_scale': 0.5,
                'finetune_lr_scale': 0.5,
                'finetune_every_n_gens': 1,
            }
        }
        
        candidates = createCandidateGeneUsingHADESlution_chunk(
            args=args,
            EA_Class=EA_Class,
            number_of_candidate_genes=100,
            build_in_params=build_in_params,
            verbose=1
        )
    """
    
    from GA_utils_scalers import ChunkedRankScaler, ChunkedLogZScoreScaler
    from GA_utils_history_loader import ChunkedGeneHistoryLoader
    
    # [Rest of the function is identical to before, but uses ImprovedDDIM]
    # [Copy from previous implementation...]
    
    if args is None or EA_Class is None or number_of_candidate_genes is None:
        raise ValueError("Required parameters: args, EA_Class, number_of_candidate_genes")
    
    if build_in_params is None:
        build_in_params = {}
    
    verbose = verbose if verbose is not None else 0
    
    use_gpu = getattr(args, 'gpu', None)
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    
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
        print(f"[HADES] Device: {device}, Candidates: {number_of_candidate_genes}")
    
    # Configuration
    if 'HADES' not in build_in_params:
        build_in_params['HADES'] = {}
    
    config = build_in_params['HADES']
    
# GET SCALED PARAMETERS (ALL 6 FIXES)
    scaled_params = get_scaled_hyperparameters(gene_length, evolution_target)
    
    defaults = {
        'use_log_scaler': True,
        'top_k_percentile': 25,
        'hidden_dims': scaled_params['hidden_dims'],           # FIX #1
        'diffusion_steps': scaled_params['diffusion_steps'],   # FIX #2
        'learning_rate': scaled_params['learning_rate'],       # FIX #4
        'weight_decay': 1e-5,
        'batch_size': scaled_params['batch_size'],             # FIX #3
        'max_epochs': scaled_params['max_epochs'],             # FIX #5
        'conditions': ['fitness'],
        'custom_conditions': {},
        'exploration_mode': 'adaptive',
        'exploration_noise': 0.1,
        'mutation_rate': 0.1,
        'use_model_cache': True,
        'force_retrain': False,
        'validation_split': scaled_params['validation_split'], # FIX #6
        'use_finetuning': True,
        'finetune_lr_scale': 0.5,
        'finetune_epoch_scale': 0.5,
        'finetune_min_epochs': 10,
        'finetune_every_n_gens': 1,
        'finetune_decay': 0.95,
        'checkpoint_dir': 'running',
        # VAE parameters
        'model_type': 'MLP',  # 'MLP' or 'VAE'
        'latent_dim': 512,    # Latent dimension for VAE
        'vae_epochs_ratio': 0.3,  # Proportion of epochs for VAE pre-training
    }
    
    for key, value in defaults.items():
        if key not in config:
            config[key] = value
    
    # Validate and convert hidden_dims
    hidden_dims_raw = config.get('hidden_dims', None)
    if hidden_dims_raw is None or hidden_dims_raw in ['None', 'none', '']:
        config['hidden_dims'] = scaled_params['hidden_dims']
    elif isinstance(hidden_dims_raw, str):
        try:
            config['hidden_dims'] = list(eval(hidden_dims_raw))
        except:
            config['hidden_dims'] = scaled_params['hidden_dims']
    else:
        config['hidden_dims'] = [int(x) for x in (hidden_dims_raw if isinstance(hidden_dims_raw, (list, tuple)) else scaled_params['hidden_dims'])]
    
    if verbose >= 1:
        print(f"[HADES] Scaled Config for {gene_length} genes:")
        print(f"  - Hidden dims: {config['hidden_dims']}")
        print(f"  - Diffusion steps: {config['diffusion_steps']}")
        print(f"  - Batch size: {config['batch_size']}")
        print(f"  - Max epochs: {config['max_epochs']}")
        print(f"  - Validation split: {config['validation_split']}")
        print(f"  - Model type: {config['model_type']}")
        if config['model_type'] == 'VAE':
            print(f"  - Latent dim: {config['latent_dim']}")


    # Create scalers - separate for fitness and genes
    progress = agent_counter / num_generations if num_generations > 0 else 0.0
    emphasis_multiplier = 0.7 if progress < 0.3 else (1.5 if progress > 0.7 else 0.7 + progress)
    emphasis_factor = 1.0 + emphasis_multiplier * torch.rand(1).item()
    
    if config['use_log_scaler']:
        fitness_scaler = ChunkedLogZScoreScaler(
            evolution_target=evolution_target,
            emphasis_factor=emphasis_factor,
            min_weight=0.01,
            device=device
        )
    else:
        fitness_scaler = ChunkedRankScaler(
            evolution_target=evolution_target,
            emphasis_factor=emphasis_factor,
            min_weight=0.01,
            device=device
        )

    # Always use per-dimension scaler for gene features (mean 0, std 1)
    try:
        from GA_utils_scalers import ChunkedPerDimensionScaler
        gene_scaler = ChunkedPerDimensionScaler(
            gene_length=gene_length,
            device=device,
            weight_mode='uniform'
        )
    except Exception:
        gene_scaler = None
    
    # Create conditions
    conditions = []
    for cond_name in config['conditions']:
        if cond_name == 'fitness':
            conditions.append(FitnessCondition())
        elif cond_name == 'novelty':
            conditions.append(NoveltyCondition())
        elif cond_name == 'custom' and cond_name in config['custom_conditions']:
            custom_def = config['custom_conditions'][cond_name]
            conditions.append(CustomCondition(
                evaluate_fn=custom_def['evaluate'],
                sample_fn=custom_def['sample']
            ))
    
    num_conditions = len(conditions)
    
    # Model caching
    model_cache_path = None
    if config['use_model_cache']:
        cache_dir = os.path.join(args.path, 'running')
        os.makedirs(cache_dir, exist_ok=True)
        model_cache_path = os.path.join(
            cache_dir,
            f'{agent_idx if agent_idx is not None else agent_counter}_diffusion_model_cache_v2.pt'
        )
    
    # Create IMPROVED diffusion model
    diffusion_steps_raw = config.get('diffusion_steps', 100)
    diffusion_steps_raw = None if diffusion_steps_raw in ['None', 'none'] else diffusion_steps_raw
    diffusion_steps = int(diffusion_steps_raw if diffusion_steps_raw is not None else 100)
    
    validation_split_raw = config.get('validation_split', 0.1)
    validation_split_raw = None if validation_split_raw in ['None', 'none'] else validation_split_raw
    validation_split = float(validation_split_raw if validation_split_raw is not None else 0.1)
    
    # Parse VAE configuration
    use_vae = config.get('model_type', 'MLP') == 'VAE'
    latent_dim_raw = config.get('latent_dim', 512)
    latent_dim = int(latent_dim_raw if latent_dim_raw is not None else 512)
    
    diffusion_model = ImprovedDDIM(
        num_params=gene_length,
        num_conditions=num_conditions,
        hidden_dims=config['hidden_dims'],
        num_steps=diffusion_steps,
        device=str(device),
        use_vae=use_vae,
        latent_dim=latent_dim
    )
    
    # Try loading cached model
    model_loaded = False
    if model_cache_path and not config['force_retrain']:
        if diffusion_model.load(model_cache_path):
            if verbose >= 1:
                print(f"[HADES] ✓ Loaded cached model")
            model_loaded = True
    
    # Determine if we should train and in what mode
    should_train = False
    finetune_mode = False
    
    max_epochs_raw = config.get('max_epochs', 777)
    max_epochs_raw = None if max_epochs_raw in ['None', 'none'] else max_epochs_raw
    max_epochs = int(max_epochs_raw if max_epochs_raw is not None else 777)
    
    learning_rate_raw = config.get('learning_rate', 1e-3)
    learning_rate_raw = None if learning_rate_raw in ['None', 'none'] else learning_rate_raw
    learning_rate = float(learning_rate_raw if learning_rate_raw is not None else 1e-3)
    
    training_epochs = max_epochs
    training_lr = learning_rate

    # Global training params used in both RAM/DISK branches
    batch_size_raw = config.get('batch_size', 32)
    batch_size_raw = None if batch_size_raw in ['None', 'none'] else batch_size_raw
    batch_size = int(batch_size_raw if batch_size_raw is not None else 32)

    weight_decay_raw = config.get('weight_decay', 1e-5)
    weight_decay_raw = None if weight_decay_raw in ['None', 'none'] else weight_decay_raw
    weight_decay = float(weight_decay_raw if weight_decay_raw is not None else 1e-5)
    
    if not model_loaded:
        # First time: full training from scratch
        should_train = True
        finetune_mode = False
        if verbose >= 1:
            print(f"[HADES] Training from scratch")
    elif config['use_finetuning']:
        # Check if it's time to fine-tune
        finetune_every_raw = config.get('finetune_every_n_gens', 1)
        finetune_every_raw = None if finetune_every_raw in ['None', 'none'] else finetune_every_raw
        finetune_every = int(finetune_every_raw if finetune_every_raw is not None else 1)
        
        if finetune_every > 0 and (agent_counter % finetune_every == 0 or agent_counter == 0):
            should_train = True
            finetune_mode = True
            
            # Calculate fine-tune epochs with decay
            finetune_epoch_scale_raw = config.get('finetune_epoch_scale', 0.5)
            finetune_epoch_scale_raw = None if finetune_epoch_scale_raw in ['None', 'none'] else finetune_epoch_scale_raw
            finetune_epoch_scale = float(finetune_epoch_scale_raw if finetune_epoch_scale_raw is not None else 0.5)
            
            finetune_decay_raw = config.get('finetune_decay', 0.95)
            finetune_decay_raw = None if finetune_decay_raw in ['None', 'none'] else finetune_decay_raw
            finetune_decay = float(finetune_decay_raw if finetune_decay_raw is not None else 0.95)
            
            finetune_min_epochs_raw = config.get('finetune_min_epochs', 10)
            finetune_min_epochs_raw = None if finetune_min_epochs_raw in ['None', 'none'] else finetune_min_epochs_raw
            finetune_min_epochs = int(finetune_min_epochs_raw if finetune_min_epochs_raw is not None else 10)
            
            finetune_lr_scale_raw = config.get('finetune_lr_scale', 0.5)
            finetune_lr_scale_raw = None if finetune_lr_scale_raw in ['None', 'none'] else finetune_lr_scale_raw
            finetune_lr_scale = float(finetune_lr_scale_raw if finetune_lr_scale_raw is not None else 0.5)
            
            base_epochs = finetune_epoch_scale * max_epochs
            decay = finetune_decay
            min_epochs = finetune_min_epochs
            
            # Apply decay based on how many times we've fine-tuned
            num_finetunes = max(1, agent_counter // max(1, finetune_every))
            training_epochs = max(min_epochs, int(base_epochs * (decay ** (num_finetunes - 1))))
            training_lr = learning_rate * finetune_lr_scale
            
            if verbose >= 1:
                print(f"[HADES] Fine-tuning (epochs={training_epochs}, lr={training_lr:.6f})")
    
    # Training
    training_loss = None
    
    if should_train:
        if genePopulation is not None:
            # RAM HIT
            if verbose >= 1:
                print(f"[HADES] RAM HIT: {len(genePopulation)} genes")
            
            valid_genes = [g for g in genePopulation 
                          if isinstance(g, dict) and 'gene' in g and 'fitnessScore' in g
                          and g['gene'] is not None 
                          and isinstance(g['fitnessScore'], (int, float))
                          and not np.isnan(g['fitnessScore'])]
            
            if len(valid_genes) == 0:
                raise RuntimeError("No valid genes")
            
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
            
            # Collect statistics for BOTH fitness_scaler AND gene_scaler
            fitness_scaler.collect_phase_1_statistics(training_fitness)
            fitness_scaler.collect_phase_2_statistics(training_fitness)
            fitness_scaler.finalize()
            
            # Normalize genes using ChunkedPerDimensionScaler for consistency
            if gene_scaler is not None:
                gene_scaler.collect_phase_1_statistics(training_genes)
                gene_scaler._finalize_pass1()
                gene_scaler.collect_phase_2_statistics(training_genes)
                gene_scaler.finalize()
                
                training_genes_normalized = gene_scaler.transform(training_genes)
                diffusion_model.use_external_normalization = True
                if verbose >= 1:
                    print(f"[HADES] Applied ChunkedPerDimensionScaler normalization")
            else:
                training_genes_normalized = training_genes
                diffusion_model.use_external_normalization = False
            
            training_fitness_scaled = fitness_scaler.scale_chunk(training_fitness)
            training_weights = fitness_scaler.get_global_weights(training_fitness)
            
            conditions_data = []
            for condition in conditions:
                cond_values = condition.evaluate(training_genes_normalized, training_fitness_scaled)
                conditions_data.append(cond_values)
            conditions_data = tuple(conditions_data)
            
            if verbose >= 1:
                print(f"[HADES] Training (epochs={training_epochs})")
            
            vae_epochs_ratio = float(config.get('vae_epochs_ratio', 0.3))
            
            training_loss = diffusion_model.fit(
                x_data=training_genes_normalized,  # Use normalized genes
                conditions_data=conditions_data,
                weights=training_weights,
                batch_size=batch_size,
                epochs=training_epochs,  # Use calculated epochs
                lr=training_lr,          # Use calculated lr
                weight_decay=weight_decay,
                verbose=(verbose >= 1),
                validation_split=validation_split,
                vae_epochs_ratio=vae_epochs_ratio
            )
            
            del training_genes, training_genes_normalized, training_fitness, training_fitness_scaled, training_weights, valid_genes
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        else:
            # DISK HIT - load with gene_scaler for normalization
            if verbose >= 1:
                print(f"[HADES] DISK HIT: Streaming from history")
            
            loader = ChunkedGeneHistoryLoader(
                savePath=args.path,
                agent_id=agent_idx if agent_idx is not None else agent_counter,
                geneFormat=gene_format,
                shuffle=True,
                shuffle_level='both',
            )
            
            top_k_percentile_raw = config.get('top_k_percentile', 40)
            top_k_percentile_raw = None if top_k_percentile_raw in ['None', 'none'] else top_k_percentile_raw
            top_k_percentile = float(top_k_percentile_raw if top_k_percentile_raw is not None else 40)
            loader_percentile = 100 - top_k_percentile
            maximize = (evolution_target == 1)
            cache_key = f"agent_{agent_idx}_HADES_top{top_k_percentile}"
            
            try:
                chunk_size = loader.calculate_chunk_size(safety_margin=0.7, chunk_allocation=0.20, verbose=False)
            except:
                chunk_size = 10000
            
            all_genes = []
            all_fitness = []
            all_weights = []
            total_genes_loaded = 0
            
            for chunk, is_done in loader.loadTopKPercentileGeneHistoryAsListOfDicts(
                args=args,
                top_k_percentile=loader_percentile,
                maximize=maximize,
                chunk_size=chunk_size,
                fitness_scaler=fitness_scaler,
                gene_scaler=gene_scaler,
                use_fitness_scaler_weights=True,
                use_gene_scaler_weights=True,
                n_feature_clusters='adaptive',
                allEpochs=True,
                cache_key=cache_key,
                use_cache=True,
                min_datapoints=min(int(population_size * 0.1), 500),
                max_datapoints=int(population_size),
                fit_fitness_scaler_on_top_k=True,
                fit_gene_scaler_on_top_k=True,

            ):
                chunk_genes, chunk_fitness_raw, chunk_weights, valid_count = _extract_chunk_to_tensors(
                    chunk, gene_length, device, extract_weights=True
                )
                
                if chunk_genes is None or valid_count == 0:
                    if is_done:
                        break
                    continue
                
                all_genes.append(chunk_genes)
                all_fitness.append(chunk_fitness_raw)
                all_weights.append(chunk_weights)
                total_genes_loaded += valid_count
                
                del chunk_genes, chunk_fitness_raw, chunk_weights
                
                if is_done:
                    break
            
            if total_genes_loaded == 0:
                raise RuntimeError("No genes loaded")
            
            training_genes = torch.cat(all_genes, dim=0)
            training_fitness = torch.cat(all_fitness, dim=0)
            training_weights = torch.cat(all_weights, dim=0)
            
            del all_genes, all_fitness, all_weights
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Apply gene normalization using ChunkedPerDimensionScaler
            # The loader has already collected statistics via fit_gene_scaler_on_top_k=True
            if gene_scaler is not None and getattr(gene_scaler, 'scaling_ready', False):
                training_genes_normalized = gene_scaler.transform(training_genes)
                diffusion_model.use_external_normalization = True
                if verbose >= 1:
                    print(f"[HADES] Applied ChunkedPerDimensionScaler normalization (DISK path)")
            else:
                training_genes_normalized = training_genes
                diffusion_model.use_external_normalization = False
                if verbose >= 1:
                    print(f"[HADES WARNING] gene_scaler not ready, using internal normalization")
            
            training_fitness_scaled = fitness_scaler.scale_chunk(training_fitness)
            
            conditions_data = []
            for condition in conditions:
                cond_values = condition.evaluate(training_genes_normalized, training_fitness_scaled)
                conditions_data.append(cond_values)
            conditions_data = tuple(conditions_data)
            
            if verbose >= 1:
                print(f"[HADES] Training on {total_genes_loaded} genes")
            
            # Ensure batch_size is defined in this branch
            batch_size_raw_stream = config.get('batch_size', 32)
            batch_size_raw_stream = None if batch_size_raw_stream in ['None', 'none'] else batch_size_raw_stream
            batch_size_stream = int(batch_size_raw_stream if batch_size_raw_stream is not None else 32)

            vae_epochs_ratio = float(config.get('vae_epochs_ratio', 0.3))
            
            training_loss = diffusion_model.fit(
                x_data=training_genes_normalized,  # Use normalized genes from ChunkedPerDimensionScaler
                conditions_data=conditions_data,
                weights=training_weights,
                batch_size=batch_size_stream,
                epochs=training_epochs,  # Use calculated epochs
                lr=training_lr,          # Use calculated lr
                weight_decay=weight_decay,
                verbose=(verbose >= 1),
                validation_split=validation_split,
                vae_epochs_ratio=vae_epochs_ratio
            )
            
            del training_genes, training_genes_normalized, training_fitness, training_fitness_scaled, training_weights, loader
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Save model
        if model_cache_path:
            diffusion_model.save(model_cache_path)
            if verbose >= 1:
                print(f"[HADES] ✓ Saved model to cache")
    
    # Generate candidates
    exploration_mode = config['exploration_mode']
    progress = agent_counter / num_generations if num_generations > 0 else 0.0
    
    if exploration_mode == 'greedy':
        exploration_factor = 0.0
    elif exploration_mode == 'exploratory':
        exploration_factor = 0.5
    elif exploration_mode == 'adaptive':
        exploration_factor = max(0.0, 0.5 * (1.0 - progress))
    else:
        exploration_factor = 0.2
    
    if np.random.rand() < 0.05:
        exploration_factor = max(exploration_factor, 0.3)
    
    if verbose >= 1:
        print(f"[HADES] Generating {number_of_candidate_genes} candidates")
    
    target_conditions = []
    for condition in conditions:
        targets = condition.sample(number_of_candidate_genes, device=device)
        if exploration_factor > 0:
            noise = torch.randn_like(targets) * exploration_factor
            targets = targets + noise
        target_conditions.append(targets)
    target_conditions = tuple(target_conditions)
    
    with torch.no_grad():
        generated_genes = diffusion_model.sample(
            num_samples=number_of_candidate_genes,
            conditions=target_conditions,
            use_ema=True,  # Use EMA for better quality
            verbose=(verbose >= 1)
        )
        
        # Apply inverse normalization if using ChunkedPerDimensionScaler
        if gene_scaler is not None and getattr(gene_scaler, 'scaling_ready', False):
            generated_genes = gene_scaler.inverse_transform(generated_genes)
            if verbose >= 1:
                print(f"[HADES] Applied inverse_transform (gene_scaler)")
    
    # Post-generation mutation
    mutation_rate_raw = config.get('mutation_rate', 0.1)
    mutation_rate_raw = None if mutation_rate_raw in ['None', 'none'] else mutation_rate_raw
    mutation_rate = float(mutation_rate_raw if mutation_rate_raw is not None else 0.1)
    
    exploration_noise_raw = config.get('exploration_noise', 0.1)
    exploration_noise_raw = None if exploration_noise_raw in ['None', 'none'] else exploration_noise_raw
    exploration_noise = float(exploration_noise_raw if exploration_noise_raw is not None else 0.1)
    
    if mutation_rate > 0:
        mutation_prob = 0.2 * exploration_factor
        mutation_mask = torch.rand(number_of_candidate_genes, device=device) < mutation_prob
        if mutation_mask.any():
            mutation_noise = torch.randn_like(generated_genes[mutation_mask]) * exploration_noise
            generated_genes[mutation_mask] += mutation_noise
    
    # Convert and clip to valid gene range
    gene_min = getattr(EA_Class, 'geneMin', -1.0)
    lower_bound = 0.0 if gene_min == 0 else -1.0
    
    candidates = []
    generated_np = generated_genes.cpu().numpy()
    
    for gene_array in generated_np:
        clipped_gene = np.clip(gene_array, lower_bound, 1.0)
        candidates.append(clipped_gene.tolist())
    
    # Log loss
    if training_loss is not None:
        final_loss = training_loss[-1]
        build_in_params['loss'] = final_loss
        config['loss'] = final_loss
    
    # Cleanup
    del diffusion_model
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    if verbose >= 1:
        print(f"[HADES] ✓ Complete: {len(candidates)} candidates")
    
    return candidates