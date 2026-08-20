"""
Latent Space Codex Predictor with Weighted Training & Novelty Search.
FIXED: Correctly unpacks (genes, fitness, weights) in training loops.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
import os
import pickle

class BioDataset(torch.utils.data.Dataset):
    """Dataset supporting Sample Weights"""
    def __init__(self, genes: np.ndarray, fitness_scores: np.ndarray, weights: np.ndarray = None):
        self.genes = torch.FloatTensor(genes)
        self.fitness_scores = torch.FloatTensor(fitness_scores).unsqueeze(1)
        
        if weights is not None:
            self.weights = torch.FloatTensor(weights).unsqueeze(1)
        else:
            self.weights = torch.ones_like(self.fitness_scores)
    
    def __len__(self) -> int:
        return len(self.genes)
    
    def __getitem__(self, idx: int):
        # RETURNS 3 VALUES: Gene, Fitness, Weight
        return self.genes[idx], self.fitness_scores[idx], self.weights[idx]

class LatentCodexModel(nn.Module):
    def __init__(self, gene_length: int, hidden_sizes: List[int] = [128, 64, 32], 
                 dropout_rate: float = 0.1):
        super(LatentCodexModel, self).__init__()
        
        self.latent_dim = hidden_sizes[-1] 
        
        # 1. Shared Encoder (Gene -> Latent)
        layers = []
        in_size = gene_length
        for h in hidden_sizes[:-1]:
            layers.extend([
                nn.Linear(in_size, h),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate)
            ])
            in_size = h
        layers.append(nn.Linear(in_size, self.latent_dim))
        self.shared_encoder = nn.Sequential(*layers)
        
        # 2. Fitness Head (Latent -> Fitness)
        self.fitness_head = nn.Sequential(
            nn.Linear(self.latent_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1)
        )
        
        # 3. Decoder (Latent -> Gene)
        dec_layers = []
        in_size = self.latent_dim
        reversed_hidden = hidden_sizes[:-1][::-1]
        
        for h in reversed_hidden:
            dec_layers.extend([
                nn.Linear(in_size, h),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate)
            ])
            in_size = h
            
        dec_layers.append(nn.Linear(in_size, gene_length))
        self.gene_decoder = nn.Sequential(*dec_layers)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, gene: torch.Tensor):
        latent = self.shared_encoder(gene)
        predicted_fitness = self.fitness_head(latent)
        reconstructed_gene = self.gene_decoder(latent)
        return predicted_fitness, reconstructed_gene, latent

    # =========================================================================
    # NOVELTY SEARCH IMPLEMENTATION
    # =========================================================================
    def evolve_latent(self, gene: torch.Tensor, 
                      population_center: torch.Tensor,
                      steps: int = 10, 
                      step_size: float = 0.01, 
                      noise_scale: float = 0.05,
                      regularization_strength: float = 1.0,
                      novelty_strength: float = 0.5) -> torch.Tensor:
        """
        Gradient Ascent with Novelty Search.
        Objective: Maximize(Fitness) + Maximize(Distance to Population Mean) - Minimize(Distance to Anchor)
        """
        # Freeze model
        for param in self.parameters():
            param.requires_grad = False
            
        with torch.no_grad():
            original_latent = self.shared_encoder(gene)
            anchor = original_latent.clone().detach() # The Tether
            
            # Ensure population center is the right shape
            if population_center.dim() == 1:
                pop_center = population_center.unsqueeze(0)
            else:
                pop_center = population_center
        
        # Optimize the Latent Vector
        latent = original_latent.detach().clone()
        latent.requires_grad = True
        
        optimizer = torch.optim.SGD([latent], lr=step_size)
        
        for _ in range(steps):
            optimizer.zero_grad()
            
            # 1. Exploitation (Fitness)
            fitness_pred = self.fitness_head(latent)
            
            # 2. Regularization (Tether to reality)
            dist_to_anchor = torch.norm(latent - anchor)
            
            # 3. Exploration (Novelty Search)
            dist_to_center = torch.norm(latent - pop_center)
            
            # Total Loss (We minimize this)
            # We want: High Fitness (-fit), Low Anchor Dist (+dist), High Center Dist (-dist)
            loss = -fitness_pred.sum() + \
                   (regularization_strength * dist_to_anchor) - \
                   (novelty_strength * dist_to_center)
            
            loss.backward()
            optimizer.step()
        
        # Inject Noise & Decode
        with torch.no_grad():
            noise = torch.randn_like(latent) * noise_scale
            evolved_latent = latent + noise
            new_gene = self.gene_decoder(evolved_latent)
            
        # Unfreeze
        for param in self.parameters():
            param.requires_grad = True
            
        return torch.clamp(new_gene, -1, 1)

class EnsembleBidirectionalPredictor:
    def __init__(self, gene_length: int, num_models: int = 3,
                 hidden_sizes: List[int] = [128, 64, 32],
                 dropout_rate: float = 0.1, device: str = 'cpu'):
        self.gene_length = gene_length
        self.num_models = num_models
        self.hidden_sizes = hidden_sizes
        self.device = device
        self.is_trained = False
        
        self.models = [
            LatentCodexModel(gene_length, hidden_sizes, dropout_rate).to(device)
            for _ in range(num_models)
        ]
        
        self.best_model = None
        self.best_model_idx = None
        self._chunk_state = None

    def start_chunk_training(self, savePath: str, agent_idx: int, epochs: int = 50,
                            batch_size: int = 32, learning_rate: float = 0.001,
                            validation_split: float = 0.2,
                            loss_weights: Tuple[float, float] = (1.0, 0.5),
                            reinitialize_models: bool = True, verbose: bool = True) -> None:
        
        if reinitialize_models:
            self.models = [
                LatentCodexModel(self.gene_length, self.hidden_sizes).to(self.device)
                for _ in range(self.num_models)
            ]
            
        for model in self.models:
            model.train()
            
        running_dir = os.path.join(savePath, 'running')
        os.makedirs(running_dir, exist_ok=True)
        temp_file_path = os.path.join(running_dir, f'{agent_idx}_LatentCodex_val.pkl')
        if os.path.exists(temp_file_path): os.remove(temp_file_path)
        
        optimizers = [optim.Adam(m.parameters(), lr=learning_rate, weight_decay=1e-5) for m in self.models]
        schedulers = [optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5) for opt in optimizers]
        
        self._chunk_state = {
            'temp_file_path': temp_file_path,
            'optimizers': optimizers,
            'schedulers': schedulers,
            'criterion': nn.MSELoss(reduction='none'), # NONE reduction to apply weights manually
            'current_epoch': 1,
            'total_epochs': epochs,
            'batch_size': batch_size,
            'loss_weights': loss_weights,
            'verbose': verbose,
            'training_active': True,
            'validation_split': validation_split,
            'epoch_train_losses': {i: [] for i in range(self.num_models)},
            'epoch_val_losses': {i: [] for i in range(self.num_models)},
            'epoch_loss_sums': {i: 0.0 for i in range(self.num_models)},
            'epoch_batch_counts': {i: 0 for i in range(self.num_models)},
        }

    # =========================================================================
    # WEIGHTED TRAINING LOOP (FIXED UNPACKING ERROR)
    # =========================================================================
    def process_chunk(self, genes: np.ndarray, fitness_scores: np.ndarray, 
                      sample_weights: np.ndarray = None, **kwargs) -> Dict[str, float]:
        
        if not self._chunk_state.get('training_active', False): return {}
        
        genes = torch.FloatTensor(genes) if not isinstance(genes, torch.Tensor) else genes.cpu()
        fitness = torch.FloatTensor(fitness_scores) if not isinstance(fitness_scores, torch.Tensor) else fitness_scores.cpu()
        
        # Handle Weights
        if sample_weights is None:
            weights = np.ones(len(genes), dtype=np.float32)
        else:
            weights = sample_weights if isinstance(sample_weights, np.ndarray) else np.array(sample_weights)
            
        if len(genes) < 2: return {}
        
        # Validation Split (Random shuffle)
        n_val = max(1, int(len(genes) * self._chunk_state['validation_split']))
        indices = torch.randperm(len(genes))
        val_idx, train_idx = indices[:n_val], indices[n_val:]
        
        self._save_validation_data(genes[val_idx].numpy(), fitness[val_idx].numpy(), weights[val_idx.numpy()])
        
        # Weighted Dataset
        dataset = BioDataset(genes[train_idx].numpy(), fitness[train_idx].numpy(), weights[train_idx.numpy()])
        loader = DataLoader(dataset, batch_size=self._chunk_state['batch_size'], shuffle=True)
        
        w_fit, w_recon = self._chunk_state['loss_weights']
        criterion = self._chunk_state['criterion']
        losses = {}
        
        for i, model in enumerate(self.models):
            model.train()
            optimizer = self._chunk_state['optimizers'][i]
            
            chunk_loss = 0.0
            count = 0
            
            # FIXED LINE: Unpack 3 values (genes, fitness, weights)
            for b_genes, b_fit, b_w in loader:
                b_genes = b_genes.to(self.device)
                b_fit = b_fit.to(self.device)
                b_w = b_w.to(self.device)
                
                optimizer.zero_grad()
                pred_fit, recon_gene, _ = model(b_genes)
                
                # Apply Weights
                l_fit = (criterion(pred_fit, b_fit) * b_w).mean()
                recon_err = criterion(recon_gene, b_genes).mean(dim=1, keepdim=True)
                l_recon = (recon_err * b_w).mean()
                
                total_loss = (w_fit * l_fit) + (w_recon * l_recon)
                total_loss.backward()
                optimizer.step()
                
                chunk_loss += total_loss.item()
                count += 1
            
            self._chunk_state['epoch_loss_sums'][i] += chunk_loss
            self._chunk_state['epoch_batch_counts'][i] += count
            losses[i] = chunk_loss / count if count > 0 else 0
            
        return losses

    def finish_chunk_epoch(self) -> Tuple[bool, Dict]:
        state = self._chunk_state
        info = {'current_epoch': state['current_epoch']}
        
        val_genes, val_fit, val_w = self._load_validation_data()
        has_val = len(val_genes) > 0
        
        if has_val:
            dataset = BioDataset(val_genes, val_fit, val_w)
            loader = DataLoader(dataset, batch_size=state['batch_size'])
        
        w_fit, w_recon = state['loss_weights']
        criterion = state['criterion']
        
        for i, model in enumerate(self.models):
            count = state['epoch_batch_counts'][i]
            train_loss = state['epoch_loss_sums'][i] / count if count > 0 else 0
            state['epoch_train_losses'][i].append(train_loss)
            
            val_loss = 0.0
            if has_val:
                model.eval()
                v_count = 0
                with torch.no_grad():
                    # FIXED LINE: Unpack 3 values here too
                    for b_genes, b_fit, b_w in loader:
                        b_genes, b_fit, b_w = b_genes.to(self.device), b_fit.to(self.device), b_w.to(self.device)
                        p_fit, recon, _ = model(b_genes)
                        
                        l_fit = (criterion(p_fit, b_fit) * b_w).mean()
                        l_recon = (criterion(recon, b_genes).mean(dim=1, keepdim=True) * b_w).mean()
                        
                        loss = (w_fit * l_fit) + (w_recon * l_recon)
                        val_loss += loss.item()
                        v_count += 1
                val_loss /= v_count
                state['schedulers'][i].step(val_loss)
            else:
                val_loss = train_loss
                
            state['epoch_val_losses'][i].append(val_loss)
            state['epoch_loss_sums'][i] = 0
            state['epoch_batch_counts'][i] = 0
            
            if state['verbose']:
                print(f" Model_{i+1}: train={train_loss:.5f}, val={val_loss:.5f}", flush=True)

        self._clear_validation_data()
        
        best_idx = min(range(self.num_models), key=lambda x: state['epoch_val_losses'][x][-1])
        self.best_model_idx = best_idx
        self.best_model = self.models[best_idx]
        info['best_val_loss'] = state['epoch_val_losses'][best_idx][-1]
        
        if state['current_epoch'] >= state['total_epochs']:
            self.is_trained = True
            state['training_active'] = False
            return True, info
            
        state['current_epoch'] += 1
        return False, info

    def evolve_population(self, seed_genes: np.ndarray, num_candidates: int,
                         steps: int = 10, step_size: float = 0.01, 
                         noise_scale: float = 0.1,
                         regularization_strength: float = 1.0,
                         novelty_strength: float = 0.5) -> np.ndarray:
        """
        Evolve seeds with Novelty Search.
        """
        if not self.is_trained: raise RuntimeError("Model not trained")
        
        self.best_model.eval()
        candidates = []
        
        seeds_tensor = torch.FloatTensor(seed_genes).to(self.device)
        
        # Calculate Population Center (Mean of seeds) in Latent Space
        with torch.no_grad():
            latents = self.best_model.shared_encoder(seeds_tensor)
            pop_center = latents.mean(dim=0) # [latent_dim]
        
        generated_count = 0
        while generated_count < num_candidates:
            model = self.models[np.random.randint(0, self.num_models)]
            
            idx = np.random.randint(0, len(seeds_tensor))
            seed = seeds_tensor[idx].unsqueeze(0)
            
            evolved = model.evolve_latent(seed, 
                                        population_center=pop_center,
                                        steps=steps, 
                                        step_size=step_size, 
                                        noise_scale=noise_scale,
                                        regularization_strength=regularization_strength,
                                        novelty_strength=novelty_strength)
            
            candidates.append(evolved.cpu().detach().numpy().flatten())
            generated_count += 1
            
        return np.array(candidates)

    def _save_validation_data(self, genes, fitness, weights):
        path = self._chunk_state['temp_file_path']
        mode = 'ab' if os.path.exists(path) else 'wb'
        with open(path, mode) as f:
            pickle.dump((genes, fitness, weights), f)
            
    def _load_validation_data(self):
        path = self._chunk_state['temp_file_path']
        if not os.path.exists(path): return np.array([]), np.array([]), np.array([])
        genes, fits, weights = [], [], []
        with open(path, 'rb') as f:
            while True:
                try:
                    g, ft, w = pickle.load(f)
                    genes.append(g)
                    fits.append(ft)
                    weights.append(w)
                except EOFError: break
        return (np.vstack(genes), np.concatenate(fits), np.concatenate(weights)) if genes else (np.array([]), np.array([]), np.array([]))

    def _clear_validation_data(self):
        if os.path.exists(self._chunk_state['temp_file_path']):
            os.remove(self._chunk_state['temp_file_path'])