import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import copy

class FeatureFitnessDataset(Dataset):
    """Custom dataset for feature-fitness pairs"""
    def __init__(self, features, fitness_scores):
        self.features = torch.FloatTensor(features)
        self.fitness_scores = torch.FloatTensor(fitness_scores).unsqueeze(1)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.fitness_scores[idx]

class FeatureFitnessNet(nn.Module):
    """Neural network for predicting fitness from features"""
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], output_size=1, dropout_rate=0.2):
        super(FeatureFitnessNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),  # Keep batch normalization
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer (regression)
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class FeatureFitnessPredictor:
    """Main class for training and using the model"""
    
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], output_size=1, dropout_rate=0.2, 
                 device='cpu', max_epochs=100, base_learning_rate=0.001):
        self.device = device
        self.model = FeatureFitnessNet(input_size, hidden_sizes, output_size, dropout_rate).to(self.device)
        self.is_trained = False
        self.dropout_rate = dropout_rate
        self.max_epochs = max_epochs
        self.base_learning_rate = base_learning_rate
        
        # Initialize optimizer and scheduler as None - will be created when training starts
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.MSELoss()
        
        # Store best model state and training history
        self.best_model_state = None
        self.training_history = {}
        
        # Adaptive statistics tracking
        self.chunk_statistics = {
            'current_epoch': 0,
            'current_chunk_in_epoch': 0,
            'epoch_history': [],  # Will store up to 50 epochs of data
            'current_epoch_chunks': [],  # Temporary storage for current epoch chunks
            'best_epoch_performance': float('inf'),
            'best_epoch_idx': -1,
            'adaptive_phase': 'collection',  # 'collection', 'transition', 'adaptive'
            'learning_rate_history': [],
            'gradient_norm_history': [],
            'early_stopping_patience': max(10, max_epochs // 10),  # Adaptive initial patience
            'consecutive_bad_epochs': 0,
            'max_grad_norm': 1.0  # Will be adaptive
        }

    def clone(self):
        """Create a clone of the current predictor"""
        cloned_predictor = FeatureFitnessPredictor(
            input_size=self.model.network[0].in_features,
            hidden_sizes=[layer.out_features for layer in self.model.network if isinstance(layer, nn.Linear)][:-1],
            output_size=self.model.network[-1].out_features,
            dropout_rate=self.dropout_rate,
            device=self.device
        )
        cloned_predictor.model.load_state_dict(self.model.state_dict())
        cloned_predictor.is_trained = self.is_trained
        return cloned_predictor
    
    def _calculate_weighted_metrics(self, y_true, y_pred, sample_weights=None):
        """Calculate weighted regression metrics (handles both weighted and unweighted cases)"""
        y_true_np = np.array(y_true).flatten()
        y_pred_np = np.array(y_pred).flatten()
        
        # Ensure all arrays have the same length
        min_len = min(len(y_true_np), len(y_pred_np))
        y_true_np = y_true_np[:min_len]
        y_pred_np = y_pred_np[:min_len]
        
        # Handle sample weights
        if sample_weights is not None:
            weights_np = np.array(sample_weights).flatten()[:min_len]
            # Normalize weights to sum to 1
            if np.sum(weights_np) > 0:
                weights_normalized = weights_np / np.sum(weights_np)
            else:
                weights_normalized = np.ones_like(weights_np) / len(weights_np)
        else:
            # Equal weights for all samples
            weights_normalized = np.ones(min_len) / min_len
        
        # Weighted metrics calculation
        # Weighted MSE
        squared_errors = (y_true_np - y_pred_np) ** 2
        weighted_mse = np.sum(weights_normalized * squared_errors)
        
        # Weighted MAE
        absolute_errors = np.abs(y_true_np - y_pred_np)
        weighted_mae = np.sum(weights_normalized * absolute_errors)
        
        # Weighted R²
        y_true_mean = np.sum(weights_normalized * y_true_np)
        ss_tot = np.sum(weights_normalized * (y_true_np - y_true_mean) ** 2)
        ss_res = np.sum(weights_normalized * squared_errors)
        
        if ss_tot > 0:
            weighted_r2 = 1 - (ss_res / ss_tot)
        else:
            weighted_r2 = 0.0
        
        return {
            'r2': weighted_r2,
            'mae': weighted_mae,
            'mse': weighted_mse
        }
        
    def train_model(self, features, fitness_scores, sample_weights=None, epochs=100, batch_size=32, 
                   learning_rate=0.001, validation_split=0.2, early_stopping_patience=15,
                   min_delta=1e-6, max_grad_norm=1.0, k_folds=None):
        """Enhanced training with early stopping, checkpointing, and sample weighting support"""
        
        if k_folds is not None:
            return self._train_with_kfold(features, fitness_scores, sample_weights, epochs, batch_size, 
                                        learning_rate, k_folds, early_stopping_patience,
                                        min_delta, max_grad_norm)
        
        # Split data with sample weights if provided
        if sample_weights is not None:
            X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
                features, fitness_scores, sample_weights, test_size=validation_split, random_state=42
            )
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                features, fitness_scores, test_size=validation_split, random_state=42
            )
            w_train, w_val = None, None
        
        # Create datasets and dataloaders
        train_dataset = FeatureFitnessDataset(X_train, y_train)
        val_dataset = FeatureFitnessDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Convert weights to tensors if provided
        if w_train is not None:
            w_train_tensor = torch.FloatTensor(w_train).to(self.device)
        if w_val is not None:
            w_val_tensor = torch.FloatTensor(w_val).to(self.device)
        
        # Loss and optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training tracking
        train_losses = []
        val_losses = []
        train_metrics_history = []
        val_metrics_history = []
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        if sample_weights is not None:
            print(f"Using sample weights (range: {np.min(sample_weights):.3f} - {np.max(sample_weights):.3f})")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            epoch_train_loss = 0
            train_predictions = []
            train_targets = []
            train_weights_used = []
            
            for batch_idx, (batch_features, batch_fitness) in enumerate(train_loader):
                batch_features = batch_features.to(self.device)
                batch_fitness = batch_fitness.to(self.device)
                
                optimizer.zero_grad()
                predictions = self.model(batch_features)
                
                # Calculate loss with proper sample weighting
                if w_train is not None:
                    # Get corresponding weights for this batch
                    batch_start_idx = batch_idx * batch_size
                    batch_end_idx = min(batch_start_idx + batch_size, len(w_train))
                    batch_weights = w_train_tensor[batch_start_idx:batch_end_idx]
                    
                    # Ensure batch_weights matches batch size
                    if len(batch_weights) != len(batch_fitness):
                        batch_weights = batch_weights[:len(batch_fitness)]
                    
                    # Apply sample weights to loss
                    individual_losses = self.criterion(predictions, batch_fitness)
                    weighted_loss = torch.mean(individual_losses * batch_weights.unsqueeze(1))
                    loss = weighted_loss
                    
                    train_weights_used.extend(batch_weights.cpu().numpy())
                else:
                    # Standard unweighted loss
                    loss = self.criterion(predictions, batch_fitness)
                    train_weights_used.extend([1.0] * len(batch_fitness))
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                
                optimizer.step()
                
                epoch_train_loss += loss.item()
                train_predictions.extend(predictions.detach().cpu().numpy())
                train_targets.extend(batch_fitness.cpu().numpy())
            
            # Validation phase
            self.model.eval()
            epoch_val_loss = 0
            val_predictions = []
            val_targets = []
            val_weights_used = []
            
            with torch.no_grad():
                for batch_idx, (batch_features, batch_fitness) in enumerate(val_loader):
                    batch_features = batch_features.to(self.device)
                    batch_fitness = batch_fitness.to(self.device)
                    predictions = self.model(batch_features)
                    
                    # Calculate validation loss with proper sample weighting
                    if w_val is not None:
                        # Get corresponding weights for this batch
                        batch_start_idx = batch_idx * batch_size
                        batch_end_idx = min(batch_start_idx + batch_size, len(w_val))
                        batch_weights = w_val_tensor[batch_start_idx:batch_end_idx]
                        
                        # Ensure batch_weights matches batch size
                        if len(batch_weights) != len(batch_fitness):
                            batch_weights = batch_weights[:len(batch_fitness)]
                        
                        # Apply sample weights to validation loss
                        individual_losses = self.criterion(predictions, batch_fitness)
                        weighted_loss = torch.mean(individual_losses * batch_weights.unsqueeze(1))
                        loss = weighted_loss
                        
                        val_weights_used.extend(batch_weights.cpu().numpy())
                    else:
                        # Standard unweighted validation loss
                        loss = self.criterion(predictions, batch_fitness)
                        val_weights_used.extend([1.0] * len(batch_fitness))
                    
                    epoch_val_loss += loss.item()
                    val_predictions.extend(predictions.cpu().numpy())
                    val_targets.extend(batch_fitness.cpu().numpy())
            
            # Calculate average losses and metrics
            avg_train_loss = epoch_train_loss / len(train_loader)
            avg_val_loss = epoch_val_loss / len(val_loader)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            # Calculate weighted metrics
            train_metrics = self._calculate_weighted_metrics(train_targets, train_predictions, train_weights_used)
            val_metrics = self._calculate_weighted_metrics(val_targets, val_predictions, val_weights_used)
            
            train_metrics_history.append(train_metrics)
            val_metrics_history.append(val_metrics)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Early stopping and model checkpointing
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_epoch = epoch
                # Save best model state
                self.best_model_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
            
            # Progress reporting
            if (epoch + 1) % 10 == 0 or epoch == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Epoch [{epoch+1:4d}/{epochs}] | '
                      f'Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | '
                      f'Train R²: {train_metrics["r2"]:.4f} | Val R²: {val_metrics["r2"]:.4f} | '
                      f'LR: {current_lr:.2e}')
            
            # Early stopping check
            if patience_counter >= early_stopping_patience:
                print(f'\nEarly stopping triggered after {epoch + 1} epochs')
                print(f'Best validation loss: {best_val_loss:.4f} at epoch {best_epoch + 1}')
                break
        
        # Load best model state
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"Loaded best model from epoch {best_epoch + 1}")
        
        # Store training history
        self.training_history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_metrics': train_metrics_history,
            'val_metrics': val_metrics_history,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss
        }
        
        self.is_trained = True
        print(f"\nTraining completed!")
        print(f"Final validation R²: {val_metrics_history[best_epoch]['r2']:.4f}")
        print(f"Final validation MAE: {val_metrics_history[best_epoch]['mae']:.4f}")
        
        return train_losses, val_losses
    
    def _train_with_kfold(self, features, fitness_scores, sample_weights, epochs, batch_size, 
                         learning_rate, k_folds, early_stopping_patience, min_delta, max_grad_norm):
        """Train with k-fold cross validation with sample weights support"""
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        fold_results = []
        
        print(f"Starting {k_folds}-fold cross validation...")
        if sample_weights is not None:
            print(f"Using sample weights (range: {np.min(sample_weights):.3f} - {np.max(sample_weights):.3f})")
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(features)):
            print(f"\n--- Fold {fold + 1}/{k_folds} ---")
            
            X_train, X_val = features[train_idx], features[val_idx]
            y_train, y_val = fitness_scores[train_idx], fitness_scores[val_idx]
            
            if sample_weights is not None:
                w_train, w_val = sample_weights[train_idx], sample_weights[val_idx]
            else:
                w_train, w_val = None, None
            
            # Reset model for each fold
            self.model.apply(self._reset_weights)
            
            # Train on this fold
            train_losses, val_losses = self._train_single_fold(
                X_train, y_train, X_val, y_val, w_train, w_val, epochs, batch_size, 
                learning_rate, early_stopping_patience, min_delta, max_grad_norm
            )
            
            fold_results.append({
                'train_losses': train_losses,
                'val_losses': val_losses,
                'final_val_loss': val_losses[-1] if val_losses else float('inf')
            })
        
        # Train final model on all data
        print(f"\n--- Training final model on all data ---")
        avg_val_loss = np.mean([fold['final_val_loss'] for fold in fold_results])
        print(f"Average CV validation loss: {avg_val_loss:.4f}")
        
        # Train on full dataset
        return self.train_model(features, fitness_scores, sample_weights, epochs, batch_size, 
                              learning_rate, validation_split=0.1, 
                              early_stopping_patience=early_stopping_patience,
                              min_delta=min_delta, max_grad_norm=max_grad_norm, k_folds=None)
    
    def _train_single_fold(self, X_train, y_train, X_val, y_val, w_train, w_val, epochs, batch_size, 
                          learning_rate, early_stopping_patience, min_delta, max_grad_norm):
        """Train a single fold with sample weights support"""
        train_dataset = FeatureFitnessDataset(X_train, y_train)
        val_dataset = FeatureFitnessDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Convert weights to tensors if provided
        if w_train is not None:
            w_train_tensor = torch.FloatTensor(w_train).to(self.device)
        if w_val is not None:
            w_val_tensor = torch.FloatTensor(w_val).to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            epoch_train_loss = 0
            for batch_idx, (batch_features, batch_fitness) in enumerate(train_loader):
                batch_features = batch_features.to(self.device)
                batch_fitness = batch_fitness.to(self.device)
                
                optimizer.zero_grad()
                predictions = self.model(batch_features)
                
                # Calculate loss with proper sample weighting
                if w_train is not None:
                    # Get corresponding weights for this batch
                    batch_start_idx = batch_idx * batch_size
                    batch_end_idx = min(batch_start_idx + batch_size, len(w_train))
                    batch_weights = w_train_tensor[batch_start_idx:batch_end_idx]
                    
                    # Ensure batch_weights matches batch size
                    if len(batch_weights) != len(batch_fitness):
                        batch_weights = batch_weights[:len(batch_fitness)]
                    
                    # Apply sample weights to loss
                    individual_losses = self.criterion(predictions, batch_fitness)
                    weighted_loss = torch.mean(individual_losses * batch_weights.unsqueeze(1))
                    loss = weighted_loss
                else:
                    # Standard unweighted loss
                    loss = self.criterion(predictions, batch_fitness)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()
                
                epoch_train_loss += loss.item()
            
            # Validation
            self.model.eval()
            epoch_val_loss = 0
            with torch.no_grad():
                for batch_idx, (batch_features, batch_fitness) in enumerate(val_loader):
                    batch_features = batch_features.to(self.device)
                    batch_fitness = batch_fitness.to(self.device)
                    predictions = self.model(batch_features)
                    
                    # Calculate validation loss with proper sample weighting
                    if w_val is not None:
                        # Get corresponding weights for this batch
                        batch_start_idx = batch_idx * batch_size
                        batch_end_idx = min(batch_start_idx + batch_size, len(w_val))
                        batch_weights = w_val_tensor[batch_start_idx:batch_end_idx]
                        
                        # Ensure batch_weights matches batch size
                        if len(batch_weights) != len(batch_fitness):
                            batch_weights = batch_weights[:len(batch_fitness)]
                        
                        # Apply sample weights to validation loss
                        individual_losses = self.criterion(predictions, batch_fitness)
                        weighted_loss = torch.mean(individual_losses * batch_weights.unsqueeze(1))
                        loss = weighted_loss
                    else:
                        # Standard unweighted validation loss
                        loss = self.criterion(predictions, batch_fitness)
                    
                    epoch_val_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            avg_val_loss = epoch_val_loss / len(val_loader)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                break
        
        return train_losses, val_losses
    
    def _reset_weights(self, m):
        """Reset model weights for k-fold CV"""
        if isinstance(m, nn.Linear):
            m.reset_parameters()
        elif isinstance(m, nn.BatchNorm1d):
            m.reset_parameters()
    
    def plot_training_history(self, show_metrics=True):
        """Plot comprehensive training history"""
        if not hasattr(self, 'training_history') or not self.training_history:
            print("No training history available. Train the model first.")
            return
        
        history = self.training_history
        
        if show_metrics:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Loss plot
            axes[0, 0].plot(history['train_losses'], label='Training Loss', color='blue')
            axes[0, 0].plot(history['val_losses'], label='Validation Loss', color='red')
            axes[0, 0].axvline(x=history['best_epoch'], color='green', linestyle='--', 
                              label=f'Best Model (Epoch {history["best_epoch"] + 1})')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('MSE Loss')
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # R² score plot
            train_r2 = [m['r2'] for m in history['train_metrics']]
            val_r2 = [m['r2'] for m in history['val_metrics']]
            axes[0, 1].plot(train_r2, label='Training R²', color='blue')
            axes[0, 1].plot(val_r2, label='Validation R²', color='red')
            axes[0, 1].axvline(x=history['best_epoch'], color='green', linestyle='--')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('R² Score')
            axes[0, 1].set_title('R² Score Progress')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # MAE plot
            train_mae = [m['mae'] for m in history['train_metrics']]
            val_mae = [m['mae'] for m in history['val_metrics']]
            axes[1, 0].plot(train_mae, label='Training MAE', color='blue')
            axes[1, 0].plot(val_mae, label='Validation MAE', color='red')
            axes[1, 0].axvline(x=history['best_epoch'], color='green', linestyle='--')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Mean Absolute Error')
            axes[1, 0].set_title('MAE Progress')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # Learning curve (validation loss zoomed)
            axes[1, 1].plot(history['val_losses'], label='Validation Loss', color='red', linewidth=2)
            axes[1, 1].axvline(x=history['best_epoch'], color='green', linestyle='--',
                              label=f'Best: {history["best_val_loss"]:.4f}')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Validation Loss')
            axes[1, 1].set_title('Validation Loss (Detailed)')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            
            plt.tight_layout()
        else:
            plt.figure(figsize=(10, 6))
            plt.plot(history['train_losses'], label='Training Loss', color='blue')
            plt.plot(history['val_losses'], label='Validation Loss', color='red')
            plt.axvline(x=history['best_epoch'], color='green', linestyle='--', 
                       label=f'Best Model (Epoch {history["best_epoch"] + 1})')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training History')
            plt.legend()
            plt.grid(True)
        
        plt.show()
        
    def train_chunk(self, features, fitness_scores, batch_size=32, 
                   validation_split=0.2, sample_weights=None):
        """Enhanced chunk training with statistics collection and adaptive optimization"""
        
        # Determine current learning rate adaptively
        current_lr = self._get_adaptive_learning_rate()
        
        # Initialize or reset optimizer if needed (first chunk of training)
        if self.optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=current_lr, weight_decay=1e-5)
        else:
            # Update learning rate for existing optimizer
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr
        
        # Handle sample weights and split chunk data for validation
        if len(features) > 10:  # Only split if chunk is large enough
            if sample_weights is not None:
                # Split all three arrays together to maintain index alignment
                X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
                    features, fitness_scores, sample_weights, 
                    test_size=validation_split, random_state=42
                )
            else:
                # No sample weights - standard split with equal importance
                X_train, X_val, y_train, y_val = train_test_split(
                    features, fitness_scores, test_size=validation_split, random_state=42
                )
                w_train, w_val = None, None
        else:
            # Chunk too small to split - use all data for both train and validation
            X_train, X_val, y_train, y_val = features, features, fitness_scores, fitness_scores
            if sample_weights is not None:
                w_train, w_val = sample_weights, sample_weights
            else:
                w_train, w_val = None, None
        
        # Adjust batch size to handle small datasets and BatchNorm requirements
        train_batch_size = min(batch_size, len(X_train))
        val_batch_size = min(batch_size, len(X_val))
        
        # Ensure batch size is at least 2 for BatchNorm (if we have enough samples)
        if len(X_train) >= 2:
            train_batch_size = max(2, train_batch_size)
        if len(X_val) >= 2:
            val_batch_size = max(2, val_batch_size)
        
        # Create datasets and dataloaders
        train_dataset = FeatureFitnessDataset(X_train, y_train)
        val_dataset = FeatureFitnessDataset(X_val, y_val)
        
        # Use drop_last=True to avoid single-sample batches with BatchNorm
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, 
                                drop_last=(len(X_train) > train_batch_size))
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, 
                              drop_last=(len(X_val) > val_batch_size))
        
        # Convert weights to tensors if provided
        if w_train is not None:
            w_train_tensor = torch.FloatTensor(w_train).to(self.device)
        if w_val is not None:
            w_val_tensor = torch.FloatTensor(w_val).to(self.device)
        
        # Training loop with statistics collection
        chunk_stats = {
            'train_losses': [],
            'val_losses': [],
            'gradient_norms': [],
            'learning_rates': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        # Get adaptive gradient clipping threshold
        max_grad_norm = self._get_adaptive_grad_norm()
        
        # Check if we have enough data to train
        if len(train_loader) == 0:
            print(f"Warning: No training batches available (chunk too small: {len(X_train)} samples)")
            # Return dummy statistics
            chunk_stats['train_losses'].append(float('inf'))
            chunk_stats['val_losses'].append(float('inf'))
            chunk_stats['gradient_norms'].append(0.0)
            chunk_stats['learning_rates'].append(current_lr)
            chunk_stats['train_metrics'].append({'r2': 0.0, 'mae': float('inf'), 'mse': float('inf')})
            chunk_stats['val_metrics'].append({'r2': 0.0, 'mae': float('inf'), 'mse': float('inf')})
            
            self.chunk_statistics['current_epoch_chunks'].append(chunk_stats)
            self.chunk_statistics['current_chunk_in_epoch'] += 1
            return chunk_stats['train_losses']
        
        self.model.train()
        # Training phase
        epoch_train_loss = 0
        gradient_norms = []
        train_predictions = []
        train_targets = []
        train_weights_used = []
        
        for batch_idx, (batch_features, batch_fitness) in enumerate(train_loader):
            batch_features = batch_features.to(self.device)
            batch_fitness = batch_fitness.to(self.device)
            
            # Skip batch if it has only 1 sample and we have BatchNorm layers
            if len(batch_features) == 1 and self._has_batchnorm():
                continue
            
            self.optimizer.zero_grad()
            predictions = self.model(batch_features)
            
            # Calculate loss with proper sample weighting
            if w_train is not None:
                # Get corresponding weights for this batch
                batch_start_idx = batch_idx * train_batch_size
                batch_end_idx = min(batch_start_idx + train_batch_size, len(w_train))
                batch_weights = w_train_tensor[batch_start_idx:batch_end_idx]
                
                # Ensure batch_weights matches batch size
                if len(batch_weights) != len(batch_fitness):
                    batch_weights = batch_weights[:len(batch_fitness)]
                
                # Apply sample weights to loss
                individual_losses = self.criterion(predictions, batch_fitness)
                weighted_loss = torch.mean(individual_losses * batch_weights.unsqueeze(1))
                loss = weighted_loss
                
                train_weights_used.extend(batch_weights.cpu().numpy())
            else:
                # Standard unweighted loss
                loss = self.criterion(predictions, batch_fitness)
                train_weights_used.extend([1.0] * len(batch_fitness))
            
            loss.backward()
            
            # Calculate gradient norm before clipping
            total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            gradient_norms.append(total_norm.item())
            
            self.optimizer.step()
            
            epoch_train_loss += loss.item()
            train_predictions.extend(predictions.detach().cpu().numpy())
            train_targets.extend(batch_fitness.cpu().numpy())
        
        # Validation phase
        self.model.eval()
        epoch_val_loss = 0
        val_predictions = []
        val_targets = []
        val_weights_used = []
        
        with torch.no_grad():
            for batch_idx, (batch_features, batch_fitness) in enumerate(val_loader):
                batch_features = batch_features.to(self.device)
                batch_fitness = batch_fitness.to(self.device)
                
                # Skip batch if it has only 1 sample and we have BatchNorm layers
                if len(batch_features) == 1 and self._has_batchnorm():
                    continue
                
                predictions = self.model(batch_features)
                
                # Calculate validation loss with proper sample weighting
                if w_val is not None:
                    # Get corresponding weights for this batch
                    batch_start_idx = batch_idx * val_batch_size
                    batch_end_idx = min(batch_start_idx + val_batch_size, len(w_val))
                    batch_weights = w_val_tensor[batch_start_idx:batch_end_idx]
                    
                    # Ensure batch_weights matches batch size
                    if len(batch_weights) != len(batch_fitness):
                        batch_weights = batch_weights[:len(batch_fitness)]
                    
                    # Apply sample weights to validation loss
                    individual_losses = self.criterion(predictions, batch_fitness)
                    weighted_loss = torch.mean(individual_losses * batch_weights.unsqueeze(1))
                    loss = weighted_loss
                    
                    val_weights_used.extend(batch_weights.cpu().numpy())
                else:
                    # Standard unweighted validation loss
                    loss = self.criterion(predictions, batch_fitness)
                    val_weights_used.extend([1.0] * len(batch_fitness))
                
                epoch_val_loss += loss.item()
                val_predictions.extend(predictions.cpu().numpy())
                val_targets.extend(batch_fitness.cpu().numpy())
        
        # Calculate metrics and store statistics
        if len(train_loader) > 0:
            avg_train_loss = epoch_train_loss / len(train_loader)
        else:
            avg_train_loss = float('inf')
            
        if len(val_loader) > 0:
            avg_val_loss = epoch_val_loss / len(val_loader)
        else:
            avg_val_loss = float('inf')
            
        if len(gradient_norms) > 0:
            avg_grad_norm = np.mean(gradient_norms)
        else:
            avg_grad_norm = 0.0
        
        # Calculate weighted metrics (handle empty predictions)
        if len(train_predictions) > 0:
            train_metrics = self._calculate_weighted_metrics(train_targets, train_predictions, train_weights_used)
        else:
            train_metrics = {'r2': 0.0, 'mae': float('inf'), 'mse': float('inf')}
            
        if len(val_predictions) > 0:
            val_metrics = self._calculate_weighted_metrics(val_targets, val_predictions, val_weights_used)
        else:
            val_metrics = {'r2': 0.0, 'mae': float('inf'), 'mse': float('inf')}
        
        chunk_stats['train_losses'].append(avg_train_loss)
        chunk_stats['val_losses'].append(avg_val_loss)
        chunk_stats['gradient_norms'].append(avg_grad_norm)
        chunk_stats['learning_rates'].append(current_lr)
        chunk_stats['train_metrics'].append(train_metrics)
        chunk_stats['val_metrics'].append(val_metrics)
                
        # Store chunk statistics for epoch-level analysis
        self.chunk_statistics['current_epoch_chunks'].append(chunk_stats)
        self.chunk_statistics['current_chunk_in_epoch'] += 1
        
        # Progress reporting (optional)
        final_train_loss = chunk_stats['train_losses'][-1]
        final_val_loss = chunk_stats['val_losses'][-1]
        final_val_r2 = chunk_stats['val_metrics'][-1]['r2']
        
        print(f'Chunk {self.chunk_statistics["current_chunk_in_epoch"]} | '
              f'Train Loss: {final_train_loss:.4f} | Val Loss: {final_val_loss:.4f} | '
              f'Val R²: {final_val_r2:.4f} | LR: {current_lr:.2e} | '
              f'Grad Norm: {chunk_stats["gradient_norms"][-1]:.3f}')
        
        self.is_trained = True
        return chunk_stats['train_losses']
    
    def _has_batchnorm(self):
        """Check if the model has BatchNorm layers"""
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm1d):
                return True
        return False
    
    def chunk_epoch_finalize(self):
        """Finalize epoch, analyze statistics, and adapt optimization strategy"""
        
        # Aggregate current epoch statistics
        epoch_stats = self._aggregate_epoch_statistics()
        
        # Store epoch statistics (keep last 50 epochs)
        self.chunk_statistics['epoch_history'].append(epoch_stats)
        if len(self.chunk_statistics['epoch_history']) > 50:
            self.chunk_statistics['epoch_history'].pop(0)
        
        # Update best performance tracking
        current_val_loss = epoch_stats['avg_val_loss']
        if current_val_loss < self.chunk_statistics['best_epoch_performance']:
            self.chunk_statistics['best_epoch_performance'] = current_val_loss
            self.chunk_statistics['best_epoch_idx'] = self.chunk_statistics['current_epoch']
            self.best_model_state = copy.deepcopy(self.model.state_dict())
            self.chunk_statistics['consecutive_bad_epochs'] = 0
        else:
            self.chunk_statistics['consecutive_bad_epochs'] += 1
        
        # Determine adaptive phase based on epoch progress
        self._update_adaptive_phase()
        
        # Make adaptive adjustments based on historical data
        should_stop = self._make_adaptive_adjustments()
        
        # Update epoch counters and reset chunk tracking
        self.chunk_statistics['current_epoch'] += 1
        self.chunk_statistics['current_epoch_chunks'] = []
        self.chunk_statistics['current_chunk_in_epoch'] = 0
        
        # Progress reporting
        print(f'\n--- Epoch {self.chunk_statistics["current_epoch"]} Summary ---')
        print(f'Avg Val Loss: {current_val_loss:.4f} | Best: {self.chunk_statistics["best_epoch_performance"]:.4f} | '
              f'Phase: {self.chunk_statistics["adaptive_phase"]} | '
              f'Bad Epochs: {self.chunk_statistics["consecutive_bad_epochs"]}')
        
        if should_stop:
            print(f'Early stopping triggered! Loading best model from epoch {self.chunk_statistics["best_epoch_idx"] + 1}')
            if self.best_model_state is not None:
                self.model.load_state_dict(self.best_model_state)
        
        return should_stop
    
    def _get_adaptive_learning_rate(self):
        """Calculate adaptive learning rate based on training history and phase"""
        
        if self.chunk_statistics['adaptive_phase'] == 'collection':
            # Conservative initial learning rate
            return self.base_learning_rate
        
        elif self.chunk_statistics['adaptive_phase'] == 'transition':
            # Gradually adapt based on limited history
            recent_epochs = self.chunk_statistics['epoch_history'][-3:]
            if len(recent_epochs) >= 2:
                # Simple trend analysis
                losses = [epoch['avg_val_loss'] for epoch in recent_epochs]
                if losses[-1] > losses[0]:  # Performance degrading
                    return self.base_learning_rate * 0.5
                else:  # Performance improving
                    return self.base_learning_rate * 1.2
            return self.base_learning_rate
        
        else:  # 'adaptive' phase
            # Full adaptive behavior based on comprehensive history
            return self._calculate_optimal_learning_rate()
    
    def _get_adaptive_grad_norm(self):
        """Calculate adaptive gradient clipping threshold"""
        
        if self.chunk_statistics['adaptive_phase'] == 'collection':
            return 1.0  # Conservative default
        
        # Analyze historical gradient norms
        if len(self.chunk_statistics['gradient_norm_history']) > 0:
            recent_norms = self.chunk_statistics['gradient_norm_history'][-20:]  # Last 20 measurements
            mean_norm = np.mean(recent_norms)
            std_norm = np.std(recent_norms)
            # Set threshold at mean + 2*std to catch outliers
            adaptive_threshold = max(0.1, min(10.0, mean_norm + 2 * std_norm))
            return adaptive_threshold
        
        return self.chunk_statistics['max_grad_norm']
    
    def _aggregate_epoch_statistics(self):
        """Aggregate statistics from all chunks in current epoch"""
        
        chunks = self.chunk_statistics['current_epoch_chunks']
        if not chunks:
            return {'avg_val_loss': float('inf'), 'num_chunks': 0}
        
        # Aggregate across all chunks in epoch
        all_train_losses = []
        all_val_losses = []
        all_grad_norms = []
        all_val_r2 = []
        
        for chunk in chunks:
            all_train_losses.extend(chunk['train_losses'])
            all_val_losses.extend(chunk['val_losses'])
            all_grad_norms.extend(chunk['gradient_norms'])
            all_val_r2.extend([m['r2'] for m in chunk['val_metrics']])
        
        epoch_stats = {
            'epoch_idx': self.chunk_statistics['current_epoch'],
            'num_chunks': len(chunks),
            'avg_train_loss': np.mean(all_train_losses),
            'avg_val_loss': np.mean(all_val_losses),
            'avg_grad_norm': np.mean(all_grad_norms),
            'avg_val_r2': np.mean(all_val_r2),
            'train_loss_std': np.std(all_train_losses),
            'val_loss_std': np.std(all_val_losses),
            'improvement_rate': 0.0  # Will be calculated based on previous epochs
        }
        
        # Calculate improvement rate if we have previous epochs
        if len(self.chunk_statistics['epoch_history']) > 0:
            prev_val_loss = self.chunk_statistics['epoch_history'][-1]['avg_val_loss']
            epoch_stats['improvement_rate'] = (prev_val_loss - epoch_stats['avg_val_loss']) / prev_val_loss
        
        # Update global gradient norm history
        self.chunk_statistics['gradient_norm_history'].extend(all_grad_norms)
        if len(self.chunk_statistics['gradient_norm_history']) > 1000:  # Keep last 1000
            self.chunk_statistics['gradient_norm_history'] = self.chunk_statistics['gradient_norm_history'][-1000:]
        
        return epoch_stats
    
    def _update_adaptive_phase(self):
        """Update the adaptive phase based on training progress"""
        
        current_epoch = self.chunk_statistics['current_epoch']
        max_epochs = self.max_epochs
        
        # Phase transitions based on epoch progress and data availability
        if current_epoch < max(3, max_epochs * 0.1):  # First 10% or minimum 3 epochs
            self.chunk_statistics['adaptive_phase'] = 'collection'
        elif current_epoch < max(8, max_epochs * 0.3):  # Next 20% or up to 8 epochs
            self.chunk_statistics['adaptive_phase'] = 'transition'
        else:
            self.chunk_statistics['adaptive_phase'] = 'adaptive'
    
    def _make_adaptive_adjustments(self):
        """Make adaptive adjustments and determine if training should stop"""
        
        current_epoch = self.chunk_statistics['current_epoch']
        max_epochs = self.max_epochs
        consecutive_bad_epochs = self.chunk_statistics['consecutive_bad_epochs']
        
        # Adaptive early stopping patience
        if self.chunk_statistics['adaptive_phase'] == 'collection':
            patience = max(10, max_epochs // 10)
        elif self.chunk_statistics['adaptive_phase'] == 'transition':
            # Analyze recent improvement patterns
            recent_epochs = self.chunk_statistics['epoch_history'][-5:]
            if len(recent_epochs) >= 3:
                improvements = [epoch.get('improvement_rate', 0) for epoch in recent_epochs]
                avg_improvement = np.mean(improvements)
                if avg_improvement > 0.01:  # Good improvement rate
                    patience = max(15, max_epochs // 8)
                else:
                    patience = max(8, max_epochs // 12)
            else:
                patience = max(10, max_epochs // 10)
        else:  # 'adaptive' phase
            patience = self._calculate_adaptive_patience()
        
        self.chunk_statistics['early_stopping_patience'] = patience
        
        # Check early stopping condition
        should_stop = consecutive_bad_epochs >= patience or current_epoch >= max_epochs
        
        return should_stop
    
    def _calculate_optimal_learning_rate(self):
        """Calculate optimal learning rate based on comprehensive history analysis"""
        
        # Assumption: Analyze recent performance trends and adjust accordingly
        recent_epochs = self.chunk_statistics['epoch_history'][-10:]  # Last 10 epochs
        
        if len(recent_epochs) < 3:
            return self.base_learning_rate
        
        # Analyze improvement trends
        improvements = [epoch.get('improvement_rate', 0) for epoch in recent_epochs]
        avg_improvement = np.mean(improvements)
        improvement_std = np.std(improvements)
        
        # Analyze loss variance (stability)
        val_losses = [epoch['avg_val_loss'] for epoch in recent_epochs]
        loss_trend = val_losses[-1] - val_losses[0]  # Overall trend
        
        # Calculate adaptive multiplier
        base_multiplier = 1.0
        
        # If improvement is good and stable, slightly increase LR
        if avg_improvement > 0.005 and improvement_std < 0.02:
            base_multiplier *= 1.1
        # If improvement is poor, decrease LR
        elif avg_improvement < -0.01:
            base_multiplier *= 0.8
        # If very unstable, decrease LR for stability
        elif improvement_std > 0.05:
            base_multiplier *= 0.9
        
        # Consider overall progress in training
        progress_ratio = self.chunk_statistics['current_epoch'] / self.max_epochs
        if progress_ratio > 0.7:  # Late in training, be more conservative
            base_multiplier *= 0.95
        
        # Apply bounds
        adaptive_lr = self.base_learning_rate * base_multiplier
        adaptive_lr = max(1e-6, min(self.base_learning_rate * 5, adaptive_lr))
        
        return adaptive_lr
    
    def _calculate_adaptive_patience(self):
        """Calculate adaptive early stopping patience based on training patterns"""
        
        # Assumption: Base patience on historical convergence patterns and remaining epochs
        base_patience = max(10, self.max_epochs // 10)
        
        # Analyze historical convergence patterns
        if len(self.chunk_statistics['epoch_history']) >= 10:
            recent_epochs = self.chunk_statistics['epoch_history'][-10:]
            improvements = [epoch.get('improvement_rate', 0) for epoch in recent_epochs]
            
            # If we're seeing consistent small improvements, be more patient
            small_improvements = [imp for imp in improvements if 0 < imp < 0.01]
            if len(small_improvements) >= 5:  # Many small improvements
                base_patience = int(base_patience * 1.5)
            
            # If we're seeing oscillating performance, be less patient
            sign_changes = sum(1 for i in range(1, len(improvements)) 
                             if improvements[i] * improvements[i-1] < 0)
            if sign_changes >= 6:  # High oscillation
                base_patience = int(base_patience * 0.8)
        
        # Consider remaining epochs
        remaining_epochs = self.max_epochs - self.chunk_statistics['current_epoch']
        adaptive_patience = min(base_patience, remaining_epochs // 2)
        
        return max(5, adaptive_patience)  # Minimum patience of 5
    
    def predict_fitness(self, features):
        """Predict fitness scores for given features"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        self.model.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).to(self.device)
            if len(features_tensor.shape) == 1:
                features_tensor = features_tensor.unsqueeze(0)
            
            predictions = self.model(features_tensor).cpu().numpy()
            
        return predictions.flatten()
    
    def suggest_features_for_fitness(self, target_fitness, num_features, 
                                   num_suggestions=10, max_iterations=5000, lr=0.003):
        """Use gradient descent to find features that produce target fitness"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making suggestions")
        
        # target_tensor = torch.FloatTensor([target_fitness]).to(self.device)
        target_tensor = target_fitness.reshape(1, -1).to(self.device)

        best_features = []
        best_losses = []
        
        for _ in range(num_suggestions):
            # Initialize random features (no range constraint)
            features = torch.randn(1, num_features, device=self.device, requires_grad=True)

            optimizer = optim.Adam([features], lr=lr, weight_decay=1e-5)

            best_loss = float('inf')
            best_feature_set = None

            self.model.eval()
            for iteration in range(max_iterations):
                optimizer.zero_grad()
                
                # Predict fitness
                predicted_fitness = self.model(features)
                loss = nn.MSELoss()(predicted_fitness, target_tensor)
                
                loss.backward()
                optimizer.step()
                
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_feature_set = features.data.clone()
                
                # Early stopping if loss is very small
                if loss.item() < 1e-6 and best_feature_set is not None:
                    break
            
            best_features.append(best_feature_set.cpu().numpy().flatten())
            best_losses.append(best_loss)
        
        # Sort by loss (best first)
        sorted_indices = np.argsort(best_losses)
        suggested_features = [best_features[i] for i in sorted_indices]
        
        return suggested_features
    
    def save_model(self, filepath):
        """Save the trained model with comprehensive statistics"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'best_model_state': self.best_model_state,
            'training_history': self.training_history,
            'chunk_statistics': self.chunk_statistics,
            'max_epochs': self.max_epochs,
            'base_learning_rate': self.base_learning_rate,
            'is_trained': self.is_trained
        }, filepath)
    
    def load_model(self, filepath):
        """Load a trained model with statistics"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_model_state = checkpoint.get('best_model_state', None)
        self.training_history = checkpoint.get('training_history', {})
        self.chunk_statistics = checkpoint.get('chunk_statistics', self._initialize_chunk_statistics())
        self.max_epochs = checkpoint.get('max_epochs', 100)
        self.base_learning_rate = checkpoint.get('base_learning_rate', 0.001)
        self.is_trained = checkpoint['is_trained']
    
    def _initialize_chunk_statistics(self):
        """Initialize chunk statistics structure"""
        return {
            'current_epoch': 0,
            'current_chunk_in_epoch': 0,
            'epoch_history': [],
            'current_epoch_chunks': [],
            'best_epoch_performance': float('inf'),
            'best_epoch_idx': -1,
            'adaptive_phase': 'collection',
            'learning_rate_history': [],
            'gradient_norm_history': [],
            'early_stopping_patience': max(10, self.max_epochs // 10),
            'consecutive_bad_epochs': 0,
            'max_grad_norm': 1.0
        }
    
    def get_training_statistics(self):
        """Get comprehensive training statistics"""
        if not self.chunk_statistics['epoch_history']:
            return {"message": "No training history available"}
        
        stats = {
            'total_epochs_trained': len(self.chunk_statistics['epoch_history']),
            'current_adaptive_phase': self.chunk_statistics['adaptive_phase'],
            'best_epoch_performance': self.chunk_statistics['best_epoch_performance'],
            'best_epoch_idx': self.chunk_statistics['best_epoch_idx'],
            'consecutive_bad_epochs': self.chunk_statistics['consecutive_bad_epochs'],
            'current_early_stopping_patience': self.chunk_statistics['early_stopping_patience']
        }
        
        # Recent performance trends
        if len(self.chunk_statistics['epoch_history']) >= 5:
            recent_epochs = self.chunk_statistics['epoch_history'][-5:]
            recent_losses = [epoch['avg_val_loss'] for epoch in recent_epochs]
            recent_improvements = [epoch.get('improvement_rate', 0) for epoch in recent_epochs]
            
            stats.update({
                'recent_avg_loss': np.mean(recent_losses),
                'recent_loss_trend': recent_losses[-1] - recent_losses[0],
                'recent_avg_improvement_rate': np.mean(recent_improvements),
                'loss_stability': np.std(recent_losses)
            })
        
        return stats

# Example usage and demo
def generate_demo_data(n_samples=1000, n_features=10):
    """Generate synthetic data for demonstration"""
    np.random.seed(42)
    
    # Generate features (no range constraint)
    features = np.random.normal(0, 1, (n_samples, n_features))
    
    # Create a complex fitness function (you can modify this)
    fitness_scores = (
        np.sum(features**2, axis=1) +  # Quadratic term
        0.5 * np.sum(features[:, :3], axis=1) +  # Linear term for first 3 features
        0.3 * np.prod(features[:, 3:6], axis=1) +  # Interaction term
        0.1 * np.random.normal(0, 1, n_samples)  # Noise
    )
    
    return features, fitness_scores

def plot_training_history(train_losses, val_losses):
    """Plot training and validation losses"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.show()

# Demo usage
if __name__ == "__main__":
    # Generate demo data
    print("Generating demo data...")
    features, fitness_scores = generate_demo_data(n_samples=2000, n_features=8)
    
    # Create and train model with sample weights support
    print("Creating and training model with sample weights support...")
    predictor = FeatureFitnessPredictor(
        input_size=8, 
        hidden_sizes=[128, 64, 32], 
        max_epochs=50,
        base_learning_rate=0.001
    )
    
    # Generate sample weights for demonstration
    np.random.seed(42)
    sample_weights = np.random.uniform(0.5, 2.0, len(features))  # Random importance weights
    
    # Train with early stopping, sample weights, and better monitoring
    train_losses, val_losses = predictor.train_model(
        features, fitness_scores, 
        sample_weights=sample_weights,
        epochs=200, 
        batch_size=32,
        early_stopping_patience=20,
        max_grad_norm=1.0
    )
    
    # Plot enhanced training history
    predictor.plot_training_history(show_metrics=True)
    
    # Test predictions
    print("\nTesting predictions...")
    test_features = np.random.normal(0, 1, (5, 8))
    predicted_fitness = predictor.predict_fitness(test_features)
    print(f"Predicted fitness scores: {predicted_fitness}")
    
    # Suggest features for target fitness
    print("\nSuggesting features for target fitness...")
    target_fitness = 2.0
    
    # Simulate chunk-based training workflow with sample weights
    print("\n" + "="*60)
    print("DEMO: Adaptive Chunk-Based Training with Sample Weights")
    print("="*60)
    
    # Create new predictor for chunk training
    chunk_predictor = FeatureFitnessPredictor(
        input_size=8, 
        hidden_sizes=[128, 64, 32], 
        max_epochs=10,
        base_learning_rate=0.001
    )
    
    # Simulate multiple epochs of chunk training
    all_data = list(zip(features, fitness_scores, sample_weights))
    chunk_size = 200
    
    for epoch in range(5):  # Train for 5 epochs max
        print(f"\n--- Starting Epoch {epoch + 1} ---")
        
        # Shuffle data for this epoch (simulating randomized chunks)
        np.random.shuffle(all_data)
        
        # Process chunks in this epoch
        for chunk_start in range(0, len(all_data), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(all_data))
            chunk_data = all_data[chunk_start:chunk_end]
            
            if len(chunk_data) < 10:  # Skip very small chunks
                continue
                
            chunk_features = np.array([item[0] for item in chunk_data])
            chunk_fitness = np.array([item[1] for item in chunk_data])
            chunk_weights = np.array([item[2] for item in chunk_data])
            
            # Train on this chunk with sample weights
            chunk_predictor.train_chunk(
                chunk_features, chunk_fitness,
                batch_size=32,
                sample_weights=chunk_weights
            )
        
        # Finalize epoch and check if we should stop
        should_stop = chunk_predictor.chunk_epoch_finalize()
        if should_stop:
            break
    
    # Show training statistics
    print("\n" + "="*60)
    print("Final Training Statistics:")
    print("="*60)
    stats = chunk_predictor.get_training_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # Test the trained model
    print(f"\nTesting adaptive trained model...")
    test_features = np.random.normal(0, 1, (5, 8))
    predicted_fitness = chunk_predictor.predict_fitness(test_features)
    print(f"Predicted fitness scores: {predicted_fitness.round(3)}")
    
    # Save enhanced model with statistics
    chunk_predictor.save_model("adaptive_feature_fitness_model.pth")
    print(f"\nAdaptive model with statistics saved successfully!")
    suggested_features = predictor.suggest_features_for_fitness(
        target_fitness, num_features=8, num_suggestions=3
    )
    
    print(f"Target fitness: {target_fitness}")
    for i, features in enumerate(suggested_features):
        predicted = predictor.predict_fitness([features])[0]
        print(f"Suggestion {i+1}: Features = {features.round(3)}, Predicted fitness = {predicted:.3f}")
    
    # Demo comparison: Standard training vs Adaptive chunk training
    print("\n" + "="*60)
    print("COMPARISON: Standard Training vs Adaptive Chunk Training")
    print("="*60)
    
    # Standard training (for comparison)
    print("\nTraining standard model...")
    standard_predictor = FeatureFitnessPredictor(input_size=8, hidden_sizes=[128, 64, 32])
    standard_train_losses, standard_val_losses = standard_predictor.train_model(
        features, fitness_scores, 
        epochs=50, 
        batch_size=32,
        early_stopping_patience=10,
        max_grad_norm=1.0
    )
    
    # Compare final performance
    print(f"\n--- Performance Comparison ---")
    test_features = np.random.normal(0, 1, (100, 8))
    
    adaptive_predictions = predictor.predict_fitness(test_features)
    standard_predictions = standard_predictor.predict_fitness(test_features)
    
    # Calculate actual fitness for comparison
    actual_fitness = (
        np.sum(test_features**2, axis=1) +
        0.5 * np.sum(test_features[:, :3], axis=1) +
        0.3 * np.prod(test_features[:, 3:6], axis=1)
    )
    
    adaptive_mae = np.mean(np.abs(adaptive_predictions - actual_fitness))
    standard_mae = np.mean(np.abs(standard_predictions - actual_fitness))
    
    adaptive_r2 = r2_score(actual_fitness, adaptive_predictions)
    standard_r2 = r2_score(actual_fitness, standard_predictions)
    
    print(f"Adaptive Chunk Model - MAE: {adaptive_mae:.4f}, R²: {adaptive_r2:.4f}")
    print(f"Standard Model - MAE: {standard_mae:.4f}, R²: {standard_r2:.4f}")
    
    if adaptive_mae < standard_mae:
        print("✓ Adaptive chunk training achieved better performance!")
    else:
        print("→ Standard training performed better (may need more epochs/data)")
    
    # Save enhanced model with statistics
    predictor.save_model("adaptive_feature_fitness_model.pth")
    print(f"\nAdaptive model with statistics saved successfully!")