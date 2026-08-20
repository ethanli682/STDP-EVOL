import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import math
import gc
# from GA_utils_scalers import *
from typing import List, Tuple, Dict, Optional, Any
import time
import os
import pickle
import warnings

class FeatureFitnessDataset(Dataset):
    """Custom dataset for feature-fitness pairs
    
    Args:
        features: Input features array
        fitness_scores: Corresponding fitness scores
    """
    def __init__(self, features: np.ndarray, fitness_scores: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.fitness_scores = torch.FloatTensor(fitness_scores).unsqueeze(1)
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.fitness_scores[idx]

def build_fc_layers(input_size: int, hidden_sizes: List[int], dropout_rate: float) -> nn.Sequential:
    """Build fully connected layers dynamically
    
    Args:
        input_size: Size of input to first FC layer
        hidden_sizes: List of hidden layer sizes
        dropout_rate: Dropout probability
        
    Returns:
        Sequential module containing FC layers
    """
    layers = []
    prev_size = input_size
    
    for hidden_size in hidden_sizes:
        layers.extend([
            nn.Linear(prev_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ])
        prev_size = hidden_size
    
    # Output layer
    layers.append(nn.Linear(prev_size, 1))
    
    return nn.Sequential(*layers)

class ANNModel(nn.Module):
    """Standard feedforward neural network
    
    Args:
        input_size: Number of input features
        hidden_sizes: List of hidden layer sizes
        dropout_rate: Dropout probability (default: 0.2)
    """
    def __init__(self, input_size: int, hidden_sizes: List[int] = [128, 64, 32], dropout_rate: float = 0.2):
        super(ANNModel, self).__init__()
        self.network = build_fc_layers(input_size, hidden_sizes, dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class BaseCNNModel(nn.Module):
    """Base class for CNN models with shared functionality
    
    Args:
        input_size: Number of input features
        hidden_sizes: List of hidden layer sizes for FC layers
        conv_channels: List of convolutional channel sizes
        dropout_rate: Dropout probability
        mode: Either '1d' or '2d' for convolution type
    """
    def __init__(self, input_size: int, hidden_sizes: List[int], 
                 conv_channels: List[int], dropout_rate: float, mode: str):
        super(BaseCNNModel, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.conv_channels = conv_channels
        self.dropout_rate = dropout_rate
        self.mode = mode
        
        if mode == '1d':
            self.conv_layers = self._build_conv_layers_1d()
            self.flattened_size = self.conv_channels[-1] * self.input_size
        elif mode == '2d':
            self.height, self.width = self._find_best_2d_shape(input_size)
            self.pad_size = self.height * self.width - input_size
            self.conv_layers = self._build_conv_layers_2d()
            self.flattened_size = self.conv_channels[-1] * self.height * self.width
        else:
            raise ValueError(f"Mode must be '1d' or '2d', got {mode}")
        
        # Build fully connected layers
        self.fc_layers = build_fc_layers(self.flattened_size, hidden_sizes, dropout_rate)
    
    def _build_conv_layers_1d(self) -> nn.Sequential:
        """Build 1D convolutional layers"""
        layers = []
        in_channels = 1
        
        for out_channels in self.conv_channels:
            layers.extend([
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(out_channels),
                nn.Dropout(self.dropout_rate)
            ])
            in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def _build_conv_layers_2d(self) -> nn.Sequential:
        """Build 2D convolutional layers"""
        layers = []
        in_channels = 1
        
        for out_channels in self.conv_channels:
            layers.extend([
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels),
                nn.Dropout2d(self.dropout_rate)
            ])
            in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def _find_best_2d_shape(self, n: int) -> Tuple[int, int]:
        """Find best factorization of n into 2D grid
        
        Args:
            n: Number to factorize
            
        Returns:
            Tuple of (height, width) closest to square shape
        """
        if n == 1:
            return 1, 1
        
        # Find closest factors to square
        sqrt_n = int(math.sqrt(n))
        
        for i in range(sqrt_n, 0, -1):
            if n % i == 0:
                return i, n // i
        
        # If n is prime, use closest square
        sqrt_n = int(math.ceil(math.sqrt(n)))
        return sqrt_n, sqrt_n
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == '1d':
            return self._forward_1d(x)
        else:
            return self._forward_2d(x)
    
    def _forward_1d(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for 1D CNN"""
        # Reshape: (batch, features) -> (batch, 1, features)
        x = x.unsqueeze(1)
        
        # Apply convolutions
        x = self.conv_layers(x)
        
        # Flatten and apply FC layers
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        
        return x
    
    def _forward_2d(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for 2D CNN"""
        batch_size = x.size(0)
        
        # Pad if necessary
        if self.pad_size > 0:
            padding = torch.zeros(batch_size, self.pad_size, device=x.device)
            x = torch.cat([x, padding], dim=1)
        
        # Reshape: (batch, features) -> (batch, 1, height, width)
        x = x.view(batch_size, 1, self.height, self.width)
        
        # Apply convolutions
        x = self.conv_layers(x)
        
        # Flatten and apply FC layers
        x = x.view(batch_size, -1)
        x = self.fc_layers(x)
        
        return x

class CNN1DModel(BaseCNNModel):
    """1D CNN treating features as a sequence
    
    Args:
        input_size: Number of input features
        hidden_sizes: List of hidden layer sizes for FC layers
        conv_channels: List of convolutional channel sizes
        dropout_rate: Dropout probability
    """
    def __init__(self, input_size: int, hidden_sizes: List[int] = [128, 64, 32], 
                 conv_channels: List[int] = [32, 64, 128], dropout_rate: float = 0.2):
        super(CNN1DModel, self).__init__(input_size, hidden_sizes, conv_channels, dropout_rate, mode='1d')

class CNN2DModel(BaseCNNModel):
    """2D CNN reshaping features into a grid
    
    Args:
        input_size: Number of input features
        hidden_sizes: List of hidden layer sizes for FC layers
        conv_channels: List of convolutional channel sizes
        dropout_rate: Dropout probability
    """
    def __init__(self, input_size: int, hidden_sizes: List[int] = [128, 64, 32], 
                 conv_channels: List[int] = [32, 64, 128], dropout_rate: float = 0.2):
        super(CNN2DModel, self).__init__(input_size, hidden_sizes, conv_channels, dropout_rate, mode='2d')

class AutoEnsemblePredictor:
    """Auto-ensemble predictor that trains multiple architectures and selects the best
    
    Trains ANN, 1D CNN, and 2D CNN models in random order with early stopping.
    Automatically selects the best performing model and discards others to save memory.
    
    Args:
        input_size: Number of input features
        hidden_sizes: FC layer sizes. CNN conv channels use this in reverse order.
                      (default: [128, 64, 32])
        dropout_rate: Dropout probability (default: 0.2)
        device: Device to train on ('cpu' or 'cuda')
        
    Attributes:
        best_model: The best performing model after training
        best_model_name: Name of the best model ('ANN', 'CNN1D', or 'CNN2D')
        is_trained: Whether the ensemble has been trained
    """
    
    def __init__(self, input_size: int, hidden_sizes: List[int] = [128, 64, 32],
                 dropout_rate: float = 0.2, device: str = 'cpu'):
        self.device = device
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.conv_channels = list(reversed(hidden_sizes))  # Reverse for conv layers
        self.dropout_rate = dropout_rate
        self.is_trained = False

        print(f"AutoEnsemblePredictor initialized on device: {self.device}", flush=True)
        print(f"Input size: {self.input_size}, Hidden sizes: {self.hidden_sizes}, Dropout rate: {self.dropout_rate}\n", flush=True)
        
        # Initialize all models
        self.models = {
            'ANN': ANNModel(input_size, hidden_sizes=self.hidden_sizes, dropout_rate=dropout_rate).to(device),
            'CNN1D': CNN1DModel(input_size, hidden_sizes=self.hidden_sizes, conv_channels=self.conv_channels, dropout_rate=dropout_rate).to(device),
            'CNN2D': CNN2DModel(input_size, hidden_sizes=self.hidden_sizes, conv_channels=self.conv_channels, dropout_rate=dropout_rate).to(device)
        }
        
        self.best_model_name = None
        self.best_model = None
    
    def _calculate_early_stop_threshold(self, fitness_scores, 
                                        early_stop_threshold: float) -> Tuple[float, float]:
        """Calculate absolute RMSE threshold for early stopping
        
        Args:
            fitness_scores: Array or tensor of fitness scores
            early_stop_threshold: Relative threshold (e.g., 0.05 for 5%)
            
        Returns:
            Tuple of (fitness_range, absolute_threshold)
        """
        # Convert to numpy if it's a PyTorch tensor
        if isinstance(fitness_scores, torch.Tensor):
            fitness_scores_np = fitness_scores.cpu().numpy()
        else:
            fitness_scores_np = np.asarray(fitness_scores)
        
        fitness_range = np.max(fitness_scores_np) - np.min(fitness_scores_np)
        absolute_threshold = early_stop_threshold * fitness_range
        return fitness_range, absolute_threshold
    
    def _prepare_data(self, features, fitness_scores, 
                      batch_size: int, validation_split: float) -> Tuple[DataLoader, DataLoader]:
        """Prepare train and validation data loaders
        
        Args:
            features: Input features (numpy array or torch tensor)
            fitness_scores: Target fitness scores (numpy array or torch tensor)
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Convert to numpy if they're PyTorch tensors
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        else:
            features = np.asarray(features)
            
        if isinstance(fitness_scores, torch.Tensor):
            fitness_scores = fitness_scores.cpu().numpy()
        else:
            fitness_scores = np.asarray(fitness_scores)
        
        X_train, X_val, y_train, y_val = train_test_split(
            features, fitness_scores, test_size=validation_split, random_state=42
        )
        
        train_dataset = FeatureFitnessDataset(X_train, y_train)
        val_dataset = FeatureFitnessDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def _check_early_stopping(self, val_loss: float, fitness_range: float, 
                              absolute_threshold: float, model_name: str, 
                              early_stop_threshold: float, verbose: bool) -> bool:
        """Check if early stopping condition is met
        
        Args:
            val_loss: Validation loss to check
            fitness_range: Range of fitness values
            absolute_threshold: Absolute RMSE threshold
            model_name: Name of current model
            early_stop_threshold: Relative threshold percentage
            verbose: Whether to print messages
            
        Returns:
            True if early stopping should trigger, False otherwise
        """
        rmse = np.sqrt(val_loss)
        relative_error = rmse / fitness_range
        
        if verbose:
            print(f"{model_name} Final Validation Loss: {val_loss:.6f}", flush=True)
            print(f"{model_name} RMSE: {rmse:.6f} ({relative_error*100:.2f}% of range)", flush=True)
        
        if rmse < absolute_threshold:
            if verbose:
                print(f"\n{'🎯 EARLY STOPPING TRIGGERED! 🎯':^60}", flush=True)
                print(f"{model_name} achieved RMSE < {early_stop_threshold*100:.1f}% threshold", flush=True)
                print(f"Skipping remaining models to save time...", flush=True)
            return True
        
        return False
    
    def _train_models_ensemble(self, model_names: List[str], train_loader: DataLoader,
                               val_loader: DataLoader, epochs: int, learning_rate: float,
                               fitness_range: float, absolute_threshold: float,
                               early_stop_threshold: float, max_models_to_train: int,
                               verbose: bool, patience: int = 20,
                               max_epochs: int = 500) -> Tuple[Dict, Dict, bool]:
        """Train all models in ensemble with early stopping

        Args:
            model_names: List of model names in random order
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs per model
            learning_rate: Learning rate
            fitness_range: Range of fitness values
            absolute_threshold: Absolute RMSE threshold for early stopping
            early_stop_threshold: Relative threshold percentage
            max_models_to_train: Maximum number of models to train
            verbose: Whether to print progress
            patience: Stop if no val improvement for this many epochs
            max_epochs: Hard ceiling on total epochs

        Returns:
            Tuple of (results, all_histories, early_stopped)
        """
        results = {}
        all_histories = {}
        early_stopped = False
        models_trained_count = 0
        
        for model_name in model_names:
            model = self.models[model_name]
            
            # Check if we've already trained max number of models or if early stop triggered
            if models_trained_count >= max_models_to_train or early_stopped:
                results[model_name] = {'final_val_loss': None, 'skipped': True}
                if verbose and not early_stopped:
                     print(f"\n{model_name}: SKIPPED (reached max_models_to_train={max_models_to_train})")
                continue
            
            if verbose:
                print(f"\nTraining {model_name}...", flush=True)
                print(f"-" * 40, flush=True)
            
            # Train single model (with adaptive epochs and convergence stopping)
            val_loss, train_losses, val_losses = self._train_single_model(
                model, train_loader, val_loader, epochs, learning_rate, verbose,
                absolute_threshold=absolute_threshold,
                patience=patience, max_epochs=max_epochs
            )
            
            results[model_name] = {
                'final_val_loss': val_loss,
                'skipped': False
            }
            all_histories[model_name] = {'train_losses': train_losses, 'val_losses': val_losses}
            
            models_trained_count += 1
            
            # ALWAYS check early stopping after each model (including first!)
            early_stopped = self._check_early_stopping(
                val_loss, fitness_range, absolute_threshold,
                model_name, early_stop_threshold, verbose
            )
        
        return results, all_histories, early_stopped
    
    def _finalize_results(self, results: Dict, all_histories: Dict, 
                          early_stopped: bool) -> None:
        """Select best model and add detailed history
        
        Args:
            results: Results dictionary
            all_histories: History of all models
            early_stopped: Whether early stopping was triggered
        """
        # Select best model
        trained_models = {k: v for k, v in results.items() if not v['skipped']}
        if not trained_models:
             raise ValueError("No models were trained!")
             
        self.best_model_name = min(trained_models, key=lambda k: trained_models[k]['final_val_loss'])
        self.best_model = self.models[self.best_model_name]
        
        # Add detailed loss history ONLY for best model
        results[self.best_model_name]['train_losses'] = all_histories[self.best_model_name]['train_losses']
        results[self.best_model_name]['val_losses'] = all_histories[self.best_model_name]['val_losses']
        results['best_model'] = self.best_model_name
        results['early_stopped'] = early_stopped
    
    def _cleanup_memory(self, verbose: bool = True):
        """Delete non-best models and free memory
        
        Args:
            verbose: Whether to print cleanup progress
        """
        if verbose:
            print(f"\nCleaning up memory: Deleting non-best models...", flush=True)
        
        models_to_delete = [name for name in self.models.keys() if name != self.best_model_name]
        for model_name in models_to_delete:
            del self.models[model_name]
            if verbose:
                print(f"  Deleted {model_name}", flush=True)
        
        # Force garbage collection
        gc.collect()
        if self.device != 'cpu':
            torch.cuda.empty_cache()
        
        if verbose:
            print(f"Memory cleanup complete. Only {self.best_model_name} retained in memory.\n", flush=True)
    
    def train_model(self, features, fitness_scores,
                    allowed_architectures: List[str] = ['ANN', 'CNN1D', 'CNN2D'],
                    epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001,
                    validation_split: float = 0.2, early_stop_threshold: float = 0.05,
                    max_models_to_train: int = 3, verbose: bool = True,
                    patience: int = 20, max_epochs: int = 500) -> Tuple[Dict, float]:
        """Train models and select the best one with early stopping

        Trains models in random order up to max_models_to_train. Early stopping can
        trigger after ANY model (including the first) if RMSE threshold is met.
        Keeps only the best model in memory.
        
        Args:
            features: Input features array/tensor of shape (n_samples, n_features)
            fitness_scores: Target fitness scores array/tensor of shape (n_samples,)
            allowed_architectures: List of model names to choose from (default: ['ANN', 'CNN1D', 'CNN2D'])
            epochs: Number of training epochs per model (default: 100)
            batch_size: Batch size for training (default: 32)
            learning_rate: Learning rate for optimizer (default: 0.001)
            validation_split: Fraction of data for validation (default: 0.2)
            early_stop_threshold: Stop if RMSE < threshold * fitness_range (default: 0.05 = 5%)
            max_models_to_train: Maximum models to train (default: 3)
            verbose: Whether to print training progress (default: True)
            
        Returns:
            Tuple of (results_dict, best_model_loss) where:
                - results_dict contains final_val_loss for all models and train/val losses for best model
                - best_model_loss is the validation loss of the best model (float)
                
        Raises:
            ValueError: If allowed_architectures contains invalid model names.
        """
        
        # 1. Validate Architectures
        valid_keys = list(self.models.keys())
        selected_architectures = [m for m in allowed_architectures if m in valid_keys]
        
        if not selected_architectures:
            raise ValueError(f"No valid architectures selected. Choose from {valid_keys}")

        # Validate max_models_to_train against the selected architectures
        if max_models_to_train < 1:
            raise ValueError("max_models_to_train must be at least 1")
            
        if verbose:
            print(f"\n{'='*60}", flush=True)
            print(f"Training Auto-Ensemble", flush=True)
            print(f"Allowed Architectures: {selected_architectures}", flush=True)
            print(f"Max Models to Train: {max_models_to_train}", flush=True)
            print(f"Early stop threshold: {early_stop_threshold*100:.1f}% of fitness range", flush=True)
            print(f"{'='*60}\n", flush=True)
        
        # Calculate early stopping threshold
        fitness_range, absolute_threshold = self._calculate_early_stop_threshold(
            fitness_scores, early_stop_threshold
        )
        
        if verbose:
            # Convert for display if needed
            if isinstance(fitness_scores, torch.Tensor):
                fs_np = fitness_scores.cpu().numpy()
            else:
                fs_np = np.asarray(fitness_scores)
            print(f"Fitness range: [{np.min(fs_np):.3f}, {np.max(fs_np):.3f}]", flush=True)
            print(f"Absolute RMSE threshold: {absolute_threshold:.6f}\n", flush=True)
        
        # Prepare data loaders
        train_loader, val_loader = self._prepare_data(
            features, fitness_scores, batch_size, validation_split
        )
        
        # Randomize model training order based ONLY on selected architectures
        np.random.shuffle(selected_architectures)
        
        if verbose:
            print(f"Training order (randomized): {' → '.join(selected_architectures)}", flush=True)
        
        # Train models with early stopping
        results, all_histories, early_stopped = self._train_models_ensemble(
            selected_architectures, train_loader, val_loader, epochs, learning_rate,
            fitness_range, absolute_threshold, early_stop_threshold,
            max_models_to_train, verbose, patience=patience, max_epochs=max_epochs
        )
        
        # Finalize results and select best model
        self._finalize_results(results, all_histories, early_stopped)
        
        # Memory cleanup
        self._cleanup_memory(verbose)
        
        if verbose:
            print(f"\n{'='*60}", flush=True)
            print(f"Best Model: {self.best_model_name} (Val Loss: {results[self.best_model_name]['final_val_loss']:.6f})", flush=True)
            print(f"{'='*60}\n", flush=True)
        
        self.is_trained = True
        
        # Return results dict and best model loss
        best_model_loss = results[self.best_model_name]['final_val_loss']
        return results, best_model_loss
    
    def _train_single_model(self, model: nn.Module, train_loader: DataLoader,
                          val_loader: DataLoader, epochs: int, learning_rate: float,
                          verbose: bool = True, absolute_threshold: float = None,
                          patience: int = 20, max_epochs: int = 500) -> Tuple[float, List[float], List[float]]:
        """Train a single model with adaptive epoch count

        Training automatically:
        - Stops early if validation loss plateaus (patience-based)
        - Stops early if RMSE drops below absolute_threshold (convergence)
        - Extends beyond initial epochs (up to max_epochs) if still improving

        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Initial number of epochs (will extend if still improving)
            learning_rate: Learning rate
            verbose: Whether to print progress
            absolute_threshold: RMSE threshold for convergence stop (None = disabled)
            patience: Stop if no val improvement for this many epochs
            max_epochs: Hard ceiling on total epochs

        Returns:
            Tuple of (final_val_loss, train_losses, val_losses)
        """
        if max_epochs < epochs:
            max_epochs = epochs

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        train_losses = []
        val_losses = []

        best_val_loss = float('inf')
        best_epoch = 0
        best_state = None

        effective_max = max_epochs
        stop_reason = None

        model.train()
        epoch = 0
        while epoch < effective_max:
            epoch_train_loss = 0

            for batch_features, batch_fitness in train_loader:
                batch_features = batch_features.to(self.device)
                batch_fitness = batch_fitness.to(self.device)

                optimizer.zero_grad()
                predictions = model(batch_features)
                loss = criterion(predictions, batch_fitness)
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item()

            # Validation
            model.eval()
            epoch_val_loss = 0
            with torch.no_grad():
                for batch_features, batch_fitness in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_fitness = batch_fitness.to(self.device)
                    predictions = model(batch_features)
                    loss = criterion(predictions, batch_fitness)
                    epoch_val_loss += loss.item()

            avg_train_loss = epoch_train_loss / len(train_loader)
            avg_val_loss = epoch_val_loss / len(val_loader)

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            scheduler.step(avg_val_loss)

            # Track best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            if verbose and (epoch + 1) % 20 == 0:
                # Show initial target until we extend, then show max
                display_total = max_epochs if epoch >= epochs else epochs
                print(f'Epoch [{epoch+1}/{display_total}], Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}', flush=True)

            # Check convergence: RMSE below threshold
            if absolute_threshold is not None:
                rmse = np.sqrt(avg_val_loss)
                if rmse < absolute_threshold:
                    stop_reason = "convergence"
                    epoch += 1
                    break

            # Check patience: no improvement for too long
            # Only activate after the initial epoch count is reached (0-based: epoch >= epochs)
            epochs_since_best = epoch - best_epoch
            if epochs_since_best >= patience and epoch >= epochs:
                stop_reason = "patience"
                epoch += 1
                break

            # Auto-extend notification: at the initial epoch limit but still improving
            if epoch == epochs - 1 and epochs_since_best < patience:
                if verbose:
                    print(f'  [Auto-extend] Still improving at epoch {epoch+1}, '
                          f'extending training (patience={patience}, max={max_epochs})', flush=True)

            model.train()
            epoch += 1

        if stop_reason is None:
            stop_reason = "max_epochs"

        # Restore best model weights
        if best_state is not None:
            model.load_state_dict(best_state)

        if verbose and epoch > epochs:
            print(f'  [Adaptive] Stopped at epoch {epoch}/{max_epochs} ({stop_reason}), best at epoch {best_epoch+1}', flush=True)
        elif verbose and stop_reason in ("convergence", "patience") and epoch < epochs:
            print(f'  [Early stop] Stopped at epoch {epoch}/{epochs} ({stop_reason}), best at epoch {best_epoch+1}', flush=True)

        return best_val_loss, train_losses, val_losses
    
    def predict_fitness(self, features) -> np.ndarray:
        """Predict fitness scores using the best model
        
        Args:
            features: Input features array/tensor of shape (n_samples, n_features) or (n_features,)
            
        Returns:
            Predicted fitness scores array of shape (n_samples,)
            
        Raises:
            ValueError: If model hasn't been trained yet
            
        Note:
            Accepts both NumPy arrays and PyTorch tensors for features
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        self.best_model.eval()
        with torch.no_grad():
            # Convert to numpy first if it's a tensor
            if isinstance(features, torch.Tensor):
                features = features.cpu().numpy()
            else:
                features = np.asarray(features)
            
            features_tensor = torch.FloatTensor(features).to(self.device)
            if len(features_tensor.shape) == 1:
                features_tensor = features_tensor.unsqueeze(0)
            
            predictions = self.best_model(features_tensor).cpu().numpy()
            
        return predictions.flatten()
    
    def suggest_features_for_fitness(self, target_fitness: float, num_features: int, 
                                   num_suggestions: int = 10, max_iterations: int = 1000, 
                                   lr: float = 0.01, verbose: bool = True) -> List[np.ndarray]:
        """Use gradient descent to find features that produce target fitness
        
        Uses backpropagation through the trained model to find feature vectors
        that yield the desired fitness score. Performs multiple random initializations
        and returns the best suggestions sorted by how close they are to the target.
        
        Args:
            target_fitness: Desired fitness score
            num_features: Number of features (must match model input size)
            num_suggestions: Number of feature suggestions to generate (default: 10)
            max_iterations: Maximum optimization iterations per suggestion (default: 1000)
            lr: Learning rate for gradient descent (default: 0.01)
            verbose: Whether to print progress (default: True)
            
        Returns:
            List of feature arrays, sorted from best to worst match
            
        Raises:
            ValueError: If model hasn't been trained yet
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making suggestions")
        
        if verbose:
            print(f"\nGenerating {num_suggestions} feature suggestions for target fitness: {target_fitness}", flush=True)
            print(f"Using best model: {self.best_model_name}", flush=True)
        
        target_tensor = torch.FloatTensor([target_fitness]).to(self.device)
        
        best_features = []
        best_losses = []
        
        for suggestion_idx in range(num_suggestions):
            # Initialize random features in [-1, 1] range
            features = torch.randn(1, num_features, device=self.device, requires_grad=True)
            features.data = torch.clamp(features.data, -1, 1)
            
            optimizer = optim.Adam([features], lr=lr)
            
            best_loss = float('inf')
            best_feature_set = None
            
            for iteration in range(max_iterations):
                optimizer.zero_grad()
                
                predicted_fitness = self.best_model(features)
                loss = nn.MSELoss()(predicted_fitness, target_tensor)
                
                loss.backward()
                optimizer.step()
                
                # Clamp features to [-1, 1] range
                with torch.no_grad():
                    features.data = torch.clamp(features.data, -1, 1)
                
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_feature_set = features.data.clone()
                
                # Early stopping
                if loss.item() < 1e-6:
                    break
            
            best_features.append(best_feature_set.cpu().numpy().flatten())
            best_losses.append(best_loss)
            
            if verbose and (suggestion_idx + 1) % 5 == 0:
                print(f"Generated {suggestion_idx + 1}/{num_suggestions} suggestions...", flush=True)
        
        # Sort by loss (best first)
        sorted_indices = np.argsort(best_losses)
        suggested_features = [best_features[i] for i in sorted_indices]
        
        return suggested_features


    # ============================Chunks===========================================
    # Helper Methods (to be added to AutoEnsemblePredictor class)
    # =============================================================================

    def _initialize_models_for_chunk_training(self, model_names: List[str]) -> None:
        """Re-initialize specified models with fresh random weights.
        
        Args:
            model_names: List of model names to initialize
        """
        for name in model_names:
            if name == 'ANN':
                self.models[name] = ANNModel(
                    self.input_size, 
                    hidden_sizes=self.hidden_sizes, 
                    dropout_rate=self.dropout_rate
                ).to(self.device)
            elif name == 'CNN1D':
                self.models[name] = CNN1DModel(
                    self.input_size, 
                    hidden_sizes=self.hidden_sizes, 
                    conv_channels=self.conv_channels, 
                    dropout_rate=self.dropout_rate
                ).to(self.device)
            elif name == 'CNN2D':
                self.models[name] = CNN2DModel(
                    self.input_size, 
                    hidden_sizes=self.hidden_sizes, 
                    conv_channels=self.conv_channels, 
                    dropout_rate=self.dropout_rate
                ).to(self.device)


    def _setup_chunk_training_state(self, savePath: str, agent_idx: int,
                                    model_names: List[str], epochs: int,
                                    max_models_to_train: int, validation_split: float,
                                    batch_size: int, learning_rate: float,
                                    early_stop_threshold: float, verbose: bool,
                                    use_validation: bool = False,
                                    gradient_clip: Optional[float] = None,
                                    patience: int = 20,
                                    max_epochs: int = 500) -> None:
        """Initialize the chunk training state dictionary.

        Args:
            savePath: Path to save temporary files
            agent_idx: Agent identifier for temp file naming
            model_names: List of model names to train
            epochs: Number of epochs to train (initial target, may extend)
            max_models_to_train: Maximum number of models to train
            validation_split: Fraction of data for validation
            batch_size: Batch size for mini-batch SGD
            learning_rate: Learning rate for optimizers
            early_stop_threshold: Early stopping threshold
            verbose: Whether to print progress
            use_validation: Whether to use validation
            gradient_clip: Max norm for gradient clipping (None = no clipping)
            patience: Stop if no val improvement for this many epochs
            max_epochs: Hard ceiling on total epochs (for auto-extend)
        """
        if max_epochs < epochs:
            max_epochs = epochs

        # Create temp file path
        running_dir = os.path.join(savePath, 'running')
        os.makedirs(running_dir, exist_ok=True)
        temp_file_path = os.path.join(running_dir, f'{str(agent_idx)}_AutoEnsemblePredictor_metadata_.pkl')

        # Check for existing temp file
        if os.path.exists(temp_file_path):
            warnings.warn(f"Existing temp file found at {temp_file_path}. Deleting and starting fresh.")
            os.remove(temp_file_path)
        
        # Initialize optimizers and schedulers for each model
        optimizers = {}
        schedulers = {}
        for name in model_names:
            optimizers[name] = optim.Adam(
                self.models[name].parameters(), 
                lr=learning_rate, 
                weight_decay=1e-5
            )
            schedulers[name] = optim.lr_scheduler.ReduceLROnPlateau(
                optimizers[name], patience=10, factor=0.5
            )
        
        # Initialize per-model tracking
        total_samples = {name: 0 for name in model_names}
        epoch_train_loss_sums = {name: 0.0 for name in model_names}
        epoch_batch_counts = {name: 0 for name in model_names}
        train_losses = {name: [] for name in model_names}
        val_losses = {name: [] for name in model_names}
        
        self._chunk_state = {
            # Configuration
            'savePath': savePath,
            'agent_idx': agent_idx,
            'temp_file_path': temp_file_path,
            'validation_split': validation_split,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'early_stop_threshold': early_stop_threshold,
            'epochs': epochs,
            'max_models_to_train': max_models_to_train,
            'verbose': verbose,
            'use_validation': use_validation,
            'gradient_clip': gradient_clip,
            
            # Model management
            'model_names': model_names,
            
            # Per-model training state
            'optimizers': optimizers,
            'schedulers': schedulers,
            'criterion': nn.MSELoss(),
            'total_samples': total_samples,
            'epoch_train_loss_sums': epoch_train_loss_sums,
            'epoch_batch_counts': epoch_batch_counts,
            
            # Per-model history
            'train_losses': train_losses,
            'val_losses': val_losses,
            
            # Global tracking
            'current_epoch': 1,
            'fitness_min': float('inf'),
            'fitness_max': float('-inf'),
            
            # Validation fitness IDs (for stable validation across epochs)
            'validation_fitness_ids': set(),  # Set of float32 fitness values
            'validation_data_collected': False,
            
            # Results
            'results': {},
            'all_histories': {},
            'early_stopped': False,
            'training_active': True,

            # Adaptive training state
            'patience': patience,
            'max_epochs': max_epochs,
            'initial_epochs': epochs,
            'best_val_loss': {name: float('inf') for name in model_names},
            'best_epoch': {name: 1 for name in model_names},  # 1-based to match current_epoch
            'best_model_states': {name: None for name in model_names},
        }


    def _is_validation_sample(self, fitness_value: float) -> bool:
        """Check if a sample belongs to validation set based on fitness ID.
        
        Args:
            fitness_value: The fitness score to check
            
        Returns:
            True if this sample should be in validation set
        """
        if not self._chunk_state.get('use_validation', False):
            return False
        return fitness_value in self._chunk_state['validation_fitness_ids']


    def _add_validation_fitness_id(self, fitness_value: float) -> None:
        """Add a fitness value to the validation ID set.
        
        Args:
            fitness_value: The fitness score to add
        """
        self._chunk_state['validation_fitness_ids'].add(fitness_value)


    def _get_validation_memory_usage(self) -> int:
        """Get approximate memory used by validation fitness IDs in bytes."""
        # Each float in a Python set uses roughly 28 bytes (object overhead)
        # Plus the set's internal hash table overhead
        n_ids = len(self._chunk_state.get('validation_fitness_ids', set()))
        return n_ids * 32  # Conservative estimate


    def _save_validation_data(self, val_features: np.ndarray, val_fitness: np.ndarray) -> None:
        """Append validation data to temp file.
        
        Args:
            val_features: Validation features to save
            val_fitness: Validation fitness scores to save
        """
        temp_file_path = self._chunk_state['temp_file_path']
        
        # Load existing data if file exists
        if os.path.exists(temp_file_path):
            with open(temp_file_path, 'rb') as f:
                data = pickle.load(f)
            data['val_features'].append(val_features)
            data['val_fitness'].append(val_fitness)
        else:
            data = {
                'val_features': [val_features],
                'val_fitness': [val_fitness]
            }
        
        # Save updated data
        with open(temp_file_path, 'wb') as f:
            pickle.dump(data, f)


    def _load_validation_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and concatenate all validation data from temp file.
        
        Returns:
            Tuple of (val_features, val_fitness) concatenated arrays
        """
        temp_file_path = self._chunk_state['temp_file_path']
        
        if not os.path.exists(temp_file_path):
            return np.array([]), np.array([])
        
        with open(temp_file_path, 'rb') as f:
            data = pickle.load(f)
        
        if not data['val_features']:
            return np.array([]), np.array([])
        
        val_features = np.concatenate(data['val_features'], axis=0)
        val_fitness = np.concatenate(data['val_fitness'], axis=0)
        
        return val_features, val_fitness


    def _clear_validation_data(self) -> None:
        """Clear the validation temp file for next epoch."""
        temp_file_path = self._chunk_state['temp_file_path']
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


    def _compute_validation_loss(self, model_name: str, val_features: np.ndarray, 
                                val_fitness: np.ndarray) -> float:
        """Compute validation loss for a model in batches.
        
        Args:
            model_name: Name of model to validate
            val_features: Validation features
            val_fitness: Validation fitness scores
            
        Returns:
            Average validation loss
        """
        model = self.models[model_name]
        criterion = self._chunk_state['criterion']
        batch_size = self._chunk_state['batch_size']
        
        model.eval()
        
        # Convert to tensors
        val_features_t = torch.FloatTensor(val_features).to(self.device)
        val_fitness_t = torch.FloatTensor(val_fitness).unsqueeze(1).to(self.device)
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for start_idx in range(0, len(val_features), batch_size):
                end_idx = min(start_idx + batch_size, len(val_features))
                batch_features = val_features_t[start_idx:end_idx]
                batch_fitness = val_fitness_t[start_idx:end_idx]
                
                predictions = model(batch_features)
                loss = criterion(predictions, batch_fitness)
                total_loss += loss.item()
                num_batches += 1
        
        model.train()
        
        # Cleanup
        del val_features_t, val_fitness_t
        
        return total_loss / num_batches if num_batches > 0 else 0.0


    def _reset_epoch_accumulators(self) -> None:
        """Reset per-epoch accumulators for all models."""
        for name in self._chunk_state['model_names']:
            self._chunk_state['total_samples'][name] = 0
            self._chunk_state['epoch_train_loss_sums'][name] = 0.0
            self._chunk_state['epoch_batch_counts'][name] = 0


    def _finalize_chunk_training_results(self) -> None:
        """Finalize results and select best model after training completes."""
        model_names = self._chunk_state['model_names']
        val_losses = self._chunk_state['val_losses']
        train_losses = self._chunk_state['train_losses']
        use_validation = self._chunk_state.get('use_validation', False)
        
        # Build results dict
        # Use the tracked best loss (from adaptive training) rather than the
        # last epoch's loss, since best model weights have already been restored.
        best_val_losses = self._chunk_state.get('best_val_loss', {})
        results = {}
        for name in model_names:
            # Use val_losses if validation enabled, otherwise train_losses
            loss_list = val_losses[name] if use_validation and val_losses[name] else train_losses[name]
            if loss_list:
                # Prefer tracked best loss; fall back to last epoch if unavailable
                tracked_best = best_val_losses.get(name, float('inf'))
                final_loss = tracked_best if tracked_best < float('inf') else loss_list[-1]
                results[name] = {
                    'final_val_loss': final_loss,
                    'skipped': False,
                    'train_losses': train_losses[name],
                    'val_losses': val_losses[name] if use_validation else train_losses[name]
                }
            else:
                results[name] = {
                    'final_val_loss': None,
                    'skipped': True
                }

        # Select best model
        trained_models = {k: v for k, v in results.items() if not v['skipped'] and v['final_val_loss'] is not None}

        if trained_models:
            self.best_model_name = min(trained_models, key=lambda k: trained_models[k]['final_val_loss'])
            self.best_model = self.models[self.best_model_name]
            results['best_model'] = self.best_model_name
            results['early_stopped'] = self._chunk_state['early_stopped']
            self.is_trained = True
        
        self._chunk_state['results'] = results


    # =============================================================================
    # Main Chunked Training Methods
    # =============================================================================

    def start_chunk_training(self,
                            savePath: str,
                            agent_idx: int,
                            allowed_architectures: List[str] = ['ANN', 'CNN1D', 'CNN2D'],
                            epochs: int = 100,
                            max_models_to_train: int = 3,
                            validation_split: float = 0.2,
                            batch_size: int = 32,
                            learning_rate: float = 0.001,
                            early_stop_threshold: float = 0.05,
                            reinitialize_models: bool = True,
                            verbose: bool = True,
                            use_validation: bool = False,
                            gradient_clip: Optional[float] = None,
                            patience: int = 20,
                            max_epochs: int = 500) -> None:
        """Initialize chunked training state. Must be called first.

        Sets up the training state for processing data in chunks. After calling this,
        use process_chunk() to feed data and finish_chunk_epoch() to complete epochs.

        Training automatically:
        - Stops early if validation loss plateaus (patience-based)
        - Stops early if RMSE drops below threshold (convergence)
        - Extends beyond initial epochs (up to max_epochs) if still improving

        Args:
            savePath: Directory path for saving temporary files
            agent_idx: Agent identifier used in temp file naming
            allowed_architectures: List of model architectures to train
                                (default: ['ANN', 'CNN1D', 'CNN2D'])
            epochs: Initial number of epochs (will extend if still improving) (default: 100)
            max_models_to_train: Maximum number of models to include in training
                                (randomly selected from allowed_architectures) (default: 3)
            validation_split: Fraction of each chunk to use for validation (default: 0.2)
            batch_size: Batch size for mini-batch SGD (default: 32)
            learning_rate: Learning rate for optimizer (default: 0.001)
            early_stop_threshold: Stop if RMSE < threshold * fitness_range (default: 0.05)
            reinitialize_models: Whether to reset models to fresh random weights (default: True)
            verbose: Whether to print progress information (default: True)
            use_validation: Whether to use validation set (default: False)
            gradient_clip: Max norm for gradient clipping, None = disabled (default: None)
            patience: Stop if no val improvement for this many epochs (default: 20)
            max_epochs: Hard ceiling on total epochs for auto-extend (default: 500)
            
        Raises:
            ValueError: If no valid architectures are provided
        """
        # Validate architectures
        valid_keys = list(self.models.keys()) if self.models else ['ANN', 'CNN1D', 'CNN2D']
        selected_architectures = [m for m in allowed_architectures if m in valid_keys or m in ['ANN', 'CNN1D', 'CNN2D']]
        
        if not selected_architectures:
            raise ValueError(f"No valid architectures selected. Choose from ['ANN', 'CNN1D', 'CNN2D']")
        
        # Limit to max_models_to_train (random selection)
        if len(selected_architectures) > max_models_to_train:
            np.random.shuffle(selected_architectures)
            selected_architectures = selected_architectures[:max_models_to_train]
        
        # Re-initialize models if requested
        if reinitialize_models:
            self._initialize_models_for_chunk_training(selected_architectures)
        else:
            # Ensure models exist
            missing = [m for m in selected_architectures if m not in self.models]
            if missing:
                self._initialize_models_for_chunk_training(missing)
        
        # Setup training state
        self._setup_chunk_training_state(
            savePath=savePath,
            agent_idx=agent_idx,
            model_names=selected_architectures,
            epochs=epochs,
            max_models_to_train=max_models_to_train,
            validation_split=validation_split,
            batch_size=batch_size,
            learning_rate=learning_rate,
            early_stop_threshold=early_stop_threshold,
            verbose=verbose,
            use_validation=use_validation,
            gradient_clip=gradient_clip,
            patience=patience,
            max_epochs=max_epochs,
        )
        
        # Set models to training mode
        for name in selected_architectures:
            self.models[name].train()
        
        if verbose:
            print(f"\n{'='*60}", flush=True)
            print(f"Chunked Training Initialized (Mini-batch SGD)", flush=True)
            print(f"Models: {selected_architectures}", flush=True)
            print(f"Epochs: {epochs}", flush=True)
            print(f"Batch size: {batch_size}", flush=True)
            print(f"Validation: {'Enabled' if use_validation else 'Disabled'}", flush=True)
            print(f"Gradient clipping: {gradient_clip if gradient_clip else 'Disabled'}", flush=True)
            print(f"Early stop threshold: {early_stop_threshold*100:.1f}% of fitness range", flush=True)
            print(f"{'='*60}\n", flush=True)


    def process_chunk(self, 
                    features: np.ndarray, 
                    fitness_scores: np.ndarray,
                    sample_weights: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Process one data chunk using mini-batch SGD with optional sample weights.
        
        Performs mini-batch stochastic gradient descent on the chunk:
        - Splits chunk into mini-batches
        - For each mini-batch: forward → backward → optimizer.step()
        - If validation enabled and epoch 1: collect validation fitness IDs
        - If validation enabled and epoch 2+: separate train/val by fitness ID
        
        Args:
            features: Input features array of shape (n_samples, n_features)
            fitness_scores: Target fitness scores array of shape (n_samples,)
                        Should already be scaled!
            sample_weights: Optional weights for each sample of shape (n_samples,)
                        Range [0.01, 1.0] from _scaler_weight. Default: uniform.
            
        Returns:
            Dictionary mapping model names to their average loss for this chunk.
            Returns dict with 0.0 if input is empty.
            
        Raises:
            ValueError: If start_chunk_training() hasn't been called
        """
        # Check training is active
        if not hasattr(self, '_chunk_state') or not self._chunk_state.get('training_active', False):
            raise ValueError("Must call start_chunk_training() before process_chunk()")
        
        # Convert inputs to numpy if needed
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        else:
            features = np.asarray(features)
            
        if isinstance(fitness_scores, torch.Tensor):
            fitness_scores = fitness_scores.cpu().numpy()
        else:
            fitness_scores = np.asarray(fitness_scores)
        
        if sample_weights is not None:
            if isinstance(sample_weights, torch.Tensor):
                sample_weights = sample_weights.cpu().numpy()
            else:
                sample_weights = np.asarray(sample_weights)
        
        # Handle empty input
        if len(features) == 0:
            return {name: 0.0 for name in self._chunk_state['model_names']}
        
        # Update fitness range tracking (for early stopping)
        self._chunk_state['fitness_min'] = min(self._chunk_state['fitness_min'], np.min(fitness_scores))
        self._chunk_state['fitness_max'] = max(self._chunk_state['fitness_max'], np.max(fitness_scores))
        
        # Get training state
        use_validation = self._chunk_state.get('use_validation', False)
        current_epoch = self._chunk_state['current_epoch']
        validation_split = self._chunk_state['validation_split']
        
        # Split into train/validation based on epoch and validation mode
        if use_validation:
            if current_epoch == 1:
                # Epoch 1: Random split, collect validation fitness IDs
                n_samples = len(features)
                n_val = max(1, int(n_samples * validation_split))
                
                indices = np.arange(n_samples)
                np.random.shuffle(indices)
                val_indices = indices[:n_val]
                train_indices = indices[n_val:]
                
                # Save validation fitness IDs for future epochs
                for idx in val_indices:
                    self._add_validation_fitness_id(float(fitness_scores[idx]))
                
                train_features = features[train_indices]
                train_fitness = fitness_scores[train_indices]
                train_weights = sample_weights[train_indices] if sample_weights is not None else None
                
                val_features = features[val_indices]
                val_fitness = fitness_scores[val_indices]
                
                # Save validation data to temp file
                if len(val_features) > 0:
                    self._save_validation_data(val_features, val_fitness)
                    
            else:
                # Epoch 2+: Use fitness IDs to identify validation samples
                validation_ids = self._chunk_state['validation_fitness_ids']
                
                train_mask = np.array([float(f) not in validation_ids for f in fitness_scores])
                val_mask = ~train_mask
                
                train_features = features[train_mask]
                train_fitness = fitness_scores[train_mask]
                train_weights = sample_weights[train_mask] if sample_weights is not None else None
                
                val_features = features[val_mask]
                val_fitness = fitness_scores[val_mask]
                
                # Save validation data to temp file
                if len(val_features) > 0:
                    self._save_validation_data(val_features, val_fitness)
        else:
            # No validation - use all data for training
            train_features = features
            train_fitness = fitness_scores
            train_weights = sample_weights
        
        # Handle empty training set (all went to validation)
        if len(train_features) == 0:
            return {name: 0.0 for name in self._chunk_state['model_names']}
        
        # Prepare weights - normalize so mean = 1.0 (preserves loss magnitude)
        if train_weights is not None:
            train_weights = train_weights.astype(np.float32)
            weight_mean = train_weights.mean()
            if weight_mean > 1e-8:
                train_weights = train_weights / weight_mean
            else:
                train_weights = np.ones_like(train_weights)
        
        # Create dataset with optional weights
        train_features_tensor = torch.FloatTensor(train_features)
        train_fitness_tensor = torch.FloatTensor(train_fitness).unsqueeze(1)
        if train_weights is not None:
            train_weights_tensor = torch.FloatTensor(train_weights)
            train_dataset = torch.utils.data.TensorDataset(
                train_features_tensor, train_fitness_tensor, train_weights_tensor
            )
            use_weights = True
        else:
            train_dataset = torch.utils.data.TensorDataset(
                train_features_tensor, train_fitness_tensor
            )
            use_weights = False
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self._chunk_state['batch_size'], 
            shuffle=True,
            drop_last=False  # Keep leftover samples
        )
        
        criterion = self._chunk_state['criterion']
        gradient_clip = self._chunk_state.get('gradient_clip', None)
        losses = {}
        
        # Process each model with mini-batch SGD
        for model_name in self._chunk_state['model_names']:
            model = self.models[model_name]
            optimizer = self._chunk_state['optimizers'][model_name]
            model.train()
            
            chunk_loss_sum = 0.0
            chunk_batch_count = 0
            
            # Mini-batch SGD within this chunk
            for batch_data in train_loader:
                if use_weights:
                    batch_features, batch_fitness, batch_weights = batch_data
                    batch_weights = batch_weights.to(self.device)
                else:
                    batch_features, batch_fitness = batch_data
                    batch_weights = None
                
                batch_features = batch_features.to(self.device)
                batch_fitness = batch_fitness.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                predictions = model(batch_features)
                
                if batch_weights is not None:
                    # Weighted MSE loss
                    per_sample_loss = (predictions - batch_fitness) ** 2
                    loss = (per_sample_loss.squeeze() * batch_weights).mean()
                else:
                    # Standard MSE loss
                    loss = criterion(predictions, batch_fitness)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping (if enabled)
                if gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
                
                # Optimizer step
                optimizer.step()
                
                chunk_loss_sum += loss.item()
                chunk_batch_count += 1
            
            # Track statistics
            avg_chunk_loss = chunk_loss_sum / chunk_batch_count if chunk_batch_count > 0 else 0.0
            self._chunk_state['epoch_train_loss_sums'][model_name] += chunk_loss_sum
            self._chunk_state['epoch_batch_counts'][model_name] += chunk_batch_count
            self._chunk_state['total_samples'][model_name] += len(train_features)
            
            losses[model_name] = avg_chunk_loss
        
        return losses


    def finish_chunk_epoch(self) -> Tuple[bool, Dict]:
        """Complete the current epoch: compute validation/training loss, check for completion.
        
        Returns:
            Tuple of (training_complete, info_dict):
            - training_complete (bool): True if all training is finished
            - info_dict (Dict): Contains:
                - 'current_epoch': Current epoch number
                - 'total_epochs': Total epochs to train
                - 'models': Dict[str, {'train_loss': float, 'val_loss': float}]
                - 'best_model_so_far': Name of best model based on loss
                - 'best_val_loss_so_far': Best loss so far
                - 'early_stopped': bool (only if training_complete)
                - 'results': Full results dict (only if training_complete)
                - 'best_loss': Best model's final loss (only if training_complete)
                
        Raises:
            ValueError: If start_chunk_training() hasn't been called
            ValueError: If no chunks were processed this epoch
        """
        # Check training is active
        if not hasattr(self, '_chunk_state') or not self._chunk_state.get('training_active', False):
            raise ValueError("Must call start_chunk_training() before finish_chunk_epoch()")
        
        model_names = self._chunk_state['model_names']
        use_validation = self._chunk_state.get('use_validation', False)
        
        # Check that chunks were processed
        total_batches = sum(self._chunk_state['epoch_batch_counts'].values())
        if total_batches == 0:
            raise ValueError("No chunks were processed this epoch. Call process_chunk() first.")
        
        verbose = self._chunk_state['verbose']
        current_epoch = self._chunk_state['current_epoch']
        total_epochs = self._chunk_state['epochs']
        
        if verbose:
            print(f"\n--- Epoch {current_epoch}/{total_epochs} ---", flush=True)
        
        # Load validation data if validation is enabled
        if use_validation:
            val_features, val_fitness = self._load_validation_data()
            has_validation = len(val_features) > 0
        else:
            has_validation = False
        
        epoch_results = {}
        
        # Compute losses for each model
        for model_name in model_names:
            model = self.models[model_name]
            scheduler = self._chunk_state['schedulers'][model_name]
            
            # Calculate average training loss for this epoch
            batch_count = self._chunk_state['epoch_batch_counts'][model_name]
            epoch_train_loss = self._chunk_state['epoch_train_loss_sums'][model_name] / batch_count if batch_count > 0 else 0.0
            self._chunk_state['train_losses'][model_name].append(epoch_train_loss)
            
            # Calculate validation loss if enabled
            if has_validation:
                epoch_val_loss = self._compute_validation_loss(model_name, val_features, val_fitness)
                self._chunk_state['val_losses'][model_name].append(epoch_val_loss)
                # Update scheduler based on validation loss
                scheduler.step(epoch_val_loss)
                loss_for_comparison = epoch_val_loss
            else:
                # Use training loss
                epoch_val_loss = epoch_train_loss
                self._chunk_state['val_losses'][model_name].append(epoch_val_loss)
                # Update scheduler based on training loss
                scheduler.step(epoch_train_loss)
                loss_for_comparison = epoch_train_loss
            
            epoch_results[model_name] = {
                'train_loss': epoch_train_loss,
                'val_loss': epoch_val_loss
            }
            
            if verbose:
                samples = self._chunk_state['total_samples'][model_name]
                batches = self._chunk_state['epoch_batch_counts'][model_name]
                val_str = f", Val={epoch_val_loss:.6f}" if has_validation else ""
                print(f"  {model_name}: Train={epoch_train_loss:.6f}{val_str} "
                    f"({samples} samples, {batches} batches)", flush=True)
        
        # Clear validation temp file for next epoch
        if use_validation:
            self._clear_validation_data()
        
        # Reset epoch accumulators
        self._reset_epoch_accumulators()
        
        # Find best model so far (use val_loss if available, else train_loss)
        loss_key = 'val_loss' if has_validation else 'train_loss'
        best_model_so_far = min(model_names, key=lambda n: epoch_results[n][loss_key])
        best_loss_so_far = epoch_results[best_model_so_far][loss_key]

        # Track per-model best loss and save best state
        for name in model_names:
            current_loss = epoch_results[name][loss_key]
            if current_loss < self._chunk_state['best_val_loss'][name]:
                self._chunk_state['best_val_loss'][name] = current_loss
                self._chunk_state['best_epoch'][name] = current_epoch
                self._chunk_state['best_model_states'][name] = {
                    k: v.clone() for k, v in self.models[name].state_dict().items()
                }

        # Check early stopping
        fitness_range = self._chunk_state['fitness_max'] - self._chunk_state['fitness_min']
        early_stop_threshold = self._chunk_state['early_stop_threshold']
        patience = self._chunk_state['patience']
        max_epochs = self._chunk_state['max_epochs']
        initial_epochs = self._chunk_state['initial_epochs']

        # Convergence check: RMSE below threshold
        early_stopped = False
        stop_reason = None
        if fitness_range > 0:
            absolute_threshold = early_stop_threshold * fitness_range
            best_rmse = np.sqrt(best_loss_so_far)

            if best_rmse < absolute_threshold:
                early_stopped = True
                stop_reason = "convergence"
                self._chunk_state['early_stopped'] = True

                if verbose:
                    print(f"\n{'🎯 EARLY STOPPING TRIGGERED! 🎯':^60}", flush=True)
                    print(f"{best_model_so_far} achieved RMSE {best_rmse:.6f} < {absolute_threshold:.6f} threshold", flush=True)

        # Patience check: no improvement for too long (only after initial epochs)
        if not early_stopped and current_epoch >= initial_epochs:
            # Check if ALL models have stagnated
            all_stagnated = True
            for name in model_names:
                epochs_since_best = current_epoch - self._chunk_state['best_epoch'][name]
                if epochs_since_best < patience:
                    all_stagnated = False
                    break

            if all_stagnated:
                early_stopped = True
                stop_reason = "patience"
                self._chunk_state['early_stopped'] = True
                if verbose:
                    best_at = self._chunk_state['best_epoch'][best_model_so_far]
                    print(f"  [Patience] No improvement for {patience} epochs, "
                          f"stopping at epoch {current_epoch} (best at epoch {best_at})", flush=True)

        # Auto-extend: if at initial epoch limit but still improving
        if not early_stopped and current_epoch >= total_epochs:
            any_improving = False
            for name in model_names:
                epochs_since_best = current_epoch - self._chunk_state['best_epoch'][name]
                if epochs_since_best < patience:
                    any_improving = True
                    break

            if any_improving and total_epochs < max_epochs:
                # Extend training
                self._chunk_state['epochs'] = max_epochs
                total_epochs = max_epochs
                if verbose:
                    print(f"  [Auto-extend] Still improving at epoch {current_epoch}, "
                          f"extending training (patience={patience}, max={max_epochs})", flush=True)

        # Check if training is complete
        training_complete = early_stopped or (current_epoch >= total_epochs)

        if not stop_reason and training_complete and not early_stopped:
            stop_reason = "max_epochs"

        # Build info dict
        info = {
            'current_epoch': current_epoch,
            'total_epochs': total_epochs,
            'models': epoch_results,
            'best_model_so_far': best_model_so_far,
            'best_val_loss_so_far': best_loss_so_far,
            'validation_ids_count': len(self._chunk_state.get('validation_fitness_ids', set())),
        }

        if training_complete:
            # Restore best model weights before finalizing
            for name in model_names:
                best_state = self._chunk_state['best_model_states'][name]
                if best_state is not None:
                    self.models[name].load_state_dict(best_state)

            if verbose and current_epoch != self._chunk_state['best_epoch'].get(best_model_so_far, current_epoch):
                best_at = self._chunk_state['best_epoch'][best_model_so_far]
                print(f"  [Adaptive] Restored {best_model_so_far} weights from epoch {best_at} "
                      f"(stopped at {current_epoch}, reason: {stop_reason})", flush=True)

            # Clean up stored states to free memory
            self._chunk_state['best_model_states'] = {}

            # Finalize results
            self._finalize_chunk_training_results()

            # Memory cleanup
            self._cleanup_memory(verbose)

            # Mark training as inactive
            self._chunk_state['training_active'] = False

            info['early_stopped'] = self._chunk_state['early_stopped']
            info['results'] = self._chunk_state['results']
            info['best_loss'] = self._chunk_state['results'].get(self.best_model_name, {}).get('final_val_loss')
            info['stop_reason'] = stop_reason

            if verbose:
                print(f"\n{'='*60}", flush=True)
                print(f"Chunked Training Complete", flush=True)
                print(f"Best Model: {self.best_model_name} (Loss: {info['best_loss']:.6f})", flush=True)
                print(f"{'='*60}\n", flush=True)
        else:
            # Move to next epoch
            self._chunk_state['current_epoch'] += 1

            # Mark validation as collected after epoch 1
            if current_epoch == 1:
                self._chunk_state['validation_data_collected'] = True

            # Set models back to train mode
            for name in model_names:
                self.models[name].train()

        return training_complete, info


    # =============================================================================
    # Method to add all chunked training methods to AutoEnsemblePredictor
    # =============================================================================

    def add_chunk_training_methods(cls):
        """Add chunked training methods to AutoEnsemblePredictor class.
        
        Usage:
            from chunk_training_methods_fixed import add_chunk_training_methods
            add_chunk_training_methods(AutoEnsemblePredictor)
        
        Args:
            cls: The AutoEnsemblePredictor class to extend
            
        Returns:
            The modified class
        """
        # Helper methods
        cls._initialize_models_for_chunk_training = _initialize_models_for_chunk_training
        cls._setup_chunk_training_state = _setup_chunk_training_state
        cls._is_validation_sample = _is_validation_sample
        cls._add_validation_fitness_id = _add_validation_fitness_id
        cls._get_validation_memory_usage = _get_validation_memory_usage
        cls._save_validation_data = _save_validation_data
        cls._load_validation_data = _load_validation_data
        cls._clear_validation_data = _clear_validation_data
        cls._compute_validation_loss = _compute_validation_loss
        cls._reset_epoch_accumulators = _reset_epoch_accumulators
        cls._finalize_chunk_training_results = _finalize_chunk_training_results
        
        # Main methods
        cls.start_chunk_training = start_chunk_training
        cls.process_chunk = process_chunk
        cls.finish_chunk_epoch = finish_chunk_epoch
        
        return cls

    
    # =============================================================================
    # Example Usage (for reference)
    # =============================================================================

    EXAMPLE_USAGE = """
    # Example 1: Basic usage with add_chunk_training_methods
    # -------------------------------------------------------
    from your_module import AutoEnsemblePredictor
    from chunk_training_methods import add_chunk_training_methods

    # Add methods to the class
    add_chunk_training_methods(AutoEnsemblePredictor)

    # Create predictor
    predictor = AutoEnsemblePredictor(input_size=100, device='cuda')

    # Initialize chunked training
    predictor.start_chunk_training(
        savePath='/path/to/save',
        agent_idx=0,
        allowed_architectures=['ANN', 'CNN1D', 'CNN2D'],
        epochs=50,
        validation_split=0.2,
        early_stop_threshold=0.05
    )

    # Training loop
    for epoch in range(50):
        # Load and process chunks
        for chunk_features, chunk_fitness in data_loader:
            losses = predictor.process_chunk(chunk_features, chunk_fitness)
            print(f"Chunk losses: {losses}")
        
        # End of epoch
        done, info = predictor.finish_chunk_epoch()
        print(f"Epoch {info['current_epoch']}: Best model = {info['best_model_so_far']}")
        
        if done:
            print(f"Training complete! Best model: {predictor.best_model_name}")
            break

    # Now use the trained model
    predictions = predictor.predict_fitness(new_features)


    # Example 2: Integration into existing file
    # -----------------------------------------
    # Simply copy all the methods above into your AutoEnsemblePredictor class definition
    # and adjust the imports (remove 'from __main__ import ...' lines)
    """
    # ============================End=of=Chunks====================================


# # Example usage
# if __name__ == "__main__":
#     # Generate synthetic data
#     np.random.seed(42)
#     n_samples, n_features = 2000, 8
    
#     features = np.random.uniform(-1, 1, (n_samples, n_features))
#     fitness_scores = (
#         np.sum(features**2, axis=1) +
#         0.5 * np.sum(features[:, :3], axis=1) +
#         0.3 * np.prod(features[:, 3:6], axis=1) +
#         0.1 * np.random.normal(0, 1, n_samples)
#     )
    
#     # Create predictor
#     predictor = AutoEnsemblePredictor(
#         input_size=n_features,
#         hidden_sizes=[128, 64, 32]
#     )
    
#     print("\n--- CASE 1: Standard usage (Training all models) ---")
#     results, best_loss = predictor.train_model(
#         features, fitness_scores, 
#         epochs=50, 
#         batch_size=32,
#         early_stop_threshold=0.05,
#         verbose=True
#     )
    
#     print("\n--- CASE 2: Forced Single Architecture (ANN Only) ---")
#     # Reset predictor
#     predictor = AutoEnsemblePredictor(input_size=n_features)
#     predictor.train_model(
#         features, fitness_scores,
#         allowed_architectures=['ANN'],
#         max_models_to_train=1,
#         verbose=True
#     )

#     print("\n--- CASE 3: Random Choice between ANN and CNN1D ---")
#     # Reset predictor
#     predictor = AutoEnsemblePredictor(input_size=n_features)
#     predictor.train_model(
#         features, fitness_scores,
#         allowed_architectures=['ANN', 'CNN1D'],
#         max_models_to_train=1,  # Will pick one at random
#         verbose=True
#     )
    
#     # Test predictions
#     print("\n" + "="*60)
#     print("Testing Predictions")
#     print("="*60)
#     test_features = np.random.uniform(-1, 1, (3, n_features))
#     predicted = predictor.predict_fitness(test_features)
#     print(f"Predicted fitness: {predicted}")
    
#     # Feature suggestions
#     print("\n" + "="*60)
#     print("Feature Suggestions")
#     print("="*60)
#     target = 2.0
#     suggestions = predictor.suggest_features_for_fitness(
#         target, n_features, num_suggestions=3, max_iterations=500
#     )
    
#     print(f"\nTarget fitness: {target:.3f}")
#     for i, feat in enumerate(suggestions):
#         pred = predictor.predict_fitness([feat])[0]
#         print(f"Suggestion {i+1}: Predicted={pred:.3f}, Error={abs(pred-target):.3f}")

#--backup----------------------------------------------------------------------------------------
# # version that works!
# def createCandidateGeneUsingFFPredictor(self, args=None, genePopulation=None, build_in_params=None, state_container=None):
#     """
#     Generates a new candidate gene using a Feed-Forward Predictor (Neural Network).
    
#     **Memory-Efficient Streaming Architecture:**
#     1. CACHE HIT: Use trained NN model to suggest genes for target fitness
#     2. RAM HIT: Train model on provided population, RECURSE
#     3. DISK HIT: Load diverse genes via streaming, train model, RECURSE
    
#     **Key Capabilities:**
#     ✓ Streaming loader: Process elite genes in 10k chunks (unified with GA/ES/CMA-ES)
#     ✓ Neural network training: AutoEnsemble predictor
#     ✓ Adaptive target z-score: Based on convergence progress
#     ✓ Reverse optimization: Suggest features for target fitness
#     ✓ Feature clustering: Adaptive clustering for diversity
#     ✓ Fitness stratification: Multiple strata
#     ✓ Unified scaler: ChunkedLogZScoreScaler
    
#     **Scalability:**
#     - Streaming loader handles 100M+ genes
#     - NN training on diverse subsets
#     - Memory-bounded operation
    
#     **Expected Performance:**
#     - Convergence: 20-50 iterations (like GA)
#     - Robustness: Very high (learns problem landscape)
#     - Memory: ~100-200 MB (model + training data)
#     - Best for: Complex, learned problem landscapes
#     """
    
#     device = torch.device('cuda' if getattr(args, 'gpu', False) and torch.cuda.is_available() else 'cpu')
    
#     if build_in_params is None:
#         build_in_params = {}
#     if 'ML' not in build_in_params:
#         build_in_params['ML'] = {}
    
#     MLparam = build_in_params['ML']

#     # ====================================================================
#     # UNPACK PREVIOUS STATE
#     # ====================================================================
#     predictor_model = None
#     FitScaler = None
#     FeatureScaler = None
#     previous_loss = None
    
#     if isinstance(state_container, list) and len(state_container) >= 4:
#         print(">> Using existing predictor model and scalers from previous generation.", flush=True)
#         previous_loss, predictor_model, FeatureScaler, FitScaler = state_container[:4]
#         build_in_params['ML']['loss'] = previous_loss
    
#     # ====================================================================
#     # CACHE HIT: Use trained NN to suggest genes
#     # ====================================================================
#     if predictor_model is not None and predictor_model.is_trained:
#         print(f"[FF Predictor Cache Hit] Using trained model to suggest new gene...", flush=True)
        
#         # ✅ ADAPTIVE TARGET Z-SCORE based on convergence progress
#         agent_counter = getattr(args, 'agent_counter', 0)
#         num_generations = getattr(args, 'numOFgenerations', 1000)
#         progress = agent_counter / num_generations if num_generations > 0 else 0.0

#         tmp_stage = 0.05
#         if np.random.rand() < tmp_stage:
#             stage = "Exploration boost" + str(tmp_stage * 100) + "%"
#             base_target_z = 2.5 + torch.rand(1).item() * 1.0
#         else:
#             stage = "Greedy phase" + str((1 - tmp_stage) * 100) + "%"
#             base_target_z = 3.0 + torch.rand(1).item() * 1.0
        
#         evolution_target = getattr(args, 'evolutionTarget', 1)
#         target_name = 'minimize' if evolution_target == -1 else 'maximize'
#         print(f"[FF Predictor] {stage.capitalize()} (progress={progress:.2%}): "
#             f"target_z={base_target_z:.2f} ({target_name})", flush=True)
        
#         # ✅ CRITICAL: Enable gradients for reverse optimization
#         # The suggestion phase does gradient descent on inputs to maximize output
#         with torch.enable_grad():
#             new_gene_tensor = predictor_model.suggest_features_for_fitness(
#                 target_fitness=base_target_z, 
#                 num_features=self.geneLength, 
#                 num_suggestions=args.total_testsGene_with_same_agent, 
#                 max_iterations=500, 
#             )
        
#         new_gene = new_gene_tensor[0]
#         lower_bound = 0 if getattr(self, 'geneMin', -1) == 0 else -1
#         new_gene = np.clip(new_gene, lower_bound, 1)
        
#         # Pack state for next generation
#         next_state_container = [
#             build_in_params['ML'].get('loss', None),
#             predictor_model,
#             FeatureScaler,
#             FitScaler
#         ]
        
#         return next_state_container, new_gene.astype(np.float32), build_in_params

#     # ====================================================================
#     # RAM HIT: Train model on provided population
#     # ====================================================================
#     def _get_hidden_sizes(input_dim):
#         """Calculate dynamic hidden layer sizes based on input dimension."""
#         base = int(64 * input_dim * np.log2(input_dim + 1))
#         base = min(base, 1000)  # Cap size to prevent explosion
#         return [base, base * 2, base // 2]

#     if genePopulation is not None:
#         print(f"[FF Predictor RAM Hit] Training on provided population ({len(genePopulation)} genes)...", 
#             flush=True)
        
#         # Extract valid genes
#         valid_genes = []
#         for gene in genePopulation:
#             if (isinstance(gene, dict) 
#                 and 'fitnessScore' in gene and isinstance(gene['fitnessScore'], (int, float)) and not np.isnan(gene['fitnessScore'])
#                 and 'gene' in gene and gene['gene'] is not None
#                 and not (isinstance(gene['gene'], str) and 'nan' in gene['gene'].lower())):
#                 valid_genes.append(gene)
        
#         if not valid_genes:
#             print("[WARNING] No valid genes in provided population. Falling back to disk loading...", flush=True)
#             # Fall through to disk hit
#             genePopulation = None
#         else:
#             # Prepare training data
#             fitness_list = np.array([g['fitnessScore'] for g in valid_genes], dtype=np.float32)
#             gene_list = np.array([g['gene'] for g in valid_genes], dtype=np.float32)
            
#             fitness_tensor = torch.tensor(fitness_list, device=device, dtype=torch.float32)
#             gene_tensor = torch.tensor(gene_list, device=device, dtype=torch.float32)
            
#             # Initialize FitScaler if not already done
#             if FitScaler is None:
#                 if args.numOFgenerations > 0:
#                     progress = args.agent_counter / args.numOFgenerations
#                     if progress < 0.3:
#                         emphasis_multiplier = 0.7
#                     elif progress > 0.7:
#                         emphasis_multiplier = 1.5
#                     else:
#                         emphasis_multiplier = 0.7 + progress
#                 else:
#                     emphasis_multiplier = 1.0
                
#                 evolution_target = getattr(args, 'evolutionTarget', 1)
                
#                 FitScaler = ChunkedLogZScoreScaler(
#                     evolution_target=evolution_target,   
#                     emphasis_factor=1.0 + emphasis_multiplier * torch.rand(1).item(),
#                     min_weight=0.01,
#                     device=device
#                 )
            
#             # Compute fitness scaling
#             FitScaler.collect_phase_1_statistics(fitness_tensor)
#             FitScaler.collect_phase_2_statistics(fitness_tensor)
#             FitScaler.finalize()
            
#             training_targets = FitScaler.scale_chunk(fitness_tensor)
            
#             # Configure and train model
#             FeatureScaler = None  # Option A: No feature scaler
#             if 'ANN_Layers_size' in MLparam and MLparam['ANN_Layers_size']:
#                 hidden_sizes = MLparam['ANN_Layers_size']
#             else:
#                 hidden_sizes = _get_hidden_sizes(self.geneLength)
            
#             print(f"[FF Predictor] Training architecture: {hidden_sizes}, "
#                 f"Dropout: {MLparam.get('Diffusion_ANN_dropout_rate', 0.2)}", flush=True)
            
#             predictor_model = AutoEnsemblePredictor(
#                 input_size=self.geneLength,            
#                 hidden_sizes=hidden_sizes,
#                 device=device,
#                 dropout_rate=MLparam.get('Diffusion_ANN_dropout_rate', 0.2),
#             )
            
#             _, val_losses = predictor_model.train_model(
#                 gene_tensor, 
#                 training_targets, 
#                 epochs=MLparam.get('Diffusion_ANN_max_epoch', 100), 
#                 batch_size=MLparam.get('batch_size', 32), 
#                 allowed_architectures=['ANN'],
#                 max_models_to_train=1,
#                 early_stop_threshold=0.05,
#                 learning_rate=MLparam.get('Diffusion_lr', 0.001),
#                 verbose=True,
#             )
            
#             build_in_params['ML']['loss'] = val_losses
            
#             # Check model quality
#             final_loss = val_losses[-1] if isinstance(val_losses, list) else val_losses
#             if final_loss and final_loss > 0.5:
#                 print(f"[WARNING] High validation loss ({final_loss:.3f}). "
#                     f"Model may not generalize well.", flush=True)
            
#             print(f"[FF Predictor RAM Hit] ✓ Model trained on {len(valid_genes)} genes", flush=True)
            
#             # Update params and RECURSE
#             next_state_container = [val_losses, predictor_model, FeatureScaler, FitScaler]
            
#             return self.createCandidateGeneUsingFFPredictor(
#                 args, None, build_in_params, next_state_container
#             )

#     # ====================================================================
#     # DISK HIT: Load diverse genes via streaming loader
#     # ====================================================================
#     print(f"[FF Predictor Disk Hit] Loading diverse population via streaming loader...", flush=True)
    
#     agent_counter = getattr(args, 'agent_counter', 0)
#     num_generations = getattr(args, 'numOFgenerations', 1000)
#     progress = agent_counter / num_generations if num_generations > 0 else 0.0
#     evolution_target = getattr(args, 'evolutionTarget', 1)
#     target_name = 'minimize' if evolution_target == -1 else 'maximize'
    
#     # ✅ FFPredictor uses BROADER selection (40%) than ES/CMA-ES for better NN training
#     top_k_percent = 0.40  # Load top 40% for NN training diversity
#     top_k_percentile = int(100 - (top_k_percent * 100))  # Convert: 0.40 → 60
    
#     print(f"[FF Predictor] Loading top {top_k_percent*100:.0f}% elite genes "
#         f"(fitness >= {top_k_percentile}th percentile) for NN training", flush=True)
    
#     # Create loader
#     loader = ChunkedGeneHistoryLoader(
#         savePath=args.path,
#         agent_id=getattr(args, 'agent_idx', agent_counter),
#         geneFormat=getattr(self, 'geneFormat', None),
#         shuffle=True,
#         shuffle_level='both',
#     )
    
#     diverse_genes = []
#     total_processed = 0
    
#     for chunk, is_done in loader.loadTopKPercentileGeneHistoryAsListOfDics(
#         args=args,
#         top_k_percentile=top_k_percentile,
#         maximize=(evolution_target == 1),
#         chunk_size=10000,
#         scaler=None,                      # Option B: Use unified scaler (None for raw, let FitScaler handle)
#         use_scaler_weights=False,
#         n_feature_clusters='adaptive',    # Option A: Keep adaptive clustering (NN benefits)
#         allEpochs=True
#     ):
#         # Option A: Remove metadata fields
#         for gene in chunk:
#             gene.pop('_scaler_weight', None)
#             gene.pop('feature_cluster_id', None)
#             gene.pop('stratum_id', None)
        
#         diverse_genes.extend(chunk)
#         total_processed += len(chunk)
        
#         # No explicit max limit (Option A: keep current behavior)
#         # FFPredictor will use whatever is loaded
        
#         if is_done:
#             print(f"[FF Predictor] Stream complete, loaded {len(diverse_genes)} genes", flush=True)
#             break
    
#     if not diverse_genes:
#         print(f"[FF Predictor ERROR] No genes loaded. Returning random gene.", flush=True)
        
#         lower_bound = 0 if getattr(self, 'geneMin', -1) == 0 else -1
#         new_gene = np.clip(torch.randn(self.geneLength, device=device).clamp(-1, 1).cpu().numpy(), 
#                         lower_bound, 1)
        
#         next_state_container = [
#             build_in_params['ML'].get('loss', None),
#             predictor_model,
#             FeatureScaler,
#             FitScaler
#         ]
        
#         return next_state_container, new_gene.astype(np.float32), build_in_params
    
#     print(f"[FF Predictor] Processed {total_processed} genes, kept {len(diverse_genes)} for NN training", flush=True)
    
#     # Initialize FitScaler if not already done
#     if FitScaler is None:
#         if args.numOFgenerations > 0:
#             progress = args.agent_counter / args.numOFgenerations
#             if progress < 0.3:
#                 emphasis_multiplier = 0.7
#             elif progress > 0.7:
#                 emphasis_multiplier = 1.5
#             else:
#                 emphasis_multiplier = 0.7 + progress
#         else:
#             emphasis_multiplier = 1.0
        
#         FitScaler = ChunkedLogZScoreScaler(
#             evolution_target=evolution_target,   
#             emphasis_factor=1.0 + emphasis_multiplier * torch.rand(1).item(),
#             min_weight=0.01,
#             device=device
#         )
    
#     # Convert to list of dicts format for RAM Hit
#     genePopulation = diverse_genes
    
#     print(f"[FF Predictor Disk Hit] ✓ Loaded {len(diverse_genes)} diverse genes "
#         f"via streaming loader", flush=True)
    
#     # RECURSE to RAM Hit with loaded population
#     return self.createCandidateGeneUsingFFPredictor(
#         args, genePopulation, build_in_params, state_container
# )


