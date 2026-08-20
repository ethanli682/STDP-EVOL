import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple

import torch.nn.functional as F
import math

# def fitness_to_probability_vector(fitness_scores, evolutionTarget=1, temperature=1.0, device=None):
#     """
#     Args:
#         temperature: Multiplier for selection pressure. 
#                      > 1.0 makes the selection "Greedier" (sharper).
#                      < 1.0 makes the selection more random (flatter).
#     """
    
#     # --- 1. Device and Type Setup ---
#     resolved_device = device
#     if isinstance(fitness_scores, torch.Tensor):
#         if resolved_device is None: resolved_device = fitness_scores.device
#         elif isinstance(resolved_device, str): resolved_device = torch.device(resolved_device)
#         scores_tensor = fitness_scores.to(dtype=torch.float32, device=resolved_device)
#     else:
#         if resolved_device is None: resolved_device = torch.device('cpu')
#         elif isinstance(resolved_device, str): resolved_device = torch.device(resolved_device)
#         try:
#             scores_tensor = torch.tensor(fitness_scores, dtype=torch.float32, device=resolved_device)
#         except Exception as e:
#             raise TypeError(f"fitness_scores must be convertible to a PyTorch tensor. Error: {e}")

#     if scores_tensor.numel() == 0:
#         return torch.tensor([], dtype=torch.float32, device=scores_tensor.device)
#     if torch.any(~torch.isfinite(scores_tensor)):
#         raise ValueError("All fitness scores must be finite numbers.")

#     # --- 2. Handle Identical Scores ---
#     min_raw = torch.min(scores_tensor)
#     max_raw = torch.max(scores_tensor)
    
#     if torch.isclose(min_raw, max_raw).item():
#         return torch.full_like(scores_tensor, 1.0 / scores_tensor.numel())

#     # --- 3. Always Shift to Relative Scale ---
#     # We always shift so the lowest score in the batch becomes 1.0.
#     # This prevents the "Vanishing Gradient" on high scores.
#     # Logic: (score - min) results in [0, range]. + 1.0 results in [1, range + 1].
#     processed_scores = (scores_tensor - min_raw) + 1.0

#     # --- 4. Log Transformation ---
#     log_scores = torch.log10(processed_scores + 1e-10)

#     # --- 5. Direction & Temperature ---
#     if evolutionTarget == -1: # Minimization
#         final_logits = -log_scores
#     elif evolutionTarget == 1: # Maximization
#         final_logits = log_scores
#     else:
#         raise ValueError("evolutionTarget must be 1 or -1")

#     # Apply Temperature (Selection Pressure)
#     final_logits = final_logits * temperature

#     # --- 6. Softmax ---
#     return F.softmax(final_logits, dim=0)

class ChunkedZScoreScaler:
    """
    A class for applying z-score normalization to feature vectors in chunks when the entire dataset cannot fit in memory.
    
    This class implements a two-pass approach:
    1. First pass: Collect statistics (mean and std) from all chunks
    2. Second pass: Scale each chunk using the global statistics
    
    The scaling applies z-score normalization: (x - μ) / σ
    - Computes global mean and standard deviation across all chunks
    - Applies feature-wise normalization (each feature normalized independently)
    - Handles edge cases like zero variance features
    """
    
    def __init__(self, device: Optional[Union[str, torch.device]] = None, eps: float = 1e-8):
        """
        Initialize the chunked z-score scaler.
        
        Args:
            device (torch.device or str, optional): Device for computations
            eps (float): Small value to prevent division by zero when std is very small
        """
        self.device = self._resolve_device(device)
        self.eps = eps
        
        # Statistics collected during first pass
        self.global_mean = None
        self.global_std = None
        self.feature_count = None
        self.total_samples = 0
        
        # Running statistics for numerical stability
        self.running_sum = None
        self.running_sum_sq = None
        
        # Status flags
        self.statistics_collected = False
        self.scaling_ready = False
        
    def _resolve_device(self, device):
        """Resolve device specification to torch.device object."""
        if device is None:
            return torch.device('cpu')
        elif isinstance(device, str):
            return torch.device(device)
        return device
    
    def _to_tensor(self, features):
        """Convert input to PyTorch tensor with proper device and dtype."""
        if isinstance(features, torch.Tensor):
            return features.to(dtype=torch.float32, device=self.device)
        else:
            try:
                return torch.tensor(features, dtype=torch.float32, device=self.device)
            except Exception as e:
                raise TypeError(f"features must be convertible to a PyTorch tensor of floats. Error: {e}")
    
    def collect_statistics(self, feature_chunk: Union[List, np.ndarray, torch.Tensor]):
        """
        Collect statistics from a chunk of feature vectors.
        Call this method for each chunk in the first pass.
        
        Args:
            feature_chunk: A chunk of feature vectors, shape (n_samples, n_features)
        """
        chunk_tensor = self._to_tensor(feature_chunk)
        
        # Handle empty chunks
        if chunk_tensor.numel() == 0:
            return
        
        # Ensure 2D tensor (samples x features)
        if chunk_tensor.dim() == 1:
            chunk_tensor = chunk_tensor.unsqueeze(0)
        elif chunk_tensor.dim() > 2:
            raise ValueError("feature_chunk must be 1D or 2D tensor")
        
        # Check for finite values
        if torch.any(~torch.isfinite(chunk_tensor)):
            raise ValueError("All feature values must be finite numbers (no NaN or Inf).")
        
        n_samples, n_features = chunk_tensor.shape
        
        # Compute min/max for this chunk
        chunk_min = torch.min(chunk_tensor, dim=0)[0]  # min per feature
        chunk_max = torch.max(chunk_tensor, dim=0)[0]  # max per feature
        
        # Initialize statistics on first chunk
        if self.running_sum is None:
            self.feature_count = n_features
            self.running_sum = torch.zeros(n_features, dtype=torch.float32, device=self.device)
            self.running_sum_sq = torch.zeros(n_features, dtype=torch.float32, device=self.device)
            
            # Initialize min/max tracking
            self.global_min = chunk_min.clone()
            self.global_max = chunk_max.clone()
        else:
            # Verify consistent feature count
            if n_features != self.feature_count:
                raise ValueError(f"Inconsistent feature count: expected {self.feature_count}, got {n_features}")
            
            # Update global min/max
            self.global_min = torch.minimum(self.global_min, chunk_min)
            self.global_max = torch.maximum(self.global_max, chunk_max)
        
        # Update running statistics
        self.running_sum += torch.sum(chunk_tensor, dim=0)
        self.running_sum_sq += torch.sum(chunk_tensor ** 2, dim=0)
        self.total_samples += n_samples

    def finalize_statistics(self):
        """
        Finalize statistics collection and prepare scaling parameters.
        Call this after processing all chunks with collect_statistics.
        """
        if self.running_sum is None:
            raise RuntimeError("No statistics collected. Call collect_statistics() first.")
        
        if self.total_samples == 0:
            raise RuntimeError("No samples collected.")
        
        # Compute global mean and standard deviation
        self.global_mean = self.running_sum / self.total_samples
        
        # Compute variance using the formula: Var(X) = E[X²] - (E[X])²
        mean_sq = self.running_sum_sq / self.total_samples
        variance = mean_sq - (self.global_mean ** 2)
        
        # Ensure non-negative variance (numerical stability)
        variance = torch.clamp(variance, min=0.0)
        
        # Compute standard deviation
        self.global_std = torch.sqrt(variance + self.eps)
        
        self.statistics_collected = True
        self.scaling_ready = True
    
    def scale_chunk(self, feature_chunk: torch.Tensor) -> None:
        """
        Scale a chunk of feature vectors IN-PLACE using the collected global statistics.
        This modifies the input tensor directly to save memory.
        
        Args:
            feature_chunk: A PyTorch tensor of feature vectors to scale in-place, 
                          shape (n_samples, n_features) or (n_features,)
        
        Note:
            - Input must be a PyTorch tensor (not list or numpy array)
            - The tensor is modified in-place, no return value
            - For non-tensor inputs, use scale_chunk_copy() method instead
        """
        if not self.scaling_ready:
            raise RuntimeError("Statistics not finalized. Call finalize_statistics() first.")
        
        if not isinstance(feature_chunk, torch.Tensor):
            raise TypeError("For in-place scaling, input must be a PyTorch tensor. Use scale_chunk_copy() for other types.")
        
        # Ensure proper device and dtype
        if feature_chunk.device != self.device or feature_chunk.dtype != torch.float32:
            raise ValueError(f"Tensor must be on device {self.device} with dtype torch.float32")
        
        # Handle empty chunks
        if feature_chunk.numel() == 0:
            return
        
        # Check dimensions
        original_shape = feature_chunk.shape
        if feature_chunk.dim() == 1:
            # Temporarily reshape for processing
            feature_chunk.unsqueeze_(0)
        elif feature_chunk.dim() > 2:
            raise ValueError("feature_chunk must be 1D or 2D tensor")
        
        # Check for finite values
        if torch.any(~torch.isfinite(feature_chunk)):
            raise ValueError("All feature values must be finite numbers (no NaN or Inf).")
        
        # Verify consistent feature count
        if feature_chunk.shape[1] != self.feature_count:
            raise ValueError(f"Inconsistent feature count: expected {self.feature_count}, got {feature_chunk.shape[1]}")
        
        # Apply z-score normalization IN-PLACE: (x - μ) / σ
        feature_chunk.sub_(self.global_mean)  # x = x - μ
        feature_chunk.div_(self.global_std)   # x = x / σ
        
        # Restore original shape if input was 1D
        if len(original_shape) == 1:
            feature_chunk.squeeze_(0)
    
    def scale_chunk_copy(self, feature_chunk: Union[List, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Scale a chunk of feature vectors using the collected global statistics.
        This creates a copy and returns the scaled result (original method behavior).
        
        Args:
            feature_chunk: A chunk of feature vectors to scale, shape (n_samples, n_features)
            
        Returns:
            torch.Tensor: Z-score normalized feature vectors
        """
        if not self.scaling_ready:
            raise RuntimeError("Statistics not finalized. Call finalize_statistics() first.")
        
        chunk_tensor = self._to_tensor(feature_chunk)
        
        # Handle empty chunks
        if chunk_tensor.numel() == 0:
            return torch.tensor([], dtype=torch.float32, device=self.device)
        
        # Ensure 2D tensor (samples x features)
        original_shape = chunk_tensor.shape
        if chunk_tensor.dim() == 1:
            chunk_tensor = chunk_tensor.unsqueeze(0)
        elif chunk_tensor.dim() > 2:
            raise ValueError("feature_chunk must be 1D or 2D tensor")
        
        # Check for finite values
        if torch.any(~torch.isfinite(chunk_tensor)):
            raise ValueError("All feature values must be finite numbers (no NaN or Inf).")
        
        # Verify consistent feature count
        if chunk_tensor.shape[1] != self.feature_count:
            raise ValueError(f"Inconsistent feature count: expected {self.feature_count}, got {chunk_tensor.shape[1]}")
        
        # Apply z-score normalization: (x - μ) / σ
        scaled_chunk = (chunk_tensor - self.global_mean) / self.global_std
        
        # Restore original shape if input was 1D
        if len(original_shape) == 1:
            scaled_chunk = scaled_chunk.squeeze(0)
        
        return scaled_chunk
    
    def reset(self):
        """Reset the scaler to process a new dataset."""
        self.global_mean = None
        self.global_std = None
        self.feature_count = None
        self.total_samples = 0
        self.running_sum = None
        self.running_sum_sq = None
        self.statistics_collected = False
        self.scaling_ready = False
    
    def get_statistics(self) -> dict:
        """Get the collected statistics."""
        return {
            'global_mean': self.global_mean.tolist() if self.global_mean is not None else None,
            'global_std': self.global_std.tolist() if self.global_std is not None else None,
            'feature_count': self.feature_count,
            'total_samples': self.total_samples,
            'statistics_collected': self.statistics_collected,
            'scaling_ready': self.scaling_ready
        }
    
    def inverse_transform(self, scaled_chunk: Union[List, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Inverse transform scaled features back to original scale.
        
        Args:
            scaled_chunk: Z-score normalized feature vectors
            
        Returns:
            torch.Tensor: Features in original scale
        """
        if not self.scaling_ready:
            raise RuntimeError("Statistics not finalized. Call finalize_statistics() first.")
        
        chunk_tensor = self._to_tensor(scaled_chunk)
        
        # Handle empty chunks
        if chunk_tensor.numel() == 0:
            return torch.tensor([], dtype=torch.float32, device=self.device)
        
        # Ensure 2D tensor (samples x features)
        original_shape = chunk_tensor.shape
        if chunk_tensor.dim() == 1:
            chunk_tensor = chunk_tensor.unsqueeze(0)
        
        # Apply inverse z-score: x = (z * σ) + μ
        original_chunk = (chunk_tensor * self.global_std) + self.global_mean
        
        # Restore original shape if input was 1D
        if len(original_shape) == 1:
            original_chunk = original_chunk.squeeze(0)
        
        return original_chunk

# class ChunkedLogFitnessScaler:
#     """
#     A scaler that calculates global log-min-max statistics in a chunked manner
#     and scales fitness values to [0, 1].
    
#     Updated: Includes Weight Normalization and Safety Clipping.
#     """
#     def __init__(self, 
#                  evolution_target: int = 1, 
#                  emphasis_factor: float = 2.0,
#                  min_weight: float = 0.1,
#                  max_weight: float = 5.0,  # NEW: Cap for weights after normalization
#                  device: Optional[Union[str, torch.device]] = None, 
#                  eps: float = 1e-8,
#                  log_eps: float = 1e-10):
#         """
#         Initialize the weighted fitness scaler.
        
#         Args:
#             evolution_target (int): 1 for maximization, -1 for minimization
#             emphasis_factor (float): Power factor. Higher = strict "winner takes all".
#             min_weight (float): Floor value for weights (prevents dying gradients).
#             max_weight (float): Ceiling value for weights (prevents exploding gradients).
#             device (torch.device or str): Device for computations.
#         """
#         if evolution_target not in [1, -1]:
#             raise ValueError("evolution_target must be 1 (maximization) or -1 (minimization)")
            
#         self.evolution_target = evolution_target
#         self.emphasis_factor = emphasis_factor
#         self.min_weight = min_weight
#         self.max_weight = max_weight # NEW
#         self.device = self._resolve_device(device)
#         self.eps = eps
#         self.log_eps = log_eps
        
#         # Statistics storage
#         self.global_min_raw = None
#         self.global_max_raw = None
#         self.global_min_log = None
#         self.global_max_log = None
#         self.log_shift_constant = None
#         self.log_range = None
#         self.total_samples = 0
        
#         self.statistics_collected = False
#         self.scaling_ready = False
        
#     def _resolve_device(self, device):
#         if device is None:
#             return torch.device('cpu')
#         elif isinstance(device, str):
#             return torch.device(device)
#         return device
    
#     def _to_tensor(self, fitness):
#         if isinstance(fitness, torch.Tensor):
#             return fitness.to(dtype=torch.float32, device=self.device)
#         else:
#             try:
#                 return torch.tensor(fitness, dtype=torch.float32, device=self.device)
#             except Exception as e:
#                 raise TypeError(f"fitness must be convertible to a PyTorch tensor. Error: {e}")
    
#     def collect_statistics(self, fitness_chunk: Union[List, np.ndarray, torch.Tensor]):
#         chunk_tensor = self._to_tensor(fitness_chunk)
#         if chunk_tensor.numel() == 0: return
#         if chunk_tensor.dim() > 1: chunk_tensor = chunk_tensor.flatten()
#         if torch.any(~torch.isfinite(chunk_tensor)):
#             raise ValueError("All fitness values must be finite numbers.")
        
#         n_samples = chunk_tensor.shape[0]
#         chunk_min = torch.min(chunk_tensor)
#         chunk_max = torch.max(chunk_tensor)
        
#         if self.global_min_raw is None:
#             self.global_min_raw = chunk_min
#             self.global_max_raw = chunk_max
#         else:
#             self.global_min_raw = torch.minimum(self.global_min_raw, chunk_min)
#             self.global_max_raw = torch.maximum(self.global_max_raw, chunk_max)
        
#         self.total_samples += n_samples
    
#     def finalize_statistics(self):
#         """
#         Finalizes global statistics after all chunks have been collected.
#         Calculates shift constants and log boundaries.
#         """
#         if self.global_min_raw is None:
#             raise RuntimeError("No statistics collected. Call collect_statistics() first.")
#         if self.total_samples == 0:
#             raise RuntimeError("No samples collected.")
            
#         # --- 1. Calculate Shift Constant ---
#         # We ALWAYS shift the global minimum to start at 1.0.
#         # This ensures that the logarithmic transformation is consistent (curved)
#         # regardless of whether the raw scores are -50 or +1,000,000.
#         # Formula: shift = 1.0 - min_val  =>  (min_val + shift) == 1.0
#         self.log_shift_constant = 1.0 - self.global_min_raw

#         # --- 2. Handle Edge Case: All scores are identical ---
#         if torch.isclose(self.global_min_raw, self.global_max_raw).item():
#             # If min == max, we can't scale meaningful ranges. 
#             # Set defaults to prevent division by zero in scale_chunk.
#             self.global_min_log = torch.tensor(0.0, dtype=torch.float32, device=self.device)
#             self.global_max_log = torch.tensor(1.0, dtype=torch.float32, device=self.device)
#             self.log_range = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        
#         # --- 3. Standard Calculation ---
#         else:
#             # Apply shift to get positive coordinates starting at 1.0
#             processed_min_raw = self.global_min_raw + self.log_shift_constant
#             processed_max_raw = self.global_max_raw + self.log_shift_constant
            
#             # Apply Log10
#             self.global_min_log = torch.log10(processed_min_raw + self.log_eps)
#             self.global_max_log = torch.log10(processed_max_raw + self.log_eps)
            
#             # Calculate Range
#             self.log_range = self.global_max_log - self.global_min_log
        
#         # --- 4. Safety Check for Range ---
#         # Even if raw values differ slightly, log values might be too close.
#         if self.log_range.item() < self.eps:
#              self.log_range = torch.tensor(1.0, dtype=torch.float32, device=self.device)

#         # --- 5. Finalize Flags ---
#         self.statistics_collected = True
#         self.scaling_ready = True
            
#     def scale_chunk(self, fitness_chunk: Union[List, np.ndarray, torch.Tensor]) -> torch.Tensor:
#         """
#         Returns normalized fitness in [0, 1] range.
#         """
#         if not self.scaling_ready:
#             raise RuntimeError("Statistics not finalized.")
        
#         chunk_tensor = self._to_tensor(fitness_chunk)
#         if chunk_tensor.numel() == 0: return torch.tensor([], device=self.device)
#         if chunk_tensor.dim() > 1: chunk_tensor = chunk_tensor.flatten()
        
#         processed_scores = chunk_tensor + self.log_shift_constant
#         log_transformed_scores = torch.log10(processed_scores + self.log_eps)
#         scaled_output = (log_transformed_scores - self.global_min_log) / (self.log_range + self.eps)
        
#         if self.evolution_target == -1: # Minimization
#             # Flip [0, 1] to [-1, 0] then shift to [0, 1]
#             scaled_output = (scaled_output * -1) + 1.0
            
#         return scaled_output
    
#     def compute_sample_weights(self, fitness_chunk: Union[List, np.ndarray, torch.Tensor], normalize: bool = True) -> torch.Tensor:
#         """
#         Compute sample weights with Normalization and Safety Clipping.
        
#         Args:
#             fitness_chunk: Fitness values.
#             normalize (bool): If True, scales weights so the batch mean is 1.0. 
#                               Highly recommended for consistent learning rates.
#         """
#         if not self.scaling_ready:
#             raise RuntimeError("Statistics not finalized.")
        
#         chunk_tensor = self._to_tensor(fitness_chunk)
#         if chunk_tensor.numel() == 0: return torch.tensor([], device=self.device)
#         if chunk_tensor.dim() > 1: chunk_tensor = chunk_tensor.flatten()
        
#         # 1. Base Logic: Calculate 'Relative' Importance [0 to 1]
#         if torch.isclose(self.global_min_raw, self.global_max_raw).item():
#             raw_weights = torch.ones_like(chunk_tensor)
#         else:
#             raw_range = self.global_max_raw - self.global_min_raw
#             fitness_normalized = (chunk_tensor - self.global_min_raw) / raw_range
            
#             if self.evolution_target == 1: 
#                 raw_weights = torch.pow(fitness_normalized, self.emphasis_factor)
#             else: 
#                 raw_weights = torch.pow(1.0 - fitness_normalized, self.emphasis_factor)

#         # 2. Normalization (CRITICAL STEP)
#         # Shift distribution so Mean is 1.0. This allows "good" samples 
#         # to go > 1.0 and "bad" samples to go < 1.0.
#         if normalize:
#             # Add eps to mean to prevent div by zero
#             batch_mean = torch.mean(raw_weights) + self.eps
#             weights = raw_weights / batch_mean
#         else:
#             weights = raw_weights

#         # 3. Safety Clipping (Max and Min)
#         # Clamp between [min_weight, max_weight]
#         # Example: clamp(weights, 0.1, 5.0)
#         weights = torch.clamp(weights, min=self.min_weight, max=self.max_weight)
        
#         return weights
    
#     def compute_chunk_probabilities(self, fitness_chunk: Union[List, np.ndarray, torch.Tensor]) -> torch.Tensor:
#             """
#             Returns a probability vector where sum(output) == 1.0.
#             Best used for Genetic Algorithm Selection (Roulette Wheel).
#             """
#             if not self.scaling_ready:
#                 raise RuntimeError("Statistics not finalized.")

#             # 1. Prepare Tensor
#             chunk_tensor = self._to_tensor(fitness_chunk)
#             if chunk_tensor.numel() == 0: return torch.tensor([], device=self.device)
#             if chunk_tensor.dim() > 1: chunk_tensor = chunk_tensor.flatten()

#             # 2. Shift & Log (Use Global Statistics)
#             processed_scores = chunk_tensor + self.log_shift_constant
#             log_scores = torch.log10(processed_scores + self.log_eps)

#             # 3. Handle Minimization
#             if self.evolution_target == -1:
#                 logits = -log_scores
#             else:
#                 logits = log_scores

#             # 4. Apply Selection Pressure (Emphasis)
#             logits = logits * self.emphasis_factor

#             # 5. Softmax -> Sums to 1.0
#             return torch.nn.functional.softmax(logits, dim=0)

#     def scale_chunk_with_weights(self, fitness_chunk: Union[List, np.ndarray, torch.Tensor]) -> tuple:
#         normalized_fitness = self.scale_chunk(fitness_chunk)
#         sample_weights = self.compute_sample_weights(fitness_chunk, normalize=True)
#         return (normalized_fitness, sample_weights)
    
#     def inverse_transform(self, normalized_fitness: Union[List, np.ndarray, torch.Tensor]) -> torch.Tensor:
#         if not self.scaling_ready: raise RuntimeError("Statistics not finalized.")
#         chunk_tensor = self._to_tensor(normalized_fitness)
#         if chunk_tensor.numel() == 0: return torch.tensor([], device=self.device)
            
#         if self.evolution_target == -1:
#             log_min_max_scaled = (chunk_tensor - 1.0) * -1
#         else:
#             log_min_max_scaled = chunk_tensor
            
#         log_space = (log_min_max_scaled * (self.log_range + self.eps)) + self.global_min_log
#         processed_raw_scores = torch.pow(10.0, log_space)
#         original_fitness = processed_raw_scores - self.log_shift_constant
        
#         return original_fitness
    
#     def reset(self):
#         self.global_min_raw = None
#         self.global_max_raw = None
#         self.global_min_log = None
#         self.global_max_log = None
#         self.log_shift_constant = None
#         self.log_range = None
#         self.total_samples = 0
#         self.statistics_collected = False
#         self.scaling_ready = False
    
#     def get_statistics(self) -> dict:
#         return {
#             'global_min_raw': self.global_min_raw.item() if self.global_min_raw is not None else None,
#             'global_max_raw': self.global_max_raw.item() if self.global_max_raw is not None else None,
#             'global_min_log': self.global_min_log.item() if self.global_min_log is not None else None,
#             'global_max_log': self.global_max_log.item() if self.global_max_log is not None else None,
#             'log_shift_constant': self.log_shift_constant.item() if self.log_shift_constant is not None else None,
#             'fs_log_shift': self.log_shift_constant.item() if self.log_shift_constant is not None else None,
#             'fs_log_range': self.log_range.item() if self.log_range is not None else None,
#             'total_samples': self.total_samples,
#             'min_weight': self.min_weight,
#             'max_weight': self.max_weight,
#             'statistics_collected': self.statistics_collected
#         }

import torch
from typing import Optional, Union

class ChunkedLogZScoreScaler:
    """
    Final Validated Scaler (v5 - Flexible Clipping) - ES Extended
    -----------------------------------------------------------
    Preserves original API. Adds Global Weighting support for Chunked ES.
    """
    
    def __init__(self, 
                 evolution_target: int = 1, 
                 emphasis_factor: float = 2.0,
                 min_weight: float = 0.1,
                 device: Optional[Union[str, torch.device]] = None, 
                 eps: float = 1e-8,
                 log_eps: float = 1e-10):
        
        if evolution_target not in [1, -1]:
            raise ValueError("evolution_target must be 1 or -1")
            
        self.evolution_target = evolution_target
        self.emphasis_factor = emphasis_factor
        self.min_weight = min_weight
        self.device = self._resolve_device(device)
        self.eps = eps
        self.log_eps = log_eps
        
        # Stats Attributes (Preserved)
        self.global_raw_min = None
        self.global_raw_max = None
        self.log_shift_constant = None
        
        # Welford's Attributes
        self.count_log = 0
        self.mean_log = None
        self.M2_log = None 
        self.global_log_std = None
        
        # --- NEW ATTRIBUTE ---
        # Stores the Z-score of the theoretical "Best Possible Candidate".
        self.global_max_z_score = None 
        
        # Flags
        self.pass1_done = False
        self.scaling_ready = False

    def _resolve_device(self, device):
        if device is None: return torch.device('cpu')
        return torch.device(device) if isinstance(device, str) else device

    def _to_tensor(self, data):
        if isinstance(data, torch.Tensor):
            return data.to(dtype=torch.float32, device=self.device)
        return torch.tensor(data, dtype=torch.float32, device=self.device)
    
    def reset(self):
        self.global_raw_min = None
        self.global_raw_max = None
        self.log_shift_constant = None
        self.count_log = 0
        self.mean_log = None
        self.M2_log = None
        self.global_log_std = None
        self.global_max_z_score = None # Reset new attribute
        self.pass1_done = False
        self.scaling_ready = False

    # --- PHASE 1: BOUNDS ---
    def collect_phase_1_statistics(self, fitness_chunk):
        chunk = self._to_tensor(fitness_chunk)
        if chunk.numel() == 0: return
        if chunk.dim() > 1: chunk = chunk.flatten()
        
        c_min = torch.min(chunk)
        c_max = torch.max(chunk)

        if self.global_raw_min is None:
            self.global_raw_min = c_min
            self.global_raw_max = c_max
        else:
            self.global_raw_min = torch.minimum(self.global_raw_min, c_min)
            self.global_raw_max = torch.maximum(self.global_raw_max, c_max)

    def _finalize_pass1(self):
        if self.global_raw_min is None:
            # Handle empty data gracefully
            self.log_shift_constant = 0.0
            self.pass1_done = True
            self.mean_log = torch.tensor(0.0, device=self.device)
            self.M2_log = torch.tensor(0.0, device=self.device)
            return

        if self.evolution_target == -1:
            # Minimization: Anchor to Min
            min_val = self.global_raw_min.item()
            if min_val <= 0:
                self.log_shift_constant = 1.0 - min_val
            else:
                self.log_shift_constant = 0.0
        else:
            # Maximization: Anchor to Max (Reflected)
            self.log_shift_constant = 1.0

        self.pass1_done = True
        self.mean_log = torch.tensor(0.0, device=self.device)
        self.M2_log = torch.tensor(0.0, device=self.device)
        self.count_log = 0

    # --- INTERNAL TRANSFORM ---
    def _transform_to_log(self, chunk):
        if self.evolution_target == -1:
            shifted = chunk + self.log_shift_constant
            return torch.log10(shifted + self.log_eps)
        else:
            # Reflection: Max + Shift - x
            reflection = (self.global_raw_max + self.log_shift_constant) - chunk
            reflection = torch.clamp(reflection, min=self.log_eps)
            return torch.log10(reflection)

    # --- PHASE 2: STATISTICS ---
    def collect_phase_2_statistics(self, fitness_chunk):
        if not self.pass1_done: self._finalize_pass1()
        chunk = self._to_tensor(fitness_chunk)
        if chunk.numel() == 0: return
        if chunk.dim() > 1: chunk = chunk.flatten()

        log_vals = self._transform_to_log(chunk)
        
        batch_count = log_vals.numel()
        if batch_count == 0: return
        
        batch_mean = torch.mean(log_vals)
        new_count = self.count_log + batch_count
        
        delta = batch_mean - self.mean_log
        batch_M2 = torch.sum((log_vals - batch_mean)**2)
        
        self.M2_log += batch_M2 + (delta**2) * (self.count_log * batch_count / new_count)
        self.mean_log += delta * (batch_count / new_count)
        self.count_log = new_count

    def finalize(self):
        if not self.pass1_done: raise RuntimeError("Phase 1 incomplete.")
        
        # 1. Calc Std
        if self.count_log < 2:
            self.global_log_std = torch.tensor(1.0, device=self.device)
        else:
            variance = self.M2_log / (self.count_log - 1)
            self.global_log_std = torch.sqrt(variance + self.eps)
        
        if self.global_log_std < self.eps:
            self.global_log_std = torch.tensor(1.0, device=self.device)
        
        # 2. (NEW) Calculate Global Max Z-Score for Stability
        # We calculate what the "Best Possible" Z-score would be.
        if self.global_raw_max is not None:
            if self.evolution_target == 1:
                best_raw = self.global_raw_max
            else:
                best_raw = self.global_raw_min
                
            best_log = self._transform_to_log(best_raw)
            # Standard Z
            best_z_standard = (best_log - self.mean_log) / self.global_log_std
            # Invert because scale_chunk inverts (Higher output = Better)
            self.global_max_z_score = -best_z_standard 
        else:
            self.global_max_z_score = torch.tensor(0.0, device=self.device)

        self.scaling_ready = True

    # --- PERSISTENCE (Preserve API) ---
    def get_statistics(self):
        return {
            'global_raw_min': self.global_raw_min.item() if self.global_raw_min is not None else None,
            'global_raw_max': self.global_raw_max.item() if self.global_raw_max is not None else None,
            'log_shift_constant': self.log_shift_constant,
            'mean_log': self.mean_log.item() if self.mean_log is not None else None,
            'global_log_std': self.global_log_std.item() if self.global_log_std is not None else None,
            'global_max_z_score': self.global_max_z_score.item() if self.global_max_z_score is not None else None,
            'pass1_done': self.pass1_done,
            'scaling_ready': self.scaling_ready,
            'evolution_target': self.evolution_target
        }

    def load_statistics(self, stats):
        def _load_t(val): return torch.tensor(val, dtype=torch.float32, device=self.device) if val is not None else None
        self.global_raw_min = _load_t(stats.get('global_raw_min'))
        self.global_raw_max = _load_t(stats.get('global_raw_max'))
        self.log_shift_constant = stats.get('log_shift_constant')
        self.mean_log = _load_t(stats.get('mean_log'))
        self.global_log_std = _load_t(stats.get('global_log_std'))
        self.global_max_z_score = _load_t(stats.get('global_max_z_score')) 
        self.pass1_done = stats.get('pass1_done', False)
        self.scaling_ready = stats.get('scaling_ready', False)
        self.evolution_target = stats.get('evolution_target', 1)

    # --- SCALING ---
    def scale_chunk(self, fitness_chunk, clip_min=None, clip_max=None):
        if not self.scaling_ready: raise RuntimeError("Not ready.")
        chunk = self._to_tensor(fitness_chunk)
        if chunk.numel() == 0: return chunk
        if chunk.dim() > 1: chunk = chunk.flatten()

        log_vals = self._transform_to_log(chunk)
        z_scores = (log_vals - self.mean_log) / self.global_log_std
        
        # Invert so that "High Z" = "Better Fitness"
        z_out = -z_scores
        
        if clip_min is not None or clip_max is not None:
            min_val = clip_min if clip_min is not None else -float('inf')
            max_val = clip_max if clip_max is not None else float('inf')
            z_out = torch.clamp(z_out, min=min_val, max=max_val)

        return z_out

    # --- LEGACY SUPPORT ---
    def compute_chunk_probabilities(self, fitness_chunk):
        if not self.scaling_ready: raise RuntimeError("Not ready.")
        z_scores = self.scale_chunk(fitness_chunk) 
        if z_scores.numel() == 0: return torch.tensor([], device=self.device)
        logits = z_scores * self.emphasis_factor
        return torch.nn.functional.softmax(logits, dim=0)

    def compute_sample_weights(self, fitness_chunk):
        if not self.scaling_ready: raise RuntimeError("Not ready.")
        z_scores = self.scale_chunk(fitness_chunk)
        z_min = z_scores.min()
        z_max = z_scores.max()
        if (z_max - z_min) < self.eps:
            p = torch.ones_like(z_scores)
        else:
            p = (z_scores - z_min) / (z_max - z_min)
        weights = torch.pow(torch.clamp(p, 0.0, 1.0), self.emphasis_factor)
        return weights * (1.0 - self.min_weight) + self.min_weight

    # --- NEW METHOD: GLOBAL WEIGHTS ---
    def get_global_weights(self, fitness_chunk):
        """
        Returns unnormalized weights exp(emphasis * (z - global_max)).
        Safe to sum across chunks because of global anchor.
        """
        if not self.scaling_ready: raise RuntimeError("Not ready.")
        if self.global_max_z_score is None: 
            return self.compute_chunk_probabilities(fitness_chunk) 
        
        # 1. Get standard inverted Z-scores (Higher is Better)
        z_scores = self.scale_chunk(fitness_chunk) 
        
        # 2. Shift by Global Max for numerical stability (exp(0) is max)
        shifted_z = z_scores - self.global_max_z_score
        
        # 3. Apply Emphasis and Exponentiate
        weights = torch.exp(shifted_z * self.emphasis_factor)
        return weights
    
    def inverse_transform(self, z_scores):
        if not self.scaling_ready: raise RuntimeError("Not ready.")
        z = self._to_tensor(z_scores)
        z_dist = -z 
        log_vals = (z_dist * self.global_log_std) + self.mean_log
        linear_val = torch.pow(10.0, log_vals)
        if self.evolution_target == -1:
            raw = linear_val - self.log_shift_constant
        else:
            raw = (self.global_raw_max + self.log_shift_constant) - linear_val
        return raw 

class PolarMapper:
    """
    Implements a scikit-learn-style transformer to map ranged values
    to 2D (x, y) coordinates and back.

    This class is designed to be memory-efficient. It can be 'fit' on
    a large dataset (iterable of batches) in a single pass to find the
    global min and max values.
    
    Args:
        use_random_r (bool): If True, multiplies (x, y) by a random radius
                             during the 'transform' step. Defaults to False.
    
    Attributes:
        min_val_ (float): The minimum value found in the dataset after fitting.
        max_val_ (float): The maximum value found in the dataset after fitting.
        is_fitted_ (bool): Flag indicating if the mapper has been fit.
    """
    
    def __init__(self, use_random_r: bool = False):
        self.use_random_r = use_random_r
        self.min_val_ = None
        self.max_val_ = None
        self.is_fitted_ = False

    def _check_is_fitted(self):
        """Helper to check if 'fit' has been called."""
        if not self.is_fitted_:
            raise RuntimeError("This PolarMapper instance is not fitted yet. "
                               "Call 'fit' with your dataset first.")

    def fit(self, X, y=None):
        """
        Finds the global min and max values from a dataset.

        Args:
            X (iterable): An iterable of batches (e.g., a PyTorch DataLoader,
                          a list of tensors, or a generator).
            y (any): Ignored. Included for scikit-learn compatibility.
        
        Returns:
            self
        """
        current_min = float('inf')
        current_max = float('-inf')
        
        is_empty = True
        print("Fitting PolarMapper...", flush=True)
        for i, batch in enumerate(X):
            is_empty = False
            # Ensure batch is a tensor, detach if it's part of a graph
            batch_tensor = torch.as_tensor(batch, dtype=torch.float32).detach()
            
            batch_min = torch.min(batch_tensor)
            batch_max = torch.max(batch_tensor)
            
            if batch_min < current_min:
                current_min = batch_min
            if batch_max > current_max:
                current_max = batch_max
            
            if (i + 1) % 100 == 0:
                 print(f"  Processed {i+1} batches...", flush=True)

        if is_empty:
            raise ValueError("Cannot fit on an empty dataset.")
        
        # Store the learned parameters as floats
        self.min_val_ = current_min.item()
        self.max_val_ = current_max.item()
        self.is_fitted_ = True
        
        print(f"Fit complete. min_val_ = {self.min_val_:.4f}, max_val_ = {self.max_val_:.4f}", flush=True)
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Encodes a tensor of values into (x, y) coordinates.
        The (x,y) pairs are flattened into the last dimension.

        Args:
            X (torch.Tensor): The input tensor of values (any shape, e.g., (*, D)).

        Returns:
            torch.Tensor: A tensor of flattened (x, y) pairs,
                          with shape (*, D*2).
        """
        self._check_is_fitted()
        values = torch.as_tensor(X, dtype=torch.float32)
        
        # 1. Normalize values from [min_val_, max_val_] to [0, 1]
        clamped_values = torch.clamp(values, self.min_val_, self.max_val_)
        
        # Handle division by zero if min_val == max_val
        range_ = self.max_val_ - self.min_val_
        if range_ == 0:
            normalized = torch.zeros_like(clamped_values)
        else:
            normalized = (clamped_values - self.min_val_) / range_
        
        # 2. Scale from [0, 1] to [0, 2*pi] to get the angle theta
        pi_tensor = torch.tensor(math.pi, device=values.device, dtype=values.dtype)
        theta = normalized * 2 * pi_tensor
        
        # 3. Calculate (x, y) coordinates
        if self.use_random_r:
            # Use a random radius, e.g., from U[0.5, 1.5]
            r = torch.rand_like(values) + 0.5
            x = r * torch.cos(theta)
            y = r * torch.sin(theta)
        else:
            # Use a fixed radius of 1
            x = torch.cos(theta)
            y = torch.sin(theta)
            
        # 4. Stack x and y into a new last dimension -> shape (*, D, 2)
        xy_pairs = torch.stack([x, y], dim=-1)
        
        # 5. Reshape to (*, D*2) to create a "1D" vector output
        # for the last dimension.
        original_shape = values.shape
        
        if len(original_shape) == 0: # Handle scalar input
            return xy_pairs.reshape(2)
            
        new_shape = (*original_shape[:-1], original_shape[-1] * 2)
        return xy_pairs.reshape(new_shape)

    def inverse_transform(self, X_xy: torch.Tensor) -> torch.Tensor:
        """
        Decodes a tensor of (x, y) coordinates back into the original values.
        Expects (x,y) pairs to be flattened into the last dimension.

        Args:
            X_xy (torch.Tensor): Input tensor of flattened (x, y) pairs.
                                 Must have shape (*, D*2).

        Returns:
            torch.Tensor: The decoded values, with shape (*, D).
        """
        self._check_is_fitted()
        X_xy_tensor = torch.as_tensor(X_xy, dtype=torch.float32)
        
        original_shape = X_xy_tensor.shape
        
        if len(original_shape) == 0:
             raise ValueError("Input cannot be a scalar.")

        last_dim = original_shape[-1]
        if last_dim % 2 != 0:
            raise ValueError(
                f"Input tensor's last dimension must be even (shape *S, D*2), "
                f"but got shape {original_shape} (last dim is {last_dim})"
            )
        
        # Reshape from (*, D*2) back to (*, D, 2)
        new_shape = (*original_shape[:-1], last_dim // 2, 2)
        xy_pairs = X_xy_tensor.reshape(new_shape)
        
        # 1. Separate x and y
        x = xy_pairs[..., 0]
        y = xy_pairs[..., 1]
        
        # 2. Calculate theta using atan2
        theta = torch.atan2(y, x)
        
        # 3. Convert range from [-pi, pi] to [0, 2*pi]
        pi_tensor = torch.tensor(math.pi, device=theta.device, dtype=theta.dtype)
        two_pi = 2 * pi_tensor
        theta_positive = (theta + two_pi) % two_pi
        
        # 4. Scale from [0, 2*pi] to [0, 1]
        normalized = theta_positive / two_pi
        
        # 5. Scale from [0, 1] to [min_val_, max_val_]
        values = normalized * (self.max_val_ - self.min_val_) + self.min_val_
        
        return values


class ChunkedPerDimensionScaler:
    """
    Per-dimension Z-score normalization for gene features.
    
    Uses Welford's online algorithm for memory-efficient streaming computation
    of mean and variance per dimension.
    
    Interface matches ChunkedLogZScoreScaler for loader compatibility:
    - collect_phase_1_statistics() - Collect global bounds per dimension
    - _finalize_pass1() - Finalize phase 1
    - collect_phase_2_statistics() - Collect mean/variance per dimension
    - finalize() - Compute final statistics
    - get_global_weights() - Return weights (for loader compatibility)
    - transform() - Apply Z-score normalization
    - inverse_transform() - Reverse normalization
    
    Args:
        gene_length: Number of dimensions in gene vector (D)
        device: PyTorch device for computations ('cpu', 'cuda', etc.)
        eps: Small constant for numerical stability
        weight_mode: How to compute weights in get_global_weights()
            - 'uniform': Return 1.0 for all genes (default, recommended for DiffEvo)
            - 'variance': Weight by inverse variance (penalize outliers)
            - 'mahalanobis': Weight by Mahalanobis distance from mean
    """
    
    def __init__(
        self,
        gene_length: int,
        device: Optional[Union[str, 'torch.device']] = None,
        eps: float = 1e-8,
        weight_mode: str = 'uniform'
    ):
        self.gene_length = gene_length
        self.device = self._resolve_device(device)
        self.eps = eps
        self.weight_mode = weight_mode
        
        # Phase 1: Global bounds per dimension
        self.dim_min = None  # (D,)
        self.dim_max = None  # (D,)
        
        # Phase 2: Welford's algorithm state per dimension
        self.count = 0
        self.mean = None     # (D,) running mean
        self.M2 = None       # (D,) sum of squared deviations
        
        # Final statistics
        self.std = None      # (D,) standard deviation per dimension
        self.var = None      # (D,) variance per dimension
        
        # Flags (match ChunkedLogZScoreScaler interface)
        self.pass1_done = False
        self.scaling_ready = False
    
    def _resolve_device(self, device):
        """Resolve device specification."""
        if torch is None:
            return None
        if device is None:
            return torch.device('cpu')
        return torch.device(device) if isinstance(device, str) else device
    
    def _to_numpy(self, data) -> np.ndarray:
        """Convert input to numpy array."""
        if torch is not None and isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy().astype(np.float32)
        return np.asarray(data, dtype=np.float32)
    
    def _to_tensor(self, data) -> 'torch.Tensor':
        """Convert input to torch tensor."""
        if torch is None:
            raise RuntimeError("PyTorch not available")
        if isinstance(data, torch.Tensor):
            return data.to(dtype=torch.float32, device=self.device)
        return torch.tensor(data, dtype=torch.float32, device=self.device)
    
    def reset(self):
        """Reset all statistics."""
        self.dim_min = None
        self.dim_max = None
        self.count = 0
        self.mean = None
        self.M2 = None
        self.std = None
        self.var = None
        self.pass1_done = False
        self.scaling_ready = False
    
    # =========================================================================
    # PHASE 1: Global Bounds Per Dimension
    # =========================================================================
    
    def collect_phase_1_statistics(self, genes_chunk: List) -> None:
        """
        Collect min/max bounds per dimension from a chunk of genes.
        
        Args:
            genes_chunk: List of gene vectors [[g1], [g2], ...] or single gene [g]
        """
        # Handle single gene vs list of genes
        if len(genes_chunk) == 0:
            return
        
        # Convert to 2D numpy array
        if isinstance(genes_chunk[0], (list, np.ndarray)):
            if hasattr(genes_chunk[0], '__len__') and len(genes_chunk[0]) == self.gene_length:
                # List of genes
                chunk_array = np.array(genes_chunk, dtype=np.float32)
            else:
                # Single gene wrapped in list
                chunk_array = np.array(genes_chunk, dtype=np.float32).reshape(-1, self.gene_length)
        else:
            # Flat list - assume single gene
            chunk_array = np.array(genes_chunk, dtype=np.float32).reshape(1, -1)
        
        if chunk_array.ndim == 1:
            chunk_array = chunk_array.reshape(1, -1)
        
        # Validate dimensions
        if chunk_array.shape[1] != self.gene_length:
            # Try to handle edge cases
            if chunk_array.size == self.gene_length:
                chunk_array = chunk_array.reshape(1, self.gene_length)
            else:
                return  # Skip incompatible data
        
        # Filter out NaN/Inf
        valid_mask = np.all(np.isfinite(chunk_array), axis=1)
        chunk_array = chunk_array[valid_mask]
        
        if chunk_array.shape[0] == 0:
            return
        
        # Update per-dimension min/max
        chunk_min = np.min(chunk_array, axis=0)
        chunk_max = np.max(chunk_array, axis=0)
        
        if self.dim_min is None:
            self.dim_min = chunk_min.copy()
            self.dim_max = chunk_max.copy()
        else:
            self.dim_min = np.minimum(self.dim_min, chunk_min)
            self.dim_max = np.maximum(self.dim_max, chunk_max)
    
    def _finalize_pass1(self) -> None:
        """Finalize phase 1 statistics."""
        if self.dim_min is None:
            # No data collected - use defaults
            self.dim_min = np.zeros(self.gene_length, dtype=np.float32)
            self.dim_max = np.ones(self.gene_length, dtype=np.float32)
        
        # Initialize phase 2 accumulators
        self.mean = np.zeros(self.gene_length, dtype=np.float32)
        self.M2 = np.zeros(self.gene_length, dtype=np.float32)
        self.count = 0
        
        self.pass1_done = True
    
    # =========================================================================
    # PHASE 2: Mean/Variance Per Dimension (Welford's Algorithm)
    # =========================================================================
    
    def collect_phase_2_statistics(self, genes_chunk: List) -> None:
        """
        Collect mean/variance statistics per dimension using Welford's algorithm.
        
        Args:
            genes_chunk: List of gene vectors [[g1], [g2], ...] or single gene [g]
        """
        if not self.pass1_done:
            self._finalize_pass1()
        
        # Handle empty input
        if len(genes_chunk) == 0:
            return
        
        # Convert to 2D numpy array (same logic as phase 1)
        if isinstance(genes_chunk[0], (list, np.ndarray)):
            if hasattr(genes_chunk[0], '__len__') and len(genes_chunk[0]) == self.gene_length:
                chunk_array = np.array(genes_chunk, dtype=np.float32)
            else:
                chunk_array = np.array(genes_chunk, dtype=np.float32).reshape(-1, self.gene_length)
        else:
            chunk_array = np.array(genes_chunk, dtype=np.float32).reshape(1, -1)
        
        if chunk_array.ndim == 1:
            chunk_array = chunk_array.reshape(1, -1)
        
        # Validate dimensions
        if chunk_array.shape[1] != self.gene_length:
            if chunk_array.size == self.gene_length:
                chunk_array = chunk_array.reshape(1, self.gene_length)
            else:
                return
        
        # Filter NaN/Inf
        valid_mask = np.all(np.isfinite(chunk_array), axis=1)
        chunk_array = chunk_array[valid_mask]
        
        if chunk_array.shape[0] == 0:
            return
        
        # Welford's parallel algorithm (batch update)
        batch_size = chunk_array.shape[0]
        batch_mean = np.mean(chunk_array, axis=0)
        batch_var = np.var(chunk_array, axis=0, ddof=0)  # Population variance
        
        new_count = self.count + batch_size
        delta = batch_mean - self.mean
        
        # Update mean
        self.mean = self.mean + delta * (batch_size / new_count)
        
        # Update M2 (sum of squared deviations)
        batch_M2 = batch_var * batch_size
        self.M2 = self.M2 + batch_M2 + (delta ** 2) * (self.count * batch_size / new_count)
        
        self.count = new_count
    
    def finalize(self) -> None:
        """Finalize statistics and prepare for transformation."""
        if not self.pass1_done:
            self._finalize_pass1()
        
        if self.count < 2:
            # Not enough data - use unit variance
            self.var = np.ones(self.gene_length, dtype=np.float32)
            self.std = np.ones(self.gene_length, dtype=np.float32)
        else:
            # Sample variance
            self.var = self.M2 / (self.count - 1)
            self.std = np.sqrt(self.var + self.eps)
        
        # Prevent division by zero for constant dimensions
        self.std = np.maximum(self.std, self.eps)
        self.var = np.maximum(self.var, self.eps)
        
        # Convert to tensors if PyTorch available
        if torch is not None:
            self.mean_tensor = torch.tensor(self.mean, dtype=torch.float32, device=self.device)
            self.std_tensor = torch.tensor(self.std, dtype=torch.float32, device=self.device)
            self.var_tensor = torch.tensor(self.var, dtype=torch.float32, device=self.device)
        
        self.scaling_ready = True
    
    # =========================================================================
    # WEIGHTS (For Loader Compatibility)
    # =========================================================================
    
    def get_global_weights(self, genes: Union[np.ndarray, 'torch.Tensor']) -> Union[np.ndarray, 'torch.Tensor']:
        """
        Compute weights for genes (for loader compatibility).
        
        The weight interpretation depends on weight_mode:
        - 'uniform': All genes get weight 1.0 (recommended for DiffEvo)
        - 'variance': Weight by how close gene is to population mean
        - 'mahalanobis': Weight by Mahalanobis distance
        
        Args:
            genes: (N, D) array of genes or (D,) single gene
        
        Returns:
            (N,) weights or scalar weight
        """
        if not self.scaling_ready:
            raise RuntimeError("Scaler not ready. Call finalize() first.")
        
        use_torch = torch is not None and isinstance(genes, torch.Tensor)
        
        if use_torch:
            genes_np = genes.detach().cpu().numpy()
        else:
            genes_np = np.asarray(genes, dtype=np.float32)
        
        # Handle 1D input
        squeeze = False
        if genes_np.ndim == 1:
            genes_np = genes_np.reshape(1, -1)
            squeeze = True
        
        if self.weight_mode == 'uniform':
            # All genes get weight 1.0
            weights = np.ones(genes_np.shape[0], dtype=np.float32)
        
        elif self.weight_mode == 'variance':
            # Weight by inverse of squared deviation from mean
            # Genes closer to mean get higher weight
            z_scores = (genes_np - self.mean) / self.std
            avg_z_sq = np.mean(z_scores ** 2, axis=1)
            weights = 1.0 / (1.0 + avg_z_sq)
        
        elif self.weight_mode == 'mahalanobis':
            # Mahalanobis distance (assuming diagonal covariance)
            z_scores = (genes_np - self.mean) / self.std
            mahal_dist = np.sqrt(np.sum(z_scores ** 2, axis=1))
            # Convert distance to weight (closer = higher weight)
            weights = np.exp(-mahal_dist / np.sqrt(self.gene_length))
        
        else:
            weights = np.ones(genes_np.shape[0], dtype=np.float32)
        
        if squeeze:
            weights = weights[0]
        
        # Convert back to tensor if input was tensor
        if use_torch:
            return torch.tensor(weights, dtype=torch.float32, device=self.device)
        return weights
    
    # =========================================================================
    # TRANSFORMATION
    # =========================================================================
    
    def transform(self, genes: Union[np.ndarray, 'torch.Tensor']) -> Union[np.ndarray, 'torch.Tensor']:
        """
        Apply Z-score normalization: (x - mean) / std
        
        Args:
            genes: (N, D) or (D,) array/tensor of genes
        
        Returns:
            Normalized genes with mean≈0, std≈1 per dimension
        """
        if not self.scaling_ready:
            raise RuntimeError("Scaler not ready. Call finalize() first.")
        
        if torch is not None and isinstance(genes, torch.Tensor):
            genes = genes.to(dtype=torch.float32, device=self.device)
            return (genes - self.mean_tensor) / self.std_tensor
        else:
            genes = np.asarray(genes, dtype=np.float32)
            return (genes - self.mean) / self.std
    
    def inverse_transform(self, normalized: Union[np.ndarray, 'torch.Tensor']) -> Union[np.ndarray, 'torch.Tensor']:
        """
        Reverse Z-score normalization: x * std + mean
        
        Args:
            normalized: (N, D) or (D,) normalized genes
        
        Returns:
            Original-scale genes
        """
        if not self.scaling_ready:
            raise RuntimeError("Scaler not ready. Call finalize() first.")
        
        if torch is not None and isinstance(normalized, torch.Tensor):
            normalized = normalized.to(dtype=torch.float32, device=self.device)
            return normalized * self.std_tensor + self.mean_tensor
        else:
            normalized = np.asarray(normalized, dtype=np.float32)
            return normalized * self.std + self.mean
    
    # =========================================================================
    # PERSISTENCE
    # =========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get scaler statistics for persistence."""
        return {
            'gene_length': self.gene_length,
            'dim_min': self.dim_min.tolist() if self.dim_min is not None else None,
            'dim_max': self.dim_max.tolist() if self.dim_max is not None else None,
            'mean': self.mean.tolist() if self.mean is not None else None,
            'std': self.std.tolist() if self.std is not None else None,
            'var': self.var.tolist() if self.var is not None else None,
            'count': self.count,
            'pass1_done': self.pass1_done,
            'scaling_ready': self.scaling_ready,
            'weight_mode': self.weight_mode,
        }
    
    def load_statistics(self, stats: Dict[str, Any]) -> 'ChunkedPerDimensionScaler':
        """Load scaler statistics from dict."""
        self.gene_length = stats.get('gene_length', self.gene_length)
        
        if stats.get('dim_min') is not None:
            self.dim_min = np.array(stats['dim_min'], dtype=np.float32)
        if stats.get('dim_max') is not None:
            self.dim_max = np.array(stats['dim_max'], dtype=np.float32)
        if stats.get('mean') is not None:
            self.mean = np.array(stats['mean'], dtype=np.float32)
        if stats.get('std') is not None:
            self.std = np.array(stats['std'], dtype=np.float32)
        if stats.get('var') is not None:
            self.var = np.array(stats['var'], dtype=np.float32)
        
        self.count = stats.get('count', 0)
        self.pass1_done = stats.get('pass1_done', False)
        self.scaling_ready = stats.get('scaling_ready', False)
        self.weight_mode = stats.get('weight_mode', 'uniform')
        
        # Create tensors if scaling is ready
        if self.scaling_ready and torch is not None:
            self.mean_tensor = torch.tensor(self.mean, dtype=torch.float32, device=self.device)
            self.std_tensor = torch.tensor(self.std, dtype=torch.float32, device=self.device)
            self.var_tensor = torch.tensor(self.var, dtype=torch.float32, device=self.device)
        
        return self
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_dimension_stats(self) -> Dict[str, Any]:
        """Get summary statistics about dimension variances."""
        if not self.scaling_ready:
            return {'error': 'Scaler not ready'}
        
        return {
            'mean_range': [float(self.mean.min()), float(self.mean.max())],
            'std_range': [float(self.std.min()), float(self.std.max())],
            'var_range': [float(self.var.min()), float(self.var.max())],
            'var_ratio': float(self.var.max() / (self.var.min() + self.eps)),
            'count': self.count,
            'constant_dims': int(np.sum(self.std < self.eps * 10)),
        }
    
    def check_normalization_quality(self, genes: Union[np.ndarray, 'torch.Tensor']) -> Dict[str, float]:
        """
        Check how well normalization worked on a sample.
        
        Returns dict with per-dimension mean and std after transformation.
        Ideally: mean ≈ 0, std ≈ 1 for all dimensions.
        """
        if not self.scaling_ready:
            return {'error': 'Scaler not ready'}
        
        normalized = self.transform(genes)
        
        if torch is not None and isinstance(normalized, torch.Tensor):
            normalized = normalized.detach().cpu().numpy()
        
        return {
            'normalized_mean_range': [float(normalized.mean(axis=0).min()), 
                                       float(normalized.mean(axis=0).max())],
            'normalized_std_range': [float(normalized.std(axis=0).min()), 
                                      float(normalized.std(axis=0).max())],
            'normalized_mean_avg': float(np.abs(normalized.mean(axis=0)).mean()),
            'normalized_std_avg': float(normalized.std(axis=0).mean()),
        }


import math
class ChunkedRankScaler:
    """
    Rank-Based Scaler (v2 - Drop-in Replacement).
    ---------------------------------------------
    Matches the API of ChunkedLogZScoreScaler for easy integration.
    
    Why this is better for Genetic Algorithms:
    1. Outlier Proof: A 'super-gene' with fitness 1,000,000 doesn't break the curve 
       if the rest are ~10. It is simply ranked #1.
    2. Minimization/Maximization Agnostic: Handles -1 and 1 targets identicaly via sorting.
    3. Constant Selection Pressure: Ensures the best genes always breed, even if 
       fitness differences become tiny in late-game convergence.
    """
    
    def __init__(self, 
                 evolution_target: int = 1, 
                 emphasis_factor: float = 2.0,
                 min_weight: float = 0.1,
                 device: Optional[Union[str, torch.device]] = None, 
                 eps: float = 1e-8,
                 log_eps: float = 1e-10,
                 max_history_items: int = 1_000_000): # Safety limit for memory
        
        if evolution_target not in [1, -1]:
            raise ValueError("evolution_target must be 1 or -1")
            
        self.evolution_target = evolution_target
        self.emphasis_factor = emphasis_factor
        self.min_weight = min_weight
        self.device = self._resolve_device(device)
        self.eps = eps
        self.max_history_items = max_history_items
        
        # Buffer for Phase 1
        self.raw_scores_buffer = []
        self.total_buffered = 0
        
        # Finalized State
        self.sorted_reference = None
        self.reference_size = 0
        self.scaling_ready = False
        
        # Stats for persistence
        self.global_min = None
        self.global_max = None

        self.pass1_done = False

    def _resolve_device(self, device):
        if device is None: return torch.device('cpu')
        return torch.device(device) if isinstance(device, str) else device
    
    def _to_tensor(self, data):
        if isinstance(data, torch.Tensor):
            return data.to(dtype=torch.float32, device=self.device)
        return torch.tensor(data, dtype=torch.float32, device=self.device)

    def reset(self):
        self.raw_scores_buffer = []
        self.total_buffered = 0
        self.sorted_reference = None
        self.reference_size = 0
        self.scaling_ready = False
        self.global_min = None
        self.global_max = None

    # --- PHASE 1: COLLECT DATA ---
    def collect_phase_1_statistics(self, fitness_chunk):
        """
        Collects scores to build the global rank reference.
        Implements Reservoir Sampling if data exceeds max_history_items.
        """
        chunk = self._to_tensor(fitness_chunk)
        if chunk.numel() == 0: return
        chunk = chunk.flatten()
        
        # 1. Update basic Min/Max stats (useful for logging)
        c_min = chunk.min()
        c_max = chunk.max()
        if self.global_min is None:
            self.global_min = c_min
            self.global_max = c_max
        else:
            self.global_min = torch.minimum(self.global_min, c_min)
            self.global_max = torch.maximum(self.global_max, c_max)

        # 2. Reservoir Logic to prevent OOM
        remaining_space = self.max_history_items - self.total_buffered
        
        if remaining_space >= chunk.numel():
            # Standard: append all
            self.raw_scores_buffer.append(chunk)
            self.total_buffered += chunk.numel()
        elif remaining_space > 0:
            # Partial fill
            self.raw_scores_buffer.append(chunk[:remaining_space])
            self.total_buffered += remaining_space
        else:
            # Buffer full: Randomly replace existing items (Reservoir Sampling)
            # Simplification: We just stop collecting or do random replace.
            # Random replacement is computationally expensive on GPU tensors list.
            # Strategy: Just keep the first 1M items (usually sufficient for GA distribution).
            pass

    def _finalize_pass1(self):
        self.pass1_done = True
        # Rank scaler doesn't need intermediate calculations
        pass

    # --- PHASE 2: NO-OP ---
    def collect_phase_2_statistics(self, fitness_chunk):
        """
        Rank Scaler gets all it needs from Pass 1. 
        We ignore Pass 2 to avoid duplicating data in the buffer.
        """
        pass

    # --- FINALIZE ---
    def finalize(self):
        if not self.raw_scores_buffer:
            # Handle empty history
            self.sorted_reference = torch.tensor([0.0], device=self.device)
            self.reference_size = 1
            self.scaling_ready = True
            return

        # 1. Concatenate
        all_scores = torch.cat(self.raw_scores_buffer)
        
        # 2. Sort ASCENDING (Always ascending for searchsorted)
        # We handle Min/Max logic during the lookup, not the sort.
        self.sorted_reference, _ = torch.sort(all_scores)
        self.reference_size = self.sorted_reference.numel()
        
        # Clear buffer to free memory, keep only the sorted tensor
        self.raw_scores_buffer = [] 
        self.scaling_ready = True

    # --- WEIGHT CALCULATION (The Core Logic) ---
    def get_global_weights(self, fitness_chunk):
        """
        Returns weights based on global rank.
        Formula: weight = (1 - normalized_rank) ^ emphasis
        """
        if not self.scaling_ready: raise RuntimeError("Call finalize() first.")
        
        chunk = self._to_tensor(fitness_chunk)
        if chunk.numel() == 0: return chunk

        # 1. Find Rank via Binary Search (searchsorted)
        # searchsorted returns the index where the value would be inserted to maintain order.
        # This effectively gives us the count of items smaller than the value.
        
        # Ensure chunk is same device
        chunk = chunk.to(self.sorted_reference.device)
        
        # ranks_asc = number of items strictly smaller
        ranks_asc = torch.searchsorted(self.sorted_reference, chunk).float()
        
        # 2. Convert to Normalized Rank (0.0 = Best, 1.0 = Worst)
        N = float(self.reference_size)
        
        if self.evolution_target == 1:
            # MAXIMIZATION: Larger is better.
            # Best item is at the end of sorted_reference.
            # rank_asc index is High for good items.
            # We want Normalized Rank to be 0 for Best.
            # norm_rank = (N - rank_asc) / N
            norm_rank = (N - ranks_asc) / N
        else:
            # MINIMIZATION: Smaller is better.
            # Best item is at start of sorted_reference.
            # rank_asc index is Low for good items.
            # norm_rank = rank_asc / N
            norm_rank = ranks_asc / N
            
        # Clamp for safety
        norm_rank = torch.clamp(norm_rank, 0.0, 1.0)
        
        # 3. Calculate Weight
        # (1 - rank) means: Best(rank 0) -> 1.0, Worst(rank 1) -> 0.0
        base_score = 1.0 - norm_rank
        weights = torch.pow(base_score, self.emphasis_factor)
        
        # 4. Apply Min Weight Floor
        weights = weights * (1.0 - self.min_weight) + self.min_weight
        
        return weights

    # --- API COMPATIBILITY ---
    def scale_chunk(self, fitness_chunk, clip_min=None, clip_max=None):
        """
        Returns a 'Z-score like' value based on rank, for compatibility.
        Maps rank to range [-2, 2] roughly, so downstream code expecting Z-scores works.
        """
        weights = self.get_global_weights(fitness_chunk)
        # Map [0, 1] weight back to roughly [-3, 3] Z-score
        # Inverse Sigmoid approximation or just linear
        # Let's simple linear map: weight 0.5 -> 0, weight 1.0 -> 3.0
        z_approx = (weights - 0.5) * 6.0 
        
        if clip_min is not None or clip_max is not None:
             z_approx = torch.clamp(z_approx, min=clip_min, max=clip_max)
        return z_approx

    def compute_sample_weights(self, fitness_chunk):
        return self.get_global_weights(fitness_chunk)

    def compute_chunk_probabilities(self, fitness_chunk):
        weights = self.get_global_weights(fitness_chunk)
        return torch.nn.functional.softmax(weights, dim=0)

    # --- PERSISTENCE ---
    def get_statistics(self):
        """
        Returns a compressed representation of the distribution (Percentiles).
        We don't want to save 1M items to JSON.
        """
        stats = {
            'global_min': self.global_min.item() if self.global_min is not None else None,
            'global_max': self.global_max.item() if self.global_max is not None else None,
            'reference_size': self.reference_size,
            'evolution_target': self.evolution_target,
            'scaling_ready': self.scaling_ready,
        }
        
        # Save compressed reference (Quantiles)
        # Save 1000 split points
        if self.sorted_reference is not None and self.reference_size > 0:
            if self.reference_size > 1000:
                indices = torch.linspace(0, self.reference_size - 1, steps=1000).long()
                compressed = self.sorted_reference[indices].tolist()
            else:
                compressed = self.sorted_reference.tolist()
            stats['compressed_reference'] = compressed
        else:
            stats['compressed_reference'] = []
            
        return stats

    def load_statistics(self, stats):
        self.global_min = self._to_tensor(stats.get('global_min'))
        self.global_max = self._to_tensor(stats.get('global_max'))
        self.evolution_target = stats.get('evolution_target', 1)
        self.scaling_ready = stats.get('scaling_ready', False)
        
        # Reconstruct Reference
        comp_ref = stats.get('compressed_reference', [])
        if comp_ref:
            # We assume the distribution between quantiles is linear enough for GA
            self.sorted_reference = self._to_tensor(comp_ref)
            self.reference_size = self.sorted_reference.numel()
        else:
             self.sorted_reference = None
             self.reference_size = 0