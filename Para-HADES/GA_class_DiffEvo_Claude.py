"""
Diffusion Evolution for Candidate Gene Generation
==================================================

Memory-efficient implementation of Diffusion Evolution algorithm based on:
"Diffusion Models are Evolutionary Algorithms" (Zhang et al., 2024)

Key Features:
- Streaming data loading for datasets larger than memory
- SLURM-aware dynamic chunk sizing
- Latent space projection for high-dimensional genes
- Integration with ChunkedLogZScoreScaler for fitness normalization

Mathematical Foundation:
- Evolution is modeled as the inverse of diffusion (denoising)
- Each individual estimates its "origin" x̂₀ via Bayesian weighted average
- Gaussian kernel provides locality: early steps = global, late steps = local
- Fitness weights determine selection pressure

Author: Implementation based on Zhang et al. (2024) arXiv:2410.02543
"""

import numpy as np
import torch
import torch.nn as nn
import math
import gc
import os
import psutil
import heapq
from typing import List, Dict, Any, Optional, Tuple, Union

# Import your existing utilities
try:
    from GA_utils_history_loader import ChunkedGeneHistoryLoader
    from GA_utils_scalers import ChunkedLogZScoreScaler, ChunkedPerDimensionScaler
except ImportError:
    # Allow standalone testing
    ChunkedGeneHistoryLoader = None
    ChunkedLogZScoreScaler = None
    ChunkedPerDimensionScaler = None


# ============================================================================
# DIFFUSION SCHEDULERS
# ============================================================================

class DiffusionScheduler:
    """
    Base scheduler for alpha values in diffusion process.
    
    Alpha schedule controls exploration vs exploitation:
    - High alpha (early): Large Gaussian kernel → global competition
    - Low alpha (late): Small Gaussian kernel → local refinement
    """
    
    def __init__(self, num_steps: int, eps: float = 1e-4):
        self.num_steps = num_steps
        self.eps = eps
        self._build_schedule()
    
    def _build_schedule(self):
        """Override in subclasses to define alpha schedule."""
        raise NotImplementedError
    
    def __iter__(self):
        """Iterate from T to 1, yielding (t, alpha_t, alpha_{t-1})."""
        for t in range(self.num_steps - 1, 0, -1):
            yield t, self.alpha[t], self.alpha[t - 1]
    
    def __len__(self):
        return self.num_steps - 1


class CosineScheduler(DiffusionScheduler):
    """
    Cosine alpha schedule (Nichol & Dhariwal, 2021).
    Recommended for fewer steps - smoother transition.
    
    α_t = (cos(πt/T) + 1) / 2
    """
    
    def _build_schedule(self):
        t = torch.linspace(0, torch.pi, self.num_steps)
        alpha = (torch.cos(t) + 1) / 2
        # Rescale to [eps, 1-eps] for numerical stability
        alpha = (alpha + self.eps) * (1 - self.eps) / (1 + self.eps)
        self.alpha = alpha


class LinearScheduler(DiffusionScheduler):
    """
    Linear alpha schedule.
    Simple but less smooth than cosine.
    
    α_t = 1 - t/T
    """
    
    def _build_schedule(self):
        self.alpha = torch.linspace(1 - self.eps, self.eps ** 2, self.num_steps)


class DDPMScheduler(DiffusionScheduler):
    """
    DDPM-style exponential schedule.
    α_t ≈ exp(-β₀t - γt²)
    """
    
    def _build_schedule(self):
        # Solve for β and γ such that α[0] ≈ 1-eps, α[T] ≈ eps
        T = self.num_steps
        beta = ((T ** 2) * np.log(1 / (1 - self.eps)) + np.log(self.eps)) / (T - 1)
        gamma = -T * (T * np.log(1 / (1 - self.eps)) + np.log(self.eps)) / (T - 1)
        t = torch.linspace(1.0 / T, 1.0, T)
        self.alpha = torch.exp(-beta * t - gamma * t.square())


# ============================================================================
# LATENT SPACE PROJECTION
# ============================================================================

class RandomProjection(nn.Module):
    """
    Random projection for latent space diffusion.
    
    Based on Johnson-Lindenstrauss lemma: random projection preserves
    pairwise distances with high probability.
    
    This allows running diffusion in low-dimensional space while
    keeping genes in their original high-dimensional space.
    
    Args:
        in_features: Original gene dimension (D)
        out_features: Latent dimension (d), typically 2-10
        normalize: Whether to normalize projection vectors
    """
    
    def __init__(self, in_features: int, out_features: int, normalize: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.normalize = normalize
        
        # Initialize random projection matrix
        # E_ij ~ N(0, 1/D) for distance preservation
        weight = torch.randn(out_features, in_features) / (in_features ** 0.5)
        
        if normalize:
            weight = weight / weight.norm(dim=1, keepdim=True)
        
        self.register_buffer('weight', weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project x from D dimensions to d dimensions."""
        return torch.mm(x, self.weight.T)


# ============================================================================
# BAYESIAN ESTIMATOR (Core of Diffusion Evolution)
# ============================================================================

class BayesianOriginEstimator:
    """
    Estimates the "origin" x̂₀ for each individual using Bayesian inference.
    
    This is the core of Diffusion Evolution (Equation 8 from paper):
    
    x̂₀(x_t) = (1/Z) Σⱼ [g(f(xⱼ)) · N(x_t; √α·xⱼ, 1-α) · xⱼ]
    
    Where:
    - g(f(x)) = fitness-to-probability mapping (from scaler)
    - N(x_t; √α·x, 1-α) = Gaussian kernel for locality
    - Z = normalization constant
    
    The Gaussian kernel makes each individual only "see" nearby neighbors,
    with the neighborhood size controlled by alpha.
    """
    
    def __init__(
        self,
        genes: torch.Tensor,          # (N, D) population genes
        fitness_weights: torch.Tensor, # (N,) fitness-based weights
        alpha: float,                  # Current alpha value
        latent: Optional[torch.Tensor] = None,  # (N, d) latent projections
        eps: float = 1e-9
    ):
        self.genes = genes
        self.fitness_weights = fitness_weights
        self.alpha = alpha
        self.latent = latent if latent is not None else genes
        self.eps = eps
        
        # Precompute scaled positions for efficiency
        self.sqrt_alpha = math.sqrt(alpha)
        # Clamp one_minus_alpha to prevent numerical issues when alpha -> 1
        self.one_minus_alpha = max(1.0 - alpha, 1e-4)
        self.sigma = math.sqrt(self.one_minus_alpha)
    
    def _gaussian_kernel(self, x_t: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """
        Compute Gaussian probability N(x_t; √α·μ, 1-α).
        
        Args:
            x_t: Query point (d,) in latent space
            mu: Reference points (N, d) in latent space
        
        Returns:
            (N,) probabilities
        """
        # μ_scaled = √α · μ
        mu_scaled = self.sqrt_alpha * mu
        
        # dist² = ||x_t - μ_scaled||²
        diff = x_t.unsqueeze(0) - mu_scaled  # (N, d)
        dist_sq = torch.sum(diff ** 2, dim=1)  # (N,)
        
        # N(x_t; μ_scaled, σ²I) ∝ exp(-dist²/(2σ²))
        # Clamp exponential argument to prevent underflow/overflow
        exp_arg = -dist_sq / (2 * self.one_minus_alpha + self.eps)
        exp_arg = torch.clamp(exp_arg, -50, 50)  # Prevent extreme values
        return torch.exp(exp_arg)
    
    def estimate_origin(self, x_t: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        """
        Estimate the high-fitness origin x̂₀ for a single individual.
        
        Args:
            x_t: Current gene (D,)
            z_t: Current latent position (d,)
        
        Returns:
            x̂₀: Estimated origin (D,)
        """
        # Compute Gaussian kernel in latent space
        p_diffusion = self._gaussian_kernel(z_t, self.latent)  # (N,)
        
        # Combine with fitness weights
        # prob_j = fitness_weight_j * p_diffusion_j
        prob = (self.fitness_weights + self.eps) * (p_diffusion + self.eps)
        
        # Normalize
        Z = torch.sum(prob) + self.eps
        prob_normalized = prob / Z  # (N,)
        
        # Weighted average of genes
        # x̂₀ = Σⱼ prob_j · xⱼ
        x0_est = torch.sum(prob_normalized.unsqueeze(1) * self.genes, dim=0)
        
        return x0_est
    
    def estimate_origins_batch(self, x_t: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        """
        Estimate origins for entire population (vectorized).
        
        Args:
            x_t: Current genes (N, D)
            z_t: Current latent positions (N, d)
        
        Returns:
            x̂₀: Estimated origins (N, D)
        """
        N = x_t.shape[0]
        device = x_t.device
        dtype = x_t.dtype
        
        # For memory efficiency, process in mini-batches if N is large
        batch_size = min(N, 256)  # Tune based on GPU memory
        x0_estimates = torch.zeros_like(x_t)
        
        for i in range(0, N, batch_size):
            end_i = min(i + batch_size, N)
            batch_z = z_t[i:end_i]  # (batch, d)
            
            # Compute all pairwise Gaussian kernels for this batch
            # batch_z: (batch, d), self.latent: (N, d)
            # mu_scaled: (N, d)
            mu_scaled = self.sqrt_alpha * self.latent  # (N, d)
            
            # diff: (batch, N, d)
            diff = batch_z.unsqueeze(1) - mu_scaled.unsqueeze(0)
            dist_sq = torch.sum(diff ** 2, dim=2)  # (batch, N)
            
            # Gaussian kernel: (batch, N)
            # Clamp exponential argument to prevent underflow/overflow
            exp_arg = -dist_sq / (2 * self.one_minus_alpha + self.eps)
            exp_arg = torch.clamp(exp_arg, -50, 50)
            p_diffusion = torch.exp(exp_arg)
            
            # Combine with fitness weights: (batch, N)
            prob = (self.fitness_weights.unsqueeze(0) + self.eps) * (p_diffusion + self.eps)
            
            # Normalize per query: (batch, N)
            Z = torch.sum(prob, dim=1, keepdim=True) + self.eps
            prob_normalized = prob / Z
            
            # Weighted sum: (batch, N) @ (N, D) -> (batch, D)
            x0_estimates[i:end_i] = torch.mm(prob_normalized, self.genes)
        
        return x0_estimates


# ============================================================================
# DDIM STEP (Denoising Update)
# ============================================================================

def ddim_step(
    x_t: torch.Tensor,      # Current population (N, D)
    x0_est: torch.Tensor,   # Estimated origins (N, D)
    alpha_t: float,         # Current alpha
    alpha_t_prev: float,    # Previous alpha (target)
    noise_scale: float = 1.0  # σ_m: mutation magnitude
) -> torch.Tensor:
    """
    One step of DDIM denoising (Equation 5 from paper).
    
    x_{t-1} = √α_{t-1} · x̂₀ + √(1-α_{t-1}-σ²) · ε̂ + σ · w
    
    Where:
    - x̂₀: Estimated origin (direction toward high fitness)
    - ε̂: Estimated noise (maintains structure)
    - w ~ N(0, I): Random noise (mutation)
    - σ: Controls exploration/exploitation trade-off
    
    Args:
        x_t: Current genes (N, D)
        x0_est: Estimated high-fitness targets (N, D)
        alpha_t: Current alpha (α_t)
        alpha_t_prev: Next alpha (α_{t-1})
        noise_scale: Mutation magnitude σ_m ∈ [0, 1]
    
    Returns:
        x_{t-1}: Updated genes (N, D)
    """
    sqrt_alpha_t = math.sqrt(alpha_t)
    sqrt_alpha_t_prev = math.sqrt(alpha_t_prev)
    sqrt_one_minus_alpha_t = math.sqrt(1.0 - alpha_t)
    
    # Compute ε̂ = (x_t - √α_t · x̂₀) / √(1-α_t)
    eps_est = (x_t - sqrt_alpha_t * x0_est) / (sqrt_one_minus_alpha_t + 1e-9)
    
    # DDPM sigma: σ = σ_m · √((1-α_{t-1})/(1-α_t)) · √(1 - α_t/α_{t-1})
    sigma_ddpm = math.sqrt((1 - alpha_t_prev) / (1 - alpha_t + 1e-9)) * \
                 math.sqrt(1 - alpha_t / (alpha_t_prev + 1e-9))
    sigma = noise_scale * sigma_ddpm
    
    # Compute coefficient for ε̂
    sqrt_one_minus_alpha_prev_minus_sigma_sq = math.sqrt(
        max(0, 1 - alpha_t_prev - sigma ** 2)
    )
    
    # x_{t-1} = √α_{t-1} · x̂₀ + √(1-α_{t-1}-σ²) · ε̂ + σ · w
    x_t_prev = sqrt_alpha_t_prev * x0_est + \
               sqrt_one_minus_alpha_prev_minus_sigma_sq * eps_est + \
               sigma * torch.randn_like(x_t)
    
    return x_t_prev


# ============================================================================
# MEMORY UTILITIES
# ============================================================================

def get_available_memory_bytes(use_slurm: bool = True) -> int:
    """
    Get available memory in bytes, respecting SLURM allocation.
    
    Priority:
    1. SLURM_MEM_PER_NODE
    2. SLURM_MEM
    3. SLURM_MEM_PER_CPU × CPUs
    4. psutil (system memory)
    """
    memory_bytes = None
    
    if use_slurm:
        try:
            # Check SLURM environment variables
            mem_per_node = os.environ.get('SLURM_MEM_PER_NODE', '').strip()
            slurm_mem = os.environ.get('SLURM_MEM', '').strip()
            mem_per_cpu = os.environ.get('SLURM_MEM_PER_CPU', '').strip()
            cpus = os.environ.get('SLURM_CPUS_PER_TASK', 
                   os.environ.get('SLURM_CPUS_ON_NODE', '')).strip()
            
            if mem_per_node:
                memory_bytes = _parse_slurm_memory(mem_per_node) * 1024 * 1024
            elif slurm_mem:
                memory_bytes = _parse_slurm_memory(slurm_mem) * 1024 * 1024
            elif mem_per_cpu and cpus:
                memory_bytes = _parse_slurm_memory(mem_per_cpu) * int(cpus) * 1024 * 1024
        except:
            pass
    
    if memory_bytes is None:
        try:
            memory_bytes = psutil.virtual_memory().available
        except:
            memory_bytes = 8 * 1024 ** 3  # Default 8GB
    
    return memory_bytes


def _parse_slurm_memory(mem_str: str) -> int:
    """Parse SLURM memory string to MB."""
    mem_str = mem_str.upper().strip()
    if mem_str.endswith('T'):
        return int(mem_str[:-1]) * 1024 * 1024
    elif mem_str.endswith('G'):
        return int(mem_str[:-1]) * 1024
    elif mem_str.endswith('M'):
        return int(mem_str[:-1])
    elif mem_str.endswith('K'):
        return max(1, int(mem_str[:-1]) // 1024)
    else:
        return int(mem_str)


def calculate_safe_pool_size(
    gene_length: int,
    safety_margin: float = 0.5,
    min_pool: int = 256,
    max_pool: int = 5000
) -> int:
    """
    Calculate safe population size based on available memory.
    
    Memory per gene ≈ gene_length × 4 (float32) + overhead
    
    Args:
        gene_length: Dimension of genes
        safety_margin: Fraction of memory to use (0.5 = 50%)
        min_pool: Minimum pool size
        max_pool: Maximum pool size
    
    Returns:
        Safe pool size
    """
    available = get_available_memory_bytes() * safety_margin
    
    # Estimate bytes per gene (gene + latent + overhead)
    bytes_per_gene = (gene_length * 4) + 1024  # float32 + dict overhead
    
    # Account for working memory (2x for computation)
    raw_capacity = int(available / (bytes_per_gene * 2))
    
    return max(min_pool, min(max_pool, raw_capacity))


def calculate_chunk_size(
    gene_length: int,
    safety_margin: float = 0.3,
    min_chunk: int = 100,
    max_chunk: int = 5000
) -> int:
    """
    Calculate optimal chunk size for streaming.
    
    Args:
        gene_length: Dimension of genes
        safety_margin: Fraction of memory per chunk
        min_chunk: Minimum chunk size
        max_chunk: Maximum chunk size
    
    Returns:
        Optimal chunk size
    """
    available = get_available_memory_bytes() * safety_margin
    bytes_per_gene = (gene_length * 4) + 1024
    raw_capacity = int(available / bytes_per_gene)
    
    return max(min_chunk, min(max_chunk, raw_capacity))


# ============================================================================
# LOSS/CONVERGENCE METRICS
# ============================================================================

def compute_diffusion_loss(
    x_t: torch.Tensor,
    x0_est: torch.Tensor,
    x_t_prev: torch.Tensor,
    fitness_weights: torch.Tensor,
    pool_fitness: torch.Tensor,
    evolution_target: int = 1
) -> Dict[str, float]:
    """
    Compute loss/convergence metrics for Diffusion Evolution.
    
    Since DiffusionEvo is not gradient-based, we compute proxy metrics:
    
    1. **movement_loss**: MSE between current position and estimated origin
       - Measures how much the population "wants" to move
       - Should decrease as population converges to high-fitness region
    
    2. **step_magnitude**: Actual movement in this step
       - ||x_{t-1} - x_t||² averaged over population
    
    3. **weighted_fitness**: Expected fitness based on selection weights
       - Σ w_i * f_i where w_i are normalized fitness weights
    
    4. **diversity_loss**: Population diversity (mean pairwise distance)
       - Low diversity = converged, high = exploring
    
    Args:
        x_t: Current genes before step (N, D)
        x0_est: Estimated origins (N, D)  
        x_t_prev: Genes after step (N, D)
        fitness_weights: Normalized selection weights (N,)
        pool_fitness: Raw fitness values of pool (N,)
        evolution_target: 1 for maximize, -1 for minimize
    
    Returns:
        Dict with loss metrics
    """
    N = x_t.shape[0]
    
    # 1. Movement loss: how far current position is from estimated origin
    # This is the "gradient" in diffusion space
    movement_loss = torch.mean((x_t - x0_est) ** 2).item()
    
    # 2. Step magnitude: how much we actually moved
    step_magnitude = torch.mean((x_t_prev - x_t) ** 2).item()
    
    # 3. Weighted fitness: expected fitness of selected population
    # Higher weights = higher fitness, so this should improve
    weights_normalized = fitness_weights / (fitness_weights.sum() + 1e-9)
    weighted_fitness = torch.sum(weights_normalized * pool_fitness).item()
    
    # 4. Diversity: sample-based pairwise distance (efficient for large N)
    if N > 100:
        # Sample 100 pairs for efficiency
        idx1 = torch.randint(0, N, (100,), device=x_t.device)
        idx2 = torch.randint(0, N, (100,), device=x_t.device)
        pairwise_dist = torch.mean(torch.sqrt(torch.sum((x_t_prev[idx1] - x_t_prev[idx2]) ** 2, dim=1) + 1e-9)).item()
    else:
        # Compute all pairwise for small populations
        diff = x_t_prev.unsqueeze(0) - x_t_prev.unsqueeze(1)  # (N, N, D)
        dist = torch.sqrt(torch.sum(diff ** 2, dim=2) + 1e-9)  # (N, N)
        # Get upper triangle (excluding diagonal)
        mask = torch.triu(torch.ones(N, N, device=x_t.device), diagonal=1).bool()
        pairwise_dist = dist[mask].mean().item() if mask.sum() > 0 else 0.0
    
    # 5. Convergence metric: ratio of step size to movement desire
    # If we're moving less than we "want" to, we're converging
    convergence_ratio = step_magnitude / (movement_loss + 1e-9)
    
    return {
        'movement_loss': movement_loss,
        'step_magnitude': step_magnitude,
        'weighted_fitness': weighted_fitness,
        'diversity': pairwise_dist,
        'convergence_ratio': convergence_ratio
    }


def compute_population_diversity(x: torch.Tensor, n_samples: int = 100) -> float:
    """Compute mean pairwise distance as diversity measure."""
    N = x.shape[0]
    if N < 2:
        return 0.0
    
    if N > n_samples:
        idx1 = torch.randint(0, N, (n_samples,), device=x.device)
        idx2 = torch.randint(0, N, (n_samples,), device=x.device)
        pairwise_dist = torch.mean(torch.sqrt(torch.sum((x[idx1] - x[idx2]) ** 2, dim=1) + 1e-9)).item()
    else:
        diff = x.unsqueeze(0) - x.unsqueeze(1)
        dist = torch.sqrt(torch.sum(diff ** 2, dim=2) + 1e-9)
        mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        pairwise_dist = dist[mask].mean().item() if mask.sum() > 0 else 0.0
    
    return pairwise_dist


def inject_diversity(
    x: torch.Tensor,
    diversity_ratio: float = 0.1,
    noise_scale: float = 0.3,
    method: str = 'hybrid'
) -> torch.Tensor:
    """
    Inject diversity into a converged population.
    
    Args:
        x: Population genes (N, D)
        diversity_ratio: Fraction of population to perturb (0.0-1.0)
        noise_scale: Magnitude of perturbation
        method: 'noise' (add noise), 'random' (replace with random), 'hybrid' (both)
    
    Returns:
        Perturbed population
    """
    N, D = x.shape
    n_perturb = max(1, int(N * diversity_ratio))
    
    # Select which individuals to perturb (randomly)
    perturb_idx = torch.randperm(N, device=x.device)[:n_perturb]
    
    x_new = x.clone()
    
    if method == 'noise':
        # Add Gaussian noise to selected individuals
        x_new[perturb_idx] = x_new[perturb_idx] + noise_scale * torch.randn(n_perturb, D, device=x.device)
    
    elif method == 'random':
        # Replace with completely random genes
        x_new[perturb_idx] = torch.randn(n_perturb, D, device=x.device)
    
    elif method == 'hybrid':
        # Half noise, half random
        n_noise = n_perturb // 2
        n_random = n_perturb - n_noise
        
        if n_noise > 0:
            noise_idx = perturb_idx[:n_noise]
            x_new[noise_idx] = x_new[noise_idx] + noise_scale * torch.randn(n_noise, D, device=x.device)
        
        if n_random > 0:
            random_idx = perturb_idx[n_noise:]
            x_new[random_idx] = torch.randn(n_random, D, device=x.device)
    
    # Clamp to valid range
    x_new = torch.clamp(x_new, -1.0, 1.0)
    
    return x_new


# ============================================================================
# MAIN DIFFUSION EVOLUTION FUNCTION
# ============================================================================

def createCandidateGene_DiffusionEvolution(
    args=None,
    EA_Class=None,
    genePopulation: Optional[List[Dict[str, Any]]] = None,
    number_of_candidate_genes: Optional[int] = None,
    build_in_params: Optional[Dict] = None,
    verbose: Optional[int] = None
) -> List[List[float]]:
    """
    Generate candidate genes using Diffusion Evolution.
    
    Memory-efficient implementation that:
    1. Streams data in chunks using ChunkedGeneHistoryLoader
    2. Uses ChunkedLogZScoreScaler for fitness normalization
    3. Projects to latent space for high-dimensional genes
    4. Runs iterative denoising to refine population toward high fitness
    
    Algorithm Overview:
    - Evolution is modeled as the inverse of diffusion (denoising)
    - Each individual estimates its "origin" x̂₀ via Bayesian weighted average
    - Gaussian kernel provides locality (early: global, late: local)
    - Output genes are interpolations toward high-fitness regions
    
    Args:
        args: Arguments object with:
            - path: Save path for disk loading
            - evolutionTarget: 1 (maximize) or -1 (minimize)
            - gpu: Whether to use GPU
            - agent_idx: Agent ID for disk loading
            - epoch: Current epoch for filtering
        EA_Class: Class with geneLength and geneFormat attributes
        genePopulation: Optional pre-loaded population [{'gene': [...], 'fitnessScore': float}, ...]
                       If None, loads from disk using streaming.
        number_of_candidate_genes: Number of genes to generate (REQUIRED)
        build_in_params: Dict with hyperparameters. Will be modified to include:
            - DiffusionEvo: Dict with algorithm parameters
            - DiffusionEvo-log: Dict with runtime statistics
            - loss: Float with the convergence loss metric
        verbose: Verbosity level:
            - None or 0: Silent
            - 1: Minimal (key results)
            - 2: Standard (progress and parameters)
            - 3: Debug (all details)
    
    Returns:
        List[List[float]]: New candidate genes [[gene1], [gene2], ...]
    
    Raises:
        ValueError: If required parameters are None
    """
    # ========================================================================
    # VALIDATION
    # ========================================================================
    if args is None:
        raise ValueError("args must be provided")
    if EA_Class is None:
        raise ValueError("EA_Class must be provided")
    if number_of_candidate_genes is None:
        raise ValueError("number_of_candidate_genes must be specified")
    if build_in_params is None:
        raise ValueError("build_in_params must be provided (can be empty dict)")
    
    # Normalize verbose level
    v = verbose if verbose is not None else 0
    
    def vprint(level: int, msg: str):
        if v >= level:
            print(msg, flush=True)
    
    vprint(1, "\n[DiffusionEvo] Starting Diffusion Evolution...")
    
    # ========================================================================
    # SETUP HYPERPARAMETERS
    # ========================================================================
    # Support both old 'DiffusionEvo' and new unified 'DiffusionEvo' parameter sections
    # Prefer 'DiffusionEvo' if it exists and has algorithm='unconditional'
    config_section = None
    if 'DiffEvo' in build_in_params:
        algo = build_in_params['DiffEvo'].get('algorithm', 'conditional')
        if algo == 'unconditional':
            config_section = 'DiffEvo'
    
    if config_section is None:
        config_section = 'DiffEvo'
    
    if config_section not in build_in_params:
        build_in_params[config_section] = {}
    
    de_params = build_in_params[config_section]
    
    # Set defaults if not provided
    de_params.setdefault('num_steps', 25)           # Denoising steps (T)
    de_params.setdefault('noise_scale', 0.5)        # Mutation magnitude σ_m
    de_params.setdefault('latent_dim', 'auto')      # Random projection dimension
    de_params.setdefault('alpha_schedule', 'cosine') # Schedule type
    de_params.setdefault('top_k_percentile', 25)    # What % to load (updated from 90)
    de_params.setdefault('pool_size', 'auto')       # Max population size
    de_params.setdefault('min_datapoints', None)    # Min genes to load
    de_params.setdefault('max_datapoints', None)    # Max genes to load
    de_params.setdefault('use_log_scaler', True)    # Use log scaler (renamed from use_scaler)
    
    # NEW: Diversity preservation parameters
    de_params.setdefault('min_diversity_ratio', 0.3)    # Min diversity as fraction of initial
    de_params.setdefault('diversity_injection_rate', 0.15)  # Fraction to perturb when stuck
    de_params.setdefault('adaptive_noise', True)        # Increase noise when diversity drops
    de_params.setdefault('early_stop_diversity', 0.1)   # Stop if diversity drops below this ratio
    de_params.setdefault('exploration_ratio', 0.1)      # Fraction of random genes to add to pool
    
    # Initialize log
    log_key = f'{config_section}-log'
    build_in_params[log_key] = {}
    de_log = build_in_params[log_key]
    
    # Initialize loss tracking
    loss_history = []
    
    # Device setup
    device = torch.device(
        'cuda' if getattr(args, 'gpu', False) and torch.cuda.is_available() else 'cpu'
    )
    vprint(2, f"[DiffusionEvo] Device: {device}")
    
    evolution_target = getattr(args, 'evolutionTarget', 1)
    gene_length = EA_Class.geneLength
    
    vprint(2, f"[DiffusionEvo] Gene length: {gene_length}, Evolution target: {evolution_target}")
    
    # ========================================================================
    # CALCULATE MEMORY-SAFE SIZES
    # ========================================================================
    if de_params['pool_size'] == 'auto':
        pool_size = calculate_safe_pool_size(gene_length)
    else:
        pool_size = de_params['pool_size']
    
    chunk_size = calculate_chunk_size(gene_length)
    
    # Auto-determine latent dimension
    if de_params['latent_dim'] == 'auto':
        # Use latent projection if gene_length > 50
        if gene_length > 50:
            latent_dim = min(10, max(2, int(np.sqrt(gene_length / 10))))
        else:
            latent_dim = gene_length  # No projection needed
    else:
        latent_dim = de_params['latent_dim']
    
    de_log['pool_size'] = pool_size
    de_log['chunk_size'] = chunk_size
    de_log['latent_dim'] = latent_dim
    de_log['use_latent_projection'] = (latent_dim < gene_length)
    
    vprint(2, f"[DiffusionEvo] Pool size: {pool_size}, Chunk size: {chunk_size}")
    vprint(2, f"[DiffusionEvo] Latent dim: {latent_dim} (projection: {latent_dim < gene_length})")
    
    # ========================================================================
    # LOAD POPULATION
    # ========================================================================
    if genePopulation is not None and len(genePopulation) > 0:
        pool = genePopulation[:pool_size]  # Cap to pool_size
        fitness_scaler = None
        gene_scaler = None
        vprint(1, f"[DiffusionEvo] Using provided population: {len(pool)} genes")
        
        # Fit gene scaler on provided population
        if len(pool) > 0 and ChunkedPerDimensionScaler is not None:
            gene_scaler = ChunkedPerDimensionScaler(
                gene_length=gene_length,
                device=device,
                weight_mode='uniform'
            )
            # Simulate the loader's phase 1/2 collection
            genes_list = [g['gene'] for g in pool if g.get('gene') is not None]
            gene_scaler.collect_phase_1_statistics(genes_list)
            gene_scaler._finalize_pass1()
            gene_scaler.collect_phase_2_statistics(genes_list)
            gene_scaler.finalize()
            vprint(2, f"[DiffusionEvo] Gene scaler fitted on provided population")
    else:
        pool, fitness_scaler, gene_scaler = _load_population_with_scaler(
            args, EA_Class, device, pool_size, chunk_size,
            de_params['top_k_percentile'],
            de_params['min_datapoints'],
            de_params['max_datapoints'],
            de_params['use_log_scaler'],
            evolution_target,
            verbose=v
        )
   
    # Fallback: return random genes if pool too small
    if not pool or len(pool) < 2:
        vprint(1, "[DiffusionEvo] Pool too small, returning random genes")
        de_log['status'] = 'fallback_random'
        build_in_params['loss'] = float('inf')  # No convergence possible
        return _generate_random_genes(number_of_candidate_genes, gene_length, device)
    
    # ========================================================================
    # LOG POOL STATISTICS
    # ========================================================================
    fitness_values = [g['fitnessScore'] for g in pool]
    de_log['pool_size_actual'] = len(pool)
    de_log['pool_fitness_min'] = min(fitness_values)
    de_log['pool_fitness_max'] = max(fitness_values)
    de_log['pool_fitness_mean'] = sum(fitness_values) / len(fitness_values)
    
    vprint(2, f"[DiffusionEvo] Pool loaded: {len(pool)} genes")
    vprint(2, f"[DiffusionEvo] Fitness range: [{de_log['pool_fitness_min']:.4f}, {de_log['pool_fitness_max']:.4f}]")
    
    # ========================================================================
    # PREPARE TENSORS
    # ========================================================================
    vprint(2, "[DiffusionEvo] Preparing tensors...")
    
    # Convert genes to tensor
    genes = torch.tensor(
        [g['gene'] for g in pool],
        dtype=torch.float32,
        device=device
    )
    
    # ========================================================================
    # ADD EXPLORATION GENES (prevents getting stuck in local optima)
    # ========================================================================
    exploration_ratio = de_params['exploration_ratio']
    if exploration_ratio > 0:
        n_explore = max(1, int(len(pool) * exploration_ratio))
        
        # Generate exploration genes: mix of random and perturbed elite
        n_random = n_explore // 2
        n_perturbed = n_explore - n_random
        
        exploration_genes = []
        
        # Completely random genes
        if n_random > 0:
            random_genes = torch.randn(n_random, gene_length, device=device)
            random_genes = torch.clamp(random_genes * 0.5, -1.0, 1.0)  # Scaled random
            exploration_genes.append(random_genes)
        
        # Perturbed copies of top genes (larger mutations)
        if n_perturbed > 0:
            top_indices = torch.randint(0, min(10, len(pool)), (n_perturbed,), device=device)
            perturbed = genes[top_indices] + torch.randn(n_perturbed, gene_length, device=device) * 0.4
            perturbed = torch.clamp(perturbed, -1.0, 1.0)
            exploration_genes.append(perturbed)
        
        if exploration_genes:
            exploration_genes = torch.cat(exploration_genes, dim=0)
            genes = torch.cat([genes, exploration_genes], dim=0)
            
            # Assign low but non-zero fitness to exploration genes
            # This lets them participate but not dominate
            median_fitness = float(np.median(fitness_values))
            if evolution_target == 1:
                explore_fitness = [median_fitness * 0.5] * n_explore  # Below median for maximize
            else:
                explore_fitness = [median_fitness * 1.5] * n_explore  # Above median for minimize
            
            fitness_values = fitness_values + explore_fitness
            vprint(2, f"[DiffusionEvo] Added {n_explore} exploration genes ({n_random} random, {n_perturbed} perturbed)")
    
    de_log['exploration_genes_added'] = int(len(genes) - len(pool))
    
    # Store raw fitness for loss computation
    pool_fitness = torch.tensor(fitness_values, dtype=torch.float32, device=device)
    
    # Compute fitness weights
    if fitness_scaler is not None and fitness_scaler.scaling_ready:
        fitness_tensor = torch.tensor(fitness_values, dtype=torch.float32, device=fitness_scaler.device)
        fitness_weights = fitness_scaler.get_global_weights(fitness_tensor).to(device)
        vprint(2, "[DiffusionEvo] Using fitness scaler weights")
    else:
        # Fallback: simple fitness-proportionate weights
        fitness_tensor = torch.tensor(fitness_values, dtype=torch.float32, device=device)
        if evolution_target == -1:
            # Minimization: lower fitness = higher weight
            fitness_weights = 1.0 / (fitness_tensor - fitness_tensor.min() + 1e-6)
        else:
            # Maximization: higher fitness = higher weight
            fitness_weights = fitness_tensor - fitness_tensor.min() + 1e-6
        fitness_weights = fitness_weights / fitness_weights.sum()
        vprint(2, "[DiffusionEvo] Using simple proportionate weights")
    
    de_log['weight_min'] = fitness_weights.min().item()
    de_log['weight_max'] = fitness_weights.max().item()
    de_log['weight_mean'] = fitness_weights.mean().item()
    
    # Apply gene normalization for better distance computation
    use_gene_normalization = (gene_scaler is not None and gene_scaler.scaling_ready)
    if use_gene_normalization:
        genes_normalized = gene_scaler.transform(genes)
        vprint(2, f"[DiffusionEvo] Applied gene Z-score normalization")
        stats = gene_scaler.get_dimension_stats()
        de_log['gene_mean_range'] = stats.get('mean_range', [None, None])
        de_log['gene_std_range'] = stats.get('std_range', [None, None])
        de_log['gene_var_ratio'] = stats.get('var_ratio', None)
    else:
        genes_normalized = genes
        vprint(2, "[DiffusionEvo] No gene normalization (scaler not available)")
    
    # Create latent projection if needed
    use_projection = (latent_dim < gene_length)
    if use_projection:
        projector = RandomProjection(gene_length, latent_dim, normalize=True)
        projector = projector.to(device)
        latent = projector(genes_normalized)  # Project normalized genes
        vprint(2, f"[DiffusionEvo] Projected to latent space: {gene_length} -> {latent_dim}")
    else:
        projector = None
        latent = genes_normalized
    
    # ========================================================================
    # RUN DIFFUSION EVOLUTION
    # ========================================================================
    vprint(2, f"[DiffusionEvo] Running {de_params['num_steps']} denoising steps...")
    
    # Initialize scheduler
    schedule_type = de_params['alpha_schedule']
    if schedule_type == 'cosine':
        scheduler = CosineScheduler(de_params['num_steps'])
    elif schedule_type == 'linear':
        scheduler = LinearScheduler(de_params['num_steps'])
    elif schedule_type == 'ddpm':
        scheduler = DDPMScheduler(de_params['num_steps'])
    else:
        scheduler = CosineScheduler(de_params['num_steps'])
    
    # Initialize population from Gaussian (standard diffusion start)
    # But we want to START from our loaded genes (evolution direction)
    # Work in normalized space for better distance computation
    x_t = genes_normalized.clone()
    z_t = latent.clone() if use_projection else x_t
    
    # Store original genes for reference (needed for weighted averaging)
    genes_for_averaging = genes_normalized  # Use normalized genes for Bayesian estimation
    
    # Denoising loop with diversity preservation
    noise_scale = de_params['noise_scale']
    min_diversity_ratio = de_params['min_diversity_ratio']
    diversity_injection_rate = de_params['diversity_injection_rate']
    adaptive_noise = de_params['adaptive_noise']
    early_stop_diversity = de_params['early_stop_diversity']
    
    # Track loss per step
    step_losses = []
    
    # Compute initial diversity for reference
    initial_diversity = compute_population_diversity(x_t)
    de_log['initial_diversity'] = initial_diversity
    vprint(2, f"[DiffusionEvo] Initial diversity: {initial_diversity:.4f}")
    
    # Track diversity injections
    diversity_injections = 0
    early_stopped = False
    effective_noise_scale = noise_scale
    
    for step_idx, (t, alpha_t, alpha_t_prev) in enumerate(scheduler):
        # Create estimator for current step
        # Use normalized genes for Bayesian weighted averaging
        estimator = BayesianOriginEstimator(
            genes=genes_for_averaging,  # Reference pool (normalized)
            fitness_weights=fitness_weights,
            alpha=alpha_t,
            latent=z_t if use_projection else genes_for_averaging
        )
        
        # Estimate origins (in normalized space)
        x0_est = estimator.estimate_origins_batch(x_t, z_t)
        
        # DDIM step with current noise scale
        x_t_prev = ddim_step(x_t, x0_est, alpha_t, alpha_t_prev, effective_noise_scale)
        
        # Compute loss metrics for this step
        loss_metrics = compute_diffusion_loss(
            x_t=x_t,
            x0_est=x0_est,
            x_t_prev=x_t_prev,
            fitness_weights=fitness_weights,
            pool_fitness=pool_fitness,
            evolution_target=evolution_target
        )
        step_losses.append(loss_metrics)
        
        current_diversity = loss_metrics['diversity']
        diversity_ratio = current_diversity / (initial_diversity + 1e-9)
        
        # ====================================================================
        # DIVERSITY PRESERVATION
        # ====================================================================
        
        # Check for diversity collapse
        if diversity_ratio < min_diversity_ratio:
            vprint(2, f"[DiffusionEvo] Step {step_idx+1}: Diversity collapse detected "
                      f"({diversity_ratio:.2%} of initial)")
            
            # Inject diversity
            x_t_prev = inject_diversity(
                x_t_prev,
                diversity_ratio=diversity_injection_rate,
                noise_scale=0.3,
                method='hybrid'
            )
            diversity_injections += 1
            
            # Adaptive noise: increase noise scale for remaining steps
            if adaptive_noise:
                effective_noise_scale = min(1.0, effective_noise_scale * 1.5)
                vprint(3, f"[DiffusionEvo] Increased noise scale to {effective_noise_scale:.3f}")
        
        # Check for severe collapse (early stop)
        if diversity_ratio < early_stop_diversity:
            vprint(2, f"[DiffusionEvo] Step {step_idx+1}: Severe diversity collapse "
                      f"({diversity_ratio:.2%}), early stopping")
            early_stopped = True
            x_t = x_t_prev
            if use_projection:
                z_t = projector(x_t)
            else:
                z_t = x_t
            break
        
        # Update x_t for next iteration
        x_t = x_t_prev
        
        # Update latent if using projection
        if use_projection:
            z_t = projector(x_t)
        else:
            z_t = x_t
        
        if v >= 3 and (step_idx + 1) % 5 == 0:
            vprint(3, f"  Step {step_idx + 1}/{len(scheduler)}, alpha={alpha_t:.4f}, "
                      f"movement_loss={loss_metrics['movement_loss']:.6f}, "
                      f"diversity={current_diversity:.4f} ({diversity_ratio:.1%})")
    
    # Log diversity preservation stats
    de_log['diversity_injections'] = diversity_injections
    de_log['early_stopped'] = early_stopped
    de_log['final_noise_scale'] = effective_noise_scale
    
    vprint(2, f"[DiffusionEvo] Denoising complete. Diversity injections: {diversity_injections}, "
              f"Early stopped: {early_stopped}")
    
    # ========================================================================
    # COMPUTE FINAL LOSS METRICS
    # ========================================================================
    
    # Average loss over all steps
    final_movement_loss = np.mean([s['movement_loss'] for s in step_losses])
    final_diversity = step_losses[-1]['diversity'] if step_losses else 0.0
    final_weighted_fitness = step_losses[-1]['weighted_fitness'] if step_losses else 0.0
    
    # Primary loss metric: movement_loss (how much population wants to move)
    # Lower = more converged to high-fitness region
    primary_loss = final_movement_loss
    
    # Store in build_in_params['loss']
    build_in_params['loss'] = primary_loss
    
    # Store detailed loss metrics in log
    de_log['loss_movement'] = final_movement_loss
    de_log['loss_diversity'] = final_diversity
    de_log['loss_weighted_fitness'] = final_weighted_fitness
    de_log['loss_initial_movement'] = step_losses[0]['movement_loss'] if step_losses else 0.0
    de_log['loss_final_movement'] = step_losses[-1]['movement_loss'] if step_losses else 0.0
    de_log['loss_reduction_ratio'] = (
        step_losses[-1]['movement_loss'] / (step_losses[0]['movement_loss'] + 1e-9)
        if step_losses else 1.0
    )
    
    # Also store step-by-step losses for detailed analysis
    de_log['loss_per_step'] = [s['movement_loss'] for s in step_losses]
    de_log['diversity_per_step'] = [s['diversity'] for s in step_losses]
    
    vprint(2, f"[DiffusionEvo] Loss: {primary_loss:.6f} (movement), "
              f"diversity: {final_diversity:.4f}, "
              f"weighted_fitness: {final_weighted_fitness:.4f}")
    vprint(2, f"[DiffusionEvo] Loss reduction: {de_log['loss_reduction_ratio']:.4f}x "
              f"({de_log['loss_initial_movement']:.6f} → {de_log['loss_final_movement']:.6f})")
    
    # ========================================================================
    # INVERSE TRANSFORM (back to original scale)
    # ========================================================================
    if use_gene_normalization:
        # Transform back to original scale
        x_t_original = gene_scaler.inverse_transform(x_t)
        vprint(2, "[DiffusionEvo] Applied inverse gene normalization")
    else:
        x_t_original = x_t
    
    # Clamp to valid gene range
    gene_min = getattr(EA_Class, 'geneMin', -1.0)
    lower_bound = 0.0 if gene_min == 0 else -1.0
    x_t_original = torch.clamp(x_t_original, lower_bound, 1.0)
    
    # ========================================================================
    # SELECT OUTPUT CANDIDATES
    # ========================================================================
    N_pool = x_t_original.shape[0]
    
    if number_of_candidate_genes <= N_pool:
        # Sample from refined population
        # Weight by fitness for higher-quality selection
        selection_probs = fitness_weights / fitness_weights.sum()
        indices = torch.multinomial(
            selection_probs,
            number_of_candidate_genes,
            replacement=False
        )
        candidates = x_t_original[indices]
    else:
        # Need more than pool size: sample with replacement + add noise for diversity
        indices = torch.multinomial(
            fitness_weights / fitness_weights.sum(),
            number_of_candidate_genes,
            replacement=True
        )
        candidates = x_t_original[indices]
        
        # Add small noise to duplicates for diversity
        noise = torch.randn_like(candidates) * 0.05
        candidates = torch.clamp(candidates + noise, lower_bound, 1.0)
    
    # ========================================================================
    # ENSURE OUTPUT DIVERSITY
    # ========================================================================
    output_diversity = compute_population_diversity(candidates)
    de_log['output_diversity'] = output_diversity
    
    # If output diversity is too low, inject some exploration
    min_output_diversity = initial_diversity * 0.2  # At least 20% of initial
    if output_diversity < min_output_diversity and number_of_candidate_genes > 2:
        vprint(2, f"[DiffusionEvo] Output diversity too low ({output_diversity:.4f}), injecting exploration")
        
        # Replace 20% of candidates with perturbed versions
        n_replace = max(1, number_of_candidate_genes // 5)
        replace_idx = torch.randperm(number_of_candidate_genes, device=device)[:n_replace]
        
        # Generate diverse replacements
        diverse_candidates = candidates[replace_idx] + torch.randn(n_replace, gene_length, device=device) * 0.3
        candidates[replace_idx] = torch.clamp(diverse_candidates, lower_bound, 1.0)
        
        # Recompute diversity
        output_diversity = compute_population_diversity(candidates)
        de_log['output_diversity_after_injection'] = output_diversity
        de_log['output_diversity_injection'] = True
    else:
        de_log['output_diversity_injection'] = False
    
    # Convert to list
    candidates_list = candidates.cpu().tolist()
    
    de_log['status'] = 'success'
    de_log['candidates_generated'] = len(candidates_list)
    de_log['gene_normalization_used'] = use_gene_normalization
    
    vprint(1, f"[DiffusionEvo] Generated {len(candidates_list)} candidates, loss={primary_loss:.6f}, "
              f"output_diversity={output_diversity:.4f}")
    
    # ========================================================================
    # CLEANUP
    # ========================================================================
    del genes, x_t, z_t, fitness_weights, genes_normalized, genes_for_averaging, x_t_original
    if use_projection:
        del latent
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return candidates_list


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _generate_random_genes(
    count: int,
    gene_length: int,
    device: torch.device
) -> List[List[float]]:
    """Generate random genes as fallback."""
    genes = torch.randn(count, gene_length, device=device)
    gene_min = getattr(EA_Class, 'geneMin', -1.0)
    lower_bound = 0.0 if gene_min == 0 else -1.0

    genes = torch.clamp(genes, lower_bound, 1.0)
    return genes.cpu().tolist()


def _load_population_with_scaler(
    args,
    EA_Class,
    device: torch.device,
    pool_size: int,
    chunk_size: int,
    top_k_percentile: float,
    min_datapoints: Optional[int],
    max_datapoints: Optional[int],
    use_scaler: bool,
    evolution_target: int,
    verbose: int = 0
) -> Tuple[List[Dict], Any, Any]:
    """
    Load elite population using streaming with scaler integration.
    
    This function leverages the loader's built-in scaler support:
    1. Creates fitness_scaler (ChunkedLogZScoreScaler) for fitness normalization
    2. Creates gene_scaler (ChunkedPerDimensionScaler) for per-dimension gene normalization
    3. Passes both scalers to loadTopKPercentileGeneHistoryAsListOfDicts
    4. The loader automatically handles all phase 1/2 statistics collection
    5. Uses min-heap to keep only the best pool_size genes
    
    Returns:
        Tuple of (population_list, fitness_scaler, gene_scaler)
        - population_list: List[Dict] with 'gene' and 'fitnessScore' keys
        - fitness_scaler: ChunkedLogZScoreScaler or None
        - gene_scaler: ChunkedPerDimensionScaler or None
    """
    def vprint(level: int, msg: str):
        if verbose >= level:
            print(msg, flush=True)
    
    if ChunkedGeneHistoryLoader is None:
        vprint(1, "[DiffusionEvo] ChunkedGeneHistoryLoader not available, returning empty pool")
        return [], None, None
    
    # Initialize fitness scaler (ChunkedLogZScoreScaler)
    if use_scaler and ChunkedLogZScoreScaler is not None:
        fitness_scaler = ChunkedLogZScoreScaler(
            evolution_target=evolution_target,
            emphasis_factor=2.0,
            min_weight=0.1,
            device=device
        )
    else:
        fitness_scaler = None
    
    # Initialize gene scaler (ChunkedPerDimensionScaler)
    # The loader will automatically call collect_phase_1/2_statistics and finalize
    if ChunkedPerDimensionScaler is not None:
        gene_scaler = ChunkedPerDimensionScaler(
            gene_length=EA_Class.geneLength,
            device=device,
            weight_mode='uniform'  # For DiffusionEvo, we just need normalization
        )
    else:
        gene_scaler = None
    
    # Initialize loader
    agent_id = getattr(args, 'agent_idx', getattr(args, 'agent_counter', 0))
    
    loader = ChunkedGeneHistoryLoader(
        savePath=args.path,
        agent_id=agent_id,
        geneFormat=getattr(EA_Class, 'geneFormat', 'json'),
        shuffle=True,
        shuffle_level='both',
        required_keys_and_types=['fitnessScore', 'gene'],
    )
    
    # Recalculate chunk size based on current memory
    current_chunk_size = calculate_chunk_size(EA_Class.geneLength)
    chunk_size = min(chunk_size, current_chunk_size)
    
    vprint(2, f"[DiffusionEvo] Loading top {100 - top_k_percentile}% genes...")
    vprint(3, f"[DiffusionEvo] Chunk size: {chunk_size}, Pool cap: {pool_size}")
    
    # Calculate min_datapoints - MUST be an integer before passing to loader
    if min_datapoints is None or not isinstance(min_datapoints, int):
        pop_size = getattr(args, 'populationSize', 1000)
        min_datapoints = max(10, min(int(pop_size * 0.05), 300))

    if max_datapoints is None or not isinstance(min_datapoints, int):
        pop_size = getattr(args, 'populationSize', 10000)
        max_datapoints = int(pop_size)
    
    vprint(3, f"[DiffusionEvo] min_datapoints: {min_datapoints}")
    
    # Stream and collect using min-heap
    import itertools
    top_k_heap = []  # (fitness_key, counter, gene_dict)
    _heap_counter = itertools.count()  # unique tiebreaker to avoid dict comparison
    total_processed = 0
    
    try:
        # The loader handles all scaler statistics collection automatically!
        for chunk, is_done in loader.loadTopKPercentileGeneHistoryAsListOfDicts(
            args=args,
            top_k_percentile=top_k_percentile,
            maximize=(evolution_target == 1),
            chunk_size=chunk_size,
            fitness_scaler=fitness_scaler,      # Loader handles phase 1/2
            gene_scaler=gene_scaler,            # Loader handles phase 1/2
            use_fitness_scaler_weights=False,   # We compute weights later
            use_gene_scaler_weights=False,      # We use transform() later
            n_feature_clusters=1,
            allEpochs=True,
            min_datapoints=min_datapoints,
            fit_fitness_scaler_on_top_k=True,
            fit_gene_scaler_on_top_k=True,
            max_datapoints=max_datapoints,
        ):
            for gene in chunk:
                fitness = gene.get('fitnessScore')
                if fitness is None:
                    continue
                try:
                    fitness = float(fitness)
                except (TypeError, ValueError):
                    continue
                if math.isnan(fitness) or math.isinf(fitness):
                    continue
                
                gene_data = gene.get('gene')
                if gene_data is None or len(gene_data) != EA_Class.geneLength:
                    continue
                
                # Min-heap key: for maximize, use fitness directly
                heap_key = fitness if evolution_target == 1 else -fitness
                
                if len(top_k_heap) < pool_size:
                    heapq.heappush(top_k_heap, (heap_key, next(_heap_counter), gene))
                elif heap_key > top_k_heap[0][0]:
                    heapq.heapreplace(top_k_heap, (heap_key, next(_heap_counter), gene))
            
            total_processed += len(chunk)
            
            if verbose >= 3:
                vprint(3, f"[DiffusionEvo] Processed {total_processed}, heap: {len(top_k_heap)}")
            
            # Recalculate chunk size periodically for memory safety
            if total_processed % 10000 == 0:
                new_chunk_size = calculate_chunk_size(EA_Class.geneLength)
                if new_chunk_size < chunk_size * 0.5:
                    vprint(2, f"[DiffusionEvo] Memory pressure detected, reducing operations")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            if is_done:
                break
    
    except Exception as e:
        vprint(1, f"[DiffusionEvo] Error during loading: {e}")
        import traceback
        traceback.print_exc()
        return [], fitness_scaler, gene_scaler
    
    # Log scaler status (the loader has already finalized them)
    if gene_scaler is not None and gene_scaler.scaling_ready:
        stats = gene_scaler.get_dimension_stats()
        vprint(2, f"[DiffusionEvo] Gene scaler fitted: count={gene_scaler.count}, "
               f"var_ratio={stats.get('var_ratio', 'N/A'):.2f}")
    
    if fitness_scaler is not None and fitness_scaler.scaling_ready:
        vprint(2, f"[DiffusionEvo] Fitness scaler fitted")
    
    # Extract from heap (best first)
    pool = []
    while top_k_heap:
        _, _, gene = heapq.heappop(top_k_heap)
        pool.append(gene)
    pool.reverse()
    
    vprint(2, f"[DiffusionEvo] Loaded {len(pool)} genes from {total_processed} processed")
    
    return pool, fitness_scaler, gene_scaler


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Diffusion Evolution - Standalone Test")
    print("=" * 60)
    
    # Mock classes
    class MockEAClass:
        geneLength = 100
        geneFormat = 'json'
    
    class MockArgs:
        path = '/tmp/test'
        evolutionTarget = 1
        gpu = False
        agent_idx = 0
        epoch = 1
    
    # Create mock population
    np.random.seed(42)
    mock_population = []
    for i in range(500):
        gene = np.random.randn(100).clip(-1, 1).tolist()
        fitness = 1.0 / (1.0 + np.sum(np.array(gene) ** 2))  # Higher fitness = closer to origin
        mock_population.append({
            'gene': gene,
            'fitnessScore': fitness
        })
    
    # Sort by fitness (for testing)
    mock_population.sort(key=lambda x: x['fitnessScore'], reverse=True)
    
    print(f"\nMock population: {len(mock_population)} genes")
    print(f"Fitness range: [{mock_population[-1]['fitnessScore']:.4f}, {mock_population[0]['fitnessScore']:.4f}]")
    
    # Test Diffusion Evolution
    build_in_params = {
        'DiffusionEvo': {
            'num_steps': 15,
            'noise_scale': 0.3,
            'latent_dim': 5,
        }
    }
    
    print("\n" + "-" * 40)
    print("Running Diffusion Evolution...")
    print("-" * 40)
    
    candidates = createCandidateGene_DiffusionEvolution(
        args=MockArgs(),
        EA_Class=MockEAClass(),
        genePopulation=mock_population,
        number_of_candidate_genes=50,
        build_in_params=build_in_params,
        verbose=2
    )
    
    print(f"\nGenerated {len(candidates)} candidates")
    print(f"Each candidate has {len(candidates[0])} genes")
    print(f"Sample gene values: {candidates[0][:5]}...")
    
    # Check loss was logged
    print(f"\n*** Loss logged: {build_in_params.get('loss', 'NOT SET')} ***")
    print(f"Loss details: {build_in_params.get('DiffusionEvo-log', {})}")
    
    # Evaluate candidates
    candidate_fitness = []
    for c in candidates:
        fitness = 1.0 / (1.0 + np.sum(np.array(c) ** 2))
        candidate_fitness.append(fitness)
    
    print(f"\nCandidate fitness range: [{min(candidate_fitness):.4f}, {max(candidate_fitness):.4f}]")
    print(f"Mean candidate fitness: {np.mean(candidate_fitness):.4f}")
    print(f"Original pool mean fitness: {np.mean([g['fitnessScore'] for g in mock_population]):.4f}")
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)