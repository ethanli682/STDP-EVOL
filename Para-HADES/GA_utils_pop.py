import numpy as np
from sklearn.cluster import MiniBatchKMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings
import sys

# Suppress warnings that frequently occur during clustering (e.g., MiniBatchKMeans initialization warnings)
warnings.filterwarnings('ignore')

# --- 1. CORE HELPER FUNCTIONS ---

def _final_sample_adjustment(selected_indices, n_samples, n_points):
    """
    Ensures the final set of indices exactly matches n_samples, handling both 
    oversampling (random subsampling) and undersampling (filling with random points).
    """
    selected_indices = np.array(list(set(selected_indices))) # Ensure uniqueness

    if len(selected_indices) == n_samples:
        return selected_indices

    # Undersampling: Need more points
    if len(selected_indices) < n_samples:
        remaining_indices = np.setdiff1d(np.arange(n_points), selected_indices)
        n_additional = n_samples - len(selected_indices)
        
        # Guard against asking for more samples than available
        n_to_take = min(n_additional, len(remaining_indices))
        
        if n_to_take > 0:
            additional_indices = np.random.choice(remaining_indices, size=n_to_take, replace=False)
            selected_indices = np.concatenate([selected_indices, additional_indices])

    # Oversampling: Have too many points
    elif len(selected_indices) > n_samples:
        selected_indices = np.random.choice(selected_indices, size=n_samples, replace=False)

    return selected_indices


def _stratified_random_fallback(fitness_scores, n_samples):
    """
    Robust fallback method: proportional stratified random sampling by fitness.
    This is guaranteed to work even if complex clustering fails.
    """
    n_points = len(fitness_scores)
    n_strata = min(10, n_samples, n_points) # Never more strata than samples or points
    
    if n_strata == 0:
        return np.array([])
        
    # Create fitness percentile bins
    fitness_percentiles = np.linspace(0, 100, n_strata + 1)
    fitness_bins = np.percentile(fitness_scores, fitness_percentiles)
    
    # Assign each point to a fitness stratum
    stratum_assignments = np.digitize(fitness_scores, fitness_bins) - 1
    stratum_assignments = np.clip(stratum_assignments, 0, n_strata - 1)
    
    selected_indices = []
    
    for stratum_id in range(n_strata):
        stratum_indices = np.where(stratum_assignments == stratum_id)[0]
        
        if len(stratum_indices) == 0:
            continue
            
        # Calculate proportional target samples for this stratum
        target_samples = max(1, int(n_samples * len(stratum_indices) / n_points))
        target_samples = min(target_samples, len(stratum_indices))
        
        if target_samples > 0:
            sampled_indices = np.random.choice(stratum_indices, size=target_samples, replace=False)
            selected_indices.extend(sampled_indices)
    
    # Final adjustment to ensure exact n_samples
    return _final_sample_adjustment(selected_indices, n_samples, n_points)


def _farthest_point_sampling(data, n_samples):
    """
    Helper function: Farthest Point Sampling.
    WARNING: O(N * n_samples) complexity. Only use on small subsets.
    """
    n_points = len(data)
    if n_samples >= n_points:
        return np.arange(n_points)
    
    # Start with a random point
    selected_indices = [np.random.randint(n_points)]
    
    # Compute squared distance from all points to the first selected point
    distances = np.sum((data - data[selected_indices[0]])**2, axis=1)
    
    for i in range(1, n_samples):
        # Find the point with the maximum minimum distance (farthest point)
        farthest_idx = np.argmax(distances)
        selected_indices.append(farthest_idx)
        
        # Update minimum distances: distance to the nearest selected point
        new_distances = np.sum((data - data[farthest_idx])**2, axis=1)
        distances = np.minimum(distances, new_distances)

    return selected_indices


def _grid_based_sampling(data, n_samples):
    """
    Helper function: Grid-based sampling for coverage (used in gmm_grid_hybrid_selection).
    """
    n_points, n_dims = data.shape
    if n_samples >= n_points:
        return np.arange(n_points)
        
    # Heuristic for grid divisions
    divisions_per_dim = int(np.ceil(n_samples ** (1.0 / n_dims)))
    divisions_per_dim = max(2, divisions_per_dim)

    grid_boundaries = []
    for dim in range(n_dims):
        dim_min, dim_max = data[:, dim].min(), data[:, dim].max()
        # Handle zero range case
        if np.isclose(dim_min, dim_max):
             boundaries = np.array([dim_min, dim_max + 1e-6])
        else:
             boundaries = np.linspace(dim_min, dim_max, divisions_per_dim + 1)
        grid_boundaries.append(boundaries)

    grid_assignments = np.zeros(n_points, dtype=int)
    for i, point in enumerate(data):
        cell_id = 0
        multiplier = 1
        for dim in range(n_dims):
            dim_cell = np.digitize(point[dim], grid_boundaries[dim]) - 1
            # Clip handles points exactly on the max boundary
            dim_cell = np.clip(dim_cell, 0, divisions_per_dim - 1)
            cell_id += dim_cell * multiplier
            multiplier *= divisions_per_dim
        grid_assignments[i] = cell_id
        
    selected_indices = []
    unique_cells = np.unique(grid_assignments)
    
    # Sample one point closest to the center from each occupied grid cell
    for cell_id in unique_cells:
        cell_points = np.where(grid_assignments == cell_id)[0]
        
        if len(cell_points) == 1:
            selected_indices.append(cell_points[0])
        else:
            cell_data = data[cell_points]
            cell_center = np.mean(cell_data, axis=0)
            distances = np.linalg.norm(cell_data - cell_center, axis=1)
            closest_idx = np.argmin(distances)
            selected_indices.append(cell_points[closest_idx])
            
    # Final adjustment to meet n_samples target
    return _final_sample_adjustment(selected_indices, n_samples, n_points)


# --- 2. THE SEVEN REQUESTED ALGORITHMS ---

def reservoir_sampling_selection(features, fitness_scores, n_samples, **kwargs):
    """
    Function: Reservoir Sampling Selection
    Performs a single-pass, memory-efficient uniform random sampling (Algorithm R), 
    ideal for streaming data of unknown length where full memory load is infeasible.
    
    NOTE: This implementation does not use fitness_scores or features to bias selection,
    as that would violate the single-pass/memory-efficient core design principle.

    Args:
        features: numpy array of shape (n_points, n_features)
        fitness_scores: numpy array of shape (n_points,)
        n_samples: int, desired number of samples to select

    Returns:
        numpy array of selected indices
    """
    n_points = len(features)
    if n_samples >= n_points:
        return np.arange(n_points)
    
    # Initialize the reservoir with the first n_samples indices
    reservoir = np.arange(n_samples, dtype=int)
    
    for i in range(n_samples, n_points):
        # Generate a random integer j between 0 and i (inclusive)
        j = np.random.randint(0, i + 1)
        
        # If j is less than n_samples, replace the j-th element of the reservoir
        # with the current index i.
        if j < n_samples:
            reservoir[j] = i
            
    return reservoir


def stratified_kmeans_selection(features, fitness_scores, n_samples, **kwargs):
    """
    Function 1: Stratified K-Means Selection (Robust Production Version)
    Stratifies by fitness, clusters by feature space. Uses MiniBatchKMeans for scale.
    """
    n_points = len(features)
    if n_samples >= n_points:
        return np.arange(n_points)
    
    # Determine number of fitness strata (adaptive)
    n_strata = min(20, max(5, n_samples // 50), n_points)
    
    # Create fitness percentile bins
    fitness_percentiles = np.linspace(0, 100, n_strata + 1)
    fitness_bins = np.percentile(fitness_scores, fitness_percentiles)
    
    # Assign each point to a fitness stratum
    stratum_assignments = np.digitize(fitness_scores, fitness_bins) - 1
    stratum_assignments = np.clip(stratum_assignments, 0, n_strata - 1)
    
    selected_indices = []
    
    for stratum_id in range(n_strata):
        stratum_indices = np.where(stratum_assignments == stratum_id)[0]
        
        if len(stratum_indices) == 0:
            continue
            
        stratum_size = len(stratum_indices)
        target_samples = max(1, int(n_samples * stratum_size / n_points))
        target_samples = min(target_samples, stratum_size)
        
        if target_samples <= 0:
            continue
            
        if target_samples == 1 or stratum_size <= 1:
            # Simple median selection for small stratum
            stratum_fitness = fitness_scores[stratum_indices]
            median_fitness = np.median(stratum_fitness)
            closest_idx = np.argmin(np.abs(stratum_fitness - median_fitness))
            selected_indices.append(stratum_indices[closest_idx])
            
        elif stratum_size > target_samples:
            # Use MiniBatchKMeans for high-throughput clustering
            stratum_features = features[stratum_indices]
            
            try:
                # Use MiniBatchKMeans for better scalability
                kmeans = MiniBatchKMeans(
                    n_clusters=target_samples,
                    random_state=kwargs.get('random_state', 42),
                    batch_size=kwargs.get('batch_size', 1024),
                    n_init=kwargs.get('n_init_kmeans', 3)
                )
                kmeans.fit(stratum_features)
                
                # Select the point closest to the centroid for each cluster
                for cluster_id in range(target_samples):
                    cluster_mask = kmeans.labels_ == cluster_id
                    if not np.any(cluster_mask):
                        continue
                        
                    cluster_indices = stratum_indices[cluster_mask]
                    cluster_features = stratum_features[cluster_mask]
                    centroid = kmeans.cluster_centers_[cluster_id]
                    
                    distances = np.sum((cluster_features - centroid) ** 2, axis=1)
                    closest_idx = np.argmin(distances)
                    selected_indices.append(cluster_indices[closest_idx])
            
            except Exception as e:
                # Fallback within stratum to random sampling if K-Means fails
                print(f"Stratum K-Means failed: {e}. Falling back to random sampling in stratum.", file=sys.stderr)
                sampled_indices = np.random.choice(stratum_indices, size=target_samples, replace=False)
                selected_indices.extend(sampled_indices)
                
        else:
            # If stratum size is slightly larger than target, take all
            selected_indices.extend(stratum_indices)

    # Final adjustment to ensure exact n_samples
    return _final_sample_adjustment(selected_indices, n_samples, n_points)


def gmm_intelligent_selection(features, fitness_scores, n_samples, fitness_weight=0.3, **kwargs):
    """
    Function 2: Gaussian Mixture Model + Intelligent Sampling (Robust Production Version)
    Models joint feature-fitness space and samples probabilistically.
    """
    n_points = len(features)
    if n_samples >= n_points:
        return np.arange(n_points)
        
    # 1. Prepare joint feature-fitness space and scale
    fitness_min, fitness_max = fitness_scores.min(), fitness_scores.max()
    normalized_fitness = (fitness_scores - fitness_min) / (fitness_max - fitness_min) if fitness_max > fitness_min else np.zeros_like(fitness_scores)
    
    weighted_fitness = normalized_fitness.reshape(-1, 1) * fitness_weight
    joint_features = np.hstack([features, weighted_fitness])
    
    # Scale data for GMM stability
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(joint_features)

    # Adaptive subsampling for model training if dataset is huge (memory management)
    subset_size = min(kwargs.get('max_gmm_subset_size', 50000), n_points)
    if n_points > subset_size:
        subset_indices = np.random.choice(n_points, size=subset_size, replace=False)
        train_data = scaled_data[subset_indices]
    else:
        train_data = scaled_data

    # 2. Determine GMM components
    n_components = min(kwargs.get('max_components', 50), n_samples // 10, len(train_data) // 100)
    best_n_components = max(n_components, 1)
    
    # Adaptive heuristic for n_init based on complexity
    default_n_init = 5
    if best_n_components < 5 and len(train_data) < 5000:
        # If the model is simple (few clusters) and data size is small, increase n_init for max stability
        default_n_init = 10 
    
    try:
        # Fit final GMM with robust settings
        final_gmm = GaussianMixture(
            n_components=best_n_components,
            covariance_type=kwargs.get('covariance_type', 'full'), # Prefer full for structure
            random_state=kwargs.get('random_state', 42),
            max_iter=kwargs.get('max_iter', 200),
            n_init=kwargs.get('n_init_gmm', default_n_init), # Increased initializations for robustness
            reg_covar=kwargs.get('reg_covar', 1e-4) # Increased regularization for stability
        )
        final_gmm.fit(train_data)
        
        # Predict assignments and probabilities on ALL data
        component_assignments = final_gmm.predict(scaled_data)
        component_probs = final_gmm.predict_proba(scaled_data)
        
    except Exception as e:
        print(f"GMM Intelligent Selection failed: {e}. Falling back to stratified random.", file=sys.stderr)
        return _stratified_random_fallback(fitness_scores, n_samples)
    
    # 3. Intelligent sampling strategy
    selected_indices = []
    min_samples_per_component = max(1, n_samples // (final_gmm.n_components * 3))
    
    for component_id in range(final_gmm.n_components):
        component_indices = np.where(component_assignments == component_id)[0]
        
        if len(component_indices) == 0:
            continue
        
        component_weight = final_gmm.weights_[component_id]
        target_samples = max(min_samples_per_component, int(n_samples * component_weight))
        target_samples = min(target_samples, len(component_indices))
        
        if target_samples > 0:
            # Use probability-weighted sampling within component
            component_probs_subset = component_probs[component_indices, component_id]
            # Guard against division by zero if all probs are zero (shouldn't happen with predict_proba)
            sum_probs = component_probs_subset.sum()
            sampling_probs = component_probs_subset / sum_probs if sum_probs > 0 else None
            
            sampled_indices = np.random.choice(
                component_indices,
                size=target_samples,
                replace=False,
                p=sampling_probs
            )
            selected_indices.extend(sampled_indices)
            
    # Final adjustment to ensure exact n_samples
    return _final_sample_adjustment(selected_indices, n_samples, n_points)


def gmm_fps_hybrid_selection(features, fitness_scores, n_samples, **kwargs):
    """
    Function 3: GMM + FPS Hybrid Selection (Robust Production Version)
    GMM pre-selects a subset, then Farthest Point Sampling maximizes diversity on that subset.
    """
    n_points = len(features)
    if n_samples >= n_points:
        return np.arange(n_points)
        
    # 1. Prepare joint space and scale
    fitness_min, fitness_max = fitness_scores.min(), fitness_scores.max()
    normalized_fitness = (fitness_scores - fitness_min) / (fitness_max - fitness_min) if fitness_max > fitness_min else np.zeros_like(fitness_scores)
    combined_data = np.column_stack([features, normalized_fitness.reshape(-1, 1)])
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(combined_data)

    # 2. Stage 1: GMM clustering (Original → 10x target subset)
    stage1_size = min(n_samples * 10, n_points)
    n_components = min(kwargs.get('max_components', 50), stage1_size // 100)
    n_components = max(n_components, 1)
    
    try:
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type='diag', # Use diag for speed in pre-selection
            random_state=kwargs.get('random_state', 42),
            n_init=kwargs.get('n_init_gmm', 3),
            reg_covar=kwargs.get('reg_covar', 1e-4)
        )
        
        gmm.fit(scaled_data)
        component_assignments = gmm.predict(scaled_data)
        
        stage1_indices = []
        samples_per_component = stage1_size // n_components
        
        for component in range(n_components):
            component_indices = np.where(component_assignments == component)[0]
            if len(component_indices) == 0: continue
            
            n_samples_comp = min(samples_per_component, len(component_indices))
            if n_samples_comp <= 0: continue
            
            # Select points closest to component mean
            component_data = scaled_data[component_indices]
            component_mean = gmm.means_[component]
            distances = np.linalg.norm(component_data - component_mean, axis=1)
            closest_indices = np.argsort(distances)[:n_samples_comp]
            
            stage1_indices.extend(component_indices[closest_indices])
            
        stage1_indices = np.array(stage1_indices)[:stage1_size]
        
    except Exception as e:
        print(f"GMM clustering failed in Stage 1: {e}. Using random sampling for stage 1.", file=sys.stderr)
        stage1_indices = np.random.choice(n_points, size=stage1_size, replace=False)
        
    if len(stage1_indices) <= n_samples:
        return stage1_indices[:n_samples]

    # 3. Stage 2: FPS for maximum diversity (10x → target)
    stage1_data = scaled_data[stage1_indices]
    
    try:
        # FPS is O(N*K), so it MUST be run on the smaller stage1_data
        stage2_local_indices = _farthest_point_sampling(stage1_data, n_samples)
        final_indices = stage1_indices[stage2_local_indices]
    except Exception as e:
        print(f"FPS failed in Stage 2: {e}. Falling back to random selection from Stage 1 subset.", file=sys.stderr)
        final_indices = np.random.choice(stage1_indices, size=n_samples, replace=False)
        
    return _final_sample_adjustment(final_indices, n_samples, n_points)


def gmm_grid_hybrid_selection(features, fitness_scores, n_samples, **kwargs):
    """
    Function 4: GMM + Grid Hybrid Selection (Robust Production Version)
    Grid-based coverage pre-selects a subset, then GMM refines the selection.
    """
    n_points = len(features)
    if n_samples >= n_points:
        return np.arange(n_points)
        
    # 1. Prepare joint space and scale
    fitness_min, fitness_max = fitness_scores.min(), fitness_scores.max()
    normalized_fitness = (fitness_scores - fitness_min) / (fitness_max - fitness_min) if fitness_max > fitness_min else np.zeros_like(fitness_scores)
    combined_data = np.column_stack([features, normalized_fitness.reshape(-1, 1)])
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(combined_data)

    # 2. Stage 1: Grid-based pre-sampling for coverage (Original → 10x target)
    stage1_size = min(n_samples * 10, n_points)
    
    try:
        stage1_indices = _grid_based_sampling(scaled_data, stage1_size)
    except Exception as e:
        print(f"Grid sampling failed: {e}. Using random subset for stage 1.", file=sys.stderr)
        stage1_indices = np.random.choice(n_points, size=stage1_size, replace=False)

    if len(stage1_indices) <= n_samples:
        return stage1_indices[:n_samples]
        
    # 3. Stage 2: GMM clustering on grid samples (10x → target)
    stage1_data = scaled_data[stage1_indices]
    
    # Adaptive component count based on the reduced subset size
    n_components = min(n_samples // 10, len(stage1_indices) // 20, kwargs.get('max_components', 30))
    best_n_components = max(n_components, 1)

    try:
        gmm = GaussianMixture(
            n_components=best_n_components,
            covariance_type='diag',
            random_state=kwargs.get('random_state', 42),
            n_init=kwargs.get('n_init_gmm', 3),
            reg_covar=kwargs.get('reg_covar', 1e-4)
        )
        gmm.fit(stage1_data)
        component_assignments = gmm.predict(stage1_data)
        
        final_indices = []
        samples_per_component = n_samples // best_n_components
        remaining_samples = n_samples % best_n_components
        
        for component in range(best_n_components):
            component_local_indices = np.where(component_assignments == component)[0]
            if len(component_local_indices) == 0: continue
            
            n_samples_comp = samples_per_component
            if component < remaining_samples:
                n_samples_comp += 1
            
            n_samples_comp = min(n_samples_comp, len(component_local_indices))
            if n_samples_comp <= 0: continue
            
            # Select most representative points (closest to centroid)
            component_data = stage1_data[component_local_indices]
            component_mean = gmm.means_[component]
            distances = np.linalg.norm(component_data - component_mean, axis=1)
            closest_indices = np.argsort(distances)[:n_samples_comp]
            
            # Map back to original indices
            original_indices = stage1_indices[component_local_indices[closest_indices]]
            final_indices.extend(original_indices)
            
    except Exception as e:
        print(f"GMM clustering failed in Stage 2: {e}. Falling back to random selection from grid subset.", file=sys.stderr)
        final_indices = np.random.choice(stage1_indices, size=n_samples, replace=False)

    # Final adjustment
    return _final_sample_adjustment(final_indices, n_samples, n_points)


def adaptive_gmm_stratification(features, fitness_scores, n_samples, **kwargs):
    """
    Custom Method 1: Adaptive Stratification + GMM (Robust Production Version)
    Uses GMM to intelligently stratify the data, then samples representatively.
    """
    n_points = len(features)
    if n_samples >= n_points:
        return np.arange(n_points)

    # 1. Prepare joint feature-fitness space and scale
    fitness_min, fitness_max = fitness_scores.min(), fitness_scores.max()
    normalized_fitness = (fitness_scores - fitness_min) / (fitness_max - fitness_min) if fitness_max > fitness_min else np.zeros_like(fitness_scores)
    combined_data = np.column_stack([features, normalized_fitness.reshape(-1, 1)])
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(combined_data)

    # 2. GMM to find natural clusters/strata
    n_components = min(kwargs.get('max_components', 50), n_samples // 5, len(scaled_data) // 100)
    best_n_components = max(n_components, 1)

    try:
        # Adaptive heuristic for n_init based on complexity
        default_n_init = 5
        if best_n_components < 5 and len(scaled_data) < 5000:
            default_n_init = 10 

        gmm = GaussianMixture(
            n_components=best_n_components,
            covariance_type=kwargs.get('covariance_type', 'full'),
            random_state=kwargs.get('random_state', 42),
            n_init=kwargs.get('n_init_gmm', default_n_init),
            reg_covar=kwargs.get('reg_covar', 1e-4)
        )
        gmm.fit(scaled_data)
        component_assignments = gmm.predict(scaled_data)
        
    except Exception as e:
        print(f"Adaptive GMM Stratification failed: {e}. Falling back to stratified random.", file=sys.stderr)
        return _stratified_random_fallback(fitness_scores, n_samples)

    # 3. Proportional, Centroid-Representative sampling from each GMM stratum
    selected_indices = []
    
    for component_id in range(gmm.n_components):
        component_indices = np.where(component_assignments == component_id)[0]
        if len(component_indices) == 0: continue

        # Calculate proportional target samples for this cluster
        samples_in_component = len(component_indices)
        target_samples = max(1, int(n_samples * (samples_in_component / n_points)))
        target_samples = min(target_samples, samples_in_component)
        
        if target_samples <= 0: continue

        # Select the points closest to the GMM mean for this component
        component_data = scaled_data[component_indices]
        component_mean = gmm.means_[component_id]
        distances = np.linalg.norm(component_data - component_mean, axis=1)
        
        # Select closest points (centroid representatives)
        closest_indices_local = np.argsort(distances)[:target_samples]
        selected_indices.extend(component_indices[closest_indices_local])

    # Final adjustment
    return _final_sample_adjustment(selected_indices, n_samples, n_points)


def density_hierarchical_selection(features, fitness_scores, n_samples, **kwargs):
    """
    Custom Method 3: Density-Based Hierarchical Sampling (Robust Production Version)
    Identifies dense regions (DBSCAN) and samples proportionally from hierarchical structure.
    """
    n_points = len(features)
    if n_samples >= n_points:
        return np.arange(n_points)

    # 1. Prepare joint feature-fitness space and scale
    fitness_min, fitness_max = fitness_scores.min(), fitness_scores.max()
    normalized_fitness = (fitness_scores - fitness_min) / (fitness_max - fitness_min) if fitness_max > fitness_min else np.zeros_like(fitness_scores)
    combined_data = np.column_stack([features, normalized_fitness.reshape(-1, 1)])
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(combined_data)

    selected_indices = []

    try:
        # 2. Stage 1: Density-Based Clustering (DBSCAN)
        dbscan = DBSCAN(
            eps=kwargs.get('dbscan_eps', 0.5),
            min_samples=kwargs.get('dbscan_min_samples', 5),
        )
        dbscan_labels = dbscan.fit_predict(scaled_data)
        
        noise_point_indices = np.where(dbscan_labels == -1)[0]
        core_point_indices = np.where(dbscan_labels != -1)[0]
        
        # Determine number of clusters for hierarchical clustering
        n_clusters_hierarchical = min(n_samples // 5, len(core_point_indices) // 5, kwargs.get('max_hac_clusters', 50))
        n_clusters_hierarchical = max(n_clusters_hierarchical, 1)
        
        # 3. Hierarchical clustering on the CORE points
        if len(core_point_indices) > n_clusters_hierarchical:
            hierarchical_clustering = AgglomerativeClustering(
                n_clusters=n_clusters_hierarchical,
                linkage=kwargs.get('hac_linkage', 'ward')
            )
            core_assignments = hierarchical_clustering.fit_predict(scaled_data[core_point_indices])
            
            # Sample from each hierarchical cluster
            for cluster_id in range(n_clusters_hierarchical):
                cluster_local_indices = np.where(core_assignments == cluster_id)[0]
                cluster_original_indices = core_point_indices[cluster_local_indices]
                
                if len(cluster_original_indices) == 0: continue

                # Proportional sampling for core points
                target_samples = max(1, int(n_samples * len(cluster_original_indices) / n_points))
                target_samples = min(target_samples, len(cluster_original_indices))
                
                # Select the point closest to the cluster's centroid
                centroid = scaled_data[cluster_original_indices].mean(axis=0)
                distances = np.linalg.norm(scaled_data[cluster_original_indices] - centroid, axis=1)
                closest_indices_local = np.argsort(distances)[:target_samples]
                selected_indices.extend(cluster_original_indices[closest_indices_local])
        else:
            # Not enough core points, take all existing core points
            selected_indices.extend(core_point_indices)
            
        # 4. Sample from noise points (outliers)
        # Allocate remaining budget to noise, prioritized for discovery
        remaining_budget = n_samples - len(selected_indices)
        
        if remaining_budget > 0 and len(noise_point_indices) > 0:
            # We want at least 10% of the budget for noise, up to remaining budget
            noise_target = max(1, int(n_samples * 0.1))
            noise_samples_to_take = min(remaining_budget, noise_target, len(noise_point_indices))
            
            noise_samples = np.random.choice(noise_point_indices, size=noise_samples_to_take, replace=False)
            selected_indices.extend(noise_samples)
        
    except Exception as e:
        print(f"Density-Based Selection failed: {e}. Falling back to stratified random.", file=sys.stderr)
        return _stratified_random_fallback(fitness_scores, n_samples)
        
    # Final adjustment
    return _final_sample_adjustment(selected_indices, n_samples, n_points)



# import numpy as np
# from sklearn.cluster import KMeans, MiniBatchKMeans
# from sklearn.mixture import GaussianMixture
# from sklearn.model_selection import GridSearchCV
# from sklearn.preprocessing import StandardScaler
# import warnings
# warnings.filterwarnings('ignore')

# def _farthest_point_sampling(data, n_samples):
#     """
#     Helper function: Farthest Point Sampling for maximum diversity
#     """
#     n_points = len(data)
#     if n_samples >= n_points:
#         return np.arange(n_points)
    
#     # Choose method based on dataset size and memory constraints
#     memory_threshold = 50000  # Adjust based on available RAM
    
#     if n_points <= memory_threshold:
#         # Use vectorized version for smaller datasets (faster)
#         try:
#             from scipy.spatial.distance import cdist
            
#             # Pre-compute distance matrix
#             distance_matrix = cdist(data, data, metric='euclidean')
            
#             # Start with random point
#             selected_indices = [np.random.randint(n_points)]
#             selected_mask = np.zeros(n_points, dtype=bool)
#             selected_mask[selected_indices[0]] = True
            
#             # Track minimum distances to selected set
#             min_distances = distance_matrix[selected_indices[0]].copy()
#             min_distances[selected_indices[0]] = 0
            
#             for _ in range(n_samples - 1):
#                 # Find point with maximum distance to selected set
#                 min_distances[selected_mask] = 0
#                 farthest_idx = np.argmax(min_distances)
#                 selected_indices.append(farthest_idx)
#                 selected_mask[farthest_idx] = True
                
#                 # Update minimum distances (vectorized)
#                 new_distances = distance_matrix[farthest_idx]
#                 min_distances = np.minimum(min_distances, new_distances)
            
#             return np.array(selected_indices)
            
#         except ImportError:
#             # Fallback if scipy not available
#             pass
    
#     # Use incremental version for larger datasets (memory efficient)
#     selected_indices = [np.random.randint(n_points)]
#     selected_mask = np.zeros(n_points, dtype=bool)
#     selected_mask[selected_indices[0]] = True
    
#     # Calculate initial distances (vectorized)
#     min_distances = np.linalg.norm(data - data[selected_indices[0]], axis=1)
#     min_distances[selected_indices[0]] = 0
    
#     for _ in range(n_samples - 1):
#         # Find farthest point from selected set
#         min_distances[selected_mask] = 0
#         farthest_idx = np.argmax(min_distances)
#         selected_indices.append(farthest_idx)
#         selected_mask[farthest_idx] = True
        
#         # Update minimum distances (vectorized)
#         new_distances = np.linalg.norm(data - data[farthest_idx], axis=1)
#         min_distances = np.minimum(min_distances, new_distances)
    
#     return np.array(selected_indices)
    
# def gmm_fps_hybrid_selection(features, fitness_scores, n_samples):
#     """
#     Function 3: GMM + FPS Hybrid Selection
    
#     Two-stage approach: GMM clustering for structure discovery + FPS for diversity.
#     Good balance of diversity and computational efficiency.
    
#     Args:
#         features: numpy array of shape (n_points, n_features) with values in [-1, 1]
#         fitness_scores: numpy array of shape (n_points,) with fitness values
#         n_samples: int, desired number of samples to select
    
#     Returns:
#         numpy array of selected indices
#     """
#     n_points = len(features)
    
#     # Handle edge cases
#     if n_samples >= n_points:
#         return np.arange(n_points)
    
#     # Normalize fitness scores and combine with features
#     fitness_min, fitness_max = fitness_scores.min(), fitness_scores.max()
#     if fitness_max > fitness_min:
#         normalized_fitness = (fitness_scores - fitness_min) / (fitness_max - fitness_min)
#     else:
#         normalized_fitness = np.zeros_like(fitness_scores)
    
#     # Create combined feature-fitness space
#     combined_data = np.column_stack([features, normalized_fitness.reshape(-1, 1)])
#     scaler = StandardScaler()
#     scaled_data = scaler.fit_transform(combined_data)
    
#     # Stage 1: GMM clustering (Original → 10x target)
#     stage1_size = min(n_samples * 10, n_points)
#     n_components = min(stage1_size // 100, 50)  # More conservative: fewer components
#     n_components = max(n_components, 1)
    
#     try:
#         # More robust GMM settings
#         gmm = GaussianMixture(n_components=n_components, 
#                              covariance_type='diag',  # Use diagonal for speed in stage 1
#                              random_state=42, 
#                              max_iter=100,
#                              reg_covar=1e-4,  # Increased regularization
#                              init_params='k-means++',  # Better initialization
#                              n_init=3)  # Multiple initializations
#         gmm.fit(scaled_data)
#         component_assignments = gmm.predict(scaled_data)
        
#         # Sample proportionally from each component
#         stage1_indices = []
#         samples_per_component = stage1_size // n_components
        
#         for component in range(n_components):
#             component_mask = component_assignments == component
#             component_indices = np.where(component_mask)[0]
            
#             if len(component_indices) == 0:
#                 continue
                
#             # Sample from this component (closest to centroid + some random)
#             n_samples_comp = min(samples_per_component, len(component_indices))
            
#             if n_samples_comp <= 0:
#                 continue
                
#             if len(component_indices) <= n_samples_comp:
#                 stage1_indices.extend(component_indices)
#             else:
#                 # Select points closest to component mean
#                 component_data = scaled_data[component_mask]
#                 component_mean = gmm.means_[component]
#                 distances = np.linalg.norm(component_data - component_mean, axis=1)
#                 closest_indices = np.argsort(distances)[:n_samples_comp]
#                 selected_component_indices = component_indices[closest_indices]
#                 stage1_indices.extend(selected_component_indices)
        
#         stage1_indices = stage1_indices[:stage1_size]
        
#     except Exception as e:
#         print(f"GMM clustering failed: {e}, using random sampling for stage 1")
#         stage1_indices = np.random.choice(n_points, size=stage1_size, replace=False)
    
#     # If stage 1 already gives us enough samples, return them
#     if len(stage1_indices) <= n_samples:
#         return np.array(stage1_indices[:n_samples])
    
#     # Stage 2: FPS for maximum diversity (10x → target)
#     stage1_data = scaled_data[stage1_indices]
#     stage2_local_indices = _farthest_point_sampling(stage1_data, n_samples)
#     final_indices = [stage1_indices[i] for i in stage2_local_indices]
    
#     return np.array(final_indices)

# def gmm_grid_hybrid_selection(features, fitness_scores, n_samples):
#     """
#     Function 4: GMM + Grid Hybrid Selection
    
#     Two-stage approach: Grid-based coverage + GMM clustering for final selection.
#     Ensures good coverage of the feature-fitness space.
    
#     Args:
#         features: numpy array of shape (n_points, n_features) with values in [-1, 1]
#         fitness_scores: numpy array of shape (n_points,) with fitness values
#         n_samples: int, desired number of samples to select
    
#     Returns:
#         numpy array of selected indices
#     """
#     n_points = len(features)
    
#     # Handle edge cases
#     if n_samples >= n_points:
#         return np.arange(n_points)
    
#     # Normalize fitness scores and combine with features
#     fitness_min, fitness_max = fitness_scores.min(), fitness_scores.max()
#     if fitness_max > fitness_min:
#         normalized_fitness = (fitness_scores - fitness_min) / (fitness_max - fitness_min)
#     else:
#         normalized_fitness = np.zeros_like(fitness_scores)
    
#     # Create combined feature-fitness space
#     combined_data = np.column_stack([features, normalized_fitness.reshape(-1, 1)])
#     scaler = StandardScaler()
#     scaled_data = scaler.fit_transform(combined_data)
    
#     # Stage 1: Grid-based pre-sampling for coverage (Original → 10x target)
#     stage1_size = min(n_samples * 10, n_points)
#     stage1_indices = _grid_based_sampling(scaled_data, stage1_size)
    
#     # If stage 1 already gives us enough samples, return them
#     if len(stage1_indices) <= n_samples:
#         return np.array(stage1_indices[:n_samples])
    
#     # Stage 2: GMM clustering on grid samples (10x → target)
#     stage1_data = scaled_data[stage1_indices]
    
#     # More conservative component selection
#     n_components = min(n_samples // 10, len(stage1_indices) // 20, 30)  # Reduced max components
#     n_components = max(n_components, 1)
    
#     try:
#         # More robust GMM settings
#         gmm = GaussianMixture(n_components=n_components, 
#                              covariance_type='diag',  # Use diagonal for speed
#                              random_state=42, 
#                              max_iter=100,
#                              reg_covar=1e-4,  # Increased regularization
#                              init_params='k-means++',  # Better initialization
#                              n_init=3)  # Multiple initializations
#         gmm.fit(stage1_data)
#         component_assignments = gmm.predict(stage1_data)
        
#         # Equal sampling from each component
#         final_indices = []
#         samples_per_component = n_samples // n_components
#         remaining_samples = n_samples % n_components
        
#         for component in range(n_components):
#             component_mask = component_assignments == component
#             component_local_indices = np.where(component_mask)[0]
            
#             if len(component_local_indices) == 0:
#                 continue
                
#             # Number of samples for this component
#             n_samples_comp = samples_per_component
#             if component < remaining_samples:
#                 n_samples_comp += 1
                
#             if len(component_local_indices) <= n_samples_comp:
#                 # Take all points in this component
#                 for local_idx in component_local_indices:
#                     final_indices.append(stage1_indices[local_idx])
#             else:
#                 # Select most representative points (closest to centroid)
#                 component_data = stage1_data[component_mask]
#                 component_mean = gmm.means_[component]
#                 distances = np.linalg.norm(component_data - component_mean, axis=1)
#                 closest_indices = np.argsort(distances)[:n_samples_comp]
                
#                 for local_idx in closest_indices:
#                     original_idx = component_local_indices[local_idx]
#                     final_indices.append(stage1_indices[original_idx])
        
#         # If we don't have enough samples, fill with remaining points
#         if len(final_indices) < n_samples:
#             remaining_needed = n_samples - len(final_indices)
#             remaining_stage1 = [idx for idx in stage1_indices if idx not in final_indices]
#             if remaining_stage1:
#                 additional = np.random.choice(remaining_stage1, 
#                                             min(remaining_needed, len(remaining_stage1)), 
#                                             replace=False)
#                 final_indices.extend(additional)
        
#         final_indices = final_indices[:n_samples]
        
#     except Exception as e:
#         print(f"GMM clustering failed: {e}, using random sampling from grid")
#         final_indices = np.random.choice(stage1_indices, size=n_samples, replace=False)
    
#     return np.array(final_indices)

# def _grid_based_sampling(data, n_samples):
#     """
#     Helper function: Grid-based sampling for coverage
#     """
#     n_points, n_dims = data.shape
#     if n_samples >= n_points:
#         return np.arange(n_points)
    
#     # Create grid divisions per dimension
#     divisions_per_dim = int(np.ceil(n_samples ** (1.0 / n_dims)))
    
#     # Create grid boundaries for each dimension
#     grid_boundaries = []
#     for dim in range(n_dims):
#         dim_min, dim_max = data[:, dim].min(), data[:, dim].max()
#         boundaries = np.linspace(dim_min, dim_max, divisions_per_dim + 1)
#         grid_boundaries.append(boundaries)
    
#     # Assign each point to a grid cell
#     grid_assignments = np.zeros(n_points, dtype=int)
#     for i, point in enumerate(data):
#         cell_id = 0
#         multiplier = 1
#         for dim in range(n_dims):
#             # Find which grid cell this dimension falls into
#             dim_cell = np.digitize(point[dim], grid_boundaries[dim]) - 1
#             dim_cell = np.clip(dim_cell, 0, divisions_per_dim - 1)
#             cell_id += dim_cell * multiplier
#             multiplier *= divisions_per_dim
#         grid_assignments[i] = cell_id
    
#     # Sample one point from each occupied grid cell
#     selected_indices = []
#     unique_cells = np.unique(grid_assignments)
    
#     for cell_id in unique_cells:
#         cell_points = np.where(grid_assignments == cell_id)[0]
#         # Select the point closest to the cell center
#         if len(cell_points) == 1:
#             selected_indices.append(cell_points[0])
#         else:
#             # Calculate cell center
#             cell_data = data[cell_points]
#             cell_center = np.mean(cell_data, axis=0)
#             distances = np.linalg.norm(cell_data - cell_center, axis=1)
#             closest_idx = np.argmin(distances)
#             selected_indices.append(cell_points[closest_idx])
    
#     # If we have fewer samples than needed, add random samples
#     if len(selected_indices) < n_samples:
#         remaining_indices = np.setdiff1d(np.arange(n_points), selected_indices)
#         n_additional = min(n_samples - len(selected_indices), len(remaining_indices))
#         additional_indices = np.random.choice(remaining_indices, size=n_additional, replace=False)
#         selected_indices.extend(additional_indices)
    
#     # If we have too many samples, randomly subsample
#     if len(selected_indices) > n_samples:
#         selected_indices = np.random.choice(selected_indices, size=n_samples, replace=False)
    
#     return selected_indices

# def stratified_kmeans_selection(features, fitness_scores, n_samples):
#     """
#     Function 1: Stratified K-Means Selection
    
#     Fast and reliable method using fitness stratification + K-means clustering.
#     Guarantees coverage across fitness spectrum while maximizing feature diversity.
    
#     Args:
#         features: numpy array of shape (n_points, n_features) with values in [-1, 1]
#         fitness_scores: numpy array of shape (n_points,) with fitness values
#         n_samples: int, desired number of samples to select
    
#     Returns:
#         numpy array of selected indices
#     """
#     n_points = len(features)
    
#     # Handle edge cases
#     if n_samples >= n_points:
#         return np.arange(n_points)
    
#     # Determine number of fitness strata (adaptive based on dataset size)
#     n_strata = min(20, max(5, n_samples // 50))  # 5-20 strata
    
#     # Create fitness percentile bins
#     fitness_percentiles = np.linspace(0, 100, n_strata + 1)
#     fitness_bins = np.percentile(fitness_scores, fitness_percentiles)
    
#     # Assign each point to a fitness stratum
#     stratum_assignments = np.digitize(fitness_scores, fitness_bins) - 1
#     stratum_assignments = np.clip(stratum_assignments, 0, n_strata - 1)
    
#     selected_indices = []
    
#     for stratum_id in range(n_strata):
#         # Get points in this fitness stratum
#         stratum_mask = stratum_assignments == stratum_id
#         stratum_indices = np.where(stratum_mask)[0]
        
#         if len(stratum_indices) == 0:
#             continue
            
#         # Calculate samples to allocate to this stratum
#         stratum_size = len(stratum_indices)
#         target_samples = max(1, int(n_samples * stratum_size / n_points))
#         target_samples = min(target_samples, stratum_size)
        
#         if target_samples == 1:
#             # Just pick one sample (closest to median fitness in stratum)
#             stratum_fitness = fitness_scores[stratum_indices]
#             median_fitness = np.median(stratum_fitness)
#             closest_idx = np.argmin(np.abs(stratum_fitness - median_fitness))
#             selected_indices.append(stratum_indices[closest_idx])
#         else:
#             # Use K-means clustering within stratum
#             stratum_features = features[stratum_indices]
            
#             # Choose between KMeans and MiniBatchKMeans based on size
#             if len(stratum_indices) > 10000:
#                 kmeans = MiniBatchKMeans(n_clusters=target_samples, 
#                                        random_state=42, 
#                                        batch_size=1000,
#                                        n_init=3)
#             else:
#                 kmeans = KMeans(n_clusters=target_samples, 
#                               random_state=42, 
#                               n_init=10)
            
#             # Fit K-means
#             kmeans.fit(stratum_features)
            
#             # For each cluster, find the point closest to centroid
#             for cluster_id in range(target_samples):
#                 cluster_mask = kmeans.labels_ == cluster_id
#                 if not np.any(cluster_mask):
#                     continue
                    
#                 cluster_indices = stratum_indices[cluster_mask]
#                 cluster_features = stratum_features[cluster_mask]
#                 centroid = kmeans.cluster_centers_[cluster_id]
                
#                 # Find closest point to centroid
#                 distances = np.sum((cluster_features - centroid) ** 2, axis=1)
#                 closest_idx = np.argmin(distances)
#                 selected_indices.append(cluster_indices[closest_idx])
    
#     # Convert to numpy array and ensure we have the right number of samples
#     selected_indices = np.array(selected_indices)
    
#     # If we have too few samples, add random ones
#     if len(selected_indices) < n_samples:
#         remaining_indices = np.setdiff1d(np.arange(n_points), selected_indices)
#         n_additional = n_samples - len(selected_indices)
#         additional_indices = np.random.choice(remaining_indices, 
#                                             size=min(n_additional, len(remaining_indices)), 
#                                             replace=False)
#         selected_indices = np.concatenate([selected_indices, additional_indices])
    
#     # If we have too many samples, randomly subsample
#     if len(selected_indices) > n_samples:
#         selected_indices = np.random.choice(selected_indices, 
#                                           size=n_samples, 
#                                           replace=False)
    
#     return selected_indices

# def gmm_intelligent_selection(features, fitness_scores, n_samples, fitness_weight=0.3):
#     """
#     Function 2: Gaussian Mixture Model + Intelligent Sampling
    
#     More sophisticated method using GMM for cluster discovery with weighted sampling.
#     Automatically discovers natural data structure and handles complex manifolds.
#     Uses full covariance to capture feature correlations.
    
#     Args:
#         features: numpy array of shape (n_points, n_features) with values in [-1, 1]
#         fitness_scores: numpy array of shape (n_points,) with fitness values
#         n_samples: int, desired number of samples to select
#         fitness_weight: float, weight for fitness in joint space (default 0.3)
    
#     Returns:
#         numpy array of selected indices
#     """
#     n_points = len(features)
    
#     # Handle edge cases
#     if n_samples >= n_points:
#         return np.arange(n_points)
    
#     # Normalize fitness scores to [0, 1]
#     fitness_min, fitness_max = fitness_scores.min(), fitness_scores.max()
#     if fitness_max > fitness_min:
#         normalized_fitness = (fitness_scores - fitness_min) / (fitness_max - fitness_min)
#     else:
#         normalized_fitness = np.zeros_like(fitness_scores)
    
#     # Create joint feature-fitness space
#     weighted_fitness = normalized_fitness.reshape(-1, 1) * fitness_weight
#     joint_features = np.hstack([features, weighted_fitness])
    
#     # Adaptive subsampling strategy for large datasets
#     if n_points > 100000:
#         # Use larger subset for better model quality with complex manifolds
#         subset_size = min(50000, n_points // 10)
#         subset_indices = np.random.choice(n_points, size=subset_size, replace=False)
#         subset_joint_features = joint_features[subset_indices]
#         use_subset_for_final = True
#     elif n_points > 20000:
#         subset_size = min(15000, n_points // 2)
#         subset_indices = np.random.choice(n_points, size=subset_size, replace=False)
#         subset_joint_features = joint_features[subset_indices]
#         use_subset_for_final = False
#     else:
#         subset_joint_features = joint_features
#         use_subset_for_final = False
    
#     # Model selection: find optimal number of components using BIC
#     max_components = min(50, n_samples // 20, len(subset_joint_features) // 100)

#     # Set sensible bounds
#     min_components = 3
#     max_components = max(min_components, max_components)  # Ensure at least min_components

#     # If we still can't fit enough components, adjust the range
#     if max_components < min_components:
#         # Fall back to a safe default
#         n_components_range = np.array([2])  # Use single component or minimum viable
#     else:
#         n_components_range = np.arange(min_components, max_components + 1, 2)

#     best_bic = np.inf
#     best_n_components = n_components_range[0]
    
#     # Try different covariance types in order of preference
#     covariance_types = ['full', 'tied', 'diag']  # Fallback hierarchy
#     best_covariance_type = 'full'
    
#     for covariance_type in covariance_types:
#         for n_components in n_components_range:
#             try:
#                 gmm = GaussianMixture(n_components=n_components, 
#                                      covariance_type=covariance_type,
#                                      random_state=42,
#                                      max_iter=100,
#                                      reg_covar=1e-6)  # Regularization for numerical stability
#                 gmm.fit(subset_joint_features)
#                 bic = gmm.bic(subset_joint_features)
                
#                 if bic < best_bic:
#                     best_bic = bic
#                     best_n_components = n_components
#                     best_covariance_type = covariance_type
#             except Exception as e:
#                 # If full covariance fails, continue to try tied/diagonal
#                 continue
        
#         # If we found a good model with current covariance type, use it
#         if best_bic < np.inf:
#             break
    
#     print(f"Selected GMM: {best_n_components} components, {best_covariance_type} covariance")
    
#     # Fit final GMM with optimal parameters
#     try:
#         final_gmm = GaussianMixture(n_components=best_n_components,
#                                    covariance_type=best_covariance_type,
#                                    random_state=42,
#                                    max_iter=200,
#                                    reg_covar=1e-6)
        
#         # For very large datasets, fit on subset then assign all points
#         if use_subset_for_final:
#             # Fit on subset
#             final_gmm.fit(subset_joint_features)
            
#             # Assign all points to components in batches to manage memory
#             batch_size = 10000
#             component_assignments = np.zeros(n_points, dtype=int)
#             component_probs = np.zeros((n_points, best_n_components))
            
#             for i in range(0, n_points, batch_size):
#                 end_idx = min(i + batch_size, n_points)
#                 batch_features = joint_features[i:end_idx]
                
#                 component_assignments[i:end_idx] = final_gmm.predict(batch_features)
#                 component_probs[i:end_idx] = final_gmm.predict_proba(batch_features)
#         else:
#             # Fit on all data (for smaller datasets)
#             final_gmm.fit(joint_features)
#             component_assignments = final_gmm.predict(joint_features)
#             component_probs = final_gmm.predict_proba(joint_features)
        
#     except Exception as e:
#         print(f"GMM fitting failed: {e}")
#         print("Falling back to stratified random sampling...")
#         return stratified_random_fallback(fitness_scores, n_samples)
    
#     # Intelligent sampling strategy
#     selected_indices = []
    
#     # Ensure minimum representation per component
#     min_samples_per_component = max(1, n_samples // (best_n_components * 3))
    
#     for component_id in range(best_n_components):
#         component_mask = component_assignments == component_id
#         component_indices = np.where(component_mask)[0]
        
#         if len(component_indices) == 0:
#             continue
        
#         # Calculate target samples for this component
#         component_weight = final_gmm.weights_[component_id]
#         target_samples = max(min_samples_per_component,
#                            int(n_samples * component_weight))
#         target_samples = min(target_samples, len(component_indices))
        
#         # Sample from this component
#         # Use probability-weighted sampling within component
#         component_probs_subset = component_probs[component_indices, component_id]
#         sampling_probs = component_probs_subset / component_probs_subset.sum()
        
#         sampled_indices = np.random.choice(component_indices,
#                                          size=target_samples,
#                                          replace=False,
#                                          p=sampling_probs)
#         selected_indices.extend(sampled_indices)
    
#     # Convert to numpy array
#     selected_indices = np.array(selected_indices)
    
#     # Adjust to exact number of samples
#     if len(selected_indices) < n_samples:
#         # Add random samples from remaining points
#         remaining_indices = np.setdiff1d(np.arange(n_points), selected_indices)
#         n_additional = n_samples - len(selected_indices)
#         if len(remaining_indices) > 0:
#             additional_indices = np.random.choice(remaining_indices,
#                                                 size=min(n_additional, len(remaining_indices)),
#                                                 replace=False)
#             selected_indices = np.concatenate([selected_indices, additional_indices])
#     elif len(selected_indices) > n_samples:
#         # Randomly subsample to exact number
#         selected_indices = np.random.choice(selected_indices,
#                                           size=n_samples,
#                                           replace=False)
    
#     return selected_indices

# def stratified_random_fallback(fitness_scores, n_samples):
#     """
#     Simple fallback method: stratified random sampling by fitness
#     """
#     n_points = len(fitness_scores)
#     n_strata = min(10, n_samples)
    
#     # Create fitness bins
#     fitness_percentiles = np.linspace(0, 100, n_strata + 1)
#     fitness_bins = np.percentile(fitness_scores, fitness_percentiles)
#     stratum_assignments = np.digitize(fitness_scores, fitness_bins) - 1
#     stratum_assignments = np.clip(stratum_assignments, 0, n_strata - 1)
    
#     selected_indices = []
    
#     for stratum_id in range(n_strata):
#         stratum_mask = stratum_assignments == stratum_id
#         stratum_indices = np.where(stratum_mask)[0]
        
#         if len(stratum_indices) == 0:
#             continue
            
#         target_samples = max(1, int(n_samples * len(stratum_indices) / n_points))
#         target_samples = min(target_samples, len(stratum_indices))
        
#         sampled_indices = np.random.choice(stratum_indices,
#                                          size=target_samples,
#                                          replace=False)
#         selected_indices.extend(sampled_indices)
    
#     selected_indices = np.array(selected_indices)
    
#     # Adjust to exact number
#     if len(selected_indices) < n_samples:
#         remaining_indices = np.setdiff1d(np.arange(n_points), selected_indices)
#         n_additional = n_samples - len(selected_indices)
#         additional_indices = np.random.choice(remaining_indices,
#                                             size=min(n_additional, len(remaining_indices)),
#                                             replace=False)
#         selected_indices = np.concatenate([selected_indices, additional_indices])
#     elif len(selected_indices) > n_samples:
#         selected_indices = np.random.choice(selected_indices,
#                                           size=n_samples,
#                                           replace=False)
    
#     return selected_indices

# # =============4==============
# def adaptive_gmm_stratification(features, fitness_scores, n_samples):
#     """
#     Adaptive Stratification + GMM: Intelligently stratifies the data using GMM
#     on the joint feature-fitness space and then samples proportionally.

#     Args:
#         features: numpy array of shape (n_points, n_features) with values in [-1, 1]
#         fitness_scores: numpy array of shape (n_points,) with fitness values
#         n_samples: int, desired number of samples to select

#     Returns:
#         numpy array of selected indices
#     """
#     n_points = len(features)
#     if n_samples >= n_points:
#         return np.arange(n_points)

#     # Normalize fitness and create a joint feature-fitness space
#     fitness_min, fitness_max = fitness_scores.min(), fitness_scores.max()
#     normalized_fitness = (fitness_scores - fitness_min) / (fitness_max - fitness_min) if fitness_max > fitness_min else np.zeros_like(fitness_scores)
#     combined_data = np.column_stack([features, normalized_fitness.reshape(-1, 1)])
#     scaler = StandardScaler()
#     scaled_data = scaler.fit_transform(combined_data)

#     # Use GMM to find natural clusters in the joint space
#     n_components = min(max(5, n_samples // 10), 50) # Heuristic for number of clusters
#     try:
#         gmm = GaussianMixture(n_components=n_components,
#                               covariance_type='full',
#                               random_state=42,
#                               reg_covar=1e-6)
#         gmm.fit(scaled_data)
#         component_assignments = gmm.predict(scaled_data)
#     except Exception:
#         # Fallback to a simpler model if GMM fails
#         gmm = GaussianMixture(n_components=n_components,
#                               covariance_type='diag',
#                               random_state=42)
#         gmm.fit(scaled_data)
#         component_assignments = gmm.predict(scaled_data)

#     selected_indices = []
    
#     # Proportional sampling from each cluster
#     for component_id in range(gmm.n_components):
#         component_indices = np.where(component_assignments == component_id)[0]
#         if len(component_indices) == 0:
#             continue

#         # Calculate number of samples to take from this cluster
#         samples_in_component = len(component_indices)
#         target_samples = max(1, int(n_samples * (samples_in_component / n_points)))
#         target_samples = min(target_samples, samples_in_component)

#         # Select the points closest to the GMM mean for this component
#         component_data = scaled_data[component_indices]
#         component_mean = gmm.means_[component_id]
#         distances = np.linalg.norm(component_data - component_mean, axis=1)
#         closest_indices_local = np.argsort(distances)[:target_samples]
#         selected_indices.extend(component_indices[closest_indices_local])

#     # Ensure the correct number of samples
#     selected_indices = np.array(selected_indices)
#     if len(selected_indices) > n_samples:
#         selected_indices = np.random.choice(selected_indices, size=n_samples, replace=False)
#     elif len(selected_indices) < n_samples:
#         remaining_indices = np.setdiff1d(np.arange(n_points), selected_indices)
#         additional_indices = np.random.choice(remaining_indices, size=n_samples - len(selected_indices), replace=False)
#         selected_indices = np.concatenate([selected_indices, additional_indices])

#     return selected_indices

# # ==============5============= 
# from sklearn.cluster import DBSCAN, AgglomerativeClustering

# def density_hierarchical_selection(features, fitness_scores, n_samples):
#     """
#     Density-Based Hierarchical Sampling: Samples a representative subset
#     by first identifying high-density core regions using DBSCAN, then
#     using hierarchical clustering to select samples proportionally
#     from all identified clusters and outliers.

#     Args:
#         features: numpy array of shape (n_points, n_features)
#         fitness_scores: numpy array of shape (n_points,)
#         n_samples: int, desired number of samples to select

#     Returns:
#         numpy array of selected indices
#     """
#     n_points = len(features)
#     if n_samples >= n_points:
#         return np.arange(n_points)

#     # 1. Combine features and fitness, then scale
#     fitness_min, fitness_max = fitness_scores.min(), fitness_scores.max()
#     normalized_fitness = (fitness_scores - fitness_min) / (fitness_max - fitness_min) if fitness_max > fitness_min else np.zeros_like(fitness_scores)
#     combined_data = np.column_stack([features, normalized_fitness.reshape(-1, 1)])
#     scaler = StandardScaler()
#     scaled_data = scaler.fit_transform(combined_data)

#     selected_indices = []

#     try:
#         # 2. Stage 1: Initial Density-Based Clustering (DBSCAN)
#         # This identifies dense clusters and noise (outliers)
#         dbscan = DBSCAN(eps=0.5, min_samples=5)
#         dbscan_labels = dbscan.fit_predict(scaled_data)
        
#         # 3. Handle core points (dense clusters)
#         core_clusters = np.unique(dbscan_labels[dbscan_labels != -1])
#         core_point_indices = np.where(dbscan_labels != -1)[0]
        
#         # 4. Handle noise points (outliers)
#         noise_point_indices = np.where(dbscan_labels == -1)[0]
        
#         # Determine number of clusters for hierarchical clustering
#         n_clusters_hierarchical = min(max(5, n_samples // 10), 50)
        
#         # Hierarchical clustering on the core points
#         if len(core_point_indices) > n_clusters_hierarchical:
#             hierarchical_clustering = AgglomerativeClustering(n_clusters=n_clusters_hierarchical, linkage='ward')
#             core_assignments = hierarchical_clustering.fit_predict(scaled_data[core_point_indices])
            
#             # Sample from each hierarchical cluster
#             for cluster_id in range(n_clusters_hierarchical):
#                 cluster_local_indices = np.where(core_assignments == cluster_id)[0]
#                 cluster_original_indices = core_point_indices[cluster_local_indices]
                
#                 # Proportional sampling based on cluster size
#                 target_samples = max(1, int(n_samples * len(cluster_original_indices) / n_points))
#                 target_samples = min(target_samples, len(cluster_original_indices))
                
#                 # Select the point closest to the cluster's centroid
#                 centroid = scaled_data[cluster_original_indices].mean(axis=0)
#                 distances = np.linalg.norm(scaled_data[cluster_original_indices] - centroid, axis=1)
#                 closest_idx_local = np.argsort(distances)[:target_samples]
#                 selected_indices.extend(cluster_original_indices[closest_idx_local])

#         else:
#             # Not enough core points for hierarchical clustering, take all
#             selected_indices.extend(core_point_indices)
            
#         # 5. Sample from noise points (outliers)
#         n_noise_samples = max(1, n_samples - len(selected_indices))
#         if len(noise_point_indices) > 0:
#             noise_samples_to_take = min(n_noise_samples, len(noise_point_indices))
#             noise_samples = np.random.choice(noise_point_indices, size=noise_samples_to_take, replace=False)
#             selected_indices.extend(noise_samples)
        
#     except Exception:
#         # Fallback to stratified random sampling if clustering fails
#         return _stratified_random_fallback(fitness_scores, n_samples)
        
#     # 6. Adjust to the exact number of samples
#     selected_indices = np.array(selected_indices)
#     if len(selected_indices) > n_samples:
#         selected_indices = np.random.choice(selected_indices, size=n_samples, replace=False)
#     elif len(selected_indices) < n_samples:
#         remaining_indices = np.setdiff1d(np.arange(n_points), selected_indices)
#         n_additional = n_samples - len(selected_indices)
#         additional_indices = np.random.choice(remaining_indices, size=min(n_additional, len(remaining_indices)), replace=False)
#         selected_indices = np.concatenate([selected_indices, additional_indices])
    
#     return selected_indices

# def _stratified_random_fallback(fitness_scores, n_samples):
#     """
#     Simple fallback method: stratified random sampling by fitness
#     """
#     n_points = len(fitness_scores)
#     n_strata = min(10, n_samples)
    
#     fitness_percentiles = np.linspace(0, 100, n_strata + 1)
#     fitness_bins = np.percentile(fitness_scores, fitness_percentiles)
#     stratum_assignments = np.digitize(fitness_scores, fitness_bins) - 1
#     stratum_assignments = np.clip(stratum_assignments, 0, n_strata - 1)
    
#     selected_indices = []
    
#     for stratum_id in range(n_strata):
#         stratum_mask = stratum_assignments == stratum_id
#         stratum_indices = np.where(stratum_mask)[0]
        
#         if len(stratum_indices) == 0:
#             continue
            
#         target_samples = max(1, int(n_samples * len(stratum_indices) / n_points))
#         target_samples = min(target_samples, len(stratum_indices))
        
#         sampled_indices = np.random.choice(stratum_indices, size=target_samples, replace=False)
#         selected_indices.extend(sampled_indices)
    
#     selected_indices = np.array(selected_indices)
    
#     if len(selected_indices) < n_samples:
#         remaining_indices = np.setdiff1d(np.arange(n_points), selected_indices)
#         n_additional = n_samples - len(selected_indices)
#         additional_indices = np.random.choice(remaining_indices, size=min(n_additional, len(remaining_indices)), replace=False)
#         selected_indices = np.concatenate([selected_indices, additional_indices])
#     elif len(selected_indices) > n_samples:
#         selected_indices = np.random.choice(selected_indices, size=n_samples, replace=False)
    
#     return selected_indices

# # =============6==============
# from sklearn.decomposition import IncrementalPCA
# from sklearn.random_projection import GaussianRandomProjection

# def incremental_pca_selection(features, fitness_scores, n_samples):
#     """
#     Selects a representative subset after reducing dimensionality with Incremental PCA.
#     This is a two-stage process for scenarios where the number of features is very large.
    
#     Args:
#         features: numpy array of shape (n_points, n_features) with values in [-1, 1]
#         fitness_scores: numpy array of shape (n_points,) with fitness values
#         n_samples: int, desired number of samples to select

#     Returns:
#         numpy array of selected indices
#     """
#     n_points, n_features = features.shape
#     if n_samples >= n_points:
#         return np.arange(n_points)

#     # 1. Dimensionality Reduction with Incremental PCA
#     # This simulates a multi-pass approach by fitting on a subset
#     # In a real-world scenario, this would be a multi-pass over chunks of data
#     n_components = min(10, n_features)
#     ipca = IncrementalPCA(n_components=n_components, batch_size=1000)
    
#     # Simulating incremental fit
#     batch_size = 10000
#     for i in range(0, n_points, batch_size):
#         ipca.partial_fit(features[i:i + batch_size])
        
#     reduced_features = ipca.transform(features)
    
#     # 2. Stratified K-Means on the reduced features
#     # This handles both feature diversity (in the reduced space) and fitness
#     # diversity by using the fitness scores for stratification
    
#     # Determine number of fitness strata (adaptive based on dataset size)
#     n_strata = min(20, max(5, n_samples // 50))
    
#     # Create fitness percentile bins
#     fitness_percentiles = np.linspace(0, 100, n_strata + 1)
#     fitness_bins = np.percentile(fitness_scores, fitness_percentiles)
    
#     # Assign each point to a fitness stratum
#     stratum_assignments = np.digitize(fitness_scores, fitness_bins) - 1
#     stratum_assignments = np.clip(stratum_assignments, 0, n_strata - 1)
    
#     selected_indices = []
    
#     for stratum_id in range(n_strata):
#         stratum_mask = stratum_assignments == stratum_id
#         stratum_indices = np.where(stratum_mask)[0]
        
#         if len(stratum_indices) == 0:
#             continue
            
#         stratum_size = len(stratum_indices)
#         target_samples = max(1, int(n_samples * stratum_size / n_points))
#         target_samples = min(target_samples, stratum_size)
        
#         if target_samples > 1 and len(stratum_indices) > target_samples:
#             # Use MiniBatchKMeans for efficiency on large strata
#             kmeans = MiniBatchKMeans(n_clusters=target_samples, random_state=42, n_init=3)
#             kmeans.fit(reduced_features[stratum_indices])
            
#             for cluster_id in range(target_samples):
#                 cluster_mask = kmeans.labels_ == cluster_id
#                 if not np.any(cluster_mask):
#                     continue
                
#                 cluster_indices = stratum_indices[cluster_mask]
#                 cluster_features = reduced_features[cluster_indices]
#                 centroid = kmeans.cluster_centers_[cluster_id]
                
#                 distances = np.sum((cluster_features - centroid) ** 2, axis=1)
#                 closest_idx = np.argmin(distances)
#                 selected_indices.append(cluster_indices[closest_idx])
#         else:
#             # Handle small strata or single-sample selection
#             stratum_fitness = fitness_scores[stratum_indices]
#             median_fitness = np.median(stratum_fitness)
#             closest_idx = np.argmin(np.abs(stratum_fitness - median_fitness))
#             selected_indices.append(stratum_indices[closest_idx])
            
#     # Final adjustment
#     final_indices = np.array(selected_indices)
#     if len(final_indices) > n_samples:
#         final_indices = np.random.choice(final_indices, size=n_samples, replace=False)
#     elif len(final_indices) < n_samples:
#         remaining = np.setdiff1d(np.arange(n_points), final_indices)
#         additional = np.random.choice(remaining, size=n_samples - len(final_indices), replace=False)
#         final_indices = np.concatenate([final_indices, additional])
        
#     return final_indices

# # =============7==============


# def random_projection_selection(features, fitness_scores, n_samples):
#     """
#     Selects a representative subset after reducing dimensionality with Random Projection.
#     This is a fast and memory-efficient pre-processing step for very high-dimensional data.
    
#     Args:
#         features: numpy array of shape (n_points, n_features) with values in [-1, 1]
#         fitness_scores: numpy array of shape (n_points,) with fitness values
#         n_samples: int, desired number of samples to select

#     Returns:
#         numpy array of selected indices
#     """
#     n_points, n_features = features.shape
#     if n_samples >= n_points:
#         return np.arange(n_points)
        
#     # 1. Dimensionality Reduction with Random Projection
#     # This is a single-pass method that works well for very large data
#     n_components = min(10, n_features)
#     rp = GaussianRandomProjection(n_components=n_components, random_state=42)
#     reduced_features = rp.fit_transform(features)
    
#     # 2. GMM + FPS Hybrid on the reduced features
#     # Uses a sophisticated method on the now-manageable data
    
#     fitness_min, fitness_max = fitness_scores.min(), fitness_scores.max()
#     normalized_fitness = (fitness_scores - fitness_min) / (fitness_max - fitness_min) if fitness_max > fitness_min else np.zeros_like(fitness_scores)
#     combined_data = np.column_stack([reduced_features, normalized_fitness.reshape(-1, 1)])
    
#     # GMM clustering stage
#     n_components_gmm = min(n_samples // 10, 50)
#     n_components_gmm = max(n_components_gmm, 1)
    
#     try:
#         gmm = GaussianMixture(n_components=n_components_gmm, covariance_type='diag', random_state=42)
#         gmm.fit(combined_data)
#         component_assignments = gmm.predict(combined_data)
        
#         # Proportional sampling from GMM clusters for stage 1
#         stage1_indices = []
#         samples_per_component = (n_samples * 10) // n_components_gmm
        
#         for component in range(n_components_gmm):
#             component_mask = component_assignments == component
#             component_indices = np.where(component_mask)[0]
#             if len(component_indices) == 0:
#                 continue
            
#             n_samples_comp = min(samples_per_component, len(component_indices))
#             if n_samples_comp <= 0: continue
            
#             component_data = combined_data[component_mask]
#             component_mean = gmm.means_[component]
#             distances = np.linalg.norm(component_data - component_mean, axis=1)
#             closest_indices = np.argsort(distances)[:n_samples_comp]
#             selected_component_indices = component_indices[closest_indices]
#             stage1_indices.extend(selected_component_indices)
            
#         stage1_indices = np.array(stage1_indices)
        
#     except Exception:
#         # Fallback to stratified random if GMM fails
#         stage1_indices = np.random.choice(n_points, size=min(n_samples * 10, n_points), replace=False)
        
#     if len(stage1_indices) <= n_samples:
#         return stage1_indices[:n_samples]
        
#     # Stage 2: Farthest Point Sampling for diversity
#     stage1_data = combined_data[stage1_indices]
    
#     # Simplified FPS helper
#     n_points_fps = len(stage1_data)
#     selected_fps_indices = [np.random.randint(n_points_fps)]
#     distances = np.full(n_points_fps, np.inf)
    
#     for i in range(n_points_fps):
#         dist = np.linalg.norm(stage1_data[i] - stage1_data[selected_fps_indices[-1]])
#         distances[i] = min(distances[i], dist)
        
#     for _ in range(n_samples - 1):
#         farthest_idx = np.argmax(distances)
#         selected_fps_indices.append(farthest_idx)
#         distances = np.minimum(distances, np.linalg.norm(stage1_data - stage1_data[farthest_idx], axis=1))

#     return stage1_indices[selected_fps_indices]

# # =============8==============


# def distributed_clustering_selection(features, fitness_scores, n_samples):
#     """
#     Simulates a distributed approach using MiniBatchKMeans. This method
#     partitions the data and clusters each part, then aggregates the results
#     to select a final, diverse subset.
    
#     Args:
#         features: numpy array of shape (n_points, n_features) with values in [-1, 1]
#         fitness_scores: numpy array of shape (n_points,) with fitness values
#         n_samples: int, desired number of samples to select

#     Returns:
#         numpy array of selected indices
#     """
#     n_points = len(features)
#     if n_samples >= n_points:
#         return np.arange(n_points)
        
#     # Combine features and fitness
#     fitness_min, fitness_max = fitness_scores.min(), fitness_scores.max()
#     normalized_fitness = (fitness_scores - fitness_min) / (fitness_max - fitness_min) if fitness_max > fitness_min else np.zeros_like(fitness_scores)
#     combined_data = np.column_stack([features, normalized_fitness.reshape(-1, 1)])
    
#     # Simulate a distributed clustering scenario
#     n_partitions = 4  # e.g., number of worker nodes
#     partition_size = n_points // n_partitions
#     all_centroids = []
    
#     for i in range(n_partitions):
#         # Cluster each partition independently
#         start_idx = i * partition_size
#         end_idx = start_idx + partition_size if i < n_partitions - 1 else n_points
#         partition_data = combined_data[start_idx:end_idx]
        
#         # Use MiniBatchKMeans for efficiency
#         partition_k = min(n_samples // n_partitions + 1, len(partition_data) // 5)
#         if partition_k < 1: partition_k = 1
        
#         kmeans = MiniBatchKMeans(n_clusters=partition_k, random_state=42 + i, n_init=3)
#         kmeans.fit(partition_data)
        
#         all_centroids.append(kmeans.cluster_centers_)
        
#     # 2. Aggregate centroids and perform final selection
#     global_centroids = np.vstack(all_centroids)
    
#     # Cluster the global centroids to get the final n_samples
#     final_kmeans = MiniBatchKMeans(n_clusters=n_samples, random_state=42, n_init=10)
#     final_kmeans.fit(global_centroids)
#     final_centers = final_kmeans.cluster_centers_
    
#     # For each final center, find the closest original data point
#     selected_indices = []
#     for center in final_centers:
#         distances = np.linalg.norm(combined_data - center, axis=1)
#         closest_idx = np.argmin(distances)
#         selected_indices.append(closest_idx)
        
#     return np.unique(selected_indices)



# # Example usage and testing
# if __name__ == "__main__":
#     # Generate synthetic test data
#     np.random.seed(42)
    
#     # Create sample data (simulating your 1M dataset with smaller version)
#     n_total = 10000  # Use 10K for testing, scales to 1M
#     n_features = 8
    
#     # Generate features in [-1, 1] range
#     features = np.random.uniform(-1, 1, (n_total, n_features))
    
#     # Generate complex fitness landscape with multiple modes
#     fitness_scores = (
#         np.sum(features**2, axis=1) +  # Quadratic term
#         0.5 * np.sum(features[:, :3], axis=1) +  # Linear term
#         0.3 * np.random.normal(0, 1, n_total)  # Noise
#     )
    
#     n_desired = 500  # Select 500 samples from 10K
    
# # Example usage and testing
# if __name__ == "__main__":
#     # Generate synthetic test data
#     np.random.seed(42)
    
#     # Create sample data (simulating your 1M dataset with smaller version)
#     n_total = 10000  # Use 10K for testing, scales to 1M
#     n_features = 8
    
#     # Generate features in [-1, 1] range
#     features = np.random.uniform(-1, 1, (n_total, n_features))
    
#     # Generate complex fitness landscape with multiple modes
#     fitness_scores = (
#         np.sum(features**2, axis=1) +  # Quadratic term
#         0.5 * np.sum(features[:, :3], axis=1) +  # Linear term for first 3 features
#         0.3 * np.prod(features[:, 3:6], axis=1) +  # Interaction term
#         0.3 * np.random.normal(0, 1, n_total)  # Noise
#     )
    
#     n_desired = 500  # Select 500 samples from 10K
    
#     print(f"Original dataset: {n_total} samples")
#     print(f"Target selection: {n_desired} samples")
#     print(f"Features shape: {features.shape}")
#     print(f"Fitness range: [{fitness_scores.min():.3f}, {fitness_scores.max():.3f}]")
    
#     # Test all four functions
#     methods = [
#         ("Stratified K-Means", stratified_kmeans_selection),
#         ("GMM Full Covariance", gmm_intelligent_selection),
#         ("GMM + FPS Hybrid", gmm_fps_hybrid_selection),
#         ("GMM + Grid Hybrid", gmm_grid_hybrid_selection)
#     ]
    
#     results = {}
    
#     for method_name, method_func in methods:
#         print(f"\n=== Testing {method_name} ===")
#         import time
#         start_time = time.time()
        
#         try:
#             selected_indices = method_func(features, fitness_scores, n_desired)
#             elapsed_time = time.time() - start_time
            
#             print(f"Selected {len(selected_indices)} samples in {elapsed_time:.2f} seconds")
#             selected_fitness = fitness_scores[selected_indices]
#             print(f"Selected fitness range: [{selected_fitness.min():.3f}, {selected_fitness.max():.3f}]")
            
#             # Calculate fitness coverage
#             fitness_coverage = (selected_fitness.max() - selected_fitness.min()) / (fitness_scores.max() - fitness_scores.min()) * 100
#             print(f"Fitness coverage: {fitness_coverage:.1f}%")
            
#             # Calculate feature diversity
#             feature_std_original = np.std(features, axis=0).mean()
#             feature_std_selected = np.std(features[selected_indices], axis=0).mean()
#             feature_diversity = feature_std_selected / feature_std_original * 100
#             print(f"Feature diversity: {feature_diversity:.1f}% of original")
            
#             results[method_name] = {
#                 'indices': selected_indices,
#                 'time': elapsed_time,
#                 'fitness_coverage': fitness_coverage,
#                 'feature_diversity': feature_diversity
#             }
            
#         except Exception as e:
#             print(f"Method failed: {e}")
#             results[method_name] = None
    
#     # Compare methods
#     print(f"\n=== Method Comparison ===")
#     valid_results = {k: v for k, v in results.items() if v is not None}
    
#     if len(valid_results) > 1:
#         # Calculate pairwise overlaps
#         method_names = list(valid_results.keys())
#         for i, method1 in enumerate(method_names):
#             for j, method2 in enumerate(method_names[i+1:], i+1):
#                 indices1 = valid_results[method1]['indices']
#                 indices2 = valid_results[method2]['indices']
#                 overlap = len(np.intersect1d(indices1, indices2))
#                 print(f"{method1} vs {method2}: {overlap} overlapping samples ({overlap/n_desired*100:.1f}%)")
    
#     print(f"\n=== Performance Summary ===")
#     for method_name, result in valid_results.items():
#         print(f"{method_name}:")
#         print(f"  Time: {result['time']:.2f}s")
#         print(f"  Fitness Coverage: {result['fitness_coverage']:.1f}%")
#         print(f"  Feature Diversity: {result['feature_diversity']:.1f}%")
    
#     print("\n=== Recommendations ===")
#     print("Method 1 (Stratified K-Means): Fast, guaranteed fitness coverage")
#     print("Method 2 (GMM Full Covariance): Best for complex manifolds, captures correlations")
#     print("Method 3 (GMM + FPS): Good balance of structure and diversity")
#     print("Method 4 (GMM + Grid): Best coverage guarantee, good for exploration")
#     print("\nAll methods successfully reduced dataset while maintaining diversity!")