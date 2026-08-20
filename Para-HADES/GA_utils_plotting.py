"""
Unified Plotting Module for Para-HADES Genetic Algorithm
=========================================================

This module provides a unified, memory-efficient plotting system that combines
the best features from all task-specific plotting implementations.

Features:
- Memory-efficient chunked data processing
- Automatic scale selection (log/linear) with intelligent decision-making
- Advanced multi-algorithm trend fitting
- Dynamic target handling (supports any number of targets)
- Configurable plot types and styles
- Task-specific customization through config dict

Author: Para-HADES Project
Date: January 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import interpolate
from scipy.optimize import curve_fit
from scipy.spatial import distance
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

import glob
import gc
import os
import time

from GA_utils_history_loader import ChunkedGeneHistoryLoader


def plotTable(args, data_plot, names):
    import matplotlib.pyplot as plt
    import numpy as np
    from datetime import datetime
    import os
    from scipy import interpolate
    import warnings
    import traceback
    
    warnings.filterwarnings('ignore')

    # --- Helper: Check Y-Axis Scale Only ---
    def check_y_log_scale(data):
        """
        Returns 'log' if Y data covers > 10x magnitude, else 'linear'.
        Uses actual min/max of all positive values for range calculation.
        """
        try:
            data = np.array(data).flatten()
            positive_data = data[data > 0]
            
            if len(positive_data) < 5: 
                print(f"check_y_log_scale: Not enough positive data ({len(positive_data)} < 5), returning linear", flush=True)
                return 'linear'
            
            data_min = np.min(positive_data)
            data_max = np.max(positive_data)
            
            print(f"check_y_log_scale: All {len(positive_data)} positive values, min={data_min:.2e}, max={data_max:.2e}", flush=True)
            
            if data_min <= 0: 
                print(f"check_y_log_scale: data_min <= 0, returning linear", flush=True)
                return 'linear'
            
            # Constraint: > 10X magnitude for Y axis (more aggressive log scaling)
            dynamic_range = data_max / data_min
            print(f"check_y_log_scale: dynamic_range = {dynamic_range:.1f}x", flush=True)
            
            if dynamic_range > 10:
                print(f"check_y_log_scale: Returning LOG (range > 10x)", flush=True)
                return 'log'
            
            print(f"check_y_log_scale: Returning LINEAR (range <= 10x)", flush=True)
            return 'linear'
        except Exception as e:
            print(f"check_y_log_scale: Exception caught: {e}", flush=True)
            return 'linear'

    # --- Helper: Trend Line ---
    def fit_advanced_trend_line(x, y):
        try:
            if len(x) < 5: return x, y, 'none'
            x = x.flatten()
            y = y.flatten()
            idx = np.argsort(x)
            x_s = x[idx]
            y_s = y[idx]
            
            num_points = int(min(200, len(x_s)*2))
            x_trend = np.linspace(x_s.min(), x_s.max(), num_points)
            
            # Polyfit
            try:
                deg = 3 if len(x_s) > 20 else 2
                z = np.polyfit(x_s, y_s, deg)
                p = np.poly1d(z)
                return x_trend, p(x_trend), 'poly'
            except:
                pass
            
            # Moving Average
            try:
                from scipy.ndimage import gaussian_filter1d
                sigma = float(max(2, len(y_s) / 10))
                y_smooth = gaussian_filter1d(y_s, sigma=sigma)
                f = interpolate.interp1d(x_s, y_smooth, kind='linear', fill_value="extrapolate")
                return x_trend, f(x_trend), 'smooth'
            except:
                pass
            
            return x, y, 'none'
        except:
            return x, y, 'none'

    # ==========================================
    # Main Plotting Logic
    # ==========================================
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple']

    plot_groups = []
    if not names: return
    if isinstance(names, (list, tuple)):
        if isinstance(names[0], str):
            plot_groups = [[n] for n in names] 
        else:
            plot_groups = names 
    else:
        print("Error: 'names' must be a list.")
        return

    if 'iteration' not in data_plot:
        print("Error: 'iteration' key missing.")
        return
    
    try:
        x_master = np.array(data_plot['iteration'], dtype=np.int32).flatten()
    except Exception as e:
        print(f"Error processing iteration data: {e}")
        return

    for group in plot_groups:
        try:
            fig, host = plt.subplots(figsize=(10, 6))
            
            if len(group) > 1:
                fig.subplots_adjust(right=0.75)
            else:
                fig.subplots_adjust(right=0.95)

            handles = []
            labels_list = []
            file_name_parts = []
            title_parts = []
            
            has_data = False
            
            # --- 1. X-Axis Logic (ALWAYS LINEAR) ---
            host.set_xscale('linear')
            host.set_xlabel("Iteration", fontweight='bold')

            for i, field_name in enumerate(group):
                if field_name not in data_plot: continue

                try:
                    raw_data = data_plot[field_name]
                    if raw_data is None or len(raw_data) == 0: continue

                    y_data = np.array(raw_data, dtype=np.float32).flatten()
                    y_data = np.nan_to_num(y_data)

                    min_len = int(min(len(x_master), len(y_data)))
                    if min_len < 2: continue
                    
                    x_curr = x_master[:min_len]
                    y_curr = y_data[:min_len]
                    
                    has_data = True
                    file_name_parts.append(field_name)
                    title_parts.append(field_name)

                    # --- 2. Y-Axis Logic (>10x check) ---
                    y_scale_type = check_y_log_scale(y_curr)
                    
                    # Add "(Log Scale)" text if applicable
                    label_text = field_name
                    if y_scale_type == 'log':
                        label_text += " (Log Scale)"

                    color = colors[i % len(colors)]
                    
                    if i == 0:
                        ax = host
                        ax.set_ylabel(label_text, fontweight='bold', color=color)
                        ax.tick_params(axis='y', labelcolor=color)
                    else:
                        ax = host.twinx()
                        offset = int((i - 1) * 60)
                        ax.spines['right'].set_position(('outward', offset))
                        ax.set_ylabel(label_text, fontweight='bold', color=color)
                        ax.tick_params(axis='y', labelcolor=color)
                        ax.spines['right'].set_visible(True)
                        ax.spines['right'].set_color(color)
                        ax.spines['right'].set_linewidth(1.5)

                    # Apply Scale with sensitive axis limits
                    if y_scale_type == 'log':
                        ax.set_yscale('log')
                        # For log scale, find positive values and set tight limits
                        positive_y = y_curr[y_curr > 0]
                        if len(positive_y) > 0:
                            y_min = np.min(positive_y)
                            y_max = np.max(positive_y)
                            # For log scale, use actual min/max (not scaled down)
                            # Add small padding factors for better visualization
                            ax.set_ylim(bottom=y_min / 1.5, top=y_max * 1.5)
                            print(f"Log scale limits set: {y_min / 1.5:.2e} to {y_max * 1.5:.2e}", flush=True)
                    else:
                        ax.set_yscale('linear')
                        # For linear scale, set tight limits with small padding
                        y_min = np.nanmin(y_curr)
                        y_max = np.nanmax(y_curr)
                        if np.isfinite(y_min) and np.isfinite(y_max):
                            y_range = y_max - y_min
                            # Add 10% padding on both sides
                            padding = y_range * 0.1 if y_range > 0 else 0.1
                            ax.set_ylim(bottom=y_min - padding, top=y_max + padding)
                            print(f"Linear scale limits set: {y_min - padding:.2e} to {y_max + padding:.2e}", flush=True)
                    
                    # Plot
                    sc = ax.scatter(x_curr, y_curr, s=10, color=color, alpha=0.3, label=field_name)
                    
                    xt, yt, mode = fit_advanced_trend_line(x_curr, y_curr)
                    if mode != 'none':
                        ln, = ax.plot(xt, yt, color=color, linewidth=2, alpha=0.9)
                        handles.append((sc, ln))
                        labels_list.append(field_name)
                    else:
                        handles.append(sc)
                        labels_list.append(field_name)

                except Exception as inner_e:
                    print(f"Error plotting field '{field_name}': {inner_e}")
                    traceback.print_exc()
                    continue

            if has_data:
                t_str = " & ".join(title_parts)
                host.set_title(f"{t_str} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                if handles:
                    host.legend(handles, labels_list, loc='best', fontsize='small')

                os.makedirs(os.path.join(args.path, 'graphs'), exist_ok=True)
                clean_name = "_".join([x for x in file_name_parts if x])
                clean_name = "".join([c for c in clean_name if c.isalnum() or c=='_'])
                if not clean_name: clean_name = "plot_unknown"
                
                out_path = os.path.join(args.path, 'graphs', f"{clean_name}.png")
                plt.savefig(out_path, dpi=150, bbox_inches='tight')
                print(f"Saved plot: {out_path}")
                
            plt.close(fig)

        except Exception as e:
            print(f"Critical error processing group {group}: {e}")
            traceback.print_exc()
            try: plt.close(fig)
            except: pass

        
def plot_PCS(args, genetic_data, target_data):
    
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler 

    scaler = StandardScaler()
    genetic_data = scaler.fit_transform(genetic_data)
    target_data = scaler.fit_transform(target_data)
    all_data = np.concatenate((genetic_data, target_data), axis=0)

    if np.isnan(all_data).sum().sum() > 0:
        print('nan in data')
        return

    # Perform PCA
    pca = PCA(n_components=2)
    _ = pca.fit(all_data)

    # use the pca to trsnform the data
    pca_target = pca.transform(target_data)
    plt.scatter(pca_target[:, 0], pca_target[:, 1])

    # Plot the distances between populations
    pca_data = pca.transform(genetic_data)
    plt.scatter(pca_data[:, 0], pca_data[:, 1])
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA: Distance between Populations' + ' - ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    # save plots to file with high resolution
    plt.savefig(args.path + "/graphs/genes_PCA.png", dpi=300)

    # Show the plot.
    # plt.show()
    plt.close()



class UnifiedPlotter:
    """
    Unified plotting system with best-of-all-worlds approach.
    
    This class handles all plotting needs for different GA tasks, providing
    a consistent interface while allowing task-specific customization.
    """
    
    def __init__(self, args, targets=None, config=None):
        """
        Initialize the UnifiedPlotter.
        
        Args:
            args: Standard GA args object with path, evolutionTarget, etc.
            targets: List/array of target values (can be None for some tasks)
            config: Dict with plotting preferences:
                {
                    'metric_type': 'distance', 'similarity', or 'error',
                    'better_direction': 'lower', 'higher', or 'auto',
                    'target_labels': list of strings or None (auto-generate),
                    'enable_clustering': bool (default: True),
                    'enable_3d_viz': bool (default: True),
                    'enable_individual_targets': bool (default: True),
                    'plot_frequency': float (0-1, probability to plot, default: 0.1),
                    'main_plot_title': str (default: auto-generated),
                    'y_axis_label': str (default: auto-generated)
                }
        """
        self.args = args
        self.targets = targets
        self.num_targets = len(targets) if targets is not None else 0
        
        # Default configuration
        self.config = {
            'metric_type': 'distance',
            'better_direction': 'lower',
            'target_labels': None,
            'enable_clustering': True,
            'enable_3d_viz': True,
            'enable_individual_targets': True,
            'plot_frequency': 0.1,
            'main_plot_title': None,
            'y_axis_label': None,
            'dpi': 300,
            'main_figsize': (12, 8),
            'sub_figsize': (10, 6)
        }
        
        # Update with user config
        if config:
            self.config.update(config)
        
        # Determine target dimensionality
        self.target_dim = 0
        if targets is not None and len(targets) > 0:
            if isinstance(targets[0], (list, np.ndarray)):
                self.target_dim = len(targets[0])
            else:
                self.target_dim = 1
        
        # Setup paths
        self.graphs_path = os.path.join(args.path, 'graphs')
        os.makedirs(self.graphs_path, exist_ok=True)
        
        # Determine file format
        self.file_format = self._detect_file_format()
        
    def _detect_file_format(self):
        """Detect whether logs are in pkl or csv format."""
        pkl_files = glob.glob(self.args.path + '/*.log.pkl')
        framed_files = glob.glob(self.args.path + '/*.log.framed.bin')
        if len(framed_files) > 0:
            return 'pkl'
        if len(pkl_files) > 0:
            return 'pkl'
        csv_files = glob.glob(self.args.path + '/*.log.csv')
        if len(csv_files) > 0:
            return 'csv'
        return None
    
    def _get_colors(self):
        """Get appropriate color scheme based on number of targets."""
        if self.num_targets == 0:
            return []
        elif self.num_targets <= 10:
            cmap = cm.get_cmap('tab10')
            return [cmap(i) for i in range(self.num_targets)]
        else:
            cmap = cm.get_cmap('jet')
            return [cmap(i / max(1, self.num_targets - 1)) for i in range(self.num_targets)]
    
    def _get_y_label(self, metric_type=None, scale='linear'):
        """Generate appropriate Y-axis label."""
        if self.config['y_axis_label']:
            return self.config['y_axis_label']
        
        metric = metric_type or self.config['metric_type']
        better = self.config['better_direction']
        
        if better == 'auto':
            if metric == 'distance' or metric == 'error':
                better = 'lower'
            elif metric == 'similarity':
                better = 'higher'
            else:
                better = None
        
        scale_text = f"({scale} scale)"
        
        if metric == 'distance':
            label = f'Distance to Closest Target {scale_text}'
            if better == 'lower':
                label += ' - Lower is Better'
        elif metric == 'error':
            label = f'Error (Absolute Distance) {scale_text}'
            if better == 'lower':
                label += ' - Lower is Better'
        elif metric == 'similarity':
            label = f'Similarity to Closest Target {scale_text}'
            if better == 'higher':
                label += ' - Higher is Better'
        else:
            label = f'Metric Value {scale_text}'
        
        return label
    
    def _get_title(self, title_type='main'):
        """Generate appropriate plot title."""
        if title_type == 'main' and self.config['main_plot_title']:
            return self.config['main_plot_title']
        
        metric = self.config['metric_type']
        
        if title_type == 'main':
            if metric == 'distance':
                return 'Evolution: Distance to Closest Target (Convergence)'
            elif metric == 'error':
                return f'Convergence to {self.num_targets} Target(s)'
            else:
                return 'Evolution: Similarity to Closest Target'
        elif title_type == 'best_results':
            direction = 'smaller' if self.args.evolutionTarget == -1 else 'bigger'
            return f'Best Results\n({direction} is better)'
        
        return 'Evolution Progress'
    
    def create_all_plots(self, method='euclidean'):
        """
        Main entry point - creates all relevant plots based on configuration.
        
        Args:
            method: Distance metric to use ('euclidean', 'cosine', etc.)
        """
        # Random sampling for performance
        if np.random.rand() >= self.config['plot_frequency']:
            return
        
        if self.file_format is None:
            print("No log files found for plotting.", flush=True)
            return
        
        print("=" * 60, flush=True)
        print("Starting Unified Plotting System", flush=True)
        print("=" * 60, flush=True)
        
        # Plot convergence (main plot)
        if self.num_targets > 0:
            self.plot_convergence(method=method)
            
            # Individual target plots (optional)
            if self.config['enable_individual_targets']:
                self.plot_individual_targets(method=method)
        
        # Plot best results
        self.plot_best_results()
        
        # Clustering plot (if enabled and applicable)
        if self.config['enable_clustering'] and self.num_targets >= 2:
            self.plot_clustering(method=method)
        
        # Gene pool visualization (if enabled)
        if self.config['enable_3d_viz'] and self.target_dim in [2, 3]:
            self.plot_gene_pool_2d_3d()
        
        print("=" * 60, flush=True)
        print("Unified Plotting Complete", flush=True)
        print("=" * 60, flush=True)
    
    def plot_convergence(self, method='euclidean'):
        """Create main convergence plot showing evolution to all targets."""
        print("Creating main convergence plot...", flush=True)
        
        # Process similarity data
        target_similarities, target_timelines = self._process_similarity_data(method)
        
        if not target_similarities or not any(len(sim) > 0 for sim in target_similarities):
            print("No valid similarity data found", flush=True)
            return
        
        # Find closest target for each data point
        all_similarities, all_timelines, all_closest_targets = self._compute_closest_targets(
            target_similarities, target_timelines
        )
        
        if len(all_similarities) == 0:
            print("No valid data points found", flush=True)
            return
        
        # Choose scale
        y_scale = self._choose_y_scale(all_similarities, "Closest Target")
        
        # Create plot
        fig = plt.figure(figsize=self.config['main_figsize'])
        colors = self._get_colors()
        
        # Prepare data
        plot_sims, plot_times, plot_colors = self._prepare_plot_data(
            all_similarities, all_timelines, all_closest_targets, colors, y_scale
        )
        
        # Scatter plot
        plt.scatter(plot_times, plot_sims, c=plot_colors, s=1.5, alpha=0.4, cmap='tab10')
        
        # Legend
        for target_idx in range(self.num_targets):
            if np.any(all_closest_targets == target_idx):
                label = self._get_target_label(target_idx)
                plt.scatter([], [], color=colors[target_idx], label=label, s=20)
        
        # Trend line
        self._add_trend_line(all_timelines, all_similarities, y_scale, color='black')
        
        # Apply scale and styling
        self._apply_plot_style(y_scale, all_similarities, 
                               xlabel='Iteration', 
                               ylabel=self._get_y_label(scale=y_scale),
                               title=self._get_title('main'))
        
        # Save
        plt.savefig(os.path.join(self.graphs_path, 'similarityGraph.png'), 
                   dpi=self.config['dpi'])
        plt.close()
        
        print(f"✓ Main convergence plot saved", flush=True)
        
        # Print statistics
        self._print_statistics(all_similarities, all_closest_targets)
    
    def plot_individual_targets(self, method='euclidean'):
        """Create individual plots for each target."""
        print("Creating individual target plots...", flush=True)
        
        # Process similarity data
        target_similarities, target_timelines = self._process_similarity_data(method)
        
        if not target_similarities:
            return
        
        # Compute closest targets
        all_similarities, all_timelines, all_closest_targets = self._compute_closest_targets(
            target_similarities, target_timelines
        )
        
        colors = self._get_colors()
        
        for target_idx in range(self.num_targets):
            target_mask = all_closest_targets == target_idx
            
            if np.sum(target_mask) < 5:  # Need at least 5 points
                continue
            
            target_times = all_timelines[target_mask]
            target_sims = all_similarities[target_mask]
            
            # Choose scale
            individual_scale = self._choose_y_scale(target_sims, f"Target {target_idx}")
            
            # Create plot
            plt.figure(figsize=self.config['sub_figsize'])
            
            # Prepare data
            plot_t_sims = target_sims.copy().astype(float)
            if individual_scale == 'log':
                pos_mask = plot_t_sims > 0
                if np.sum(pos_mask) == 0:
                    individual_scale = 'linear'
                else:
                    plot_t_sims[~pos_mask] = np.nan
            
            # Scatter
            label = self._get_target_label(target_idx)
            plt.scatter(target_times, plot_t_sims, s=3.0, alpha=0.6,
                       color=colors[target_idx], label=f'Points closest to {label}')
            
            # Trend
            self._add_trend_line(target_times, target_sims, individual_scale,
                               color='black', label_prefix='Trend')
            
            # Styling
            self._apply_plot_style(individual_scale, target_sims,
                                 xlabel='Iteration',
                                 ylabel=self._get_y_label(scale=individual_scale),
                                 title=f'Evolution of Points Closest to {label}')
            
            # Save
            plt.savefig(os.path.join(self.graphs_path, f'similarityGraph_target_{target_idx}.png'),
                       dpi=self.config['dpi'])
            plt.close()
        
        print(f"✓ Individual target plots saved", flush=True)
    
    def plot_best_results(self):
        """Create bar chart of best results per target."""
        print("Creating best results plot...", flush=True)
        
        best_results = self._process_best_results()
        
        if best_results is None or len(best_results) == 0:
            print("No best results data available", flush=True)
            return
        
        # Choose scale
        results_scale = self._choose_y_scale(best_results, "Best Results")
        
        # Prepare data
        plot_results = best_results.copy().astype(float)
        if results_scale == 'log':
            plot_results[plot_results <= 0] = np.nan
        
        # Create plot
        plt.figure(figsize=self.config['sub_figsize'])
        bars = plt.bar(range(len(plot_results)), plot_results, alpha=0.7, color='skyblue')
        
        # Add trend line
        if len(best_results) > 3:
            self._add_bar_trend_line(best_results, results_scale)
        
        # Styling
        plt.xlabel('Target', fontsize=12)
        
        if results_scale == 'log':
            plt.yscale('log')
            pos_vals = best_results[best_results > 0]
            if len(pos_vals) > 0:
                min_val, max_val = np.min(pos_vals), np.max(pos_vals)
                if min_val == max_val:
                    min_val, max_val = min_val * 0.9, max_val * 1.1
                # Sensitive to max value with 25% padding above
                plt.ylim(bottom=min_val * 0.8, top=max_val * 1.25)
            plt.ylabel('Best Result (log scale)', fontsize=12)
        else:
            plt.ylabel('Best Result (linear scale)', fontsize=12)
            # Set tight limits for linear scale sensitive to min/max
            valid_results = best_results[np.isfinite(best_results)]
            if len(valid_results) > 0:
                min_val, max_val = np.min(valid_results), np.max(valid_results)
                y_range = max_val - min_val
                padding = y_range * 0.1 if y_range > 0 else 0.1
                plt.ylim(bottom=min_val - padding, top=max_val + padding)
        
        plt.title(self._get_title('best_results'), fontsize=14)
        plt.grid(True, alpha=0.3, axis='y')
        
        try:
            plt.tight_layout()
        except:
            pass
        
        plt.savefig(os.path.join(self.graphs_path, 'bestResultsGraph.png'),
                   dpi=self.config['dpi'])
        plt.close()
        
        print(f"✓ Best results plot saved", flush=True)
        
        del best_results
        gc.collect()
    
    def plot_clustering(self, method='euclidean'):
        """Create 2D clustering plot (Target 0 vs Target 1)."""
        if self.num_targets < 2:
            return
        
        print("Creating clustering visualization (Target 0 vs Target 1)...", flush=True)
        
        # Process similarity data
        target_similarities, target_timelines = self._process_similarity_data(method)
        
        if not target_similarities or len(target_similarities) < 2:
            return
        
        # Get data for first two targets
        x_data = np.array(target_similarities[0])
        y_data = np.array(target_similarities[1])
        
        min_len = min(len(x_data), len(y_data))
        if min_len < 10:
            print("Not enough data for clustering plot", flush=True)
            return
        
        x_data = x_data[:min_len]
        y_data = y_data[:min_len]
        t_data = np.array(target_timelines[0])[:min_len]
        
        # Create plot
        plt.figure(figsize=(10, 10))
        scatter = plt.scatter(x_data, y_data, c=t_data, cmap='viridis', s=3, alpha=0.5)
        plt.colorbar(scatter, label='Iteration (Time)')
        
        # Labels
        label_0 = self._get_target_label(0)
        label_1 = self._get_target_label(1)
        
        better_text = " - Lower is Better" if self.config['better_direction'] == 'lower' else ""
        plt.xlabel(f'{self.config["metric_type"].title()} to {label_0}{better_text}', fontsize=12)
        plt.ylabel(f'{self.config["metric_type"].title()} to {label_1}{better_text}', fontsize=12)
        plt.title(f'Population Clustering: {label_0} vs {label_1}', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Diagonal reference line
        lims = [np.min([plt.xlim(), plt.ylim()]), np.max([plt.xlim(), plt.ylim()])]
        plt.plot(lims, lims, 'k--', alpha=0.2, zorder=0)
        
        try:
            plt.tight_layout()
        except:
            pass
        
        plt.savefig(os.path.join(self.graphs_path, 'clusteringGraph.png'),
                   dpi=self.config['dpi'])
        plt.close()
        
        print(f"✓ Clustering plot saved", flush=True)
    
    def plot_gene_pool_2d_3d(self):
        """Create 2D or 3D visualization of gene pool evolution."""
        if self.target_dim not in [2, 3]:
            return
        
        print(f"Creating {self.target_dim}D gene pool visualization...", flush=True)
        
        # Load gene pool samples
        sample_params, sample_iterations, max_iteration = self._load_gene_pool_samples()
        
        if len(sample_params) == 0:
            print("No gene pool data available", flush=True)
            return
        
        sample_params = np.array(sample_params)
        sample_iterations = np.array(sample_iterations)
        max_iteration = max_iteration if max_iteration > 0 else 1
        
        colors_time = plt.cm.viridis(sample_iterations / max_iteration)
        
        if self.target_dim == 2:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot samples
            scatter = ax.scatter(sample_params[:, 0], sample_params[:, 1],
                               c=sample_iterations, cmap='viridis', s=2, alpha=0.5)
            plt.colorbar(scatter, label='Iteration', ax=ax)
            
            # Plot targets
            if self.targets is not None:
                target_array = np.array(self.targets)
                ax.scatter(target_array[:, 0], target_array[:, 1],
                          c='red', marker='*', s=200, edgecolors='black',
                          linewidths=2, label='Targets', zorder=5)
            
            ax.set_xlabel('Dimension 0', fontsize=12)
            ax.set_ylabel('Dimension 1', fontsize=12)
            ax.set_title('2D Gene Pool Evolution', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        else:  # 3D
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot samples
            scatter = ax.scatter(sample_params[:, 0], sample_params[:, 1], sample_params[:, 2],
                               c=sample_iterations, cmap='viridis', s=1, alpha=0.4)
            plt.colorbar(scatter, label='Iteration', ax=ax, pad=0.1)
            
            # Plot targets
            if self.targets is not None:
                target_array = np.array(self.targets)
                ax.scatter(target_array[:, 0], target_array[:, 1], target_array[:, 2],
                          c='red', marker='*', s=300, edgecolors='black',
                          linewidths=2, label='Targets', zorder=5)
            
            ax.set_xlabel('Dimension 0', fontsize=11)
            ax.set_ylabel('Dimension 1', fontsize=11)
            ax.set_zlabel('Dimension 2', fontsize=11)
            ax.set_title('3D Gene Pool Evolution', fontsize=14)
            ax.legend()
        
        try:
            plt.tight_layout()
        except:
            pass
        
        filename = '2d_gene_pool.png' if self.target_dim == 2 else '3d_gene_pool.png'
        plt.savefig(os.path.join(self.graphs_path, filename), dpi=self.config['dpi'])
        plt.close()
        
        print(f"✓ {self.target_dim}D gene pool visualization saved", flush=True)
        
        del sample_params, sample_iterations
        gc.collect()
    
    # ========== Internal Helper Methods ==========
    
    def _process_similarity_data(self, method='euclidean', chunk_size=1000):
        """Process similarity/distance data in chunks."""
        loader = ChunkedGeneHistoryLoader(
            savePath=self.args.path,
            geneFormat=self.file_format,
            required_keys_and_types=['fitnessScore', 'epoch', 'iteration', 'loss', 'tests'],
            agent_id=getattr(self.args, 'agent_idx', 0),
            shuffle=False
        )
        
        target_similarities = [[] for _ in range(self.num_targets)]
        target_timelines = [[] for _ in range(self.num_targets)]
        
        chunk_vec = []
        chunk_timeline = []
        
        is_complete = False
        while not is_complete:
            chunk, is_complete = loader.loadGeneHistoryChunkAsListOfDics(
                self.args, chunk_size, allEpochs=True
            )
            
            for gene in chunk:
                if 'fitnessScore' not in gene or gene['fitnessScore'] is None:
                    continue
                if 'param' in gene and 'blabla0' in gene['param']:
                    chunk_vec.append(gene['param']['blabla0'])
                    chunk_timeline.append(gene['iteration'])
            
            if len(chunk_vec) >= chunk_size:
                vec_array = np.array(chunk_vec)
                timeline_array = np.array(chunk_timeline)
                
                for i, tar in enumerate(self.targets):
                    tmp = distance.cdist(vec_array, np.array(tar).reshape(1, -1), method)
                    similarities = tmp.flatten()
                    target_similarities[i].extend(similarities)
                    target_timelines[i].extend(timeline_array)
                
                chunk_vec.clear()
                chunk_timeline.clear()
                gc.collect()
        
        # Process remaining
        if chunk_vec:
            vec_array = np.array(chunk_vec)
            timeline_array = np.array(chunk_timeline)
            
            for i, tar in enumerate(self.targets):
                tmp = distance.cdist(vec_array, np.array(tar).reshape(1, -1), method)
                similarities = tmp.flatten()
                target_similarities[i].extend(similarities)
                target_timelines[i].extend(timeline_array)
        
        return target_similarities, target_timelines
    
    def _compute_closest_targets(self, target_similarities, target_timelines):
        """Find closest target for each data point."""
        all_similarities = []
        all_timelines = []
        all_closest_targets = []
        
        max_len = max(len(sim) for sim in target_similarities)
        
        for point_idx in range(max_len):
            point_similarities = []
            point_timeline = None
            
            for target_idx in range(len(target_similarities)):
                if point_idx < len(target_similarities[target_idx]):
                    point_similarities.append(target_similarities[target_idx][point_idx])
                    if point_timeline is None:
                        point_timeline = target_timelines[target_idx][point_idx]
                else:
                    point_similarities.append(float('inf'))
            
            if point_similarities and point_timeline is not None:
                closest_target_idx = np.argmin(point_similarities)
                closest_similarity = point_similarities[closest_target_idx]
                
                if np.isfinite(closest_similarity):
                    all_similarities.append(closest_similarity)
                    all_timelines.append(point_timeline)
                    all_closest_targets.append(closest_target_idx)
        
        return (np.array(all_similarities), np.array(all_timelines), 
                np.array(all_closest_targets))
    
    def _process_best_results(self, chunk_size=1000):
        """Process test results to find best results."""
        loader = ChunkedGeneHistoryLoader(
            savePath=self.args.path,
            geneFormat=self.file_format,
            required_keys_and_types=['fitnessScore', 'epoch', 'iteration', 'tests'],
            agent_id=getattr(self.args, 'agent_idx', 0),
            shuffle=False
        )
        
        best_results = None
        chunk_tests = []
        
        is_complete = False
        while not is_complete:
            chunk, is_complete = loader.loadGeneHistoryChunkAsListOfDics(
                self.args, chunk_size, allEpochs=True
            )
            
            for gene in chunk:
                if 'fitnessScore' in gene and gene['fitnessScore'] is not None:
                    if 'tests' in gene:
                        chunk_tests.append(gene['tests'])
            
            if len(chunk_tests) >= chunk_size:
                chunk_array = np.array(chunk_tests)
                
                if self.args.evolutionTarget == 1:
                    chunk_best = chunk_array.max(axis=0)
                else:
                    chunk_best = chunk_array.min(axis=0)
                
                if best_results is None:
                    best_results = chunk_best
                else:
                    if self.args.evolutionTarget == 1:
                        best_results = np.maximum(best_results, chunk_best)
                    else:
                        best_results = np.minimum(best_results, chunk_best)
                
                chunk_tests.clear()
                gc.collect()
        
        # Process remaining
        if chunk_tests:
            chunk_array = np.array(chunk_tests)
            
            if self.args.evolutionTarget == 1:
                chunk_best = chunk_array.max(axis=0)
            else:
                chunk_best = chunk_array.min(axis=0)
            
            if best_results is None:
                best_results = chunk_best
            else:
                if self.args.evolutionTarget == 1:
                    best_results = np.maximum(best_results, chunk_best)
                else:
                    best_results = np.minimum(best_results, chunk_best)
        
        return best_results
    
    def _load_gene_pool_samples(self, max_samples=10000):
        """Load samples from gene pool for visualization."""
        gene_files = glob.glob(self.args.path + '/genes/*.pkl')
        if len(gene_files) == 0:
            gene_files = glob.glob(self.args.path + '/genes/*.json')
        
        if len(gene_files) == 0:
            return [], [], 0
        
        gene_loader = ChunkedGeneHistoryLoader(
            savePath=self.args.path + '/genes',
            geneFormat='pkl' if glob.glob(self.args.path + '/genes/*.pkl') else 'csv',
            required_keys_and_types=['fitnessScore', 'epoch', 'iteration', 'loss', 'tests'],
            agent_id=getattr(self.args, 'agent_idx', 0),
            shuffle=False
        )
        
        max_iteration = 0
        sample_params = []
        sample_iterations = []
        
        is_complete = False
        while not is_complete:
            chunk, is_complete = gene_loader.loadGeneHistoryChunkAsListOfDics(
                self.args, 1000, allEpochs=True
            )
            
            for gene in chunk:
                if 'param' not in gene or 'blabla0' not in gene['param']:
                    continue
                
                if gene['iteration'] > max_iteration:
                    max_iteration = gene['iteration']
                
                if len(sample_params) < max_samples:
                    sample_params.append(gene['param']['blabla0'])
                    sample_iterations.append(gene['iteration'])
            
            if len(sample_params) >= max_samples:
                break
        
        return sample_params, sample_iterations, max_iteration
    
    def _choose_y_scale(self, data, scale_name="Y"):
        """
        Automatically choose between linear and log scale.
        Uses log scale when max/min dynamic range > 100x.
        """
        data = np.array(data)
        data = data[np.isfinite(data)]
        
        if len(data) == 0:
            return 'linear'
        
        positive_data = data[data > 0]
        
        if len(positive_data) == 0:
            print(f"{scale_name}: Using linear scale (no positive values)", flush=True)
            return 'linear'
        
        data_min = np.min(positive_data)
        data_max = np.max(positive_data)
        
        if data_max <= 0 or data_min <= 0:
            return 'linear'
        
        dynamic_range = data_max / data_min if data_min > 0 else 1
        
        # Primary criterion: Use log scale if dynamic range > 100x
        if dynamic_range > 100:
            print(f"{scale_name}: Using log scale (dynamic range: {dynamic_range:.1f}x > 100x)", flush=True)
            return 'log'
        
        try:
            log_data = np.log10(positive_data)
            log_std = np.std(log_data)
            linear_std = np.std(positive_data)
            data_mean = np.mean(positive_data)
            
            log_cv = log_std / np.abs(np.mean(log_data)) if np.mean(log_data) != 0 else float('inf')
            linear_cv = linear_std / data_mean if data_mean != 0 else float('inf')
        except:
            log_cv = float('inf')
            linear_cv = 1.0
        
        log_score = 0
        linear_score = 0
        reasons = []
        
        # Dynamic range (secondary scoring for < 100x range)
        if dynamic_range > 10:
            log_score += 1
            reasons.append(f"moderate dynamic range ({dynamic_range:.1f}x)")
        else:
            linear_score += 1
            reasons.append(f"small dynamic range ({dynamic_range:.1f}x)")
        
        # Distribution
        if len(positive_data) > 5:
            try:
                log_bins = np.logspace(np.log10(data_min), np.log10(data_max), 10)
                log_hist, _ = np.histogram(positive_data, bins=log_bins)
                filled_log_bins = np.sum(log_hist > 0)
                
                linear_bins = np.linspace(data_min, data_max, 10)
                linear_hist, _ = np.histogram(positive_data, bins=linear_bins)
                filled_linear_bins = np.sum(linear_hist > 0)
                
                if filled_log_bins >= filled_linear_bins:
                    log_score += 1
                    reasons.append("better log distribution")
                else:
                    linear_score += 1
                    reasons.append("better linear distribution")
            except:
                pass
        
        # Coefficient of variation
        if log_cv < linear_cv * 0.8:
            log_score += 2
            reasons.append("lower log CV")
        elif linear_cv < log_cv * 0.8:
            linear_score += 2
            reasons.append("lower linear CV")
        
        scale_type = 'log' if log_score > linear_score else 'linear'
        print(f"{scale_name}: Using {scale_type} scale (score: {log_score} vs {linear_score}) - {', '.join(reasons[:3])}", 
              flush=True)
        
        return scale_type
    
    def _fit_trend_line(self, x, y, method='auto'):
        """Fit advanced trend line to data."""
        if len(x) < 2:
            return x, y, 'none'
        
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]
        
        mask = np.isfinite(x_sorted) & np.isfinite(y_sorted)
        x_clean = x_sorted[mask]
        y_clean = y_sorted[mask]
        
        if len(x_clean) < 2:
            return x, y, 'none'
        
        x_trend = np.linspace(x_clean.min(), x_clean.max(), min(300, max(50, len(x_clean) * 4)))
        successful_fits = []
        
        methods_to_try = ['polynomial', 'exponential', 'moving_average'] if method == 'auto' else [method]
        
        for try_method in methods_to_try:
            try:
                if try_method == 'polynomial':
                    for degree in range(2, min(7, len(x_clean))):
                        try:
                            coeffs = np.polyfit(x_clean, y_clean, degree)
                            y_trend = np.polyval(coeffs, x_trend)
                            if np.all(np.isfinite(y_trend)):
                                y_pred = np.polyval(coeffs, x_clean)
                                corr = np.corrcoef(y_clean, y_pred)[0, 1] if len(y_clean) > 1 else 0
                                if not np.isnan(corr) and corr > 0.1:
                                    successful_fits.append((x_trend, y_trend, f'poly{degree}', abs(corr)))
                        except:
                            continue
                
                elif try_method == 'exponential':
                    try:
                        popt, _ = curve_fit(lambda x, a, b: a * np.exp(b * x), 
                                          x_clean, y_clean, p0=[y_clean[0], -0.001], maxfev=3000)
                        y_trend = popt[0] * np.exp(popt[1] * x_trend)
                        if np.all(np.isfinite(y_trend)) and np.all(y_trend > 0):
                            successful_fits.append((x_trend, y_trend, 'exponential', 0.7))
                    except:
                        pass
                
                elif try_method == 'moving_average':
                    window = max(2, len(x_clean) // 10)
                    if window < len(x_clean):
                        smoothed = np.convolve(y_clean, np.ones(window)/window, mode='valid')
                        smoothed_x = np.convolve(x_clean, np.ones(window)/window, mode='valid')
                        if len(smoothed) > 1:
                            f = interpolate.interp1d(smoothed_x, smoothed, kind='linear',
                                                   bounds_error=False, fill_value='extrapolate')
                            y_trend = f(x_trend)
                            if np.all(np.isfinite(y_trend)):
                                successful_fits.append((x_trend, y_trend, f'moving_avg', 0.5))
            except:
                continue
        
        if successful_fits:
            best_fit = max(successful_fits, key=lambda f: f[3])
            return best_fit[0], best_fit[1], best_fit[2]
        
        # Fallback
        try:
            f = interpolate.interp1d(x_clean, y_clean, kind='linear', 
                                   bounds_error=False, fill_value='extrapolate')
            y_trend = f(x_trend)
            return x_trend, y_trend, 'linear'
        except:
            return x, y, 'none'
    
    def _prepare_plot_data(self, similarities, timelines, closest_targets, colors, y_scale):
        """Prepare data for plotting with proper handling of log scale."""
        plot_sims = similarities.copy().astype(float)
        plot_times = timelines.copy()
        plot_colors = np.array([colors[i] for i in closest_targets])
        
        if y_scale == 'log':
            pos_mask = plot_sims > 0
            if np.sum(pos_mask) > 0:
                plot_sims[~pos_mask] = np.nan
            else:
                y_scale = 'linear'
        
        return plot_sims, plot_times, plot_colors
    
    def _add_trend_line(self, x, y, y_scale, color='black', label_prefix='Overall trend'):
        """Add trend line to current plot."""
        try:
            valid_mask = y > 0 if y_scale == 'log' else np.isfinite(y)
            if np.sum(valid_mask) > 2:
                x_trend, y_trend, trend_method = self._fit_trend_line(
                    x[valid_mask], y[valid_mask], method='auto'
                )
                
                if trend_method != 'none':
                    if y_scale == 'log':
                        y_trend = np.array(y_trend)
                        y_trend[y_trend <= 0] = np.nan
                    
                    plt.plot(x_trend, y_trend, color=color, linewidth=3, alpha=0.8,
                           label=f'{label_prefix} ({trend_method})')
        except Exception as e:
            print(f"Could not fit trend line: {e}", flush=True)
    
    def _add_bar_trend_line(self, data, y_scale):
        """Add trend line to bar chart."""
        x_bars = np.array(range(len(data)))
        try:
            valid_mask = data > 0 if y_scale == 'log' else np.isfinite(data)
            if np.sum(valid_mask) > 2:
                x_trend, y_trend, trend_method = self._fit_trend_line(
                    x_bars[valid_mask], data[valid_mask]
                )
                if y_scale == 'log':
                    y_trend = np.array(y_trend)
                    y_trend[y_trend <= 0] = np.nan
                plt.plot(x_trend, y_trend, color='red', linewidth=3,
                       label=f'Trend ({trend_method})', alpha=0.8)
                plt.legend()
        except:
            pass
    
    def _apply_plot_style(self, y_scale, data, xlabel, ylabel, title):
        """Apply consistent styling to plot with sensitive y-axis limits."""
        if y_scale == 'log':
            plt.yscale('log')
            pos_vals = data[data > 0]
            if len(pos_vals) > 0:
                min_val, max_val = np.min(pos_vals), np.max(pos_vals)
                if min_val == max_val:
                    min_val, max_val = min_val * 0.9, max_val * 1.1
                # Sensitive to max value with 25% padding above
                plt.ylim(bottom=min_val * 0.8, top=max_val * 1.25)
        else:
            # Linear scale: tight limits sensitive to min/max
            valid_data = data[np.isfinite(data)]
            if len(valid_data) > 0:
                min_val, max_val = np.min(valid_data), np.max(valid_data)
                y_range = max_val - min_val
                padding = y_range * 0.1 if y_range > 0 else 0.1
                plt.ylim(bottom=min_val - padding, top=max_val + padding)
        
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        try:
            plt.tight_layout()
        except:
            pass
    
    def _get_target_label(self, target_idx):
        """Get label for a target."""
        if self.config['target_labels'] and target_idx < len(self.config['target_labels']):
            return f"Target {target_idx} ({self.config['target_labels'][target_idx]})"
        return f"Target {target_idx}"
    
    def _print_statistics(self, similarities, closest_targets):
        """Print summary statistics."""
        print(f"Similarity analysis summary:", flush=True)
        print(f"Total data points: {len(similarities)}", flush=True)
        for target_idx in range(self.num_targets):
            target_count = np.sum(closest_targets == target_idx)
            target_percentage = (target_count / len(similarities)) * 100
            print(f"  Target {target_idx}: {target_count} points ({target_percentage:.1f}%)", flush=True)
