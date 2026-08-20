import numpy as np
from sklearn.metrics import pairwise_distances_chunked
import pickle
import tempfile
import os

import numpy as np
from sklearn.metrics import pairwise_distances_chunked
import pickle
import tempfile
import os

class IncrementalDiversityCalculator:
    def __init__(self, distance_metric: str = 'euclidean'):
        self.distance_metric = distance_metric
        
        # Accumulators
        self.total_distance_sum = 0.0
        self.total_pairs_count = 0
        self.chunk_count = 0
        
        # storage for chunk filenames
        self.stored_chunk_filenames = []
        
        # AUTOMATIC CLEANUP: 
        # assigning TemporaryDirectory to self keeps the folder alive 
        # as long as this class instance exists. 
        # When this instance is garbage collected, the folder deletes itself.
        self.temp_dir_obj = tempfile.TemporaryDirectory()
        self.temp_dir_path = self.temp_dir_obj.name
        
        print(f"Initialized diversity calculator using temp storage at: {self.temp_dir_path}", flush=True)

    def add_chunk(self, new_vectors: np.ndarray):
        """
        Process a new chunk:
        1. Calc internal distances
        2. Calc distances vs historical chunks
        3. Save this chunk to disk for future comparisons
        """
        # Ensure input is numpy array
        if not isinstance(new_vectors, np.ndarray):
            new_vectors = np.array(new_vectors)
            
        n_new = len(new_vectors)
        
        # --- Step 1: Internal Diversity (Within this new chunk) ---
        # We use a generator to process this incrementally
        # WITHOUT reduce_func - just sum the distance matrices directly
        chunk_gen = pairwise_distances_chunked(
            new_vectors, 
            metric=self.distance_metric
        )
        # Sum all distance chunks (each D is a 2D array of distances)
        total_internal_dist = sum(np.sum(D) for D in chunk_gen)
        
        # Add to global sum (divide by 2: symmetric matrix, we only want upper triangle)
        self.total_distance_sum += (total_internal_dist / 2.0)
        self.total_pairs_count += (n_new * (n_new - 1) // 2)

        # --- Step 2: Historical Diversity (New vs Old) ---
        # Compare against every chunk previously saved to disk
        for filename in self.stored_chunk_filenames:
            file_path = os.path.join(self.temp_dir_path, filename)
            
            with open(file_path, 'rb') as f:
                old_vectors = pickle.load(f)
            
            n_old = len(old_vectors)
            
            # Compare New Block vs Old Block
            inter_chunk_gen = pairwise_distances_chunked(
                new_vectors, 
                old_vectors, 
                metric=self.distance_metric
            )
            dist_sum = sum(np.sum(D) for D in inter_chunk_gen)
            
            self.total_distance_sum += dist_sum
            self.total_pairs_count += (n_new * n_old)
            
            # Free memory of old chunk immediately
            del old_vectors

        # --- Step 3: Save Current Chunk to Disk ---
        filename = f"chunk_{self.chunk_count}.pkl"
        save_path = os.path.join(self.temp_dir_path, filename)
        
        with open(save_path, 'wb') as f:
            pickle.dump(new_vectors, f)
        
        self.stored_chunk_filenames.append(filename)
        self.chunk_count += 1
        
        # Free memory of new chunk
        del new_vectors

    def get_current_score(self) -> float:
        """Get the diversity score based on data processed SO FAR."""
        if self.total_pairs_count == 0:
            return 0.0
        return self.total_distance_sum / self.total_pairs_count

from GA_utils_history_loader import ChunkedGeneHistoryLoader

import gc
try:
    import psutil
except Exception:
    psutil = None


def _memory_percent():
    if psutil is not None:
        try:
            return psutil.virtual_memory().percent
        except Exception:
            pass
    return 0


def _available_memory_gb():
    if psutil is not None:
        try:
            return psutil.virtual_memory().available / 1e9
        except Exception:
            pass
    try:
        page_size = os.sysconf('SC_PAGE_SIZE')
        avail_pages = os.sysconf('SC_AVPHYS_PAGES')
        return (page_size * avail_pages) / 1e9
    except Exception:
        return 0.0

def calculateDiversityMatrix(args=None, geneFormat=None):
    '''
    Calculates diversity matrix using dynamic memory estimation 
    to prevent OOM errors while maximizing throughput.
    '''
    
    # 1. Initialize Loader
    loader = ChunkedGeneHistoryLoader(
        savePath=args.path,
        agent_id=args.agent_idx,
        geneFormat=geneFormat,
        required_keys_and_types=['gene', 'epoch', 'fitnessScore'], # Only load what we need
        shuffle=False
    )

    # 2. Dynamic Memory Estimation
    # We ask the loader how many records fit in 50% of available RAM.
    # We use a sample_size of 50 to get a reliable average record size.
    raw_capacity = loader.estimate_cpu_memory_capacity(
        sample_size=50, 
        percentage_of_memory_to_be_used=0.5, 
        use_slurm_allocation=getattr(args, 'slurm', False)
    )

    # 3. Apply Safety Factor
    # raw_capacity is how many records fit in RAM at rest.
    # The Diversity Calculator needs:
    #   - 1 chunk in memory (current)
    #   - 1 chunk loaded from disk (history comparison)
    #   - Overhead for pairwise distance calculation buffers
    # Therefore, we divide the raw storage capacity by 4 to be safe.
    safe_chunk_size = int(raw_capacity / 4)

    # 4. Clamp Limits
    # Min: 100 (prevent tiny chunks causing overhead)
    # Max: 20,000 (prevent huge chunks causing timeouts/fragmentation)
    safe_chunk_size = max(100, min(safe_chunk_size, 20000))

    # Get actual allocated memory (SLURM-aware)
    import os
    slurm_alloc = None
    use_slurm = getattr(args, 'slurm', False)
    
    if use_slurm:
        # Try SLURM_MEM_PER_NODE first (most reliable)
        mem_per_node = os.environ.get('SLURM_MEM_PER_NODE')
        if mem_per_node and mem_per_node.strip():
            try:
                # Parse MB value
                slurm_alloc = float(mem_per_node) * 1024 ** 2  # Convert MB to bytes
            except (ValueError, TypeError):
                pass
    
    if slurm_alloc:
        print(f"Dynamic Memory Config: SLURM allocated {slurm_alloc / (1024**3):.2f} GB (via SLURM_MEM_PER_NODE).", flush=True)
    else:
        print(f"Dynamic Memory Config: System has {_available_memory_gb():.2f} GB free.", flush=True)
    
    print(f" Calculated Chunk Size: {safe_chunk_size} records (Targeting <50% RAM usage)", flush=True)

    # 5. Initialize Calculator
    diversity_calc = IncrementalDiversityCalculator(distance_metric='euclidean')

    try:
        # Load only the required number of iterations/population
        num_to_load = args.populationSize
        skipped_missing_gene = 0
        skipped_malformed_gene = 0
        
        # Generator: Streams data without holding it all in RAM
        chunk_generator = loader.loadLastGeneHistoryIterations_inChunks(
            args=args, 
            population_size=num_to_load, 
            chunk_size=safe_chunk_size
        )

        for chunk in chunk_generator:
            # Extract just the gene vectors
            # (Assuming chunk is list of dicts: [{'gene': [...], ...}, ...])
            geneData = []
            expected_gene_length = None
            while chunk:
                item = chunk.pop()  # Remove from end (faster)
                if not isinstance(item, dict) or ('gene' not in item):
                    skipped_missing_gene += 1
                    del item
                    continue

                try:
                    gene_vector = np.asarray(item['gene'], dtype=np.float32).reshape(-1)
                    if gene_vector.size == 0:
                        skipped_malformed_gene += 1
                        del item
                        continue

                    if expected_gene_length is None:
                        expected_gene_length = int(gene_vector.size)
                    elif int(gene_vector.size) != expected_gene_length:
                        skipped_malformed_gene += 1
                        del item
                        continue

                    geneData.append(gene_vector)
                except Exception:
                    skipped_malformed_gene += 1
                del item  # Delete dict immediately

            if len(geneData) == 0:
                del chunk
                continue

            geneData = np.vstack(geneData)
            del chunk
            
            # Add to calculator (Computes, saves to disk, then clears RAM)
            diversity_calc.add_chunk(geneData)
            
            # Explicit Memory Cleanup per chunk
            del geneData
            
            # Force Garbage Collection periodically (optional, but good for tight memory)
            # We check if we are getting close to the memory limit
            if _memory_percent() > 80:
                gc.collect()

        # 6. Compute Final Score
        diversityMeanReturn = dict()
        diversityMeanReturn['Average pairwise distances'] = diversity_calc.get_current_score()
        print(
            f"GA - Part 2 - DiversitySkipCounter missing_gene={skipped_missing_gene} malformed_gene={skipped_malformed_gene}",
            flush=True,
        )
        
        return diversityMeanReturn

    except Exception as e:
        print(f"Error during diversity calculation: {e}", flush=True)
        return None

    finally:
        # 7. Final Cleanup
        # Explicitly delete the calculator to trigger its internal cleanup (removing temp files)
        if 'diversity_calc' in locals():
            del diversity_calc
        
        # Force full garbage collection to release 50% memory back to OS
        gc.collect()
        print("Memory cleanup complete.", flush=True)



# Example usage
if __name__ == "__main__":
    # Create diversity calculator
    diversity_calc = ChunkedGeneHistoryLoader(distance_metric='euclidean', use_disk_storage=True)
    
    # Simulate adding chunks
    for i in range(3):
        # Generate random vectors between -1 and 1
        chunk = np.random.uniform(-1, 1, size=(100, 50))  # 100 vectors, 50 features
        diversity_calc.add_chunk(chunk)
    
    # Compute diversity score
    diversity_score = diversity_calc.compute_diversity()
    print(f"Final diversity score: {diversity_score:.6f}", flush=True)
    
    # Get statistics
    stats = diversity_calc.get_stats()
    print("Statistics:", stats, flush=True)
    
    # Reset for new calculation
    diversity_calc.reset()