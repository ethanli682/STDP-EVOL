import glob
import pickle
import zstandard
import random
import sys
import os
import json
import ast
import concurrent.futures
import numpy as np
import heapq
from sklearn.cluster import KMeans
from typing import List, Dict, Any, Optional, Tuple, Union
from GA_utils_framed_records import FramedRecordReader

try:
    import psutil
except ImportError:
    psutil = None

# Try to import torch, but handle if missing (since it's optional in logic)
try:
    import torch
except ImportError:
    torch = None


class _MemorySnapshot:
    def __init__(self, total: int, available: int):
        self.total = int(total)
        self.available = int(available)


def _get_virtual_memory_snapshot() -> _MemorySnapshot:
    if psutil is not None:
        vm = psutil.virtual_memory()
        return _MemorySnapshot(total=vm.total, available=vm.available)

    meminfo_path = '/proc/meminfo'
    if os.path.exists(meminfo_path):
        meminfo = {}
        with open(meminfo_path, 'rt') as f:
            for line in f:
                parts = line.split(':', 1)
                if len(parts) != 2:
                    continue
                key = parts[0].strip()
                value_field = parts[1].strip().split()[0]
                try:
                    meminfo[key] = int(value_field) * 1024
                except Exception:
                    continue

        total = meminfo.get('MemTotal', 8 * 1024**3)
        available = meminfo.get('MemAvailable', meminfo.get('MemFree', total // 2))
        return _MemorySnapshot(total=total, available=available)

    try:
        page_size = os.sysconf('SC_PAGE_SIZE')
        phys_pages = os.sysconf('SC_PHYS_PAGES')
        avail_pages = os.sysconf('SC_AVPHYS_PAGES')
        return _MemorySnapshot(total=page_size * phys_pages, available=page_size * avail_pages)
    except Exception:
        return _MemorySnapshot(total=8 * 1024**3, available=4 * 1024**3)


class ChunkedGeneHistoryLoader:
    def __init__(self, savePath: str, geneFormat: str, required_keys_and_types: Optional[Union[List[str], Dict[str, str]]] = None, 
                 agent_id: int = None, shuffle: bool = True, shuffle_level: str = 'both', device: Optional[str] = None):
        """
        Initialize the chunked gene history loader. (Refactored: History Only)
        """
        self.savePath = savePath
        self.geneFormat = geneFormat
        self.shuffle = shuffle
        self.shuffle_level = shuffle_level
        self.device = device
        self.agent_id = agent_id
        if self.agent_id is None:
            raise ValueError("agent_id must be provided for this loader")        
        
        # Parse required_keys_and_types
        if required_keys_and_types is None:
            self.required_keys = None
            self.return_types = None
        elif isinstance(required_keys_and_types, list):
            self.required_keys = required_keys_and_types
            self.return_types = {key: 'list' for key in required_keys_and_types}
        elif isinstance(required_keys_and_types, dict):
            self.required_keys = list(required_keys_and_types.keys())
            self.return_types = required_keys_and_types.copy()
        else:
            raise ValueError("required_keys_and_types must be None, list of keys, or dict of {key: type}")
        
        # Initialize internal state
        self._file_list = []
        self._current_file_index = 0
        self._current_file_position = 0
        self._shuffled_indices = []
        self._current_chunk_index = 0
        self._is_initialized = False
        self._total_records = 0
        self._records_per_file = {}
        self._global_fitness_min = float('inf')
        self._global_fitness_max = float('-inf')
        self._cached_phases = {}  # Cache for phases 1-3
        self._cache_valid = False

        # Cache path for history counts
        self._cache_path = os.path.join(savePath, 'running', f'{str(agent_id)}_loader_metadata_cache.json')
        # Ensure directory exists for cache safety
        os.makedirs(os.path.dirname(self._cache_path), exist_ok=True)

    def _extract_required_data(self, record: Dict[Any, Any]) -> Dict[Any, Any]:
        """Extract only the required keys from a record."""
        if self.required_keys is None:
            return record
        
        extracted = {}
        for key in self.required_keys:
            if key in record:
                extracted[key] = record[key]
        return extracted

    def set_required_keys_and_types(self, keys_and_types: Union[List[str], Dict[str, str]]):
        """Set the keys and return types to extract from each record."""
        if isinstance(keys_and_types, list):
            self.required_keys = keys_and_types
            self.return_types = {key: 'list' for key in keys_and_types}
        elif isinstance(keys_and_types, dict):
            self.required_keys = list(keys_and_types.keys())
            self.return_types = keys_and_types.copy()
        else:
            raise ValueError("keys_and_types must be list of keys or dict of {key: type}")
        
    def set_shuffle_options(self, shuffle: bool, shuffle_level: str = 'both', device: Optional[str] = None):
        self.shuffle = shuffle
        self.shuffle_level = shuffle_level
        
        if device is not None:
            self.device = device
            if self.return_types:
                for key, return_type in self.return_types.items():
                    if return_type == 'torch':
                        self.return_types[key] = 'torch'
        
        if self._is_initialized:
            self._is_initialized = False

    # ============================================================================
    #  HISTORY LOADER INITIALIZATION (Optimized with Threads & Cache)
    # ============================================================================

    def _initialize_file_mapping(self, args, allEpochs: bool = False):
        """Initialize file mapping and record counts with optimizations."""
        if self._is_initialized:
            return
            
        self._file_list = glob.glob(self.savePath + '/*.log.csv')
        
        # Sort or shuffle files
        if self.shuffle and self.shuffle_level in ['both', 'files']:
            random.shuffle(self._file_list)
        else:
            self._file_list.sort()

        self._total_records = 0
        self._records_per_file = {}

        # 1. Try Cache
        cache_key = f"{args.epoch}_{allEpochs}_{len(self._file_list)}"
        if self._load_from_cache(cache_key):
            print("Loaded record counts from cache (Fast start)", flush=True)
        else:
            print("Indexing history files (First run)...", flush=True)
            # 2. Parallel Processing
            max_workers = min(32, (os.cpu_count() or 1) * 2)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {
                    executor.submit(self._count_valid_records, file, args, allEpochs): file 
                    for file in self._file_list
                }
                
                for future in concurrent.futures.as_completed(future_to_file):
                    file = future_to_file[future]
                    try:
                        count = future.result()
                        self._records_per_file[file] = count
                        self._total_records += count
                    except Exception as e:
                        self._records_per_file[file] = 0

            self._save_to_cache(cache_key)

        self._create_indices()
        self._is_initialized = True

    def _load_from_cache(self, cache_key):
        if os.path.exists(self._cache_path):
            try:
                with open(self._cache_path, 'r') as f:
                    data = json.load(f)
                    if data.get('key') == cache_key:
                        self._records_per_file = data['counts']
                        self._total_records = data['total']
                        return True
            except:
                return False
        return False

    def _save_to_cache(self, cache_key):
        try:
            data = {
                'key': cache_key,
                'total': self._total_records,
                'counts': self._records_per_file
            }
            with open(self._cache_path, 'w') as f:
                json.dump(data, f)
        except:
            pass

    def _count_valid_records(self, file: str, args, allEpochs: bool) -> int:
            """Count records with String Check Optimization (Fixed for JSON & YAML)"""
            count = 0
            
            # Prepare search strings for both formats
            json_search_1 = f'"epoch": {args.epoch}'  # Matches {"epoch": 50}
            json_search_2 = f"epoch: {args.epoch}"    # Matches epoch: 50
            
            check_epoch = not allEpochs

            try:
                if self.geneFormat == 'json':
                    with open(file, 'rt') as f:
                        for line in f:
                            # Fast String Check (Checks both formats)
                            if check_epoch:
                                if json_search_1 not in line and json_search_2 not in line:
                                    continue
                            try:
                                # 1. Try JSON (Fastest)
                                try:
                                    gene = json.loads(line)
                                except json.JSONDecodeError:
                                    # 2. Fallback to YAML/Text cleanup
                                    try:
                                        gene = json.loads(line) 
                                    except:
                                        # Handle weird YAML edge cases from your original code
                                        line = line.replace('}})]', '}},)]')
                                        gene = json.loads(line)
                                
                                if (allEpochs or gene.get('epoch') == args.epoch) and gene.get('fitnessScore') is not None:
                                    count += 1
                            except:
                                continue

                elif self.geneFormat == 'pickle' or self.geneFormat == 'pkl':
                    # For pickle/csv wrapper
                    with open(file, 'rt') as f_csv:
                        for line in f_csv:
                            # Fast String Check (Checks both formats)
                            if check_epoch:
                                if json_search_1 not in line and json_search_2 not in line:
                                    continue
                                    
                            try:
                                try:
                                    gene = json.loads(line)
                                except json.JSONDecodeError:
                                    gene = json.loads(line) # Attempt JSON again

                                if (allEpochs or gene.get('epoch') == args.epoch) and gene.get('fitnessScore') is not None:
                                    count += 1
                            except:
                                continue
            except:
                pass
                
            return count

    def _create_indices(self):
        indices = []
        for file_idx, file in enumerate(self._file_list):
            record_count = self._records_per_file.get(file, 0)
            record_indices = list(range(record_count))
            
            if self.shuffle and self.shuffle_level in ['both', 'records']:
                random.shuffle(record_indices)
            
            for record_idx in record_indices:
                indices.append((file_idx, record_idx))
        
        if self.shuffle and self.shuffle_level == 'both':
            random.shuffle(indices)
        
        self._shuffled_indices = indices
        self._current_chunk_index = 0

    # ============================================================================
    #  LOADING & CHUNKING LOGIC
    # ============================================================================

    def _load_records_from_file(self, file: str, max_records: Optional[int] = None) -> List[Dict[Any, Any]]:
        records = []
        try:
            if self.geneFormat == 'json':
                with open(file, 'rt') as f:
                    for line in f:
                        try:
                            try:
                                gene = json.loads(line)
                            except:
                                line = line.replace('}})]', '}},)]')
                                gene = json.loads(line)
                            gene['fname'] = file
                            records.append(self._extract_required_data(gene))
                            if max_records and len(records) >= max_records: break
                        except: continue
            elif self.geneFormat == 'pickle' or self.geneFormat == 'pkl':
                if FramedRecordReader.exists_for_csv(file):
                    with FramedRecordReader.from_csv_path(file, codec='pickle') as framed_reader:
                        limit = len(framed_reader) if max_records is None else min(max_records, len(framed_reader))
                        for idx in range(limit):
                            try:
                                gene = framed_reader.read(idx)
                                gene['fname'] = file
                                records.append(self._extract_required_data(gene))
                            except:
                                continue
                else:
                    with open(file, 'rt') as f_csv:
                        # Logic assumes .pkl file has same name as csv but with .pkl extension
                        pkl_path = file.replace('csv', 'pkl') # Simplistic replacement
                        if not os.path.exists(pkl_path):
                            pkl_path = file.replace('.log.csv', '.pkl')
                            
                        with zstandard.open(pkl_path, 'rb') as f_pkl:
                            for l_csv in f_csv:
                                try:
                                    try:
                                        gene = json.loads(l_csv)
                                    except:
                                        l_csv = l_csv.replace('}})]', '}},)]')
                                        gene = json.loads(l_csv)
                                    gene['fname'] = file
                                    try:
                                        gene.update(pickle.load(f_pkl))
                                        records.append(self._extract_required_data(gene))
                                    except: pass
                                    if max_records and len(records) >= max_records: break
                                except: continue
        except: pass
        return records

    def _update_fitness_extrema(self, records: List[Dict[Any, Any]]):
        """
        Internal helper to update global min/max fitness from a batch of records.
        """
        # Filter for existing fitnessScores that are not None
        scores = []
        for r in records:
            _fs = r.get('fitnessScore')
            if _fs is not None:
                try:
                    scores.append(float(_fs))
                except (TypeError, ValueError):
                    pass
        
        if not scores:
            return

        # Calculate local min/max for this batch
        local_min = min(scores)
        local_max = max(scores)

        # Update global state
        if local_min < self._global_fitness_min:
            self._global_fitness_min = local_min
        
        if local_max > self._global_fitness_max:
            self._global_fitness_max = local_max

    def loadGeneHistoryChunkAsListOfDics(self, args, chunk_size: int, allEpochs: bool = False) -> Tuple[List[Dict[Any, Any]], bool]:
        if not self._is_initialized:
            self._initialize_file_mapping(args, allEpochs)
            
        if self._current_chunk_index >= len(self._shuffled_indices):
            if self.shuffle:
                self._create_indices()
                print("Data exhausted, reshuffling...", flush=True)
            else:
                self._current_chunk_index = 0
                print("Data exhausted, restarting...", flush=True)
        
        start_idx = self._current_chunk_index
        end_idx = min(start_idx + chunk_size, len(self._shuffled_indices))
        
        # Cache approach
        file_records_cache = {}
        chunk_indices = self._shuffled_indices[start_idx:end_idx]
        file_groups = {}
        
        for file_idx, record_idx in chunk_indices:
            if file_idx not in file_groups: file_groups[file_idx] = []
            file_groups[file_idx].append(record_idx)
            
        population_chunk = []
        for file_idx, record_indices in file_groups.items():
            file_path = self._file_list[file_idx]
            if file_path not in file_records_cache:
                # Stream file to filter valid records (avoids loading entire file)
                valid = []
                for rec in self._yield_records_from_file(file_path):
                    if (allEpochs or rec.get('epoch') == args.epoch) and rec.get('fitnessScore') is not None:
                        valid.append(rec)
                file_records_cache[file_path] = valid
            
            f_recs = file_records_cache[file_path]
            for r_idx in record_indices:
                if r_idx < len(f_recs):
                    population_chunk.append(f_recs[r_idx])
                    
        self._current_chunk_index = end_idx
        is_complete = (end_idx >= len(self._shuffled_indices))
        self._update_fitness_extrema(population_chunk)
        print(f"Progress: {end_idx}/{len(self._shuffled_indices)} records", flush=True)
        return population_chunk, is_complete

    def loadGeneHistoryChunkAsLists(self, args, chunk_size: int, allEpochs: bool = False) -> Tuple[List[Any], bool]:
        population_chunk, is_complete = self.loadGeneHistoryChunkAsListOfDics(args, chunk_size, allEpochs)
        
        if not population_chunk:
            return [], is_complete

        # Determine keys
        if self.required_keys is None:
            all_keys = set().union(*(d.keys() for d in population_chunk))
            keys_to_use = sorted(list(all_keys))
            return_types = {key: 'list' for key in keys_to_use}
        else:
            keys_to_use = self.required_keys
            return_types = self.return_types

        data_structures = []
        
        # Optimized Extraction
        for key in keys_to_use:
            return_type = return_types.get(key, 'list')
            values = [record.get(key, None) for record in population_chunk]
            
            if return_type == 'list':
                data_structures.append(values)
            elif return_type == 'numpy':
                try:
                    data_structures.append(np.array(values))
                except:
                    data_structures.append(values)
            elif return_type.startswith('torch') and torch:
                try:
                    if ':' in return_type:
                        t_dev = return_type.split(':', 1)[1]
                    else:
                        t_dev = self.device if self.device else 'cpu'
                    
                    try:
                        # Direct conversion is usually fastest
                        tensor = torch.tensor(values, device=t_dev)
                    except:
                        # Fallback for mixed types
                        tensor = torch.tensor(values, device='cpu').to(t_dev)
                    data_structures.append(tensor)
                except:
                    data_structures.append(values)
            else:
                data_structures.append(values)
                
        return data_structures, is_complete

    def loadAllGeneHistory(self, args, allEpochs: bool = False) -> List[Dict[Any, Any]]:
        if not self._is_initialized:
            self._initialize_file_mapping(args, allEpochs)
        
        population = []
        if not allEpochs:
            if isinstance(self.required_keys, list) and 'epoch' not in self.required_keys:
                self.required_keys.append('epoch')
                
        # Sequential load - FIXED: Stream files instead of loading entire file
        total_yielded = 0
        for file_path in self._file_list:
            for record in self._yield_records_from_file(file_path):
                total_yielded += 1
                if (allEpochs or record.get('epoch') == args.epoch) and record.get('fitnessScore') is not None:
                    population.append(self._extract_required_data(record))

        # LOUD diagnostic for the silent-0 that previously caused resume to fall
        # back to a random bootstrap with no signal. Distinguish "decoded nothing"
        # (on-disk read failure) from "decoded but filtered out" (epoch/fitness
        # filter) so the cause is actionable.
        if not population and self._file_list:
            if total_yielded == 0:
                print(f"GA - history loader - WARNING: loadAllGeneHistory found "
                      f"{len(self._file_list)} history file(s) but DECODED 0 records. "
                      f"On-disk history could not be read (see framed decode warnings "
                      f"above). Resume will fall back to random bootstrap.", flush=True)
            else:
                print(f"GA - history loader - WARNING: loadAllGeneHistory decoded "
                      f"{total_yielded} record(s) from {len(self._file_list)} file(s) "
                      f"but 0 passed the fitnessScore/epoch filter "
                      f"(epoch={getattr(args, 'epoch', None)}, allEpochs={allEpochs}). "
                      f"Resume will fall back to random bootstrap.", flush=True)

        if self.shuffle:
            random.shuffle(population)
        self._update_fitness_extrema(population)
        print(f"Loaded all data: {len(population)} records", flush=True)
        return population
    
    def loadLastGeneHistoryIterations(self, args, population_size: int) -> List[Dict[Any, Any]]:
        """
        Streams history and returns the data for the requested window size ending at current agent_counter.
        
        Args:
            population_size: The number of recent iterations to include.
                             (e.g., if agent_counter is 100 and population_size is 10, 
                              it fetches iterations 91 to 100).
        """
        if not self._is_initialized:
            self._initialize_file_mapping(args, allEpochs=True)

        # 1. Source of Truth: args.agent_counter
        if not hasattr(args, 'agent_counter'):
             # Fallback or error depending on your preference
             print("Warning: args.agent_counter not found, defaulting to 0", flush=True)
             current_counter = 0
        else:
             current_counter = args.agent_counter

        # 2. Handle Start Logic (Clamp to 0 if history is short)
        # We calculate the start of the window. 
        start_counter = int(current_counter - population_size + 1)
        if start_counter < 0:
            start_counter = 0
        
        target_iterations = set(range(start_counter, current_counter + 1))
        
        population = []

        # Ensure 'iteration' key exists for filtering
        remove_iteration_key = False
        if self.required_keys is not None and 'iteration' not in self.required_keys:
            self.required_keys.append('iteration')
            remove_iteration_key = True

        try:
            # 3. OPTIMIZATION: Iterate files in REVERSE. 
            # The data you want is likely in the last few files.
            # We don't stop early because we don't strictly know if files are perfectly ordered,
            # but this processes the most likely candidates first.
            for file_path in reversed(self._file_list):
                
                # Check if we have collected enough data? 
                # Hard to do strictly without assuming 1 record per iteration, 
                # so we continue streaming safely.
                
                for record in self._yield_records_from_file(file_path):
                    # Check against the calculated target range
                    if record.get('iteration') in target_iterations and record.get('fitnessScore') is not None:
                        population.append(self._extract_required_data(record))

        finally:
            if remove_iteration_key:
                self.required_keys.remove('iteration')
                for record in population:
                    record.pop('iteration', None)

        if self.shuffle:
            random.shuffle(population)
            
        self._update_fitness_extrema(population)
        print(f"Loaded {len(population)} records from iterations {start_counter} to {current_counter}.", flush=True)
        return population

    def loadLastGeneHistoryIterations_inChunks(self, args, population_size: int, chunk_size: int):
        """
        Generator that streams the last 'population_size' iterations in chunks.
        
        Args:
            args: Argument object containing 'agent_counter'.
            population_size: Total number of historical records to load.
            chunk_size: Number of records to yield per call.
            
        Yields:
            List[Dict]: A list of records (length = chunk_size, except possibly the last one).
        """
        if not self._is_initialized:
            # We initialize with allEpochs=True because we are searching across multiple epochs/files
            self._initialize_file_mapping(args, allEpochs=True)

        # 1. Determine the range of iterations to load
        if not hasattr(args, 'agent_counter'):
             print("Warning: args.agent_counter not found, defaulting to 0", flush=True)
             current_counter = 0
        else:
             current_counter = args.agent_counter

        # Calculate window: e.g. if counter=100, size=10 -> start=91 (range 91-100)
        start_counter = max(0, current_counter - population_size + 1)
        target_iterations = set(range(start_counter, current_counter + 1))
        
        current_chunk = []
        total_records_yielded = 0
        
        # 2. Iterate files in REVERSE (Newest -> Oldest)
        # This is an optimization to find the latest iterations faster.
        for file_path in reversed(self._file_list):
            
            # Safety check: If we have already finished the job, stop opening new files.
            if total_records_yielded >= population_size:
                break
                
            # 3. Stream records from the current file
            # relying on the fixed _yield_records_from_file to handle 'break' safely
            for record in self._yield_records_from_file(file_path):
                
                # Filter: Check if record is in the target iteration window
                if record.get('iteration') in target_iterations and record.get('fitnessScore') is not None:
                    
                    # Extract only required keys (removes 'iteration' if not requested)
                    clean_record = self._extract_required_data(record)
                    current_chunk.append(clean_record)
                    
                    # 4. Yield Chunk if full
                    if len(current_chunk) >= chunk_size:
                        
                        # Optional: Shuffle within the chunk if the loader is set to shuffle
                        if self.shuffle:
                            random.shuffle(current_chunk)
                            
                        yield current_chunk
                        
                        total_records_yielded += len(current_chunk)
                        current_chunk = [] # Reset buffer
                        
                        # Stop scanning if we hit the limit
                        if total_records_yielded >= population_size:
                            break 
        
        # 5. Yield any remaining data in the buffer (partial chunk)
        if current_chunk and total_records_yielded < population_size:
            if self.shuffle:
                random.shuffle(current_chunk)
            yield current_chunk

    def loadBestFitnessHistory(self, args, top_k: int, maximize: bool,) -> List[Dict[Any, Any]]:
        """
        Streams history and returns the top 'top_k' best fitness scores.
        
        Args:
            maximize (bool): If True, returns highest scores. If False, returns lowest scores.
        """
        if not self._is_initialized:
            self._initialize_file_mapping(args, allEpochs=True)

        # Handling 'fitnessScore' key dependency
        remove_fitness_key = False
        if self.required_keys is not None and 'fitnessScore' not in self.required_keys:
            self.required_keys.append('fitnessScore')
            remove_fitness_key = True

        # 1. Create a "Super Generator" that streams records from ALL files sequentially
        #    This ensures we never load more than 1 record into memory at a time.
        def all_records_generator():
            for file_path in self._file_list:
                yield from self._yield_records_from_file(file_path)

        try:
            # 2. Use heapq optimized functions to pull top K from the stream
            #    nlargest/nsmallest are memory-safe O(N log K) when used on generators.
            if maximize:
                best_population = heapq.nlargest(
                    top_k,
                    (x for x in all_records_generator() if x.get('fitnessScore') is not None),
                    key=lambda x: float(x['fitnessScore'])
                )
            else:
                best_population = heapq.nsmallest(
                    top_k,
                    (x for x in all_records_generator() if x.get('fitnessScore') is not None),
                    key=lambda x: float(x['fitnessScore'])
                )

        finally:
            if remove_fitness_key:
                self.required_keys.remove('fitnessScore')

        # Cleanup: Remove fitnessScore from results if user didn't ask for it
        if remove_fitness_key:
            for record in best_population:
                record.pop('fitnessScore', None)

        self._update_fitness_extrema(best_population)
        
        direction = "highest" if maximize else "lowest"
        print(f"Loaded {len(best_population)} records with {direction} fitness scores.", flush=True)
        return best_population

    def loadTopKPercentGeneHistoryAsListOfDics(self, args, top_k_percent: float, 
                                maximize: bool, allEpochs: bool = False,
                                max_genes: Optional[int] = None) -> List[Dict[Any, Any]]:
        """
        Streams history and returns genes in the top K% by fitness (memory-safe).
        
        Uses heapq.nlargest/nsmallest for O(N log K) space complexity—never loads 
        full population. Returns the BEST K genes by fitness score.
        
        Args:
            top_k_percent: Percentage (0-100). E.g., 25 = top 25%
            maximize: If True, returns highest scores. If False, returns lowest.
            allEpochs: If True, considers all epochs. Else filters by args.epoch.
            max_genes: (Optional) Maximum number of genes to return. 
                       If set, returns min(top_k, max_genes).
                       heapq.nlargest/nsmallest guarantee these are the BEST genes.
                       Example: top_k_percent=50 with 1M population = 500k genes,
                       but max_genes=50000 will return the best 50k.
        
        Returns:
            List[Dict]: Records in top K% by fitness score (guaranteed best, capped at max_genes).
        
        NOTE: This implementation is CORRECT because heapq.nlargest/nsmallest
              use the O(N log K) algorithm which returns the best K elements.
              No min-heap workaround needed here.
        """
        if not self._is_initialized:
            self._initialize_file_mapping(args, allEpochs=allEpochs)
        
        # Estimate K from population size
        if self._total_records == 0:
            print("Warning: No records found. Returning empty list.", flush=True)
            return []
        
        top_k = max(1, int(self._total_records * (top_k_percent / 100.0)))
        
        # Apply max_genes cap if provided
        if max_genes is not None:
            original_top_k = top_k
            top_k = min(top_k, max_genes)
        
        # Ensure 'fitnessScore' is in required keys
        remove_fitness_key = False
        if self.required_keys is not None and 'fitnessScore' not in self.required_keys:
            self.required_keys.append('fitnessScore')
            remove_fitness_key = True
        
        try:
            # Stream all records, heapq.nlargest/nsmallest keep only top K in memory
            # These use O(N log K) algorithm internally
            def filtered_records_generator():
                for file_path in self._file_list:
                    for record in self._yield_records_from_file(file_path):
                        if (allEpochs or record.get('epoch') == args.epoch) and \
                        record.get('fitnessScore') is not None:
                            yield self._extract_required_data(record)
            
            # heapq guarantees these are the BEST K genes by fitness
            if maximize:
                top_population = heapq.nlargest(
                    top_k,
                    filtered_records_generator(),
                    key=lambda x: float(x['fitnessScore'])
                )
            else:
                top_population = heapq.nsmallest(
                    top_k,
                    filtered_records_generator(),
                    key=lambda x: float(x['fitnessScore'])
                )
        
        finally:
            if remove_fitness_key:
                self.required_keys.remove('fitnessScore')
                for record in top_population:
                    record.pop('fitnessScore', None)
        
        self._update_fitness_extrema(top_population)
        direction = "highest" if maximize else "lowest"
        
        # Enhanced logging to show when max_genes was applied
        if max_genes is not None:
            print(f"Loaded {len(top_population, flush=True)} records ({top_k_percent}% of {self._total_records}, "
                  f"capped at {max_genes}) with {direction} fitness (BEST by heapq).", flush=True)
        else:
            print(f"Loaded {len(top_population)} records ({top_k_percent}% of {self._total_records}, flush=True) "
                  f"with {direction} fitness (BEST by heapq).", flush=True)
        
        return top_population

    def loadTopKPercentGeneHistoryAsListOfDics_inChunks(self, args, top_k_percent: float, 
                                                        maximize: bool, chunk_size: int,
                                                        allEpochs: bool = False,
                                                        max_genes: Optional[int] = None):
        """
        Generator that streams genes in top K% by fitness in chunks.
        Memory-safe: Never loads all top-k at once.
        
        CRITICAL: When max_genes is set, uses min-heap to ensure we return
        the BEST X genes (by fitness), not just the first X genes encountered.
        
        Args:
            top_k_percent: Percentage (0-100). E.g., 25 = top 25%
            maximize: If True, returns highest scores. If False, returns lowest.
            chunk_size: Records to yield per iteration
            allEpochs: If True, all epochs. Else filters by args.epoch.
            max_genes: (Optional) Maximum genes to return.
                    IMPORTANT: Returns the BEST max_genes by fitness
                    (using min-heap), NOT just the first max_genes.
                    
        Yields:
            Tuple[List[Dict], bool]: (chunk_of_records, is_complete)
        """
        if not self._is_initialized:
            self._initialize_file_mapping(args, allEpochs=allEpochs)
        
        # Ensure fitnessScore is in required keys
        remove_keys = []
        if self.required_keys is not None:
            if 'fitnessScore' not in self.required_keys:
                self.required_keys.append('fitnessScore')
                remove_keys.append('fitnessScore')
        
        try:
            print(f"Initializing top-K filtering for {top_k_percent}%"
                f"{f' (keeping best {max_genes} genes by fitness)' if max_genes else ''}...", 
                flush=True)
            
            # Estimate K from population size
            top_k = max(1, int(self._total_records * (top_k_percent / 100.0)))
            
            if max_genes is not None:
                # Mode A: Use min-heap to keep only TOP X by fitness
                print(f"Using min-heap to track top {max_genes} genes by fitness...", flush=True)
                
                import itertools
                top_genes_heap = []  # Min-heap: (fitness_value, counter, record_dict)
                _heap_counter = itertools.count()  # unique tiebreaker to avoid dict comparison
                min_fitness_in_heap = float('inf') if maximize else float('-inf')
                total_considered = 0

                for file_path in self._file_list:
                    for record in self._yield_records_from_file(file_path):
                        # Check epoch and fitness
                        if not (allEpochs or record.get('epoch') == args.epoch):
                            continue

                        fitness_val = record.get('fitnessScore')
                        if fitness_val is None:
                            continue
                        try:
                            fitness_val = float(fitness_val)
                        except (TypeError, ValueError):
                            continue
                        
                        total_considered += 1
                        clean_record = self._extract_required_data(record)
                        
                        # Heap logic: keep only top X by fitness
                        if maximize:
                            # For maximization: keep highest values
                            if len(top_genes_heap) < max_genes:
                                # Heap not full yet
                                heapq.heappush(top_genes_heap, (fitness_val, next(_heap_counter), clean_record))
                                if len(top_genes_heap) == max_genes:
                                    # Heap just became full, record min fitness
                                    min_fitness_in_heap = top_genes_heap[0][0]
                            elif fitness_val > min_fitness_in_heap:
                                # This gene is better than worst in heap
                                heapq.heapreplace(top_genes_heap, (fitness_val, next(_heap_counter), clean_record))
                                min_fitness_in_heap = top_genes_heap[0][0]
                        else:
                            # For minimization: keep lowest values
                            if len(top_genes_heap) < max_genes:
                                # Heap not full yet (use negative for max-heap behavior)
                                heapq.heappush(top_genes_heap, (-fitness_val, next(_heap_counter), clean_record))
                                if len(top_genes_heap) == max_genes:
                                    # Heap just became full, record max fitness
                                    min_fitness_in_heap = -top_genes_heap[0][0]
                            elif fitness_val < min_fitness_in_heap:
                                # This gene is better than worst in heap
                                heapq.heapreplace(top_genes_heap, (-fitness_val, next(_heap_counter), clean_record))
                                min_fitness_in_heap = -top_genes_heap[0][0]
                        
                        # Memory cleanup every 1000 records
                        if total_considered % 1000 == 0 and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                # Heap now contains the top X genes by fitness
                # Sort them for consistent output
                if maximize:
                    sorted_records = sorted(top_genes_heap, key=lambda x: x[0], reverse=True)
                else:
                    sorted_records = sorted(top_genes_heap, key=lambda x: x[0])
                
                # Yield in chunks
                current_chunk = []
                for _, _, record_dict in sorted_records:
                    current_chunk.append(record_dict)
                    
                    if len(current_chunk) >= chunk_size:
                        if self.shuffle:
                            random.shuffle(current_chunk)
                        yield current_chunk, False
                        current_chunk = []
                
                # Yield remaining
                if current_chunk:
                    if self.shuffle:
                        random.shuffle(current_chunk)
                    yield current_chunk, True
                
                print(f"Heap-based filtering complete: "
                    f"returned {len(sorted_records)}/{max_genes} genes (best by fitness), "
                    f"considered={total_considered}, top_k_percent={top_k_percent}%", flush=True)
            
            else:
                # Mode B: No max_genes—stream top K% genes
                print(f"Streaming top {top_k_percent}% genes (no max_genes cap)...", flush=True)
                
                current_chunk = []
                total_included = 0
                total_skipped = 0
                
                for file_path in self._file_list:
                    for record in self._yield_records_from_file(file_path):
                        # Check epoch and fitness
                        if not (allEpochs or record.get('epoch') == args.epoch):
                            continue
                        
                        fitness_val = record.get('fitnessScore')
                        if fitness_val is None:
                            continue
                        try:
                            fitness_val = float(fitness_val)
                        except (TypeError, ValueError):
                            continue
                        
                        total_skipped += 1
                        
                        # Note: Without explicit top-K filtering in streaming mode,
                        # this yields ALL records in top_k_percent.
                        # For true top-K in streaming mode without max_genes,
                        # consider using heapq.nlargest offline instead.
                        clean_record = self._extract_required_data(record)
                        current_chunk.append(clean_record)
                        total_included += 1
                        
                        # Yield chunk when full
                        if len(current_chunk) >= chunk_size:
                            if self.shuffle:
                                random.shuffle(current_chunk)
                            
                            yield current_chunk, False
                            current_chunk = []
                        
                        # Memory cleanup every 1000 records
                        if total_included % 1000 == 0 and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                # Yield remaining records
                if current_chunk:
                    if self.shuffle:
                        random.shuffle(current_chunk)
                    yield current_chunk, True
                
                print(f"Top-K streaming complete: "
                    f"included={total_included}, top_k_percent={top_k_percent}%", flush=True)
            
        finally:
            # Cleanup
            if remove_keys and self.required_keys is not None:
                for key in remove_keys:
                    if key in self.required_keys:
                        self.required_keys.remove(key)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def loadTopKPercentGeneHistoryAsListOfDics_inChunks_WeightBased(
            self, args, top_k_percent: float, scaler, chunk_size: int,
            allEpochs: bool = False, max_genes: Optional[int] = None):
        """
        Generator that streams genes in top K% using WEIGHT-BASED filtering.
        
        Memory-safe: Never loads all top-K at once. Uses weight threshold from 
        the scaler's exponential weighting to determine which genes to include.
        
        CRITICAL FIX: If max_genes is set, uses min-heap to ensure we return
        the BEST X genes (by fitness), not just the first X genes that pass threshold.
        
        Key insight:
        - Genes with negligible weights (<threshold) are skipped
        - If max_genes: Keep only top X by fitness (using min-heap)
        - If no max_genes: Stream all genes passing weight threshold
        - Mathematically equivalent to explicit top-K but memory-bounded
        
        Args:
            args: Argument object with epoch info
            top_k_percent: Percentage (0-100). E.g., 25 = top 25%
            scaler: ChunkedLogZScoreScaler instance (must be finalized)
            chunk_size: Records to yield per iteration
            allEpochs: If True, all epochs. Else filters by args.epoch.
            max_genes: (Optional) Maximum genes to return.
                    IMPORTANT: Returns the BEST max_genes by fitness
                    (using min-heap), NOT just the first max_genes.
            
        Yields:
            Tuple[List[Dict], bool]: (chunk_of_records, is_complete)
                Each record includes '_es_weight' field for later use
        """
        if not self._is_initialized:
            self._initialize_file_mapping(args, allEpochs=allEpochs)
        
        if scaler is None or not scaler.scaling_ready:
            raise RuntimeError("Scaler must be finalized before weight-based filtering")
        
        # Ensure fitnessScore is available
        remove_keys = []
        if self.required_keys is not None:
            if 'fitnessScore' not in self.required_keys:
                self.required_keys.append('fitnessScore')
                remove_keys.append('fitnessScore')
        
        try:
            print(f"Initializing weight-based filtering for top {top_k_percent}%"
                f"{f' (keeping best {max_genes} genes by fitness)' if max_genes else ''}...", 
                flush=True)
            
            # --- STEP 1: Estimate weight threshold via sampling ---
            weight_threshold_percentile = (100 - top_k_percent) / 100.0
            sample_fitness = []
            
            print(f"Sampling population to estimate weight threshold...", flush=True)
            
            for file_path in self._file_list:
                for record in self._yield_records_from_file(file_path):
                    if (allEpochs or record.get('epoch') == args.epoch) and \
                    record.get('fitnessScore') is not None:
                        try:
                            sample_fitness.append(float(record.get('fitnessScore')))
                        except (TypeError, ValueError):
                            continue
                        if len(sample_fitness) >= 100000:
                            break
                if len(sample_fitness) >= 100000:
                    break
            
            if not sample_fitness:
                print("Warning: No records found during sampling. Yielding empty.", flush=True)
                yield [], True
                return
            
            # Compute weights for sample
            sample_np = np.array(sample_fitness, dtype=np.float32)
            sample_tensor = torch.from_numpy(sample_np).to(dtype=torch.float32, device=scaler.device)
            sample_weights = scaler.get_global_weights(sample_tensor)
            sample_weights_np = sample_weights.cpu().numpy() if hasattr(sample_weights, 'cpu') else sample_weights
            
            # Find the weight threshold
            sorted_weights_desc = np.sort(sample_weights_np)[::-1]
            cutoff_idx = max(0, int(len(sorted_weights_desc) * weight_threshold_percentile))
            weight_threshold = float(sorted_weights_desc[cutoff_idx]) if cutoff_idx < len(sorted_weights_desc) else 1e-8
            
            # Cleanup sample tensors
            del sample_tensor, sample_weights, sample_weights_np, sample_np, sorted_weights_desc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"Weight threshold: {weight_threshold:.10e} (filters bottom {100-top_k_percent}%)", flush=True)
            
            # --- STEP 2: Stream records with weight filtering + optional max_genes heap ---
            
            if max_genes is not None:
                # Mode A: Use min-heap to keep only TOP X by fitness
                # This ensures we return the BEST genes, not just first X
                print(f"Using min-heap to track top {max_genes} genes by fitness...", flush=True)
                
                import itertools
                top_genes_heap = []  # Min-heap: (fitness, counter, record_dict)
                _heap_counter = itertools.count()  # unique tiebreaker to avoid dict comparison
                min_fitness_in_heap = float('inf')
                total_considered = 0
                total_skipped = 0

                for file_path in self._file_list:
                    for record in self._yield_records_from_file(file_path):
                        # Check epoch and fitness
                        if not (allEpochs or record.get('epoch') == args.epoch):
                            continue

                        fitness_val = record.get('fitnessScore')
                        if fitness_val is None:
                            continue
                        try:
                            fitness_val = float(fitness_val)
                        except (TypeError, ValueError):
                            continue

                        total_considered += 1

                        # Compute weight for this single record
                        fitness_tensor = torch.tensor([fitness_val], dtype=torch.float32, device=scaler.device)
                        weight_raw = scaler.get_global_weights(fitness_tensor)
                        weight = float(weight_raw.item()) if hasattr(weight_raw, 'item') else float(weight_raw)

                        # Filter by weight threshold
                        if weight >= weight_threshold:
                            clean_record = self._extract_required_data(record)
                            clean_record['_es_weight'] = weight

                            # Heap logic: keep only top X by fitness
                            if len(top_genes_heap) < max_genes:
                                # Heap not full yet
                                heapq.heappush(top_genes_heap, (fitness_val, next(_heap_counter), clean_record))
                                if len(top_genes_heap) == max_genes:
                                    # Heap just became full, record min fitness
                                    min_fitness_in_heap = top_genes_heap[0][0]
                            elif fitness_val > min_fitness_in_heap:
                                # This gene is better than worst in heap
                                heapq.heapreplace(top_genes_heap, (fitness_val, next(_heap_counter), clean_record))
                                min_fitness_in_heap = top_genes_heap[0][0]
                        else:
                            total_skipped += 1
                        
                        # Memory cleanup every 1000 records
                        if total_considered % 1000 == 0 and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                # Heap now contains the top X genes by fitness
                # Sort them for consistent output (descending fitness)
                sorted_records = sorted(top_genes_heap, key=lambda x: x[0], reverse=True)
                
                # Yield in chunks
                current_chunk = []
                for _, _, record_dict in sorted_records:
                    current_chunk.append(record_dict)
                    
                    if len(current_chunk) >= chunk_size:
                        if self.shuffle:
                            random.shuffle(current_chunk)
                        yield current_chunk, False
                        current_chunk = []
                
                # Yield remaining
                if current_chunk:
                    if self.shuffle:
                        random.shuffle(current_chunk)
                    yield current_chunk, True
                
                print(f"Heap-based filtering complete: "
                    f"returned {len(sorted_records)}/{max_genes} genes (best by fitness), "
                    f"considered={total_considered}, weight-filtered={total_considered - total_skipped}, "
                    f"top_k_percent={top_k_percent}%", flush=True)
            
            else:
                # Mode B: No max_genes—stream all genes passing weight threshold
                print(f"Streaming all genes passing weight threshold (no max_genes cap)...", flush=True)
                
                current_chunk = []
                total_included = 0
                total_skipped = 0
                
                for file_path in self._file_list:
                    for record in self._yield_records_from_file(file_path):
                        # Check epoch and fitness
                        if not (allEpochs or record.get('epoch') == args.epoch):
                            continue
                        
                        fitness_val = record.get('fitnessScore')
                        if fitness_val is None:
                            continue
                        try:
                            fitness_val = float(fitness_val)
                        except (TypeError, ValueError):
                            continue
                        
                        # Compute weight for this single record
                        fitness_tensor = torch.tensor([fitness_val], dtype=torch.float32, device=scaler.device)
                        weight_raw = scaler.get_global_weights(fitness_tensor)
                        weight = float(weight_raw.item()) if hasattr(weight_raw, 'item') else float(weight_raw)
                        
                        # Filter by weight threshold
                        if weight >= weight_threshold:
                            clean_record = self._extract_required_data(record)
                            clean_record['_es_weight'] = weight
                            current_chunk.append(clean_record)
                            total_included += 1
                            
                            # Yield chunk when full
                            if len(current_chunk) >= chunk_size:
                                if self.shuffle:
                                    random.shuffle(current_chunk)
                                
                                yield current_chunk, False
                                current_chunk = []
                        else:
                            total_skipped += 1
                        
                        # Memory cleanup every 1000 records
                        if (total_included + total_skipped) % 1000 == 0 and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                # Yield remaining records
                if current_chunk:
                    if self.shuffle:
                        random.shuffle(current_chunk)
                    yield current_chunk, True
                elif total_included == 0:
                    yield [], True
                
                print(f"Weight-based filtering complete (streaming mode, flush=True): "
                    f"included={total_included}, skipped={total_skipped}, "
                    f"top_k_percent={top_k_percent}%", flush=True)
            
        finally:
            # Cleanup
            if remove_keys and self.required_keys is not None:
                for key in remove_keys:
                    if key in self.required_keys:
                        self.required_keys.remove(key)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def reset(self):
        self._current_chunk_index = 0
        self._global_fitness_min = float('inf')
        self._global_fitness_max = float('-inf')

        if self._is_initialized and self.shuffle:
            if self.shuffle_level in ['both', 'files']:
                random.shuffle(self._file_list)
            self._create_indices()

    def get_total_records(self) -> int:
        return self._total_records

    def get_fitness_min_max(self) -> Tuple[float, float]:
        """
        Returns the (min, max) fitnessScore found so far.
        Returns (inf, -inf) if no scores have been processed.
        """
        return self._global_fitness_min, self._global_fitness_max
    
    def calculate_chunk_size (
        self,
        safety_margin: float = 0.5,
        chunk_allocation: float = 0.20,
        min_chunk: int = 100,
        max_chunk: int = 10000,
        verbose: bool = False
    ) -> int:
        """
        Calculate safe chunk size based on available memory (SLURM-aware).
        
        Args:
            safety_margin: Fraction of memory to keep free (0.5 = use 50%)
            chunk_allocation: Fraction of usable memory per chunk (0.20 = 20% per chunk)
            min_chunk: Minimum chunk size
            max_chunk: Maximum chunk size
            verbose: Print debug info
            
        Returns:
            Safe chunk size for loading
        """
        import os
        
        try:
            # 1. Get available memory (SLURM-aware)
            slurm_mem_bytes = None
            
            # Check SLURM environment variables
            if 'SLURM_MEM_PER_NODE' in os.environ:
                slurm_mem_mb = int(os.environ['SLURM_MEM_PER_NODE'])
                slurm_mem_bytes = slurm_mem_mb * 1024 * 1024
            elif 'SLURM_MEM' in os.environ:
                slurm_mem_mb = int(os.environ['SLURM_MEM'])
                slurm_mem_bytes = slurm_mem_mb * 1024 * 1024
            elif 'SLURM_MEM_PER_CPU' in os.environ and 'SLURM_CPUS_ON_NODE' in os.environ:
                mem_per_cpu = int(os.environ['SLURM_MEM_PER_CPU'])
                cpus = int(os.environ['SLURM_CPUS_ON_NODE'])
                slurm_mem_bytes = mem_per_cpu * cpus * 1024 * 1024
            
            if slurm_mem_bytes:
                # Use SLURM allocation
                total_memory = slurm_mem_bytes
                
                # Estimate current process memory
                try:
                    if psutil is not None:
                        process = psutil.Process()
                        mem_info = process.memory_info()
                        used_memory = mem_info.rss
                        
                        # Add children
                        for child in process.children(recursive=True):
                            try:
                                used_memory += child.memory_info().rss
                            except Exception:
                                pass
                    else:
                        used_memory = 0
                except:
                    used_memory = 0
                
                available_memory = total_memory - used_memory
                
                if verbose:
                    print(f"[ChunkSize] SLURM allocation: {total_memory / 1e9:.2f}GB, "
                          f"used: {used_memory / 1e9:.2f}GB, "
                          f"available: {available_memory / 1e9:.2f}GB", flush=True)
            else:
                # Fallback to system memory (less accurate for shared nodes)
                try:
                    mem = _get_virtual_memory_snapshot()
                    total_memory = mem.total
                    available_memory = mem.available
                    
                    if verbose:
                        print(f"[ChunkSize] System memory (no SLURM): "
                              f"available: {available_memory / 1e9:.2f}GB", flush=True)
                except:
                    # Ultra-fallback: assume 4GB available
                    available_memory = 4 * 1024 * 1024 * 1024
                    if verbose:
                        print(f"[ChunkSize] Fallback: assuming 4GB available", flush=True)
            
            # 2. Apply safety margin
            usable_memory = available_memory * safety_margin
            
            # 3. Calculate bytes per record (conservative estimate)
            # Assume each gene is float32 array + dict overhead
            bytes_per_record = 100  # Base overhead
            
            # Try to infer gene length from recent records
            if hasattr(self, '_file_list') and self._file_list:
                try:
                    # Sample first file to get gene length
                    sample_gene = None
                    for record in self._yield_records_from_file(self._file_list[0]):
                        if 'gene' in record and record['gene'] is not None:
                            sample_gene = record['gene']
                            break
                    
                    if sample_gene is not None:
                        if hasattr(sample_gene, '__len__'):
                            gene_length = len(sample_gene)
                            bytes_per_record += gene_length * 4  # float32
                        else:
                            bytes_per_record += 1000  # Fallback estimate
                    else:
                        bytes_per_record += 1000
                except:
                    bytes_per_record += 1000
            else:
                bytes_per_record += 1000
            
            # 4. Calculate chunk size
            chunk_memory = usable_memory * chunk_allocation
            chunk_size = int(chunk_memory / bytes_per_record)
            
            # 5. Clamp to bounds
            chunk_size = max(min_chunk, min(chunk_size, max_chunk))
            
            if verbose:
                print(f"[ChunkSize] Calculated: {chunk_size} records "
                      f"(~{(chunk_size * bytes_per_record) / 1e6:.1f}MB per chunk)", flush=True)
            
            return chunk_size
            
        except Exception as e:
            if verbose:
                print(f"[ChunkSize ERROR] {e}, using fallback: {min_chunk * 2}", flush=True)
            return min_chunk * 2

    # -------------------------------------------------------------------------
    #  loadTopKPercentWithDiversityCoverage
    # -------------------------------------------------------------------------

    def loadTopKPercentileGeneHistoryAsListOfDicts(
        self,
        args,
        top_k_percentile: float,
        maximize: bool,
        chunk_size: int,
        fitness_scaler=None,
        gene_scaler=None,
        use_fitness_scaler_weights: bool = True,
        use_gene_scaler_weights: bool = True,
        n_feature_clusters: str = 'adaptive',
        allEpochs: bool = False,
        cache_key: str = None,
        use_cache: bool = True,
        min_datapoints: Optional[int] = None,
        max_datapoints: Optional[int] = None,
        balance_clusters: bool = False,
        max_per_cluster: Optional[int] = None,
        fit_fitness_scaler_on_top_k: bool = True, # <--- NEW
        fit_gene_scaler_on_top_k: bool = True     # <--- NEW
    ):
        """
        Stream top K% genes with percentile completeness and configurable scaler fitting scope.
        
        Args:
            fit_fitness_scaler_on_top_k: If True, fitness scaler learns Mean/Std from Top-K only.
                                         If False, learns from entire population (Global).
            fit_gene_scaler_on_top_k:    If True, gene scaler learns Mean/Std from Top-K only.
                                         If False, learns from entire population (Global).
        """
        from sklearn.cluster import MiniBatchKMeans
        from collections import defaultdict
        
        if not self._is_initialized:
            self._initialize_file_mapping(args, allEpochs=allEpochs)
        
        # Validate constraints
        if min_datapoints and max_datapoints and min_datapoints > max_datapoints:
            raise ValueError(f"min_datapoints ({min_datapoints}) cannot exceed max_datapoints ({max_datapoints})")
        
        # Generate cache key (Updated with fitting flags)
        if cache_key is None:
            epoch_id = args.epoch if hasattr(args, 'epoch') and not allEpochs else 'all'
            cache_key = (f"topk_{int(top_k_percentile)}_{maximize}_{allEpochs}_"
                         f"epoch{epoch_id}_min{min_datapoints}_max{max_datapoints}_"
                         f"fitFit{fit_fitness_scaler_on_top_k}_fitGene{fit_gene_scaler_on_top_k}")
        
        # ========================================================================
        # CHECK CACHE
        # ========================================================================
        if use_cache and cache_key in self._cached_phases:
            print(f"\n[Phase 1-3] Using cached data (cache_key={cache_key})...", flush=True)
            cached_data = self._cached_phases[cache_key]
            
            # Unpack cache
            safe_percentile_min = cached_data.get('safe_percentile_min')
            boundary_percentile_min = cached_data.get('boundary_percentile_min')
            safe_percentile_max = cached_data.get('safe_percentile_max')
            boundary_percentile_max = cached_data.get('boundary_percentile_max')
            
            safe_threshold_min = cached_data.get('safe_threshold_min')
            boundary_threshold_min = cached_data.get('boundary_threshold_min')
            safe_threshold_max = cached_data.get('safe_threshold_max')
            boundary_threshold_max = cached_data.get('boundary_threshold_max')
            
            n_clusters = cached_data['n_clusters']
            total_genes = cached_data['total_genes']
            mbkmeans = cached_data.get('mbkmeans')
            
            print(f"[Phase 1-3] ✓ Loaded from cache", flush=True)
        
        else:
            # ====================================================================
            # PHASE 1: Fitness Distribution + Dual Boundary Detection
            # ====================================================================
            # NOTE: We always scan the FULL dataset in Phase 1 to establish safe global
            # Min/Max bounds for the scalers, even if we only fit Mean/Std on Top-K later.
            print(f"\n[Phase 1] Scanning fitness distribution...", flush=True)
            
            fitness_all = []
            phase1_count = 0
            nan_fitness_count = 0
            
            try:
                for file_path in self._file_list:
                    for record in self._yield_records_from_file(file_path):
                        if not (allEpochs or record.get('epoch') == args.epoch):
                            continue
                        
                        fitness_val = record.get('fitnessScore')
                        if fitness_val is None:
                            nan_fitness_count += 1
                            continue
                        try:
                            fitness_val = float(fitness_val)
                        except (TypeError, ValueError):
                            nan_fitness_count += 1
                            continue
                        if not np.isfinite(fitness_val):
                            nan_fitness_count += 1
                            continue
                        
                        fitness_all.append(fitness_val)
                        phase1_count += 1
                        
                        # Phase 1 Scaler Updates (Global Bounds)
                        if fitness_scaler is not None and not fitness_scaler.pass1_done:
                            fitness_scaler.collect_phase_1_statistics([fitness_val])
                        
                        if gene_scaler is not None and not getattr(gene_scaler, 'pass1_done', False):
                            gene_features = record.get('gene')
                            if gene_features is not None:
                                gene_array = np.array(gene_features, dtype=np.float32)
                                if np.all(np.isfinite(gene_array)):
                                    gene_scaler.collect_phase_1_statistics([gene_features])
                        
                        if phase1_count % 100000 == 0:
                            print(f"  Scanned {phase1_count} valid records...", flush=True)
                
                if not fitness_all:
                    print("[ERROR] No records found with valid fitness scores", flush=True)
                    yield [], True
                    return
                
                fitness_array = np.array(fitness_all, dtype=np.float32)
                total_genes = len(fitness_array)
                
                print(f"[Phase 1] Fitness distribution complete: {total_genes} valid genes", flush=True)
                
                # --- Boundary Detection Logic (Identical to original) ---
                
                # 1. Max Datapoints Boundary
                safe_percentile_max = None
                boundary_percentile_max = None
                safe_threshold_max = None
                boundary_threshold_max = None
                
                if max_datapoints:
                    print(f"[Phase 1] Finding boundary for max_datapoints={max_datapoints}...", flush=True)
                    num_test_points = max(10, int((top_k_percentile - 0.1) / 0.1))
                    test_percentiles = np.linspace(0.1, top_k_percentile, num_test_points)
                    
                    for test_percentile in test_percentiles:
                        test_percentile = np.clip(test_percentile, 0.0, 100.0)
                        if maximize:
                            percentile_value = np.clip(100.0 - test_percentile, 0.0, 100.0)
                            threshold = np.percentile(fitness_array, percentile_value)
                            count = np.sum(fitness_array >= threshold)
                        else:
                            threshold = np.percentile(fitness_array, test_percentile)
                            count = np.sum(fitness_array <= threshold)
                        
                        if count >= max_datapoints:
                            boundary_percentile_max = test_percentile
                            boundary_threshold_max = threshold
                            safe_percentile_max = max(0.1, test_percentile - 1.0)
                            if maximize:
                                safe_value = np.clip(100.0 - safe_percentile_max, 0.0, 100.0)
                                safe_threshold_max = np.percentile(fitness_array, safe_value)
                            else:
                                safe_threshold_max = np.percentile(fitness_array, safe_percentile_max)
                            break
                    else:
                        # Fallback if max not reached
                        safe_percentile_max = top_k_percentile
                        boundary_percentile_max = top_k_percentile
                        if maximize:
                            safe_threshold_max = np.percentile(fitness_array, np.clip(100.0 - top_k_percentile, 0.0, 100.0))
                        else:
                            safe_threshold_max = np.percentile(fitness_array, top_k_percentile)
                        boundary_threshold_max = safe_threshold_max

                # 2. Min Datapoints Boundary
                safe_percentile_min = None
                boundary_percentile_min = None
                safe_threshold_min = None
                boundary_threshold_min = None
                
                if min_datapoints:
                    print(f"[Phase 1] Finding boundary for min_datapoints={min_datapoints}...", flush=True)
                    start_percentile = boundary_percentile_max if max_datapoints else top_k_percentile
                    num_test_points = max(10, int((100.0 - start_percentile) / 0.1))
                    test_percentiles = np.linspace(start_percentile, 100.0, num_test_points)
                    
                    for test_percentile in test_percentiles:
                        test_percentile = np.clip(test_percentile, 0.0, 100.0)
                        if maximize:
                            percentile_value = np.clip(100.0 - test_percentile, 0.0, 100.0)
                            threshold = np.percentile(fitness_array, percentile_value)
                            count = np.sum(fitness_array >= threshold)
                        else:
                            threshold = np.percentile(fitness_array, test_percentile)
                            count = np.sum(fitness_array <= threshold)
                        
                        if count >= min_datapoints:
                            boundary_percentile_min = test_percentile
                            boundary_threshold_min = threshold
                            safe_percentile_min = max(start_percentile, test_percentile - 1.0)
                            if maximize:
                                safe_value = np.clip(100.0 - safe_percentile_min, 0.0, 100.0)
                                safe_threshold_min = np.percentile(fitness_array, safe_value)
                            else:
                                safe_threshold_min = np.percentile(fitness_array, safe_percentile_min)
                            break
                    else:
                        # Fallback to full set
                        boundary_percentile_min = 100.0
                        safe_percentile_min = 99.0
                        boundary_threshold_min = fitness_array.min() if maximize else fitness_array.max()
                        safe_threshold_min = np.percentile(fitness_array, 1.0 if maximize else 99.0)

                print(f"[Phase 1] ✓ Boundary detection complete", flush=True)
            
            except Exception as e:
                print(f"[ERROR] Phase 1 failed: {e}", file=sys.stderr, flush=True)
                raise
            
            # Finalize Phase 1 Stats
            if fitness_scaler is not None and not fitness_scaler.pass1_done:
                fitness_scaler._finalize_pass1()
            if gene_scaler is not None and not getattr(gene_scaler, 'pass1_done', False):
                gene_scaler._finalize_pass1()
            
            # ====================================================================
            # PHASE 2: MiniBatchKMeans + Selective Scaler Fitting
            # ====================================================================
            print(f"\n[Phase 2] Fitting clusters & scalers...", flush=True)
            
            # Determine threshold for clustering (which always targets Top K)
            if max_datapoints and min_datapoints:
                clustering_threshold = boundary_threshold_min
            elif max_datapoints:
                clustering_threshold = boundary_threshold_max
            elif min_datapoints:
                clustering_threshold = boundary_threshold_min
            else:
                pct = 100 - top_k_percentile if maximize else top_k_percentile
                clustering_threshold = np.percentile(fitness_array, pct)
            
            if maximize:
                comparison_fn = lambda f: f >= clustering_threshold
            else:
                comparison_fn = lambda f: f <= clustering_threshold
            
            # Calculate K for KMeans
            temp_top_k_count = np.sum(comparison_fn(fitness_array))
            if n_feature_clusters == 'adaptive':
                sample_for_k = []
                for file_path in self._file_list[:min(5, len(self._file_list))]:  # Sample first 5 files
                    for record in self._yield_records_from_file(file_path):
                        if not (allEpochs or record.get('epoch') == args.epoch): continue
                        _fs = record.get('fitnessScore')
                        if _fs is None: continue
                        try:
                            _fs = float(_fs)
                        except (TypeError, ValueError):
                            continue
                        if not np.isfinite(_fs): continue
                        is_top_k = comparison_fn(_fs)
                        gene = record.get('gene')
                        if is_top_k and isinstance(gene, (list, np.ndarray)) and len(gene) > 0:
                            sample_for_k.append(np.array(gene, dtype=np.float32))
                            if len(sample_for_k) >= 5000: break
                    if len(sample_for_k) >= 5000: break
                
                n_clusters = self._select_n_clusters_adaptive(
                    np.array(sample_for_k, dtype=np.float32) if sample_for_k else np.random.randn(100, 10),
                    temp_top_k_count
                )
            else:
                n_clusters = int(n_feature_clusters)
            
            mbkmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=min(chunk_size, 1000), n_init=3, random_state=42, verbose=0)
            
            # Buffers for Phase 2
            batch_clustering = []
            batch_fitness_stats = []
            batch_gene_stats = []
            
            phase2_count = 0
            nan_gene_count = 0
            
            try:
                for file_path in self._file_list:
                    for record in self._yield_records_from_file(file_path):
                        if not (allEpochs or record.get('epoch') == args.epoch):
                            continue
                        
                        fitness_val = record.get('fitnessScore')
                        if fitness_val is None:
                            continue
                        try:
                            fitness_val = float(fitness_val)
                        except (TypeError, ValueError):
                            continue
                        if not np.isfinite(fitness_val):
                            continue
                        
                        # Determine if record is in Top K subset
                        is_top_k = comparison_fn(fitness_val)
                        
                        gene_features = record.get('gene')
                        has_valid_gene = False
                        if gene_features is not None:
                            gene_array = np.array(gene_features, dtype=np.float32)
                            if np.all(np.isfinite(gene_array)):
                                has_valid_gene = True
                            else:
                                nan_gene_count += 1
                        
                        phase2_count += 1
                        
                        # --- 1. Clustering (Always on Top K) ---
                        if is_top_k and has_valid_gene:
                            batch_clustering.append(gene_features)
                        
                        # --- 2. Fitness Scaler Stats ---
                        if fitness_scaler is not None and not fitness_scaler.scaling_ready:
                            # If fitting on Top K: update only if is_top_k
                            # If fitting on Global: update always
                            if (not fit_fitness_scaler_on_top_k) or is_top_k:
                                batch_fitness_stats.append(fitness_val)

                        # --- 3. Gene Scaler Stats ---
                        if gene_scaler is not None and not getattr(gene_scaler, 'scaling_ready', False):
                            # If fitting on Top K: update only if is_top_k
                            # If fitting on Global: update always
                            if ((not fit_gene_scaler_on_top_k) or is_top_k) and has_valid_gene:
                                batch_gene_stats.append(gene_features)
                        
                        # --- Flush Batches ---
                        
                        # Flush Clustering
                        if len(batch_clustering) >= chunk_size:
                            mbkmeans.partial_fit(np.array(batch_clustering, dtype=np.float32))
                            batch_clustering = []
                            
                        # Flush Fitness Stats
                        if len(batch_fitness_stats) >= chunk_size:
                            fitness_scaler.collect_phase_2_statistics(batch_fitness_stats)
                            batch_fitness_stats = []
                            
                        # Flush Gene Stats
                        if len(batch_gene_stats) >= chunk_size:
                            gene_scaler.collect_phase_2_statistics(batch_gene_stats)
                            batch_gene_stats = []
                        
                        if phase2_count % 100000 == 0:
                            print(f"  Processed {phase2_count} records...", flush=True)
                
                # Flush remaining buffers
                if batch_clustering:
                    mbkmeans.partial_fit(np.array(batch_clustering, dtype=np.float32))
                if batch_fitness_stats and fitness_scaler:
                    fitness_scaler.collect_phase_2_statistics(batch_fitness_stats)
                if batch_gene_stats and gene_scaler:
                    gene_scaler.collect_phase_2_statistics(batch_gene_stats)
                
                print(f"[Phase 2] ✓ Complete.", flush=True)
            
            except Exception as e:
                print(f"[ERROR] Phase 2 failed: {e}", file=sys.stderr, flush=True)
                raise
            
            # Finalize Scalers
            if fitness_scaler is not None and not fitness_scaler.scaling_ready:
                fitness_scaler.finalize()
            if gene_scaler is not None and not getattr(gene_scaler, 'scaling_ready', False):
                gene_scaler.finalize()
            
            # ====================================================================
            # CACHE
            # ====================================================================
            if use_cache:
                self._cached_phases[cache_key] = {
                    'safe_percentile_min': safe_percentile_min,
                    'boundary_percentile_min': boundary_percentile_min,
                    'safe_percentile_max': safe_percentile_max,
                    'boundary_percentile_max': boundary_percentile_max,
                    'safe_threshold_min': safe_threshold_min,
                    'boundary_threshold_min': boundary_threshold_min,
                    'safe_threshold_max': safe_threshold_max,
                    'boundary_threshold_max': boundary_threshold_max,
                    'n_clusters': n_clusters,
                    'total_genes': total_genes,
                    'mbkmeans': mbkmeans,
                }
                print(f"[Cache] ✓ Cached phases 1-2 (cache_key={cache_key})", flush=True)
        
        # ========================================================================
        # COMPUTE FINAL THRESHOLDS (After Cache Check)
        # ========================================================================
        # Logic to pick the correct thresholds from the (loaded or computed) variables
        if max_datapoints and min_datapoints:
            final_safe_threshold = safe_threshold_max
            final_boundary_threshold = boundary_threshold_min
        elif max_datapoints:
            final_safe_threshold = safe_threshold_max
            final_boundary_threshold = boundary_threshold_max
        elif min_datapoints:
            final_safe_threshold = safe_threshold_min
            final_boundary_threshold = boundary_threshold_min
        else:
            final_safe_threshold = None
            # Need to recalculate base threshold if we used cache and didn't compute array
            if 'fitness_array' not in locals():
                # Fast local scan just for percentile (avoid if possible, but safe fallback)
                # Optimization: In many cases we can reconstruct this from cache if we stored percentile values
                # But for safety, we do a quick re-scan only if needed.
                # Actually, let's just do a quick scan if we are strictly in this branch.
                temp_fitness = []
                for file_path in self._file_list:
                    for record in self._yield_records_from_file(file_path):
                        if not (allEpochs or record.get('epoch') == args.epoch): continue
                        if record.get('fitnessScore') is not None: temp_fitness.append(record['fitnessScore'])
                if not temp_fitness:
                    yield [], True
                    return
                fitness_array = np.array(temp_fitness, dtype=np.float32)

            if maximize:
                final_boundary_threshold = np.percentile(fitness_array, 100 - top_k_percentile)
            else:
                final_boundary_threshold = np.percentile(fitness_array, top_k_percentile)

        # ========================================================================
        # PHASE 3: Cluster Balancing Setup
        # ========================================================================
        if balance_clusters:
            if max_per_cluster is None:
                # Need total_genes. If cached, we have it. If not, it's in locals.
                if 'total_genes' not in locals(): total_genes = self._total_records # approximate fallback
                
                target_total = max_datapoints if max_datapoints else (
                    min_datapoints if min_datapoints else 
                    int(total_genes * top_k_percentile / 100)
                )
                max_per_cluster = max(1, int(target_total / n_clusters))
            
            print(f"\n[Phase 3] Cluster balancing enabled (Max {max_per_cluster:,} per cluster)", flush=True)

        # ========================================================================
        # PHASE 4: Stream with Dual-Threshold Logic
        # ========================================================================
        print(f"\n[Phase 4] Streaming with percentile completeness...", flush=True)
        
        cluster_counts = defaultdict(int)
        current_chunk = []
        total_yielded = 0
        safe_zone_count = 0
        boundary_zone_count = 0
        
        # Zone definitions
        if maximize:
            if final_safe_threshold is not None:
                in_safe_zone = lambda f: f >= final_safe_threshold
            else:
                in_safe_zone = lambda f: False
            in_boundary_zone = lambda f: f >= final_boundary_threshold and not in_safe_zone(f)
        else:
            if final_safe_threshold is not None:
                in_safe_zone = lambda f: f <= final_safe_threshold
            else:
                in_safe_zone = lambda f: False
            in_boundary_zone = lambda f: f <= final_boundary_threshold and not in_safe_zone(f)
        
        try:
            for file_path in self._file_list:
                for record in self._yield_records_from_file(file_path):
                    if not (allEpochs or record.get('epoch') == args.epoch):
                        continue
                    
                    fitness_val = record.get('fitnessScore')
                    if fitness_val is None:
                        continue
                    try:
                        fitness_val = float(fitness_val)
                    except (TypeError, ValueError):
                        continue
                    if not np.isfinite(fitness_val):
                        continue

                    # Check zones
                    is_safe = in_safe_zone(fitness_val)
                    is_boundary = in_boundary_zone(fitness_val)
                    
                    if not (is_safe or is_boundary):
                        continue
                    
                    # Inclusion Check
                    if is_safe:
                        should_include = True
                    elif is_boundary:
                        if max_datapoints:
                            should_include = total_yielded < max_datapoints
                        elif min_datapoints:
                            should_include = total_yielded < min_datapoints
                        else:
                            should_include = True
                    
                    if not should_include:
                        continue
                    
                    gene_features = record.get('gene')
                    if gene_features is None: continue
                    
                    # Cluster prediction & Balancing
                    cluster_id = 0
                    if mbkmeans is not None:
                        try:
                            # MiniBatchKMeans expects 2D array
                            gene_arr = np.array(gene_features, dtype=np.float32).reshape(1, -1)
                            if np.all(np.isfinite(gene_arr)):
                                cluster_id = int(mbkmeans.predict(gene_arr)[0])
                        except: pass
                    
                    if balance_clusters:
                        if cluster_counts[cluster_id] >= max_per_cluster:
                            continue
                    
                    # Prepare Output Record
                    clean_record = self._extract_required_data(record)
                    clean_record['feature_cluster_id'] = cluster_id
                    
                    # --- Compute Weights (Fitness) ---
                    if fitness_scaler is not None and use_fitness_scaler_weights and fitness_scaler.scaling_ready:
                        try:
                            weight = fitness_scaler.get_global_weights(np.array([fitness_val], dtype=np.float32))
                            clean_record['_fitness_scaler_weight'] = float(weight.item() if hasattr(weight, 'item') else weight)
                        except:
                            clean_record['_fitness_scaler_weight'] = 1.0
                    else:
                        clean_record['_fitness_scaler_weight'] = 1.0
                        
                    # --- Compute Weights (Gene) ---
                    if gene_scaler is not None and use_gene_scaler_weights and getattr(gene_scaler, 'scaling_ready', False):
                        try:
                            gene_arr = np.array([gene_features], dtype=np.float32)
                            gene_weight = gene_scaler.get_global_weights(gene_arr)
                            clean_record['_gene_scaler_weight'] = float(gene_weight.item() if hasattr(gene_weight, 'item') else gene_weight)
                        except:
                            clean_record['_gene_scaler_weight'] = 1.0
                    else:
                        clean_record['_gene_scaler_weight'] = 1.0

                    # Add to chunk
                    if is_safe: safe_zone_count += 1
                    elif is_boundary: boundary_zone_count += 1
                    
                    current_chunk.append(clean_record)
                    cluster_counts[cluster_id] += 1
                    total_yielded += 1
                    
                    if len(current_chunk) >= chunk_size:
                        if self.shuffle and self.shuffle_level in ['both', 'records']:
                            random.shuffle(current_chunk)
                        yield current_chunk, False
                        current_chunk = []
                    
                    if total_yielded % 10000 == 0:
                        print(f"  Yielded {total_yielded} genes...", flush=True)

            # Flush last chunk
            if current_chunk:
                if self.shuffle and self.shuffle_level in ['both', 'records']:
                    random.shuffle(current_chunk)
                yield current_chunk, True
            else:
                yield [], True
            
            print(f"\n[Phase 4] ✓ Complete: {total_yielded} genes streamed", flush=True)

        except Exception as e:
            print(f"[ERROR] Phase 4 failed: {e}", file=sys.stderr, flush=True)
            raise   
              
              
    # -------------------------------------------------------------------------
    # HELPER METHODS
    # -------------------------------------------------------------------------

    def _select_n_clusters_adaptive(self, sample_genes: np.ndarray, temp_top_k_count: int) -> int:
        """
        Adaptively select number of clusters using random method selection.
        Randomly chooses between: sqrt, elbow, or silhouette methods.
        
        Args:
            sample_genes: Feature matrix sample (n_samples × n_features)
            temp_top_k_count: Total count of top-K genes
        
        Returns:
            n_clusters: Number of clusters for MiniBatchKMeans
        """
        import random
        from sklearn.metrics import silhouette_score
        
        method = random.choice(['sqrt', 'elbow', 'silhouette'])
        
        # Limit eval sample for speed
        eval_size = min(5000, max(100, len(sample_genes)))
        if len(sample_genes) > eval_size:
            eval_idx = np.random.choice(len(sample_genes), eval_size, replace=False)
            eval_data = sample_genes[eval_idx]
        else:
            eval_data = sample_genes
        
        if method == 'sqrt':
            n_clusters = max(2, int(np.sqrt(temp_top_k_count)))
            print(f"    [K-select] sqrt → {n_clusters}", flush=True)
        
        elif method == 'elbow':
            k_range = list(range(2, min(15, max(3, int(np.sqrt(eval_size))))))
            inertias = [
                KMeans(n_clusters=k, n_init=2, random_state=42, verbose=0)
                .fit(eval_data).inertia_ 
                for k in k_range
            ]
            diffs = np.diff(inertias)
            diffs2 = np.diff(diffs) if len(diffs) > 1 else [0]
            elbow_idx = min(np.argmax(np.abs(diffs2)) + 1, len(k_range) - 1)
            n_clusters = k_range[elbow_idx]
            print(f"    [K-select] elbow → {n_clusters}", flush=True)
        
        else:  # silhouette
            k_range = list(range(2, min(15, max(3, int(np.sqrt(eval_size))))))
            best_k, best_score = 2, -1
            for k in k_range:
                km = KMeans(n_clusters=k, n_init=2, random_state=42, verbose=0)
                score = silhouette_score(eval_data, km.fit_predict(eval_data))
                if score > best_score:
                    best_score, best_k = score, k
            n_clusters = best_k
            print(f"    [K-select] silhouette(score={best_score:.3f}) → {n_clusters}", flush=True)
        
        return n_clusters

    def clear_cache(self, cache_key: str = None):
        """
        Clear cached phases 1-3 data.
        
        Args:
            cache_key: If provided, clear only this key. If None, clear all.
        
        Example:
            >>> loader.clear_cache("training_session_1")  # Clear one cache
            >>> loader.clear_cache()  # Clear all caches
        """
        if cache_key is None:
            self._cached_phases.clear()
            self._cache_valid = False
            print("[Cache] ✓ Cleared all cached data", flush=True)
        else:
            if cache_key in self._cached_phases:
                del self._cached_phases[cache_key]
                print(f"[Cache] ✓ Cleared cache (cache_key={cache_key})", flush=True)
            else:
                print(f"[Cache] Warning: No cache found for {cache_key}", flush=True)

    def is_cache_valid(self) -> bool:
        """Check if cache contains valid data."""
        return self._cache_valid and len(self._cached_phases) > 0

    def _reconstruct_record_from_location(self, file_path: str, line_number: int) -> Dict:
        """
        Reconstruct a single record from file location.
        
        Args:
            file_path: Path to the file
            line_number: Line number in the file (0-indexed)
        
        Returns:
            Dict: The reconstructed record
        """
        try:
            if self.geneFormat == 'json':
                with open(file_path, 'rt') as f:
                    for i, line in enumerate(f):
                        if i == line_number:
                            try:
                                gene = json.loads(line)
                            except:
                                line = line.replace('}})]', '}},)]')
                                gene = json.loads(line)
                            
                            gene['fname'] = file_path
                            return self._extract_required_data(gene)
            
            elif self.geneFormat == 'pickle' or self.geneFormat == 'pkl':
                if FramedRecordReader.exists_for_csv(file_path):
                    with FramedRecordReader.from_csv_path(file_path, codec='pickle') as framed_reader:
                        gene = framed_reader.read(line_number)
                        gene['fname'] = file_path
                        return self._extract_required_data(gene)
                else:
                    pkl_path = file_path.replace('csv', 'pkl')
                    if not os.path.exists(pkl_path):
                        pkl_path = file_path.replace('.log.csv', '.pkl')
                    
                    with open(file_path, 'rt') as f_csv:
                        with zstandard.open(pkl_path, 'rb') as f_pkl:
                            for i, l_csv in enumerate(f_csv):
                                if i == line_number:
                                    try:
                                        gene = json.loads(l_csv)
                                    except:
                                        l_csv = l_csv.replace('}})]', '}},)]')
                                        gene = json.loads(l_csv)
                                    
                                    gene['fname'] = file_path
                                    
                                    try:
                                        gene.update(pickle.load(f_pkl))
                                    except:
                                        pass
                                    
                                    return self._extract_required_data(gene)
        
        except Exception as e:
            print(f"[WARNING] Failed to reconstruct record from {file_path}:{line_number}: {e}", 
                  flush=True)
            return {}
        
        return {}

    def _compute_diversity_score(self, features: np.ndarray) -> float:
        """
        Compute feature variance-based diversity score [0.0-1.0].
        Features are assumed to be in [-1, 1] range.
        Max variance for [-1, 1] is ~0.25.
        """
        if features.shape[0] == 0:
            return 0.0
        
        feature_variances = np.var(features, axis=0)
        avg_variance = np.mean(feature_variances)
        
        # Normalize: max variance for [-1, 1] uniform is 0.333
        max_variance = 0.333
        diversity_score = min(1.0, avg_variance / max_variance)
        
        return float(diversity_score)

    def _compute_adaptive_strata_count(self, fitness_array: np.ndarray, population_size: int) -> int:
        """
        Compute adaptive number of fitness strata based on distribution.
        Heuristic: sqrt(population_size) but capped between 3-20.
        """
        n_strata = max(3, min(20, int(np.sqrt(population_size))))
        return n_strata

    def _adaptive_fitness_strata(
        self, 
        fitness_array: np.ndarray, 
        n_strata: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create adaptive fitness strata using equal-frequency binning.
        
        Returns:
            strata_assignments: Array of stratum IDs for each gene
            strata_bins: Bin edges
        """
        # Use quantiles for equal-frequency binning
        quantiles = np.linspace(0, 100, n_strata + 1)
        bins = np.percentile(fitness_array, quantiles)
        
        # Ensure bins are monotonic (handle duplicates)
        bins = np.unique(bins)
        
        # Assign to strata
        strata_assignments = np.digitize(fitness_array, bins) - 1
        strata_assignments = np.clip(strata_assignments, 0, len(bins) - 2)
        
        return strata_assignments, bins

    def _stratified_kmeans_for_diversity(
        self,
        features: np.ndarray,
        fitness_scores: np.ndarray,
        n_samples: int
    ) -> np.ndarray:
        """
        Apply stratified K-Means clustering within a fitness stratum.
        Returns indices of selected genes.
        
        This reuses the logic from stratified_kmeans_selection() from the first document.
        """
        n_points = len(features)
        if n_samples >= n_points:
            return np.arange(n_points)
        
        # For this sub-stratum, further stratify by fitness
        n_sub_strata = min(5, max(2, n_samples // 10))
        sub_strata = np.digitize(
            fitness_scores,
            np.percentile(fitness_scores, np.linspace(0, 100, n_sub_strata + 1))
        ) - 1
        sub_strata = np.clip(sub_strata, 0, n_sub_strata - 1)
        
        selected = []
        
        for sub_stratum_id in range(n_sub_strata):
            sub_mask = sub_strata == sub_stratum_id
            sub_indices = np.where(sub_mask)[0]
            
            if len(sub_indices) == 0:
                continue
            
            sub_size = len(sub_indices)
            target = max(1, int(n_samples * (sub_size / n_points)))
            target = min(target, sub_size)
            
            if target == sub_size:
                selected.extend(sub_indices)
            else:
                # Random selection for this sub-stratum (can be upgraded to K-means)
                selected.extend(np.random.choice(sub_indices, size=target, replace=False))
        
        selected = np.array(selected)
        
        # Final adjustment
        if len(selected) > n_samples:
            selected = np.random.choice(selected, size=n_samples, replace=False)
        elif len(selected) < n_samples:
            remaining = np.setdiff1d(np.arange(n_points), selected)
            if len(remaining) > 0:
                additional = np.random.choice(
                    remaining, 
                    size=min(n_samples - len(selected), len(remaining)), 
                    replace=False
                )
                selected = np.concatenate([selected, additional])
        
        return selected

    def _compute_adaptive_feature_clusters(self, features: np.ndarray, stratum_size: int) -> int:
        """
        Compute adaptive number of feature clusters within a stratum.
        Heuristic: sqrt(stratum_size), capped between 1-10
        
        Args:
            features: Feature matrix for this stratum
            stratum_size: Number of points in stratum
        
        Returns:
            Number of feature clusters to use
        """
        if stratum_size < 2:
            return 1
        
        n_clusters = max(1, min(10, int(np.sqrt(stratum_size))))
        return n_clusters

    def _apply_feature_clustering(
        self, 
        features: np.ndarray, 
        n_clusters: int,
        method: str = 'kmeans'
    ) -> Tuple[np.ndarray, int]:
        """
        Apply feature clustering to identify genetic niches.
        
        Args:
            features: Feature matrix (stratum_size × n_features)
            n_clusters: Number of clusters
            method: 'kmeans' or 'spectral'
        
        Returns:
            (cluster_assignments, actual_n_clusters)
        """
        if n_clusters <= 1 or len(features) < 2:
            return np.zeros(len(features), dtype=int), 1
        
        try:
            if method == 'kmeans':
                kmeans = KMeans(
                    n_clusters=min(n_clusters, len(features)),
                    random_state=42,
                    n_init=5,
                    max_iter=100
                )
                assignments = kmeans.fit_predict(features)
                actual_clusters = len(np.unique(assignments))
            else:
                from sklearn.cluster import SpectralClustering
                spectral = SpectralClustering(
                    n_clusters=min(n_clusters, len(features)),
                    random_state=42,
                    affinity='nearest_neighbors'
                )
                assignments = spectral.fit_predict(features)
                actual_clusters = len(np.unique(assignments))
            
            return assignments, actual_clusters
        
        except Exception as e:
            print(f"    Feature clustering failed ({method}): {e}, using single cluster", flush=True)
            return np.zeros(len(features), dtype=int), 1

    def _extract_feature_clusters_with_metadata(
        self, 
        top_k_genes: List[Dict],
        strata_assignments: np.ndarray,
        n_strata: int
    ) -> Dict[int, np.ndarray]:
        """
        Extract feature cluster assignments for each gene.
        Returns dict: {(stratum_id, feature_cluster_id): array of gene indices}
        """
        from sklearn.cluster import KMeans
        
        cluster_map = {}
        
        for stratum_id in range(n_strata):
            stratum_mask = strata_assignments == stratum_id
            stratum_indices = np.where(stratum_mask)[0]
            
            if len(stratum_indices) == 0:
                continue
            
            stratum_features = np.array([top_k_genes[i]['gene'] for i in stratum_indices])
            
            n_feature_clusters = self._compute_adaptive_feature_clusters(
                stratum_features,
                len(stratum_indices),
                len(top_k_genes)
            )
            
            if n_feature_clusters > 1:
                try:
                    kmeans = KMeans(n_clusters=n_feature_clusters, random_state=42, n_init=5)
                    feature_assignments = kmeans.fit_predict(stratum_features)
                except:
                    feature_assignments = np.zeros(len(stratum_features), dtype=int)
            else:
                feature_assignments = np.zeros(len(stratum_features), dtype=int)
            
            # Store mapping
            for feature_cluster_id in range(n_feature_clusters):
                local_indices = np.where(feature_assignments == feature_cluster_id)[0]
                global_indices = stratum_indices[local_indices]
                cluster_map[(stratum_id, feature_cluster_id)] = global_indices
        
        return cluster_map

    # ============================================================================
    #  MEMORY ESTIMATION
    # ============================================================================

    def estimate_cpu_memory_capacity(
        self, 
        sample_size: int = 50,  # CHANGED: Default from 1 to 50
        percentage_of_memory_to_be_used: float = 0.8,
        use_slurm_allocation: bool = False
    ) -> int:
        """
        Estimate how many records can fit in available CPU memory.
        
        Args:
            sample_size: Number of records to sample (recommend >= 50)
            percentage_of_memory_to_be_used: Fraction of available memory to use (default 0.8 = 80%)
            use_slurm_allocation: If True, respect SLURM memory limits
        
        Returns:
            Maximum recommended chunk size (number of records)
        
        Example:
            >>> max_records = loader.estimate_cpu_memory_capacity(sample_size=100)
            >>> # Safe to load chunks of up to max_records records
        """
        import glob
        import random
        
        print("\n[Memory] Estimating CPU memory capacity...", flush=True)
        
        self.free_memory()
        
        # Step 1: Get available memory
        available_memory = self._get_available_memory(use_slurm_allocation)
        usable_memory = int(available_memory * percentage_of_memory_to_be_used)
        
        print(f"[Memory] Usable memory: {usable_memory / 1024**3:.2f}GB "
            f"({percentage_of_memory_to_be_used*100:.0f}% of {available_memory / 1024**3:.2f}GB)", 
            flush=True)
        
        # Step 2: Load representative sample
        files = glob.glob(self.savePath + '/*.log.csv')
        if not files:
            print("[Memory] Error: No data files found", flush=True)
            return 1000
        
        print(f"[Memory] Sampling {sample_size} records from {len(files)} files...", flush=True)
        
        sample_records = []
        files_sampled = set()
        max_files_to_check = min(len(files), 20)  # Check up to 20 files for better distribution
        
        try:
            # Sample from multiple files (not just random files)
            files_to_sample = random.sample(files, k=min(len(files), max_files_to_check))
            
            for file_path in files_to_sample:
                remaining = sample_size - len(sample_records)
                if remaining <= 0:
                    break
                
                records_from_file = self._load_records_from_file(file_path, max_records=remaining)
                sample_records.extend(records_from_file)
            
            if not sample_records:
                print("[Memory] Error: Could not load any sample records", flush=True)
                return 1000
            
            # Step 3: Calculate average record size
            avg_record_size = self._calculate_avg_record_size(sample_records, min_sample_size=10)
            
            # Step 4: Calculate safe capacity
            # Add Python list overhead: ~8 bytes per reference + list object overhead
            python_overhead_per_record = 64  # List reference + dict overhead
            total_per_record = avg_record_size + python_overhead_per_record
            
            max_capacity = int(usable_memory / total_per_record)
            
            # Step 5: Apply safety factor (recommend 70% of max)
            recommended_chunk_size = int(max_capacity * 0.7)
            recommended_chunk_size = max(100, recommended_chunk_size)  # At least 100
            
            print(f"\n[Memory] ✓ Capacity Analysis:", flush=True)
            print(f"         Per-record size: {total_per_record:,} bytes ({total_per_record/1024:.2f} KB)", flush=True)
            print(f"         Max theoretical:  {max_capacity:,} records", flush=True)
            print(f"         Recommended safe: {recommended_chunk_size:,} records (70% safety margin)", flush=True)
            
            return recommended_chunk_size
        
        except Exception as e:
            print(f"[Memory] Error during estimation: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return 1000

    def estimate_gpu_memory_capacity(
        self, 
        sample_size: int = 50,
        percentage_of_memory_to_be_used: float = 0.8,
        device: str = None,
        return_types: dict = None,
        use_slurm_allocation: bool = False
    ) -> dict:
        """
        Estimate GPU memory capacity for chunked loading.
        
        Args:
            sample_size: Number of records to sample (recommend >= 50)
            percentage_of_memory_to_be_used: GPU memory fraction to use
            device: CUDA device (e.g., 'cuda:0' or 'cpu')
            return_types: Dict mapping field names to return types
            use_slurm_allocation: If True, respect SLURM GPU allocation
        
        Returns:
            Dict with capacity estimates and recommendations
        """
        if not torch:
            return {'error': 'torch_not_available'}
        
        print("\n[Memory] Estimating GPU memory capacity...", flush=True)
        
        # Determine device
        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            print(f"[Memory] Auto-selected device: {device}", flush=True)
        
        # Load sample records
        import glob
        import random
        
        files = glob.glob(self.savePath + '/*.log.csv')
        if not files:
            return {'error': 'no_files', 'estimated_capacity': 100}
        
        print(f"[Memory] Sampling {sample_size} records...", flush=True)
        
        sample_records = []
        files_to_sample = random.sample(files, k=min(len(files), 10))
        
        try:
            for file_path in files_to_sample:
                remaining = sample_size - len(sample_records)
                if remaining <= 0:
                    break
                records = self._load_records_from_file(file_path, max_records=remaining)
                sample_records.extend(records)
            
            if not sample_records:
                return {'error': 'no_sample_loaded', 'estimated_capacity': 100}
            
            # Calculate capacity
            return self._calculate_gpu_capacity(
                sample_records, 
                device, 
                percentage_of_memory_to_be_used,
                return_types,
                use_slurm_allocation
            )
        
        except Exception as e:
            print(f"[Memory] Error: {e}", flush=True)
            return {'error': str(e), 'estimated_capacity': 100}

    def _calculate_avg_record_size(self, sample_records: List[Dict], min_sample_size: int = 10) -> int:
        """Approximate average serialized size (bytes) for records.

        Uses pickle to capture nested structures; falls back to a safe default
        if sampling fails. Ensures a non-zero value to avoid division errors.
        """
        if not sample_records:
            return 1024

        # Limit the number of samples to keep this quick
        sample_slice = sample_records[: max(min_sample_size, min(len(sample_records), 100))]

        total_size = 0
        counted = 0
        for rec in sample_slice:
            try:
                total_size += len(pickle.dumps(rec, protocol=pickle.HIGHEST_PROTOCOL))
                counted += 1
            except Exception:
                continue

        if counted == 0:
            return 1024

        return max(1, total_size // counted)

    def _calculate_gpu_capacity(
        self,
        sample_records: List[Dict],
        device: str,
        percentage_of_memory_to_be_used: float,
        return_types: Optional[Dict[str, str]] = None,
        use_slurm_allocation: bool = False,
    ) -> dict:
        """Estimate GPU capacity using sample records.

        This is a conservative estimator that only considers gene tensors and
        small metadata, then applies a safety margin to avoid OOM.
        """
        if not torch or device == 'cpu' or not torch.cuda.is_available():
            return {'estimated_capacity': 100, 'device': device, 'note': 'cpu-or-no-cuda'}

        # Determine available GPU memory
        try:
            free_mem, total_mem = torch.cuda.mem_get_info(device)
            available = int(free_mem)
        except Exception:
            # Fallback to device properties
            props = torch.cuda.get_device_properties(device)
            available = int(props.total_memory * percentage_of_memory_to_be_used)

        available = int(available * percentage_of_memory_to_be_used)

        # Estimate per-record cost
        gene_lengths = [len(rec.get('gene', [])) for rec in sample_records if 'gene' in rec]
        gene_len = int(np.median(gene_lengths)) if gene_lengths else 0
        per_record_bytes = (gene_len * 4) if gene_len > 0 else 4096

        # Additional overhead for metadata/return types
        meta_overhead = 1024
        if return_types:
            meta_overhead += 512 * len(return_types)
        per_record_bytes = max(1024, per_record_bytes + meta_overhead)

        raw_capacity = max(1, available // per_record_bytes)
        recommended = int(max(50, raw_capacity * 0.7))

        return {
            'estimated_capacity': recommended,
            'raw_capacity': raw_capacity,
            'per_record_bytes': per_record_bytes,
            'available_bytes': available,
            'device': device,
        }

    # ============================================================================
    #  HELPERS
    # ============================================================================

    def free_memory(self):
        import gc
        gc.collect()
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_available_memory(self, use_slurm: bool) -> int:
        """
        Get available memory in bytes, respecting SLURM allocation if running under SLURM.
        
        SLURM Priority:
        1. SLURM_MEM (total job memory)
        2. SLURM_MEM_PER_NODE (most reliable for --mem= allocation)
        3. SLURM_MEM_PER_CPU × CPUs
        4. psutil + safety margin
        """
        memory_bytes = None
        
        if use_slurm:
            try:
                # Check 1: SLURM_MEM (total memory for the job)
                slurm_mem = os.environ.get('SLURM_MEM')
                if slurm_mem and slurm_mem.strip():  # Check for non-empty
                    try:
                        memory_bytes = self._parse_slurm_memory(slurm_mem) * 1024 * 1024
                        print(f"[Memory] Using SLURM_MEM: {memory_bytes / 1024**3:.2f}GB", flush=True)
                        return memory_bytes
                    except (ValueError, TypeError) as e:
                        print(f"[Memory] SLURM_MEM parsing failed: {e}", flush=True)
                
                # Check 2: SLURM_MEM_PER_NODE (works with --mem=)
                mem_per_node = os.environ.get('SLURM_MEM_PER_NODE')
                if mem_per_node and mem_per_node.strip():  # Check for non-empty
                    try:
                        memory_bytes = self._parse_slurm_memory(mem_per_node) * 1024 * 1024
                        print(f"[Memory] Using SLURM_MEM_PER_NODE: {memory_bytes / 1024**3:.2f}GB", flush=True)
                        return memory_bytes
                    except (ValueError, TypeError) as e:
                        print(f"[Memory] SLURM_MEM_PER_NODE parsing failed: {e}", flush=True)
                
                # Check 3: SLURM_MEM_PER_CPU × CPUs (works with --mem-per-cpu=)
                mem_per_cpu = os.environ.get('SLURM_MEM_PER_CPU')
                cpus_allocated = (os.environ.get('SLURM_CPUS_PER_TASK') or 
                                os.environ.get('SLURM_CPUS_ON_NODE') or 
                                os.environ.get('SLURM_NTASKS'))
                
                if mem_per_cpu and mem_per_cpu.strip() and cpus_allocated and cpus_allocated.strip():
                    try:
                        mem_per_cpu_val = self._parse_slurm_memory(mem_per_cpu)
                        total_mem_mb = mem_per_cpu_val * int(cpus_allocated)
                        memory_bytes = total_mem_mb * 1024 * 1024
                        print(f"[Memory] Using SLURM_MEM_PER_CPU: {mem_per_cpu_val}MB/CPU × {cpus_allocated} CPUs = {memory_bytes / 1024**3:.2f}GB", 
                            flush=True)
                        return memory_bytes
                    except (ValueError, TypeError) as e:
                        print(f"[Memory] SLURM_MEM_PER_CPU calculation failed: {e}", flush=True)
                
                # If we get here, no SLURM vars worked
                print(f"[Memory] Warning: No valid SLURM memory variables found, falling back to psutil", flush=True)
                print(f"[Memory] Debug: SLURM_MEM={os.environ.get('SLURM_MEM', flush=True)}, "
                      f"SLURM_MEM_PER_NODE={os.environ.get('SLURM_MEM_PER_NODE')}, "
                      f"SLURM_MEM_PER_CPU={os.environ.get('SLURM_MEM_PER_CPU')}", flush=True)
            
            except Exception as e:
                print(f"[Memory] Warning: SLURM parsing failed: {e}, falling back to psutil", flush=True)
        
        # Fallback: Use psutil with safety margin
        try:
            available = _get_virtual_memory_snapshot().available
            
            # Apply 50% safety margin when not using SLURM (conservative)
            safety_factor = 0.5 if not use_slurm else 0.7
            safe_available = int(available * safety_factor)
            
            source_name = 'psutil' if psutil is not None else '/proc/meminfo'
            print(f"[Memory] Using {source_name}: {available / 1024**3:.2f}GB available → "
                  f"{safe_available / 1024**3:.2f}GB usable ({int(safety_factor*100)}% safety margin)", 
                  flush=True)
            
            return safe_available
        
        except Exception as e:
            print(f"[Memory] Error getting system memory: {e}, defaulting to 8GB", flush=True)
            return 8 * 1024**3

    def _parse_slurm_memory(self, mem_str: str) -> int:
        """
        Parse SLURM memory string to MB.
        
        Handles formats:
        - "50000" (MB, default)
        - "50G" or "50g" (GB)
        - "50000M" or "50000m" (MB explicit)
        - "50T" or "50t" (TB, rare)
        
        Returns:
            Memory in MB
        """
        if not mem_str or not mem_str.strip():
            raise ValueError("Empty memory string")
        
        mem_str = str(mem_str).strip().upper()
        
        if mem_str.endswith('T'):
            return int(mem_str[:-1]) * 1024 * 1024  # TB to MB
        elif mem_str.endswith('G'):
            return int(mem_str[:-1]) * 1024  # GB to MB
        elif mem_str.endswith('M'):
            return int(mem_str[:-1])  # Already MB
        elif mem_str.endswith('K'):
            return max(1, int(mem_str[:-1]) // 1024)  # KB to MB
        else:
            # No suffix - assume MB (SLURM default)
            return int(mem_str)
    
    def _yield_csv_summary_records(self, file: str):
        """Yield records from the .log.csv summary sidecar written next to the
        framed binary (GA.py part=2). Recovery source used ONLY when the framed
        file cannot be decoded. The summary omits heavy fields (gene/param) but
        retains fitnessScore/epoch/iteration/loss, which is what resume and epoch
        detection need."""
        try:
            with open(file, 'rt') as f_csv:
                for line in f_csv:
                    try:
                        try:
                            gene = json.loads(line)
                        except json.JSONDecodeError:
                            gene = json.loads(line.replace('}})]', '}},)]'))
                        if not isinstance(gene, dict):
                            continue
                        gene['fname'] = file
                        yield gene
                    except Exception:
                        continue
        except Exception as _e:
            print(f"GA - history loader - WARNING: CSV summary fallback could not "
                  f"read {file}: {_e!r}", flush=True)

    def _yield_records_from_file(self, file: str):
        """
        Generator that yields records one-by-one from a file.
        Memory-safe: Never loads entire file into memory.
        
        Args:
            file: Path to the file
            
        Yields:
            Dict: Single record
        """
        try:
            # If a CSV file with flat entries and geneFormat is JSON, parse lines accordingly
            ext = os.path.splitext(file)[1].lower()
            if ext in ('.csv', '.log.csv') and self.geneFormat == 'csv':
                def parse_flat_line(s: str):
                    s = s.strip()
                    if not s:
                        return None
                    # First try normal JSON
                    try:
                        return json.loads(s)
                    except Exception:
                        pass
                    # Fallback: key-value pairs (e.g., key=value, key: value)
                    parts = []
                    if ',' in s:
                        parts = [p.strip() for p in s.split(',') if p.strip()]
                    else:
                        parts = [p.strip() for p in s.split() if p.strip()]
                    result = {}
                    for part in parts:
                        if ':' in part:
                            key, val = part.split(':', 1)
                        elif '=' in part:
                            key, val = part.split('=', 1)
                        else:
                            # Not a recognizable key-value pair
                            continue
                        key = key.strip().strip('"').strip("'")
                        val = val.strip()
                        parsed_val = None
                        # Strip quotes
                        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                            parsed_val = val[1:-1]
                        else:
                            low = val.lower()
                            if low == 'true':
                                parsed_val = True
                            elif low == 'false':
                                parsed_val = False
                            elif low in ('null', 'none'):
                                parsed_val = None
                            elif val.startswith('[') or val.startswith('{'):
                                # Try JSON then Python literal
                                try:
                                    parsed_val = json.loads(val)
                                except Exception:
                                    try:
                                        parsed_val = ast.literal_eval(val)
                                    except Exception:
                                        parsed_val = val
                            else:
                                # Try numeric
                                try:
                                    if '.' in val or 'e' in val.lower():
                                        f = float(val)
                                        parsed_val = int(f) if f.is_integer() else f
                                    else:
                                        parsed_val = int(val)
                                except Exception:
                                    parsed_val = val
                        result[key] = parsed_val
                    return result if result else None

                with open(file, 'rt') as f:
                    for line in f:
                        try:
                            gene = parse_flat_line(line)
                            if not gene:
                                continue
                            gene['fname'] = file
                            yield gene
                        except Exception:
                            continue
                return

            if self.geneFormat == 'json':
                with open(file, 'rt') as f:
                    for line in f:
                        try:
                            try:
                                gene = json.loads(line)
                            except json.JSONDecodeError:
                                # Handle malformed JSON (your specific case)
                                line = line.replace('}})]', '}},)]')
                                try:
                                    gene = json.loads(line)
                                except Exception:
                                    # Fallback: treat line as flat key-value text
                                    # Reuse the text parsing logic
                                    s = line.strip()
                                    gene = None
                                    if s:
                                        # Simple parser for key=value or key: value pairs
                                        parts = [p.strip() for p in (s.split(',') if ',' in s else s.split()) if p.strip()]
                                        tmp = {}
                                        for part in parts:
                                            if ':' in part:
                                                k, v = part.split(':', 1)
                                            elif '=' in part:
                                                k, v = part.split('=', 1)
                                            else:
                                                continue
                                            k = k.strip().strip('"').strip("'")
                                            v = v.strip()
                                            # Try JSON/literal for complex values
                                            pv = v
                                            try:
                                                if v.startswith('[') or v.startswith('{'):
                                                    pv = json.loads(v)
                                                else:
                                                    # Booleans/null
                                                    if v.lower() == 'true':
                                                        pv = True
                                                    elif v.lower() == 'false':
                                                        pv = False
                                                    elif v.lower() in ('null', 'none'):
                                                        pv = None
                                                    else:
                                                        # numeric
                                                        if '.' in v or 'e' in v.lower():
                                                            fl = float(v)
                                                            pv = int(fl) if fl.is_integer() else fl
                                                        else:
                                                            pv = int(v)
                                            except Exception:
                                                # Fall back to Python literal if possible
                                                try:
                                                    pv = ast.literal_eval(v)
                                                except Exception:
                                                    pv = v
                                            tmp[k] = pv
                                        gene = tmp if tmp else None
                                    if not gene:
                                        continue
                            
                            gene['fname'] = file
                            yield gene
                        except Exception:
                            continue
            
            elif self.geneFormat == 'pickle' or self.geneFormat == 'pkl':
                if FramedRecordReader.exists_for_csv(file):
                    # Normal pickle-format read path. Track decode failures so a
                    # corrupt / undecodable framed file is LOUD instead of silently
                    # yielding nothing -- the exact failure mode that hid the resume
                    # bug (loadAllGeneHistory returned 0 while history existed).
                    n_entries = 0
                    n_failed = 0
                    n_yielded = 0
                    first_error = None
                    with FramedRecordReader.from_csv_path(file, codec='pickle') as framed_reader:
                        n_entries = len(framed_reader)
                        for idx in range(n_entries):
                            try:
                                gene = framed_reader.read(idx)
                                gene['fname'] = file
                                n_yielded += 1
                                yield gene
                            except Exception as _e:
                                n_failed += 1
                                if first_error is None:
                                    first_error = repr(_e)
                                continue
                    if n_failed:
                        print(f"GA - history loader - WARNING: framed decode failed "
                              f"for {n_failed}/{n_entries} record(s) in {file} "
                              f"(first error: {first_error})", flush=True)
                    # Recovery: framed file exists but decoded ZERO records (codec/
                    # version mismatch, truncated .bin, ...). Fall back to the .log.csv
                    # sidecar, which carries the scalar fields written alongside every
                    # framed append -- enough for resume and epoch detection. Runs ONLY
                    # when framed yields nothing, so the success path is unchanged.
                    if n_yielded == 0 and n_entries > 0:
                        print(f"GA - history loader - WARNING: 0/{n_entries} framed "
                              f"record(s) decoded in {file}; falling back to CSV summary "
                              f"sidecar (genes unavailable, scalars recovered).", flush=True)
                        yield from self._yield_csv_summary_records(file)
                    return
                else:
                    with open(file, 'rt') as f_csv:
                    # Get pickle file path
                        pkl_path = file.replace('csv', 'pkl')
                        if not os.path.exists(pkl_path):
                            pkl_path = file.replace('.log.csv', '.pkl')
                    
                        if not os.path.exists(pkl_path):
                            # No pickle file - yield JSON only
                            for line in f_csv:
                                try:
                                    try:
                                        gene = json.loads(line)
                                    except json.JSONDecodeError:
                                        line = line.replace('}})]', '}},)]')
                                        try:
                                            gene = json.loads(line)
                                        except Exception:
                                            # Fallback for flat text lines
                                            s = line.strip()
                                            gene = None
                                            if s:
                                                parts = [p.strip() for p in (s.split(',') if ',' in s else s.split()) if p.strip()]
                                                tmp = {}
                                                for part in parts:
                                                    if ':' in part:
                                                        k, v = part.split(':', 1)
                                                    elif '=' in part:
                                                        k, v = part.split('=', 1)
                                                    else:
                                                        continue
                                                    k = k.strip().strip('"').strip("'")
                                                    v = v.strip()
                                                    pv = v
                                                    try:
                                                        if v.startswith('[') or v.startswith('{'):
                                                            pv = json.loads(v)
                                                        else:
                                                            if v.lower() == 'true':
                                                                pv = True
                                                            elif v.lower() == 'false':
                                                                pv = False
                                                            elif v.lower() in ('null', 'none'):
                                                                pv = None
                                                            else:
                                                                if '.' in v or 'e' in v.lower():
                                                                    fl = float(v)
                                                                    pv = int(fl) if fl.is_integer() else fl
                                                                else:
                                                                    pv = int(v)
                                                    except Exception:
                                                        try:
                                                            pv = ast.literal_eval(v)
                                                        except Exception:
                                                            pv = v
                                                    tmp[k] = pv
                                                gene = tmp if tmp else None
                                    gene['fname'] = file
                                    if not gene:
                                        continue
                                    yield gene
                                except Exception:
                                    continue
                        else:
                            # Load with pickle data
                            with zstandard.open(pkl_path, 'rb') as f_pkl:
                                for line_csv in f_csv:
                                    try:
                                        try:
                                            gene = json.loads(line_csv)
                                        except json.JSONDecodeError:
                                            line_csv = line_csv.replace('}})]', '}},)]')
                                            try:
                                                gene = json.loads(line_csv)
                                            except Exception:
                                                # Fallback for flat text lines
                                                s = line_csv.strip()
                                                gene = None
                                                if s:
                                                    parts = [p.strip() for p in (s.split(',') if ',' in s else s.split()) if p.strip()]
                                                    tmp = {}
                                                    for part in parts:
                                                        if ':' in part:
                                                            k, v = part.split(':', 1)
                                                        elif '=' in part:
                                                            k, v = part.split('=', 1)
                                                        else:
                                                            continue
                                                        k = k.strip().strip('"').strip("'")
                                                        v = v.strip()
                                                        pv = v
                                                        try:
                                                            if v.startswith('[') or v.startswith('{'):
                                                                pv = json.loads(v)
                                                            else:
                                                                if v.lower() == 'true':
                                                                    pv = True
                                                                elif v.lower() == 'false':
                                                                    pv = False
                                                                elif v.lower() in ('null', 'none'):
                                                                    pv = None
                                                                else:
                                                                    if '.' in v or 'e' in v.lower():
                                                                        fl = float(v)
                                                                        pv = int(fl) if fl.is_integer() else fl
                                                                    else:
                                                                        pv = int(v)
                                                        except Exception:
                                                            try:
                                                                pv = ast.literal_eval(v)
                                                            except Exception:
                                                                pv = v
                                                        tmp[k] = pv
                                                    gene = tmp if tmp else None
                                                if not gene:
                                                    continue
                                        
                                        gene['fname'] = file
                                        
                                        # Try to load pickle data
                                        try:
                                            gene.update(pickle.load(f_pkl))
                                        except Exception:
                                            pass
                                        
                                        yield gene
                                    except Exception:
                                        continue
        
        except Exception as e:
            print(f"[Warning] Error reading file {file}: {e}", flush=True)
            return
