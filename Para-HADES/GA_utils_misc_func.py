try:
    import matplotlib.pyplot as plt
except ImportError:
    pass
import numpy as np
import torch
import math
import pickle
import sys
from datetime import datetime

try:
    from sklearn.decomposition import PCA
except ImportError:
    pass
from sklearn.preprocessing import StandardScaler 


def pickle_loads_compat(payload):
    try:
        return pickle.loads(payload)
    except ModuleNotFoundError as exc:
        if exc.name == 'numpy._core':
            import numpy.core as numpy_core
            sys.modules.setdefault('numpy._core', numpy_core)
            return pickle.loads(payload)
        if exc.name == 'numpy.core':
            import numpy._core as numpy_core_private
            sys.modules.setdefault('numpy.core', numpy_core_private)
            return pickle.loads(payload)
        raise

def check_batchSize(x, weights, batch_size=256, device=None):
    if x.shape[0] - ((x.shape[0] // batch_size) * batch_size) == 1:
        # if the last batch is not full, we need to duplicate the random element
        idx = np.random.randint(0, x.shape[0])
        x = torch.cat((x, x[idx].unsqueeze(0)), dim=0)
        weights = torch.cat((weights, weights[idx].unsqueeze(0)), dim=0)

    return x.to(device), weights.to(device)

def is_valid_fitness(value):
    # None or empty string check
    if value is None or value == '':
        return False
    
    # String variants of None
    if isinstance(value, str) and value.lower() == 'none':
        return False
    
    # NaN/Inf check for float
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return False
    
    # NaN/Inf check for numpy types
    if hasattr(value, 'dtype') and (np.isnan(value).any() if hasattr(np, 'isnan') else False or 
                                    np.isinf(value).any() if hasattr(np, 'isinf') else False):
        return False
    
    return True


def fast_converter(obj):
    """Helper to convert Numpy/Torch types to JSON-friendly formats."""
    # Handle Numpy
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    
    # Handle PyTorch
    elif torch is not None and isinstance(obj, torch.Tensor):
        # Move to CPU, detach from graph, convert to list
        return obj.detach().cpu().tolist()
        
    raise TypeError(f"Type {type(obj)} is not serializable")

import json
class JSONEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy and non-serializable types."""
    def default(self, obj):
        # Handle numpy types
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, bool):
            return bool(obj)
        # Let the base class handle it
        return super().default(obj)


import yaml
class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True

#------------------------------- 

"""
Example demonstrating the function with different output types

# Create sample data (simulate your large dataset)
sample_data = [
    {'feature1': 1.5, 'feature2': 10, 'feature3': 'A', 'unwanted': 'trash1'},
    {'feature1': 2.3, 'feature2': 20, 'feature3': 'B', 'unwanted': 'trash2'},
    {'feature1': 3.1, 'feature2': 30, 'feature3': 'C', 'unwanted': 'trash3'},
    {'feature1': 4.7, 'feature2': 40, 'feature3': 'D', 'unwanted': 'trash4'},
]

print("=== Example 1: Mixed output types ===")
# Extract specific keys with different output types
keys_to_extract = ['feature1', 'feature2', 'feature3']
output_types = ['torch', 'numpy', 'list']

result = convert_dicts_to_objects(
    sample_data.copy(),  # Use copy so we can reuse data
    keys_to_extract, 
    output_types,
    chunk_size=2  # Small chunk for demo
)

print(f"Result types: {[type(x).__name__ for x in result]}")
print(f"Feature1 (torch): {result[0]}")
print(f"Feature2 (numpy): {result[1]}")
print(f"Feature3 (list): {result[2]}")

print("\n=== Example 2: All PyTorch tensors ===")
# Convert everything to PyTorch tensors
sample_data2 = [
    {'x': 1.0, 'y': 2.0},
    {'x': 3.0, 'y': 4.0},
    {'x': 5.0, 'y': 6.0}
]

result2 = convert_dicts_to_objects(
    sample_data2,
    ['x', 'y'],
    ['torch', 'torch'],
    torch_dtype='float32'
)

print(f"X tensor: {result2[0]}")
print(f"Y tensor: {result2[1]}")

print("\n=== Example 3: Custom conversion function ===")
# Using custom conversion
sample_data3 = [{'values': i} for i in range(5)]

def custom_converter(data_list):
    #Custom function that squares all values
    return [x**2 for x in data_list]

result3 = convert_dicts_to_objects(
    sample_data3,
    ['values'],
    [custom_converter]
)

print(f"Custom result (squared): {result3[0]}")
"""


import gc
import numpy as np
from typing import List, Dict, Any, Union, Callable, Optional

def convert_dicts_to_objects(
    data_list: List[Dict[str, Any]], 
    keys_to_extract: List[str],
    output_types: List[Union[str, Callable]] = None,
    chunk_size: int = 100000
) -> List[Any]:
    """
    Memory-efficient conversion of large list of dictionaries to transposed list of objects.
    
    Handles 90% memory usage datasets without OOM by processing in chunks and immediate cleanup.
    
    Args:
        data_list: Large list of dictionaries (modified in-place, will be cleared)
        keys_to_extract: List of keys to extract from each dictionary
        output_types: List of output types for each key. Options:
                     - torch.float32: PyTorch tensor with float32 dtype
                     - torch.float64: PyTorch tensor with float64 dtype
                     - torch.int32: PyTorch tensor with int32 dtype
                     - torch.int64: PyTorch tensor with int64 dtype
                     - np.float32: NumPy array with float32 dtype
                     - np.float64: NumPy array with float64 dtype
                     - np.int32: NumPy array with int32 dtype
                     - np.int64: NumPy array with int64 dtype
                     - list: Python list (default)
                     - Custom callable: Any function that takes a list and returns desired object
        chunk_size: Number of items to process at once (tune based on available memory)
    
    Returns:
        List of objects in transposed format: [obj_for_key1, obj_for_key2, ...]
        
    Example:
        data = [{'a': 1, 'b': 2.5}, {'a': 3, 'b': 4.1}]
        result = convert_dicts_to_objects(data, ['a', 'b'], [torch.float32, np.float64])
        # Returns: [torch.tensor([1., 3.], dtype=torch.float32), np.array([2.5, 4.1], dtype=float64)]
    """
    
    if not data_list:
        return []
    
    # Set default output types
    if output_types is None:
        output_types = ["list"] * len(keys_to_extract)
    
    if len(output_types) != len(keys_to_extract):
        raise ValueError("output_types length must match keys_to_extract length")
    
    num_items = len(data_list)
    num_keys = len(keys_to_extract)
    
    # Pre-allocate result columns as lists (most memory efficient intermediate format)
    result_columns = [[] for _ in range(num_keys)]
    
    print(f"Starting conversion of {num_items} items with {num_keys} keys...")
    print(f"Processing in chunks of {chunk_size} items")
    
    # Process data in chunks to control memory usage
    for start_idx in range(0, num_items, chunk_size):
        end_idx = min(start_idx + chunk_size, num_items)
        
        # Extract values from current chunk
        for i in range(start_idx, end_idx):
            item = data_list[i]
            
            if isinstance(item, dict):
                # First pass: check if all keys exist and are not None
                if all(item.get(key) is not None for key in keys_to_extract):
                    # Second pass: extract values (we know they're all valid)
                    for key_idx, key in enumerate(keys_to_extract):
                        value = item.pop(key)
                        result_columns[key_idx].append(value)
                
                # Clear the entire dict to free memory immediately
                item.clear()
            
            # Set to None to indicate processed (helps with garbage collection)
            data_list[i] = None
        
        # Force garbage collection after each chunk
        gc.collect()
        
        if end_idx % (chunk_size * 5) == 0 or end_idx == num_items:
            print(f"Extracted data: {end_idx}/{num_items} items ({end_idx/num_items*100:.1f}%)")
    
    # Clear the original data list completely
    data_list.clear()
    gc.collect()
    print("Data extraction complete. Converting to requested object types...")
    
    # Convert each column to requested output type
    final_result = []
    
    for key_idx, (column_data, output_type) in enumerate(zip(result_columns, output_types)):
        print(f"Converting column {key_idx+1}/{num_keys} ({keys_to_extract[key_idx]}) to {output_type}...")
        
        converted_obj = _convert_to_type_safe(column_data, output_type, chunk_size)
        final_result.append(converted_obj)
        
        # Clear the intermediate list to free memory
        column_data.clear()
        gc.collect()
    
    # Clear all intermediate data
    result_columns.clear()
    gc.collect()
    
    print("Conversion complete!")
    return final_result


def _convert_to_type_safe(data_list: List[Any], output_type: Union[str, Callable], 
                         chunk_size: int) -> Any:
    """
    Memory-safe conversion of large list to specified object type.
    """
    
    # Handle direct type objects (torch.float32, np.float64, etc.)
    if hasattr(output_type, '__module__'):
        # Check if it's a torch dtype
        if hasattr(output_type, '__module__') and 'torch' in str(output_type.__module__):
            return _convert_to_torch_safe(data_list, output_type, chunk_size)
        # Check if it's a numpy dtype
        elif hasattr(output_type, '__module__') and 'numpy' in str(output_type.__module__):
            return _convert_to_numpy_safe(data_list, output_type, chunk_size)
    
    # Handle string specifications
    if isinstance(output_type, str):
        output_type = output_type.lower()
    
    # Convert based on type
    if output_type == "list" or output_type is list:
        return data_list  # No conversion needed
        
    elif callable(output_type):
        # Custom conversion function
        try:
            result = output_type(data_list)
            data_list.clear()
            return result
        except MemoryError:
            raise MemoryError(f"Custom conversion function caused OOM. Consider chunked processing.")
    
    else:
        raise ValueError(f"Unsupported output_type: {output_type}")


def _convert_to_torch_safe(data_list: List[Any], torch_dtype, chunk_size: int):
    """Memory-safe PyTorch tensor conversion"""
    try:
        import torch
        
        # Try numpy intermediate conversion (often more memory efficient)
        try:
            # Convert to numpy first (usually more memory efficient than direct torch conversion)
            np_dtype = _torch_to_numpy_dtype(torch_dtype)
            np_array = np.array(data_list, dtype=np_dtype)
            data_list.clear()
            gc.collect()
            
            # Convert numpy to torch (zero-copy when possible)
            tensor = torch.from_numpy(np_array).to(torch_dtype)
            return tensor
            
        except MemoryError:
            # Fallback: chunked tensor conversion
            return _convert_to_torch_chunked(data_list, torch_dtype, chunk_size)
            
    except ImportError:
        raise ImportError("PyTorch not available. Install with: pip install torch")


def _convert_to_numpy_safe(data_list: List[Any], np_dtype, chunk_size: int):
    """Memory-safe numpy array conversion"""
    try:
        # Try direct conversion first
        result = np.array(data_list, dtype=np_dtype)
        data_list.clear()
        return result
    except MemoryError:
        # If direct conversion fails, process in chunks
        return _convert_to_numpy_chunked(data_list, np_dtype, chunk_size)


def _convert_to_torch_chunked(data_list: List[Any], torch_dtype, chunk_size: int):
    """Chunked PyTorch tensor conversion for very large data"""
    import torch
    
    tensor_parts = []
    total_items = len(data_list)
    
    for i in range(0, total_items, chunk_size):
        chunk = data_list[i:i+chunk_size]
        tensor_chunk = torch.tensor(chunk, dtype=torch_dtype)
        tensor_parts.append(tensor_chunk)
        
        # Clear processed chunk
        del data_list[i:i+chunk_size]
        gc.collect()
        
        print(f"  Converted chunk {i//chunk_size + 1}/{(total_items-1)//chunk_size + 1}")
    
    # Concatenate all chunks
    final_tensor = torch.cat(tensor_parts)
    del tensor_parts
    gc.collect()
    
    return final_tensor


def _convert_to_numpy_chunked(data_list: List[Any], np_dtype, chunk_size: int):
    """Chunked numpy array conversion"""
    chunks = []
    total_items = len(data_list)
    
    for i in range(0, total_items, chunk_size):
        chunk = data_list[i:i+chunk_size]
        np_chunk = np.array(chunk, dtype=np_dtype)
        chunks.append(np_chunk)
        
        # Clear processed chunk
        del data_list[i:i+chunk_size]
        gc.collect()
    
    # Concatenate all chunks
    final_array = np.concatenate(chunks)
    del chunks
    gc.collect()
    
    return final_array


def _torch_to_numpy_dtype(torch_dtype):
    """Map torch dtype to compatible numpy dtype"""
    import torch
    
    mapping = {
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.int32: np.int32,
        torch.int64: np.int64,
        torch.bool: np.bool_,
        torch.uint8: np.uint8,
        torch.int8: np.int8,
        torch.int16: np.int16,
    }
    return mapping.get(torch_dtype, np.float32)


def extract_chunk_to_tensors(
    chunk: list,
    gene_length: int,
    device: torch.device,
    extract_weights: bool = False
) -> tuple:
    """
    Extract genes, fitness, and weights from chunk into tensors (inline).
    Deletes original data to free memory after extraction.
    
    Args:
        chunk: List of dictionaries containing 'gene' and 'fitnessScore'
        gene_length: Length of gene vectors
        device: PyTorch device (cpu or cuda)
        extract_weights: Whether to extract _fitness_scaler_weight
        
    Returns:
        Tuple: (genes_tensor, fitness_tensor, weights_tensor, valid_count)
    """
    chunk_size = len(chunk)
    if chunk_size == 0:
        return None, None, None, 0
    
    with torch.no_grad():
        genes_tensor = torch.empty((chunk_size, gene_length), dtype=torch.float32, device=device)
        fitness_tensor = torch.empty(chunk_size, dtype=torch.float32, device=device)
        weights_tensor = torch.empty(chunk_size, dtype=torch.float32, device=device) if extract_weights else None
        
        idx = 0
        for gene_dict in chunk:
            gene = gene_dict.get('gene')
            fitness = gene_dict.get('fitnessScore')
            
            if gene is None or fitness is None or (isinstance(fitness, float) and np.isnan(fitness)):
                continue
            
            if isinstance(gene, torch.Tensor):
                genes_tensor[idx] = gene.to(device)
            elif isinstance(gene, np.ndarray):
                genes_tensor[idx] = torch.from_numpy(gene).to(device)
            else:
                genes_tensor[idx] = torch.tensor(gene, dtype=torch.float32, device=device)
            
            fitness_tensor[idx] = float(fitness)
            
            if extract_weights:
                weight = gene_dict.get('_fitness_scaler_weight', 1.0)
                weights_tensor[idx] = float(weight) if weight is not None and not np.isnan(weight) else 1.0
            
            # Delete original data to free memory
            gene_dict['gene'] = None
            gene_dict['fitnessScore'] = None
            if '_fitness_scaler_weight' in gene_dict:
                gene_dict['_fitness_scaler_weight'] = None
            
            idx += 1
        
        if idx == 0:
            return None, None, None, 0
        elif idx < chunk_size:
            genes_tensor = genes_tensor[:idx].contiguous()
            fitness_tensor = fitness_tensor[:idx].contiguous()
            if extract_weights:
                weights_tensor = weights_tensor[:idx].contiguous()
        
        return genes_tensor, fitness_tensor, weights_tensor, idx

#------------------------------