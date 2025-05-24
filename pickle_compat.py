import pickle
import sys
import logging
import importlib.util
import types

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Handle numpy._core module not found error
        if module == 'numpy._core':
            # Redirect to numpy
            module = 'numpy'
        
        # Handle other potential module renames or missing modules
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError) as e:
            logging.warning(f"Error finding {module}.{name}: {str(e)}")
            
            # Try to find the class in numpy
            if module.startswith('numpy.'):
                try:
                    return super().find_class('numpy', name)
                except:
                    pass
            
            # Create a dummy class/module as a fallback
            logging.warning(f"Creating dummy for {module}.{name}")
            
            # For array-like objects, return numpy array
            if name in ['ndarray', '_reconstruct', 'dtype']:
                import numpy as np
                if name == 'ndarray' or name == '_reconstruct':
                    return np.ndarray
                elif name == 'dtype':
                    return np.dtype
            
            # For other cases, create a dummy object
            class DummyClass:
                def __init__(self, *args, **kwargs):
                    pass
                
                def __getattr__(self, attr):
                    return None
            
            return DummyClass

def safe_load_pickle(file_path):
    """Load a pickle file safely with custom unpickler to handle version mismatches."""
    try:
        with open(file_path, 'rb') as f:
            unpickler = CustomUnpickler(f)
            return unpickler.load()
    except Exception as e:
        logging.error(f"Error loading pickle with custom unpickler: {str(e)}")
        return None
