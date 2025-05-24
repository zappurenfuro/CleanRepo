import pickle
import numpy as np
import logging
import os
import sys

class CustomUnpickler(pickle.Unpickler):
    """Custom unpickler to handle numpy version mismatches."""
    
    def find_class(self, module, name):
        """Handle numpy version differences by redirecting old module paths."""
        # Handle numpy._core module not found
        if module == 'numpy._core':
            module = 'numpy.core'
        
        # Handle other potential numpy module changes
        if module.startswith('numpy.') and not module.startswith('numpy.core'):
            try:
                return getattr(np, name)
            except AttributeError:
                pass
        
        # Handle pandas module changes
        if module.startswith('pandas.'):
            import pandas as pd
            try:
                return getattr(pd, name)
            except AttributeError:
                pass
        
        # Default behavior
        return super().find_class(module, name)

def safe_load_pickle(file_path):
    """Safely load a pickle file with version compatibility handling."""
    try:
        if not os.path.exists(file_path):
            logging.error(f"Pickle file does not exist: {file_path}")
            return None
        
        # Check if file is compressed (gzip)
        is_gzipped = False
        try:
            with open(file_path, 'rb') as f:
                magic_number = f.read(2)
                if magic_number == b'\x1f\x8b':  # gzip magic number
                    is_gzipped = True
        except:
            pass
        
        # Load the file with custom unpickler
        if is_gzipped:
            import gzip
            with gzip.open(file_path, 'rb') as f:
                unpickler = CustomUnpickler(f)
                loaded_data = unpickler.load()
        else:
            with open(file_path, 'rb') as f:
                unpickler = CustomUnpickler(f)
                loaded_data = unpickler.load()
        
        # Check if the loaded data has metadata
        if isinstance(loaded_data, dict) and 'data' in loaded_data and 'metadata' in loaded_data:
            logging.info(f"Loaded pickle file with metadata: {file_path}")
            return loaded_data['data']
        else:
            logging.info(f"Loaded pickle file: {file_path}")
            return loaded_data
            
    except Exception as e:
        logging.error(f"Error loading pickle file {file_path}: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return None