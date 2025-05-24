import pickle
import numpy as np
import logging
import os
import sys
import traceback

def robust_load_pickle(file_path):
    """
    Load a pickle file with multiple fallback methods to handle version compatibility issues.
    """
    if not os.path.exists(file_path):
        logging.error(f"Pickle file does not exist: {file_path}")
        return None
    
    # Try different methods in order of preference
    methods = [
        "direct_load",
        "custom_unpickler",
        "encoding_latin1",
        "pickle_protocol_2"
    ]
    
    for method in methods:
        try:
            logging.info(f"Trying to load {file_path} with method: {method}")
            
            if method == "direct_load":
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
            
            elif method == "custom_unpickler":
                class CustomUnpickler(pickle.Unpickler):
                    def find_class(self, module, name):
                        # Handle numpy._core module not found
                        if module == 'numpy._core':
                            module = 'numpy.core'
                        
                        # Handle other potential numpy module changes
                        if module.startswith('numpy.'):
                            try:
                                return getattr(np, name)
                            except AttributeError:
                                pass
                        
                        # Default behavior
                        return super().find_class(module, name)
                
                with open(file_path, 'rb') as f:
                    data = CustomUnpickler(f).load()
            
            elif method == "encoding_latin1":
                # Try with latin1 encoding (helps with Python 2/3 compatibility)
                with open(file_path, 'rb') as f:
                    data = pickle.load(f, encoding='latin1')
            
            elif method == "pickle_protocol_2":
                # Try with pickle protocol 2 (more compatible)
                with open(file_path, 'rb') as f:
                    data = pickle.load(f, fix_imports=True, encoding='bytes')
            
            # Check if the loaded data has metadata
            if isinstance(data, dict) and 'data' in data and 'metadata' in data:
                logging.info(f"Successfully loaded pickle file with metadata using method: {method}")
                return data['data']
            else:
                logging.info(f"Successfully loaded pickle file using method: {method}")
                return data
                
        except Exception as e:
            logging.warning(f"Method {method} failed: {str(e)}")
            continue
    
    # If all methods failed
    logging.error(f"All methods failed to load pickle file: {file_path}")
    return None