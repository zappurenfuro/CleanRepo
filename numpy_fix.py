"""
This module creates a fake numpy._core module to handle pickle compatibility issues.
It needs to be imported before any pickle loading operations.
"""
import sys
import numpy as np
import logging

# Create a fake numpy._core module that redirects to numpy.core
class FakeNumpy:
    def __init__(self):
        self._core = np.core
        
    def __getattr__(self, name):
        return getattr(np, name)

class FakeCore:
    def __getattr__(self, name):
        try:
            return getattr(np.core, name)
        except AttributeError:
            # Try to find it in the main numpy module
            try:
                return getattr(np, name)
            except AttributeError:
                logging.warning(f"Could not find {name} in numpy.core or numpy")
                raise

# Create the fake modules
fake_numpy = FakeNumpy()
fake_core = FakeCore()

# Add the fake modules to sys.modules
sys.modules['numpy._core'] = fake_core

logging.info("Installed numpy._core compatibility layer")