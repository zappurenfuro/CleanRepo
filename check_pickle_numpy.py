import pickle
import sys
import os

def inspect_pickle(file_path):
    """Inspect a pickle file to determine the numpy version used."""
    print(f"Inspecting pickle file: {file_path}")
    
    try:
        with open(file_path, 'rb') as f:
            # Read the first 1000 bytes to look for numpy version info
            data = f.read(1000)
            
            # Look for numpy version strings
            if b'numpy' in data:
                print("  Contains numpy references")
                
                # Look for specific numpy modules
                if b'numpy._core' in data:
                    print("  Uses numpy._core module (likely numpy < 1.20)")
                if b'numpy.core' in data:
                    print("  Uses numpy.core module")
                
            else:
                print("  No numpy references found in header")
    
    except Exception as e:
        print(f"  Error inspecting file: {e}")

def main():
    # Check output directory for pickle files
    output_dir = "output"
    
    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} not found")
        return
    
    # Find all pickle files
    pickle_files = []
    for root, _, files in os.walk(output_dir):
        for file in files:
            if file.endswith('.pkl'):
                pickle_files.append(os.path.join(root, file))
    
    if not pickle_files:
        print("No pickle files found")
        return
    
    print(f"Found {len(pickle_files)} pickle files")
    
    # Inspect each pickle file
    for file_path in pickle_files:
        inspect_pickle(file_path)
        print()

if __name__ == "__main__":
    main()