import os
import shutil
import glob
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def ensure_dir(directory):
    """Ensure a directory exists, creating it if necessary."""
    os.makedirs(directory, exist_ok=True)
    logging.info(f"Ensured directory exists: {directory}")

def copy_pickle_files():
    """Copy pickle files from various possible locations to the correct destinations."""
    # Define destination directories
    output_dir = "/app/output"
    tfidf_dir = "/app/output/tfidf_enhanced_1748065889"
    
    # Ensure destination directories exist
    ensure_dir(output_dir)
    ensure_dir(tfidf_dir)
    
    # Define file mappings (filename -> destination path)
    file_mappings = {
        "processed_resumes.pkl": os.path.join(output_dir, "processed_resumes.pkl"),
        "tfidf_bge_embeddings.pkl": os.path.join(tfidf_dir, "tfidf_bge_embeddings.pkl"),
        "tfidf_vectorizer.pkl": os.path.join(tfidf_dir, "tfidf_vectorizer.pkl"),
        "tfidf_matrix.pkl": os.path.join(tfidf_dir, "tfidf_matrix.pkl")
    }
    
    # Possible source directories to search for pickle files
    source_dirs = [
        "/app/pickle_files",  # For local Docker with volume mounting
        "/app",               # Root of the application
        "/app/output",        # Output directory
        ".",                  # Current directory
        "./output",           # Relative output directory
    ]
    
    # Copy each file if it exists in any of the source directories
    success = True
    for filename, dest_path in file_mappings.items():
        # Skip if the destination file already exists
        if os.path.exists(dest_path):
            logging.info(f"File already exists at destination: {dest_path}")
            continue
        
        # Try to find the file in any of the source directories
        found = False
        for source_dir in source_dirs:
            # Try direct path
            source_path = os.path.join(source_dir, filename)
            if os.path.exists(source_path):
                try:
                    shutil.copy2(source_path, dest_path)
                    logging.info(f"Copied {source_path} to {dest_path}")
                    found = True
                    break
                except Exception as e:
                    logging.error(f"Error copying {source_path} to {dest_path}: {str(e)}")
            
            # Try recursive search
            if not found:
                try:
                    found_files = glob.glob(os.path.join(source_dir, f"**/{filename}"), recursive=True)
                    if found_files:
                        try:
                            shutil.copy2(found_files[0], dest_path)
                            logging.info(f"Copied {found_files[0]} to {dest_path}")
                            found = True
                            break
                        except Exception as e:
                            logging.error(f"Error copying {found_files[0]} to {dest_path}: {str(e)}")
                except Exception as e:
                    logging.error(f"Error searching in {source_dir}: {str(e)}")
        
        if not found:
            logging.error(f"Could not find {filename} in any source directory")
            success = False
    
    # Check if all required files are now in place
    all_files_exist = True
    for dest_path in file_mappings.values():
        if not os.path.exists(dest_path):
            logging.error(f"Required file still missing: {dest_path}")
            all_files_exist = False
        else:
            logging.info(f"Required file exists: {dest_path}")
    
    if all_files_exist:
        logging.info("All required pickle files are in place")
    else:
        logging.warning("Some required pickle files are still missing")
    
    return success and all_files_exist

if __name__ == "__main__":
    logging.info("Starting pickle file copy process")
    copy_pickle_files()
    logging.info("Finished pickle file copy process")
