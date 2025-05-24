import os
import requests
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

def download_pickle_files():
    """Download pickle files from URLs specified in environment variables."""
    # Define destination directories
    output_dir = "/app/output"
    tfidf_dir = "/app/output/tfidf_enhanced_1748065889"
    
    # Ensure destination directories exist
    ensure_dir(output_dir)
    ensure_dir(tfidf_dir)
    
    # Define file mappings (env var -> destination path)
    file_mappings = {
        "PROCESSED_RESUMES_URL": os.path.join(output_dir, "processed_resumes.pkl"),
        "TFIDF_BGE_EMBEDDINGS_URL": os.path.join(tfidf_dir, "tfidf_bge_embeddings.pkl"),
        "TFIDF_VECTORIZER_URL": os.path.join(tfidf_dir, "tfidf_vectorizer.pkl"),
        "TFIDF_MATRIX_URL": os.path.join(tfidf_dir, "tfidf_matrix.pkl")
    }
    
    # Download each file if the environment variable is set
    success = True
    for env_var, dest_path in file_mappings.items():
        # Skip if the destination file already exists
        if os.path.exists(dest_path):
            logging.info(f"File already exists at destination: {dest_path}")
            continue
        
        # Check if the environment variable is set
        url = os.environ.get(env_var)
        if not url:
            logging.info(f"Environment variable {env_var} not set, skipping download")
            continue
        
        # Download the file
        try:
            logging.info(f"Downloading file from {url} to {dest_path}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logging.info(f"Successfully downloaded file to {dest_path}")
        except Exception as e:
            logging.error(f"Error downloading file from {url} to {dest_path}: {str(e)}")
            success = False
    
    # Check if all required files are now in place
    all_files_exist = True
    for dest_path in file_mappings.values():
        if not os.path.exists(dest_path):
            logging.warning(f"Required file still missing: {dest_path}")
            all_files_exist = False
        else:
            logging.info(f"Required file exists: {dest_path}")
    
    if all_files_exist:
        logging.info("All required pickle files are in place")
    else:
        logging.warning("Some required pickle files are still missing")
    
    return success

if __name__ == "__main__":
    logging.info("Starting pickle file download process")
    download_pickle_files()
    logging.info("Finished pickle file download process")
