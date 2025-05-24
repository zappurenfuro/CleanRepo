import os
import logging
import gdown
from pathlib import Path

def download_from_gdrive(folder_url, output_dir):
    """Download all files from a Google Drive folder"""
    logging.info(f"Downloading files from Google Drive folder: {folder_url}")
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Download the entire folder
        gdown.download_folder(url=folder_url, output=output_dir, quiet=False, use_cookies=False)
        
        logging.info(f"Successfully downloaded files to {output_dir}")
        return True
    except Exception as e:
        logging.error(f"Error downloading from Google Drive: {str(e)}")
        logging.error(traceback.format_exc())
        return False

def check_data_exists(output_dir):
    """Check if data files already exist"""
    # Check for the tfidf_enhanced folder
    tfidf_dirs = []
    if os.path.exists(output_dir):
        for item in os.listdir(output_dir):
            if item.startswith("tfidf_enhanced_"):
                tfidf_dir = os.path.join(output_dir, item)
                if os.path.isdir(tfidf_dir):
                    tfidf_dirs.append(tfidf_dir)
    
    # Check for processed_resumes.json
    processed_resumes = os.path.join(output_dir, "processed_resumes.json")
    
    if tfidf_dirs and os.path.exists(processed_resumes):
        logging.info(f"Found existing data files")
        logging.info(f"TF-IDF directories: {tfidf_dirs}")
        logging.info(f"Processed resumes: {processed_resumes}")
        return True
    
    return False

def setup_data_directories(base_dir):
    """Set up the necessary directories"""
    # Define directories
    input_dir = os.path.join(base_dir, "input")
    output_dir = os.path.join(base_dir, "output")
    cv_dir = os.path.join(base_dir, "cv_dummy")
    
    # Create directories if they don't exist
    for directory in [input_dir, output_dir, cv_dir]:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Ensured directory exists: {directory}")
    
    return input_dir, output_dir, cv_dir