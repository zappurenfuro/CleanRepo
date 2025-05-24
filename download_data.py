import os
import logging
import gdown
from pathlib import Path

def download_from_gdrive(folder_url, parent_dir):
    """Download all files from a Google Drive folder"""
    logging.info(f"Downloading files from Google Drive folder: {folder_url}")
    
    try:
        # Create parent directory if it doesn't exist
        os.makedirs(parent_dir, exist_ok=True)
        
        # Create a temporary directory for download
        temp_dir = os.path.join(parent_dir, "temp_download")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Download the entire folder to temp directory
        gdown.download_folder(url=folder_url, output=temp_dir, quiet=False, use_cookies=False)
        
        # Check if the download created an "output" folder inside our temp directory
        downloaded_output = os.path.join(temp_dir, "output")
        target_output = os.path.join(parent_dir, "output")
        
        if os.path.exists(downloaded_output) and os.path.isdir(downloaded_output):
            # If output directory already exists, remove it
            if os.path.exists(target_output):
                logging.info(f"Removing existing output directory: {target_output}")
                shutil.rmtree(target_output)
            
            # Move the downloaded output directory to the correct location
            logging.info(f"Moving downloaded output from {downloaded_output} to {target_output}")
            shutil.move(downloaded_output, target_output)
        else:
            # If no "output" folder was created, the files were downloaded directly to temp_dir
            # Move all files from temp_dir to output directory
            logging.info(f"Moving all files from {temp_dir} to {target_output}")
            
            # Create output directory if it doesn't exist
            os.makedirs(target_output, exist_ok=True)
            
            # Move all files and directories
            for item in os.listdir(temp_dir):
                source = os.path.join(temp_dir, item)
                destination = os.path.join(target_output, item)
                
                if os.path.exists(destination):
                    if os.path.isdir(destination):
                        shutil.rmtree(destination)
                    else:
                        os.remove(destination)
                
                shutil.move(source, destination)
        
        # Clean up temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        logging.info(f"Successfully downloaded files to {target_output}")
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