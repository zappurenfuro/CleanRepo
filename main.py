from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import tempfile
import shutil
import logging
import time
import json
import traceback
import glob
from pathlib import Path
import sys

# Import the resume scanner model
from scp import TFIDFEnhancedResumeScanner, load_from_pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize FastAPI app
app = FastAPI(
    title="CVScan API",
    description="API for scanning and matching resumes with job titles",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use existing directories
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
CV_DIR = BASE_DIR / "cv_dummy"

# Global variables
scanner = None
df = None
embeddings = None
tfidf_vectorizer = None
tfidf_matrix = None
current_cv_file = None
current_results_dir = None

# Response models
class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    preloaded_data: bool

class MatchResult(BaseModel):
    title: str
    similarity_percentage: float
    embedding_text: Optional[str] = None

class ScanResponse(BaseModel):
    matches: List[MatchResult]
    avg_similarity: float

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None

def find_latest_pickle_file(directory, filename_pattern):
    """Find the most recent pickle file matching the pattern in the directory tree"""
    matches = []
    
    # First look directly in the output directory
    direct_matches = list(Path(directory).glob(filename_pattern))
    if direct_matches:
        matches.extend(direct_matches)
    
    # Then look in subdirectories
    for tfidf_dir in Path(directory).glob("tfidf_enhanced_*"):
        if tfidf_dir.is_dir():
            subdir_matches = list(tfidf_dir.glob(filename_pattern))
            matches.extend(subdir_matches)
    
    if not matches:
        return None
    
    # Sort by modification time (newest first)
    matches.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(matches[0])

def load_precomputed_data():
    """Load precomputed data from pickle files"""
    global df, embeddings, tfidf_vectorizer, tfidf_matrix
    
    try:
        logging.info("Loading precomputed data from pickle files...")
        
        # Find the most recent pickle files
        processed_resumes_file = find_latest_pickle_file(OUTPUT_DIR, "processed_resumes.pkl")
        embeddings_file = find_latest_pickle_file(OUTPUT_DIR, "tfidf_bge_embeddings.pkl")
        tfidf_vectorizer_file = find_latest_pickle_file(OUTPUT_DIR, "tfidf_vectorizer.pkl")
        tfidf_matrix_file = find_latest_pickle_file(OUTPUT_DIR, "tfidf_matrix.pkl")
        
        logging.info(f"Found processed_resumes_file: {processed_resumes_file}")
        logging.info(f"Found embeddings_file: {embeddings_file}")
        logging.info(f"Found tfidf_vectorizer_file: {tfidf_vectorizer_file}")
        logging.info(f"Found tfidf_matrix_file: {tfidf_matrix_file}")
        
        # Load the data if all files are found
        if all([processed_resumes_file, embeddings_file, tfidf_vectorizer_file, tfidf_matrix_file]):
            df = load_from_pickle(processed_resumes_file)
            embeddings = load_from_pickle(embeddings_file)
            tfidf_vectorizer = load_from_pickle(tfidf_vectorizer_file)
            tfidf_matrix = load_from_pickle(tfidf_matrix_file)
            
            # Verify data was loaded correctly
            if all([df is not None, embeddings is not None, tfidf_vectorizer is not None, tfidf_matrix is not None]):
                logging.info(f"Successfully loaded precomputed data:")
                logging.info(f"  - DataFrame with {len(df)} records")
                logging.info(f"  - Embeddings with shape {embeddings.shape}")
                logging.info(f"  - TF-IDF vectorizer with {len(tfidf_vectorizer.get_feature_names_out())} features")
                logging.info(f"  - TF-IDF matrix with shape {tfidf_matrix.shape}")
                return True
            else:
                logging.error("Failed to load one or more precomputed data files")
                return False
        else:
            logging.warning("One or more precomputed data files not found")
            return False
            
    except Exception as e:
        logging.error(f"Error loading precomputed data: {str(e)}")
        logging.error(traceback.format_exc())
        return False

def initialize_scanner_with_preloaded_data():
    """Initialize scanner with preloaded data to avoid retraining"""
    global scanner, df, embeddings, tfidf_vectorizer, tfidf_matrix
    
    try:
        logging.info("Initializing scanner with preloaded data...")
        
        # Create scanner instance
        scanner = TFIDFEnhancedResumeScanner(
            input_folder=str(INPUT_DIR),
            output_folder=str(OUTPUT_DIR),
            cv_folder=str(CV_DIR)
        )
        
        # Load the model (this doesn't retrain)
        scanner._load_model()
        
        # Assign preloaded data to scanner
        scanner.df = df
        scanner.embeddings = embeddings
        scanner.tfidf_vectorizer = tfidf_vectorizer
        scanner.tfidf_matrix = tfidf_matrix
        
        # Create results directory
        scanner.results_dir = os.path.join(scanner.output_folder, f"api_results_{int(time.time())}")
        os.makedirs(scanner.results_dir, exist_ok=True)
        
        # Create evaluation directory
        scanner.eval_dir = os.path.join(scanner.results_dir, "evaluation")
        os.makedirs(scanner.eval_dir, exist_ok=True)
        
        logging.info("Scanner initialized with preloaded data successfully")
        return True
        
    except Exception as e:
        logging.error(f"Error initializing scanner with preloaded data: {str(e)}")
        logging.error(traceback.format_exc())
        return False

# API endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the API is running and the model is loaded"""
    return {
        "status": "ok",
        "version": "1.0.0",
        "model_loaded": scanner is not None,
        "preloaded_data": all([df is not None, embeddings is not None, 
                              tfidf_vectorizer is not None, tfidf_matrix is not None])
    }

@app.post("/initialize", response_model=dict)
async def initialize_model():
    """Initialize the resume scanner model with preloaded data"""
    global scanner
    
    if scanner is not None:
        return {"message": "Model already initialized"}
    
    # First load precomputed data
    if not all([df is not None, embeddings is not None, tfidf_vectorizer is not None, tfidf_matrix is not None]):
        if not load_precomputed_data():
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to load precomputed data. Check server logs for details."}
            )
    
    # Initialize scanner with preloaded data
    if initialize_scanner_with_preloaded_data():
        return {"message": "Model initialized successfully with preloaded data"}
    else:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to initialize scanner. Check server logs for details."}
        )

@app.post("/scan/text", response_model=ScanResponse)
async def scan_text(text: str = Form(...), top_n: int = Form(5)):
    """Scan resume text and match against job titles"""
    if scanner is None:
        raise HTTPException(status_code=503, detail="Model not initialized. Call /initialize first.")
    
    try:
        # Match the text against job titles
        results = scanner.match_text(text, top_n=top_n)
        
        # Convert DataFrame to list of dictionaries
        matches = []
        for _, row in results['matches'].iterrows():
            match_dict = {
                "title": row['title'],
                "similarity_percentage": row['similarity_percentage']
            }
            if 'embedding_text' in row:
                match_dict["embedding_text"] = row['embedding_text']
            matches.append(match_dict)
        
        return {
            "matches": matches,
            "avg_similarity": results['avg_similarity']
        }
    except Exception as e:
        logging.error(f"Error scanning text: {str(e)}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/scan/file", response_model=ScanResponse)
async def scan_file(file: UploadFile = File(...), top_n: int = Form(5)):
    """Scan resume file and match against job titles"""
    global current_cv_file, current_results_dir
    
    if scanner is None:
        raise HTTPException(status_code=503, detail="Model not initialized. Call /initialize first.")
    
    # Check file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ['.pdf', '.docx', '.doc']:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload PDF or Word documents.")
    
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
    temp_file_path = temp_file.name
    
    try:
        # Save uploaded file to temp file
        with temp_file:
            shutil.copyfileobj(file.file, temp_file)
        
        # Process the resume file
        results = scanner.process_resume_file(temp_file_path, top_n=top_n)
        
        # Store the current CV file name and results directory for later use
        current_cv_file = file.filename
        current_results_dir = scanner.results_dir
        
        # Convert DataFrame to list of dictionaries
        matches = []
        for _, row in results['matches'].iterrows():
            match_dict = {
                "title": row['title'],
                "similarity_percentage": row['similarity_percentage']
            }
            if 'embedding_text' in row:
                match_dict["embedding_text"] = row['embedding_text']
            matches.append(match_dict)
        
        return {
            "matches": matches,
            "avg_similarity": results['avg_similarity']
        }
    except Exception as e:
        logging.error(f"Error scanning file: {str(e)}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the temp file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

@app.get("/download/visualization")
async def download_visualization():
    """Download the visualization PNG file for the current CV"""
    if current_cv_file is None or current_results_dir is None:
        raise HTTPException(status_code=404, detail="No CV has been uploaded yet")
    
    try:
        # Get the base name of the CV file (without extension)
        base_name = os.path.splitext(os.path.basename(current_cv_file))[0]
        
        # Look for any PNG files in the results directory
        png_files = glob.glob(os.path.join(current_results_dir, "*.png"))
        
        if not png_files:
            raise HTTPException(status_code=404, detail="No visualization files found")
        
        # Sort by modification time to get the most recent one
        png_files.sort(key=os.path.getmtime, reverse=True)
        visualization_file = png_files[0]
        
        logging.info(f"Found visualization file: {visualization_file}")
        
        # Return the file as a response
        return FileResponse(
            path=visualization_file,
            filename=f"{base_name}_visualization.png",
            media_type="image/png"
        )
    except Exception as e:
        logging.error(f"Error downloading visualization: {str(e)}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/check/visualization")
async def check_visualization():
    """Check if a visualization is available for download"""
    if current_cv_file is None or current_results_dir is None:
        return {"available": False, "message": "No CV has been uploaded yet"}
    
    try:
        # Look for any PNG files in the results directory
        png_files = glob.glob(os.path.join(current_results_dir, "*.png"))
        
        if png_files:
            # Sort by modification time to get the most recent one
            png_files.sort(key=os.path.getmtime, reverse=True)
            return {"available": True, "filename": os.path.basename(png_files[0])}
        else:
            return {"available": False, "message": "No visualization files found"}
    except Exception as e:
        logging.error(f"Error checking visualization: {str(e)}")
        return {"available": False, "message": str(e)}

@app.get("/debug/paths")
async def debug_path_info():
    """Debug endpoint to check paths and file existence"""
    try:
        # Find all pickle files in the output directory
        pickle_files = []
        for root, _, files in os.walk(OUTPUT_DIR):
            for file in files:
                if file.endswith('.pkl'):
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
                    mod_time = time.strftime('%Y-%m-%d %H:%M:%S', 
                                            time.localtime(os.path.getmtime(file_path)))
                    pickle_files.append({
                        "path": file_path,
                        "size_mb": round(file_size, 2),
                        "modified": mod_time
                    })
        
        # Check for tfidf_enhanced directories
        tfidf_dirs = []
        for item in os.listdir(OUTPUT_DIR):
            item_path = os.path.join(OUTPUT_DIR, item)
            if os.path.isdir(item_path) and item.startswith("tfidf_enhanced_"):
                tfidf_dirs.append(item_path)
        
        return {
            "base_dir": str(BASE_DIR),
            "input_dir": str(INPUT_DIR),
            "output_dir": str(OUTPUT_DIR),
            "cv_dir": str(CV_DIR),
            "input_dir_exists": os.path.exists(INPUT_DIR),
            "output_dir_exists": os.path.exists(OUTPUT_DIR),
            "cv_dir_exists": os.path.exists(CV_DIR),
            "pickle_files": pickle_files,
            "tfidf_dirs": tfidf_dirs,
            "scanner_initialized": scanner is not None,
            "data_preloaded": all([df is not None, embeddings is not None, 
                                 tfidf_vectorizer is not None, tfidf_matrix is not None])
        }
    except Exception as e:
        logging.error(f"Error in debug paths: {str(e)}")
        logging.error(traceback.format_exc())
        return {"error": str(e), "traceback": traceback.format_exc()}

# Startup event to initialize the scanner
@app.on_event("startup")
async def startup_event():
    # Check if input directories exist
    for directory in [INPUT_DIR, OUTPUT_DIR, CV_DIR]:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created directory: {directory}")
    
    # Debug paths and file existence
    debug_paths()
    
    # Try to load precomputed data
    if load_precomputed_data():
        # Initialize scanner with preloaded data
        if initialize_scanner_with_preloaded_data():
            logging.info("Scanner initialized with preloaded data on startup")
        else:
            logging.warning("Failed to initialize scanner with preloaded data")
    else:
        logging.warning("Failed to load precomputed data on startup")
        logging.warning("Please call /initialize endpoint manually after ensuring data files are available")

# Shutdown event to clean up resources
@app.on_event("shutdown")
async def shutdown_event():
    global scanner
    if scanner is not None:
        scanner.cleanup()
        scanner = None
        logging.info("Scanner resources cleaned up")

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)