from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
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
from scp import TFIDFEnhancedResumeScanner

def debug_paths():
    """Debug function to print paths and check for embedding files"""
    print("\n===== PATH DEBUGGING =====")
    print(f"Current working directory: {os.getcwd()}")
    
    # Check input/output directories
    input_dir = os.path.join(os.getcwd(), "input")
    output_dir = os.path.join(os.getcwd(), "output")
    
    print(f"Input directory exists: {os.path.exists(input_dir)}")
    print(f"Output directory exists: {os.path.exists(output_dir)}")
    
    # List all tfidf_enhanced directories
    print("\nSearching for tfidf_enhanced directories:")
    tfidf_dirs = []
    for root, dirs, files in os.walk(output_dir):
        for dir in dirs:
            if dir.startswith("tfidf_enhanced_"):
                tfidf_dir = os.path.join(root, dir)
                tfidf_dirs.append(tfidf_dir)
                print(f"  Found: {tfidf_dir}")
    
    # Check for embedding files (both PKL and JSON)
    print("\nSearching for embedding files:")
    embedding_files = []
    for tfidf_dir in tfidf_dirs:
        for root, dirs, files in os.walk(tfidf_dir):
            for file in files:
                if file == "tfidf_bge_embeddings.pkl" or file == "tfidf_bge_embeddings.json":
                    embedding_file = os.path.join(root, file)
                    embedding_files.append(embedding_file)
                    print(f"  Found: {embedding_file}")
                    print(f"  File size: {os.path.getsize(embedding_file) / (1024*1024):.2f} MB")
                    print(f"  Last modified: {os.path.getmtime(embedding_file)}")
    
    # Check file permissions
    print("\nChecking file permissions:")
    for embedding_file in embedding_files:
        try:
            with open(embedding_file, 'rb') as f:
                # Just read a small part to test access
                f.read(10)
            print(f"  Can read: {embedding_file}")
        except Exception as e:
            print(f"  ERROR reading {embedding_file}: {str(e)}")
    
    print("\n===== END PATH DEBUGGING =====")

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
    allow_origins=["*"],  # Adjust this in production to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use existing directories
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
CV_DIR = BASE_DIR / "cv_dummy"

# Global scanner instance
scanner = None

# Background task to initialize the scanner
def initialize_scanner():
    global scanner
    try:
        logging.info("Initializing resume scanner...")
        scanner = TFIDFEnhancedResumeScanner(
            input_folder=str(INPUT_DIR),
            output_folder=str(OUTPUT_DIR),
            cv_folder=str(CV_DIR)
        )
        
        # Check if input files exist
        input_files_exist = all(
            os.path.exists(os.path.join(INPUT_DIR, f)) 
            for f in ['01_people.csv', '02_abilities.csv', '03_education.csv', '04_experience.csv', '05_person_skills.csv']
        )
        
        if not input_files_exist:
            error_msg = "Input CSV files are missing. Please add the required CSV files to the input directory."
            logging.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Load data
        scanner.load_data()
        
        # Create TF-IDF vectors
        scanner.create_tfidf_vectors()
        
        # Create embeddings
        scanner.create_embeddings()
        
        logging.info("Resume scanner initialized successfully")
    except Exception as e:
        logging.error(f"Error initializing scanner: {str(e)}")
        logging.error(traceback.format_exc())
        scanner = None
        raise

# Response models
class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    file_format: str

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

# API endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the API is running and the model is loaded"""
    # Determine which file format is being used
    file_format = "unknown"
    if scanner is not None:
        # Check if the results directory exists
        if hasattr(scanner, 'results_dir') and os.path.exists(scanner.results_dir):
            # Check for JSON files
            json_files = glob.glob(os.path.join(scanner.results_dir, "*.json"))
            pkl_files = glob.glob(os.path.join(scanner.results_dir, "*.pkl"))
            
            if json_files:
                file_format = "json"
            elif pkl_files:
                file_format = "pkl"
    
    return {
        "status": "ok",
        "version": "1.0.0",
        "model_loaded": scanner is not None,
        "file_format": file_format
    }

@app.post("/initialize", response_model=dict)
async def initialize_model(background_tasks: BackgroundTasks):
    """Initialize the resume scanner model in the background"""
    if scanner is not None:
        return {"message": "Model already initialized"}
    
    background_tasks.add_task(initialize_scanner)
    return {"message": "Model initialization started in background"}

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

# Add these new variables to track the current session
current_cv_file = None
current_results_dir = None

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

# Add an endpoint to check if a visualization is available
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

@app.post("/evaluate", response_model=dict)
async def evaluate_model(k_values: List[int] = [3, 5, 10], relevance_threshold: Optional[float] = None):
    """Evaluate the model using ground truth data"""
    if scanner is None:
        raise HTTPException(status_code=503, detail="Model not initialized. Call /initialize first.")
    
    try:
        # Run evaluation
        metrics = scanner.evaluate_model(k_values=k_values, relevance_threshold=relevance_threshold)
        
        if metrics is None:
            return JSONResponse(
                status_code=500,
                content={"error": "Evaluation failed. Check server logs for details."}
            )
        
        return {
            "message": "Evaluation completed successfully",
            "metrics": metrics
        }
    except Exception as e:
        logging.error(f"Error evaluating model: {str(e)}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# Add a new endpoint to check file formats
@app.get("/check/file-formats")
async def check_file_formats():
    """Check which file formats are available in the output directory"""
    try:
        formats = {
            "json": [],
            "pkl": []
        }
        
        # Search for files in the output directory
        for root, dirs, files in os.walk(str(OUTPUT_DIR)):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith('.json'):
                    formats["json"].append({
                        "path": file_path,
                        "size_mb": os.path.getsize(file_path) / (1024 * 1024),
                        "modified": os.path.getmtime(file_path)
                    })
                elif file.endswith('.pkl'):
                    formats["pkl"].append({
                        "path": file_path,
                        "size_mb": os.path.getsize(file_path) / (1024 * 1024),
                        "modified": os.path.getmtime(file_path)
                    })
        
        return {
            "json_files": len(formats["json"]),
            "pkl_files": len(formats["pkl"]),
            "details": formats
        }
    except Exception as e:
        logging.error(f"Error checking file formats: {str(e)}")
        return {"error": str(e)}

# Startup event to initialize the scanner
@app.on_event("startup")
async def startup_event():
    # Check if input directories exist
    for directory in [INPUT_DIR, OUTPUT_DIR, CV_DIR]:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created directory: {directory}")
    
    # Debug paths and file existence
    try:
        import glob
        logging.info(f"Current working directory: {os.getcwd()}")
        logging.info(f"Input directory exists: {os.path.exists(str(INPUT_DIR))}")
        logging.info(f"Output directory exists: {os.path.exists(str(OUTPUT_DIR))}")
        
        # List all PKL and JSON files in the output directory
        pkl_files = glob.glob(str(OUTPUT_DIR / "**/*.pkl"), recursive=True)
        json_files = glob.glob(str(OUTPUT_DIR / "**/*.json"), recursive=True)
        
        logging.info(f"Found {len(pkl_files)} PKL files in output directory")
        for pkl_file in pkl_files:
            file_size = os.path.getsize(pkl_file) / (1024 * 1024)
            logging.info(f"  {pkl_file} ({file_size:.2f} MB)")
        
        logging.info(f"Found {len(json_files)} JSON files in output directory")
        for json_file in json_files:
            file_size = os.path.getsize(json_file) / (1024 * 1024)
            logging.info(f"  {json_file} ({file_size:.2f} MB)")
    except Exception as e:
        logging.error(f"Error in path debugging: {str(e)}")
    
    # Check if input files exist
    input_files_exist = all(
        os.path.exists(os.path.join(INPUT_DIR, f)) 
        for f in ['01_people.csv', '02_abilities.csv', '03_education.csv', '04_experience.csv', '05_person_skills.csv']
    )
    
    if not input_files_exist:
        logging.warning("Input CSV files are missing. The API will start, but initialization will fail until files are provided.")
        logging.warning(f"Please add the required CSV files to: {INPUT_DIR}")
    else:
        # Initialize the scanner in the background
        try:
            initialize_scanner()
        except Exception as e:
            logging.error(f"Failed to initialize scanner on startup: {str(e)}")
            logging.error("The API will start, but you'll need to call /initialize endpoint manually.")

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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)