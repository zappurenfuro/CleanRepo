from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os
import tempfile
import shutil
import logging
import time
import json
import traceback
import glob
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import re
from functools import lru_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize FastAPI app
app = FastAPI(
    title="CVScan API",
    description="API for scanning and matching resumes with job titles",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
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

# Global variables to store loaded data
df = None
embeddings = None
tfidf_vectorizer = None
tfidf_matrix = None
tfidf_feature_names = None
results_dir = None

# Response models
class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool = Field(..., description="Whether the model is loaded")

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

# Helper functions
def safe_load_pickle(file_path):
    """Safely load a pickle file with error handling."""
    try:
        if not os.path.exists(file_path):
            logging.error(f"Pickle file does not exist: {file_path}")
            return None
        
        with open(file_path, 'rb') as f:
            loaded_data = pickle.load(f)
        
        # Check if the loaded data has metadata
        if isinstance(loaded_data, dict) and 'data' in loaded_data and 'metadata' in loaded_data:
            logging.info(f"Loaded pickle file with metadata: {file_path}")
            return loaded_data['data']
        else:
            logging.info(f"Loaded pickle file: {file_path}")
            return loaded_data
            
    except Exception as e:
        logging.error(f"Error loading pickle file {file_path}: {str(e)}")
        logging.error(traceback.format_exc())
        return None

@lru_cache(maxsize=1000)
def clean_text(text):
    """Clean text by removing quotes and exclamation marks."""
    if not text or pd.isna(text):
        return ""
    
    # Convert to string if not already
    text = str(text)
    
    # Remove single quotes, double quotes, and exclamation marks
    text = text.replace("'", "").replace('"', "").replace('!', "")
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

@lru_cache(maxsize=1000)
def clean_standardize_title(title):
    """Clean and standardize job titles."""
    if not title or pd.isna(title):
        return "Unknown Title"
    
    # Convert to string and take only the first title if multiple exist
    title = str(title).split(';')[0].strip()
    
    # Dictionary of terms that should be preserved as-is
    preserved_terms = {
        'devops': 'DevOps',
        'mysql': 'MySQL',
        'postgresql': 'PostgreSQL',
        'mongodb': 'MongoDB',
        'javascript': 'JavaScript',
        'typescript': 'TypeScript',
        'nodejs': 'NodeJS',
        'reactjs': 'ReactJS',
        'vuejs': 'VueJS',
        'angularjs': 'AngularJS',
        'dotnet': '.NET',
        'aspnet': 'ASP.NET',
        'fullstack': 'Fullstack',
        'frontend': 'Frontend',
        'backend': 'Backend',
        'nosql': 'NoSQL',
        'graphql': 'GraphQL',
        'restapi': 'RESTful API',
        'cicd': 'CI/CD',
        'android': 'Android',
        'ios': 'iOS'
    }
    
    # Dictionary of terms that should be fully capitalized
    special_terms = {
        'to': 'to',
        'xml': 'XML',
        'mean': 'MEAN',
        'js': 'JS',
        'php': 'PHP',
        'css': 'CSS',
        'html': 'HTML',
        'ui': 'UI',
        'ux': 'UX',
        'api': 'API',
        'aws': 'AWS',
        'gcp': 'GCP',
        'qa': 'QA',
        'sre': 'SRE',
        'ml': 'ML',
        'ai': 'AI',
        'ci': 'CI',
        'cd': 'CD',
        'it': 'IT'
    }
    
    # Handle special cases for common job titles
    title = re.sub(r'\bfront[\s-]*end\b', 'Frontend', title, flags=re.IGNORECASE)
    title = re.sub(r'\bback[\s-]*end\b', 'Backend', title, flags=re.IGNORECASE)
    title = re.sub(r'\bfull[\s-]*stack\b', 'Fullstack', title, flags=re.IGNORECASE)
    
    # Split into words for processing
    words = title.split()
    processed_words = []
    
    for word in words:
        word_lower = word.lower()
        
        # Check if it's a special term that should be fully capitalized
        if word_lower in special_terms:
            processed_words.append(special_terms[word_lower])
        elif word_lower in preserved_terms:
            processed_words.append(preserved_terms[word_lower])
        else:
            # Capitalize the first letter of each word
            processed_words.append(word.capitalize())
    
    # Join the words back together
    processed_title = ' '.join(processed_words)
    
    # Remove any double spaces
    processed_title = re.sub(r'\s+', ' ', processed_title).strip()
    
    return processed_title

@lru_cache(maxsize=1000)
def normalize_title_for_comparison(title):
    """Create a normalized version of a title for comparison and deduplication."""
    if not title or pd.isna(title):
        return ""
    
    # First apply the standard cleaning
    title = clean_standardize_title(title)
    
    # Convert to lowercase for comparison
    title = title.lower()
    
    # Remove all punctuation and special characters
    title = re.sub(r'[^\w\s]', '', title)
    
    # Replace multiple spaces with a single space
    title = re.sub(r'\s+', ' ', title).strip()
    
    # Remove common words that don't add meaning for comparison
    stop_words = ['and', 'or', 'the', 'a', 'an', 'in', 'on', 'at', 'for', 'with', 'to']
    title_words = title.split()
    title_words = [word for word in title_words if word not in stop_words]
    
    # Join words back together
    return ' '.join(title_words)

def normalize_job_titles(matches_df):
    """Normalize job titles in the matches dataframe."""
    if 'title' not in matches_df.columns:
        return matches_df
    
    # Clean up titles first
    matches_df['title'] = matches_df['title'].apply(clean_standardize_title)
    
    # Add a normalized title column for comparison
    matches_df['normalized_title'] = matches_df['title'].apply(normalize_title_for_comparison)
    
    # Group by normalized title and keep the one with highest similarity
    result = matches_df.loc[matches_df.groupby('normalized_title')['similarity_percentage'].idxmax()]
    
    # Drop the normalized_title column as it's no longer needed
    result = result.drop(columns=['normalized_title'])
    
    # Sort by similarity percentage in descending order
    result = result.sort_values('similarity_percentage', ascending=False)
    
    return result

def apply_tfidf_weighting(text):
    """Apply TF-IDF weighting to text before embedding."""
    global tfidf_vectorizer, tfidf_feature_names
    
    if tfidf_vectorizer is None:
        raise ValueError("TF-IDF vectorizer not initialized.")
    
    # Transform the text to TF-IDF vector
    text_tfidf = tfidf_vectorizer.transform([text])
    
    # Get the top N terms with highest TF-IDF scores
    N = 20  # Number of top terms to include
    
    # Get non-zero elements and their indices
    nonzero = text_tfidf.nonzero()[1]
    scores = text_tfidf.data
    
    # Sort by score and get top N
    if len(nonzero) > N:
        top_indices = nonzero[scores.argsort()[-N:]]
        top_scores = scores[scores.argsort()[-N:]]
    else:
        top_indices = nonzero
        top_scores = scores
    
    # Get the corresponding terms
    top_terms = [tfidf_feature_names[i] for i in top_indices]
    
    # Weight the original text by repeating important terms
    # The number of repetitions is proportional to the TF-IDF score
    weighted_text = text
    for term, score in zip(top_terms, top_scores):
        # Normalize score to a reasonable number of repetitions (1-5)
        repetitions = min(5, max(1, int(score * 10)))
        weighted_text += f" {term * repetitions}"
    
    return weighted_text

def find_most_recent_pickle_files():
    """Find the most recent pickle files in the output directory."""
    global OUTPUT_DIR
    
    # Find all tfidf_enhanced directories
    tfidf_dirs = []
    for item in OUTPUT_DIR.glob("tfidf_enhanced_*"):
        if item.is_dir():
            tfidf_dirs.append(item)
    
    if not tfidf_dirs:
        logging.warning("No tfidf_enhanced directories found in output directory")
        return None, None, None, None
    
    # Sort by modification time (most recent first)
    tfidf_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    most_recent_dir = tfidf_dirs[0]
    logging.info(f"Using most recent directory: {most_recent_dir}")
    
    # Find pickle files
    processed_resumes_file = OUTPUT_DIR / "processed_resumes.pkl"
    embeddings_file = most_recent_dir / "tfidf_bge_embeddings.pkl"
    tfidf_vectorizer_file = most_recent_dir / "tfidf_vectorizer.pkl"
    tfidf_matrix_file = most_recent_dir / "tfidf_matrix.pkl"
    
    # Check if files exist
    if not processed_resumes_file.exists():
        logging.warning(f"Processed resumes file not found: {processed_resumes_file}")
    else:
        logging.info(f"Found processed_resumes_file: {processed_resumes_file}")
    
    if not embeddings_file.exists():
        logging.warning(f"Embeddings file not found: {embeddings_file}")
    else:
        logging.info(f"Found embeddings_file: {embeddings_file}")
    
    if not tfidf_vectorizer_file.exists():
        logging.warning(f"TF-IDF vectorizer file not found: {tfidf_vectorizer_file}")
    else:
        logging.info(f"Found tfidf_vectorizer_file: {tfidf_vectorizer_file}")
    
    if not tfidf_matrix_file.exists():
        logging.warning(f"TF-IDF matrix file not found: {tfidf_matrix_file}")
    else:
        logging.info(f"Found tfidf_matrix_file: {tfidf_matrix_file}")
    
    return processed_resumes_file, embeddings_file, tfidf_vectorizer_file, tfidf_matrix_file

def load_precomputed_data():
    """Load precomputed data from pickle files."""
    global df, embeddings, tfidf_vectorizer, tfidf_matrix, tfidf_feature_names, results_dir
    
    logging.info("Loading precomputed data from pickle files...")
    
    # Find the most recent pickle files
    processed_resumes_file, embeddings_file, tfidf_vectorizer_file, tfidf_matrix_file = find_most_recent_pickle_files()
    
    if not all([processed_resumes_file, embeddings_file, tfidf_vectorizer_file, tfidf_matrix_file]):
        logging.error("One or more required pickle files not found")
        return False
    
    # Load processed resumes
    df = safe_load_pickle(processed_resumes_file)
    if df is None:
        logging.error("Failed to load processed resumes")
        return False
    
    # Load embeddings
    embeddings = safe_load_pickle(embeddings_file)
    if embeddings is None:
        logging.error("Failed to load embeddings")
        return False
    
    # Load TF-IDF vectorizer
    tfidf_vectorizer = safe_load_pickle(tfidf_vectorizer_file)
    if tfidf_vectorizer is None:
        logging.error("Failed to load TF-IDF vectorizer")
        return False
    
    # Load TF-IDF matrix
    tfidf_matrix = safe_load_pickle(tfidf_matrix_file)
    if tfidf_matrix is None:
        logging.error("Failed to load TF-IDF matrix")
        return False
    
    # Get feature names from vectorizer
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    
    # Set results directory
    results_dir = embeddings_file.parent
    
    logging.info(f"Successfully loaded all precomputed data")
    logging.info(f"DataFrame shape: {df.shape}")
    logging.info(f"Embeddings shape: {embeddings.shape}")
    logging.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    logging.info(f"Number of TF-IDF features: {len(tfidf_feature_names)}")
    
    return True

def match_text(text: str, top_n: int = 5, file_name: str = None) -> Dict:
    """Match text against TF-IDF weighted BGE embeddings."""
    global df, embeddings, tfidf_vectorizer, tfidf_feature_names, results_dir
    
    logging.info(f"Matching text against TF-IDF weighted BGE embeddings...")
    
    # Clean the input text
    text = clean_text(text)
    
    # Debug: Print a hash of the input text
    import hashlib
    text_hash = hashlib.md5(text.encode()).hexdigest()
    logging.info(f"Matching text (hash: {text_hash[:8]}...)")
    
    # Apply TF-IDF weighting to the query text
    weighted_text = apply_tfidf_weighting(text)
    
    # Use the precomputed embeddings directly
    # We need to find the most similar embeddings in the precomputed set
    
    # First, transform the weighted text using the TF-IDF vectorizer
    text_tfidf = tfidf_vectorizer.transform([weighted_text])
    
    # Calculate similarities using the TF-IDF vectors
    # This is a simpler approach that doesn't require the embedding model
    similarities = cosine_similarity(text_tfidf, tfidf_matrix)[0]
    
    # Get more candidates than needed to allow for deduplication
    top_indices = similarities.argsort()[-(top_n*3):][::-1]  # Get 3x needed to filter duplicates
    top_similarities = similarities[top_indices]
    
    # Get the corresponding rows from the dataframe
    top_matches = df.iloc[top_indices].copy()
    top_matches['similarity_percentage'] = (top_similarities * 100).round(2)
    
    # Apply proper title formatting to ensure consistent display
    top_matches['title'] = top_matches['title'].apply(lambda x: clean_standardize_title(x))
    
    # Normalize and deduplicate job titles
    top_matches = normalize_job_titles(top_matches)
    
    # Take only the top N after deduplication
    top_matches = top_matches.head(top_n)
    
    # Save results with unique filename if provided
    if file_name and results_dir:
        base_name = os.path.splitext(os.path.basename(file_name))[0]
        
        # Save results
        results_file = os.path.join(results_dir, f"{base_name}_matches.csv")
        try:
            top_matches.to_csv(results_file, index=False)
            logging.info(f"Saved {len(top_matches)} matches to {results_file}")
        except Exception as e:
            logging.error(f"Failed to save matches to {results_file}: {str(e)}")
    
    # Calculate average similarity score
    avg_similarity = top_matches['similarity_percentage'].mean()
    
    logging.info(f"Average similarity score: {avg_similarity:.2f}%")
    
    # Return results
    return {
        'matches': top_matches,
        'avg_similarity': avg_similarity
    }

def extract_text_from_file(file_path: str) -> str:
    """Extract text from various document formats (doc, docx, pdf)."""
    import PyPDF2
    import docx2txt
    import textract
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == '.pdf':
            text = ""
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    page_text = reader.pages[page_num].extract_text()
                    if page_text:
                        text += page_text + " "  # Use space instead of newline
            
            # If PyPDF2 fails to extract meaningful text, try textract as backup
            if not text.strip():
                logging.info(f"PyPDF2 failed to extract text from {file_path}, trying textract...")
                text = textract.process(file_path).decode('utf-8')
                text = text.replace('\n', ' ')  # Replace newlines with spaces
                
        elif file_ext in ['.docx', '.doc', '.docs']:
            # Try docx2txt first (for .docx)
            text = docx2txt.process(file_path)
            text = text.replace('\n', ' ')  # Replace newlines with spaces
            
            # If docx2txt fails to extract meaningful text, try textract as backup
            if not text.strip():
                logging.info(f"docx2txt failed to extract text from {file_path}, trying textract...")
                text = textract.process(file_path).decode('utf-8')
                text = text.replace('\n', ' ')  # Replace newlines with spaces
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        # Clean the extracted text
        text = clean_text(text)
            
        # Debug: Print a hash of the extracted text to verify it's different
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        logging.info(f"Extracted text from {file_path} (hash: {text_hash[:8]}...)")
        
        # Debug: Print the first 200 characters of the text
        preview = text[:200].replace('\n', ' ').strip()
        logging.info(f"Text preview: {preview}...")
        
        return text
        
    except Exception as e:
        logging.error(f"Error extracting text from {file_path}: {str(e)}")
        logging.error(traceback.format_exc())
        raise

def generate_visualization(file_path, results):
    """Generate visualization for results."""
    if 'error' in results:
        return None
    
    # Get the base name of the file
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Check if this is a temporary file (starts with 'tmp')
    if base_name.startswith('tmp'):
        # Use a generic title instead of the temp filename
        title = 'Resume Analysis - Top Job Matches'
    else:
        title = f'Top Matches for {base_name}'
    
    # Create a bar chart of top matches
    plt.figure(figsize=(12, 8))
    
    # Get top matches
    titles = results['matches']['title'].tolist()
    scores = results['matches']['similarity_percentage'].tolist()
    
    # Shorten long titles for display
    display_titles = [t[:30] + '...' if len(t) > 30 else t for t in titles]
    
    # Plot
    plt.barh(range(len(display_titles)), scores, color='#2ecc71')
    plt.yticks(range(len(display_titles)), display_titles)
    plt.xlabel('Similarity Score (%)')
    plt.title(title)  # Use the cleaned title here
    plt.xlim(0, 100)
    
    # Add score labels
    for i, score in enumerate(scores):
        plt.text(score + 1, i, f'{score:.2f}%', va='center')
    
    plt.tight_layout()
    
    # Save the chart with a meaningful name regardless of input filename
    if base_name.startswith('tmp'):
        output_filename = "resume_matches.png"
    else:
        output_filename = f"{base_name}_matches.png"
    
    # Create a temporary file to save the visualization
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
        chart_file = temp_file.name
        plt.savefig(chart_file)
        plt.close()
    
    logging.info(f"Generated visualization chart for {file_path}")
    
    return chart_file, output_filename

# API endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the API is running and the model is loaded"""
    global df, embeddings, tfidf_vectorizer, tfidf_matrix
    
    model_loaded = all([df is not None, 
                        embeddings is not None, 
                        tfidf_vectorizer is not None, 
                        tfidf_matrix is not None])
    
    return {
        "status": "ok",
        "version": "1.0.0",
        "model_loaded": model_loaded
    }

@app.post("/initialize", response_model=dict)
async def initialize_model(background_tasks: BackgroundTasks):
    """Initialize the model by loading precomputed data"""
    global df, embeddings, tfidf_vectorizer, tfidf_matrix
    
    if all([df is not None, embeddings is not None, tfidf_vectorizer is not None, tfidf_matrix is not None]):
        return {"message": "Model already initialized"}
    
    success = load_precomputed_data()
    
    if success:
        return {"message": "Model initialized successfully"}
    else:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to initialize model. Check server logs for details."}
        )

@app.post("/scan/text", response_model=ScanResponse)
async def scan_text(text: str = Form(...), top_n: int = Form(5)):
    """Scan resume text and match against job titles"""
    global df, embeddings, tfidf_vectorizer, tfidf_matrix
    
    if not all([df is not None, embeddings is not None, tfidf_vectorizer is not None, tfidf_matrix is not None]):
        # Try to initialize the model
        success = load_precomputed_data()
        if not success:
            raise HTTPException(
                status_code=503, 
                detail="Model not initialized. Call /initialize first or check server logs for details."
            )
    
    try:
        # Match the text against job titles
        results = match_text(text, top_n=top_n)
        
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
    global df, embeddings, tfidf_vectorizer, tfidf_matrix
    
    if not all([df is not None, embeddings is not None, tfidf_vectorizer is not None, tfidf_matrix is not None]):
        # Try to initialize the model
        success = load_precomputed_data()
        if not success:
            raise HTTPException(
                status_code=503, 
                detail="Model not initialized. Call /initialize first or check server logs for details."
            )
    
    # Check file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ['.pdf', '.docx', '.doc']:
        raise HTTPException(
            status_code=400, 
            detail="Unsupported file format. Please upload PDF or Word documents."
        )
    
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
    temp_file_path = temp_file.name
    
    try:
        # Save uploaded file to temp file
        with temp_file:
            shutil.copyfileobj(file.file, temp_file)
        
        # Extract text from file
        resume_text = extract_text_from_file(temp_file_path)
        
        # Match the text against job titles
        results = match_text(resume_text, top_n=top_n, file_name=file.filename)
        
        # Generate visualization
        visualization_file, _ = generate_visualization(file.filename, results)
        
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
async def download_visualization(filename: str):
    """Download the visualization PNG file for a specific resume"""
    global results_dir
    
    if not results_dir:
        raise HTTPException(
            status_code=404, 
            detail="Results directory not set. Process a resume first."
        )
    
    try:
        # Get the base name of the CV file (without extension)
        base_name = os.path.splitext(filename)[0]
        
        # Look for the visualization file
        visualization_file = os.path.join(results_dir, f"{base_name}_matches.png")
        
        if not os.path.exists(visualization_file):
            # Try to find any PNG file with a similar name
            png_files = glob.glob(os.path.join(results_dir, f"{base_name}*.png"))
            
            if not png_files:
                raise HTTPException(
                    status_code=404, 
                    detail="No visualization file found for this resume."
                )
            
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

# Startup event to initialize the model
@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    global df, embeddings, tfidf_vectorizer, tfidf_matrix
    
    # Check if input directories exist
    for directory in [INPUT_DIR, OUTPUT_DIR, CV_DIR]:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created directory: {directory}")
    
    # Try to load precomputed data
    try:
        success = load_precomputed_data()
        if success:
            logging.info("Successfully loaded precomputed data on startup")
        else:
            logging.warning("Failed to load precomputed data on startup")
            logging.warning("Please call /initialize endpoint manually after ensuring data files are available")
    except Exception as e:
        logging.error(f"Error loading precomputed data on startup: {str(e)}")
        logging.error(traceback.format_exc())
        logging.warning("Please call /initialize endpoint manually after ensuring data files are available")

# Shutdown event to clean up resources
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    global df, embeddings, tfidf_vectorizer, tfidf_matrix
    
    # Clear variables
    df = None
    embeddings = None
    tfidf_vectorizer = None
    tfidf_matrix = None
    
    # Force garbage collection
    import gc
    gc.collect()
    
    logging.info("Resources cleaned up")

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("optimized_api:app", host="0.0.0.0", port=8000, reload=True)