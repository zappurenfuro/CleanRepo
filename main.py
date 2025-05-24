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
import sys
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import docx2txt
import PyPDF2
import textract
import pickle
from functools import lru_cache
import scipy.sparse

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
current_cv_file = None
current_results_dir = None

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

# Helper functions from scp.py
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
    """Clean and standardize job titles according to specific requirements."""
    if not title or pd.isna(title):
        return "Unknown Title"
    
    # Convert to string and take only the first title if multiple exist
    title = str(title).split(';')[0].strip()
    
    # Dictionary of terms that should be preserved as-is (no splitting)
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
    
    # List of invalid prefixes to remove
    invalid_prefixes = ['fl', 'ft', 'pt', 'temp', 'contract', 'freelance']
    
    # Remove text after pipe or vertical bar
    if '|' in title:
        title = title.split('|')[0].strip()
    
    # Remove text after parentheses
    if '(' in title:
        title = title.split('(')[0].strip()

    # Replace "&" with "/"
    title = title.replace('&', ' / ')
    
    # Replace "and" with "/"
    title = re.sub(r'\band\b', ' / ', title, flags=re.IGNORECASE)
    
    # Remove numbers from the title
    title = re.sub(r'\b\d+\b', '', title)
    
    # Remove invalid prefixes at the beginning of the title
    prefix_pattern = r'^(' + '|'.join(invalid_prefixes) + r')\.?\s+'
    title = re.sub(prefix_pattern, '', title, flags=re.IGNORECASE)
    
    # Handle commas in titles - typically keep only the main role before the comma
    if ',' in title:
        title = title.split(',')[0].strip()
    
    # Remove seniority prefixes
    seniority_terms = ['senior', 'sr.', 'sr', 'junior', 'jr.', 'jr', 'lead', 'principal', 'staff', 'intern']
    seniority_pattern = r'^(' + '|'.join(seniority_terms) + r')\.?\s*'
    title = re.sub(seniority_pattern, '', title, flags=re.IGNORECASE)
    
    # First check for preserved compound terms and replace them with placeholders
    placeholder_map = {}
    for i, (term, replacement) in enumerate(preserved_terms.items()):
        placeholder = f"__PRESERVED_{i}__"
        pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
        if pattern.search(title.lower()):
            title = pattern.sub(placeholder, title.lower())
            placeholder_map[placeholder] = replacement
    
    # Remove hyphens in compound words
    title = re.sub(r'(\w+)-(\w+)', r'\1\2', title)
    title = re.sub(r'(\w+) -(\w+)', r'\1\2', title)
    title = re.sub(r'(\w+)- (\w+)', r'\1\2', title)
    title = re.sub(r'(\w+) - (\w+)', r'\1\2', title)
    
    # Standardize spacing around slashes
    title = re.sub(r'\s*/\s*', ' / ', title)
    
    # Handle special cases for common job titles
    title = re.sub(r'\bfront[\s-]*end\b', 'Frontend', title, flags=re.IGNORECASE)
    title = re.sub(r'\bback[\s-]*end\b', 'Backend', title, flags=re.IGNORECASE)
    title = re.sub(r'\bfull[\s-]*stack\b', 'Fullstack', title, flags=re.IGNORECASE)
    
    # Fix run-together words
    title = re.sub(r'([a-z])([A-Z])', r'\1 \2', title)
    
    # Split into words for processing
    words = title.split()
    processed_words = []
    
    skip_next = False
    for i, word in enumerate(words):
        if skip_next:
            skip_next = False
            continue
            
        word_lower = word.lower()
        
        # Skip seniority terms that might appear in the middle of the title
        if word_lower in seniority_terms:
            continue
        
        # Handle special cases like "Front End" vs "Frontend"
        if i < len(words) - 1:
            combined = word_lower + words[i+1].lower()
            if combined in ['frontend', 'backend', 'fullstack']:
                processed_words.append(combined.capitalize())
                skip_next = True
                continue
        
        # Check if it's a special term that should be fully capitalized
        if word_lower in special_terms:
            processed_words.append(special_terms[word_lower])
        else:
            # Check if this is a placeholder for a preserved term
            if word in placeholder_map:
                processed_words.append(placeholder_map[word])
            else:
                # Capitalize the first letter of each word
                processed_words.append(word.capitalize())
    
    # Join the words back together
    processed_title = ' '.join(processed_words)
    
    # Remove any double spaces
    processed_title = re.sub(r'\s+', ' ', processed_title).strip()
    
    # Restore any preserved terms that might have been missed
    for placeholder, replacement in placeholder_map.items():
        processed_title = processed_title.replace(placeholder, replacement)
    
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
    
    # Standardize common variations
    replacements = {
        'frontend': 'frontend',
        'front': 'frontend',
        'frontenddev': 'frontend developer',
        'backend': 'backend',
        'back': 'backend',
        'fullstack': 'fullstack',
        'full': 'fullstack',
        'ui': 'user interface',
        'ux': 'user experience',
        'dev': 'developer',
        'developer': 'developer',
        'engineer': 'engineer',
        'eng': 'engineer',
        'engg': 'engineering',
        'architect': 'architect',
        'arch': 'architect',
        'consultant': 'consultant',
        'cons': 'consultant'
    }
    
    normalized_words = []
    for word in title_words:
        if word in replacements:
            normalized_words.append(replacements[word])
        else:
            normalized_words.append(word)
    
    # Join words back together
    return ' '.join(normalized_words)

def normalize_job_titles(matches_df):
    """Normalize job titles in the matches dataframe to group similar titles."""
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

def load_from_pickle(file_path: str) -> Any:
    """Load an object from a pickle file."""
    try:
        if not os.path.exists(file_path):
            logging.error(f"Pickle file does not exist: {file_path}")
            return None
        
        # Import our custom unpickler
        from pickle_compat import safe_load_pickle
        
        # Try loading with custom unpickler first
        logging.info(f"Attempting to load pickle file with custom unpickler: {file_path}")
        loaded_data = safe_load_pickle(file_path)
        
        if loaded_data is None:
            # Fall back to standard pickle if custom unpickler fails
            logging.info(f"Custom unpickler failed, trying standard pickle: {file_path}")
            # Check if file is compressed (gzip)
            is_gzipped = False
            try:
                with open(file_path, 'rb') as f:
                    magic_number = f.read(2)
                    if magic_number == b'\x1f\x8b':  # gzip magic number
                        is_gzipped = True
            except:
                pass
            
            # Load the file
            if is_gzipped:
                import gzip
                with gzip.open(file_path, 'rb') as f:
                    loaded_data = pickle.load(f)
            else:
                with open(file_path, 'rb') as f:
                    loaded_data = pickle.load(f)
        
        # Check if the loaded data has metadata
        if isinstance(loaded_data, dict) and 'data' in loaded_data and 'metadata' in loaded_data:
            logging.info(f"Loaded pickle file with metadata: {file_path}")
            # Log metadata if needed
            if 'created_at' in loaded_data:
                logging.info(f"File created at: {loaded_data['created_at']}")
            if 'version' in loaded_data:
                logging.info(f"File version: {loaded_data['version']}")
            return loaded_data['data']
        else:
            logging.info(f"Loaded pickle file: {file_path}")
            return loaded_data
            
    except Exception as e:
        logging.error(f"Error loading pickle file {file_path}: {str(e)}")
        logging.error(traceback.format_exc())
        return None

def find_pickle_files():
    """Find the pickle files in the output directory with recursive search."""
    global OUTPUT_DIR, results_dir
    
    logging.info(f"Searching for pickle files in {OUTPUT_DIR}")
    
    # Find processed_resumes.pkl with recursive search
    processed_resumes_files = list(Path(OUTPUT_DIR).glob("**/processed_resumes.pkl"))
    if not processed_resumes_files:
        logging.error(f"Processed resumes file not found anywhere in {OUTPUT_DIR}")
        return None, None, None, None
    
    # Use the first one found
    processed_resumes_file = processed_resumes_files[0]
    logging.info(f"Found processed_resumes.pkl at: {processed_resumes_file}")
    
    # Find tfidf_enhanced directories with recursive search
    tfidf_dirs = []
    for item in Path(OUTPUT_DIR).glob("**/tfidf_enhanced_*"):
        if item.is_dir():
            tfidf_dirs.append(item)
    
    if not tfidf_dirs:
        logging.error(f"No tfidf_enhanced directories found anywhere in {OUTPUT_DIR}")
        return None, None, None, None
    
    # Sort by modification time (most recent first)
    tfidf_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    most_recent_dir = tfidf_dirs[0]
    results_dir = most_recent_dir
    logging.info(f"Using most recent directory: {most_recent_dir}")
    
    # Find pickle files
    embeddings_file = most_recent_dir / "tfidf_bge_embeddings.pkl"
    tfidf_vectorizer_file = most_recent_dir / "tfidf_vectorizer.pkl"
    tfidf_matrix_file = most_recent_dir / "tfidf_matrix.pkl"
    
    # Check if files exist
    files_exist = True
    if not embeddings_file.exists():
        logging.error(f"Embeddings file not found: {embeddings_file}")
        files_exist = False
    else:
        logging.info(f"Found embeddings file: {embeddings_file}")
    
    if not tfidf_vectorizer_file.exists():
        logging.error(f"TF-IDF vectorizer file not found: {tfidf_vectorizer_file}")
        files_exist = False
    else:
        logging.info(f"Found TF-IDF vectorizer file: {tfidf_vectorizer_file}")
    
    if not tfidf_matrix_file.exists():
        logging.error(f"TF-IDF matrix file not found: {tfidf_matrix_file}")
        files_exist = False
    else:
        logging.info(f"Found TF-IDF matrix file: {tfidf_matrix_file}")
    
    if not files_exist:
        return None, None, None, None
    
    return processed_resumes_file, embeddings_file, tfidf_vectorizer_file, tfidf_matrix_file

def load_precomputed_data():
    """Load precomputed data from pickle files."""
    global df, embeddings, tfidf_vectorizer, tfidf_matrix, tfidf_feature_names, current_results_dir
    
    logging.info("Loading precomputed data from pickle files...")
    
    # Find pickle files
    processed_resumes_file, embeddings_file, tfidf_vectorizer_file, tfidf_matrix_file = find_pickle_files()
    
    if not all([processed_resumes_file, embeddings_file, tfidf_vectorizer_file, tfidf_matrix_file]):
        logging.error("One or more required pickle files not found")
        return False
    
    # Load processed resumes
    df = load_from_pickle(str(processed_resumes_file))
    if df is None:
        logging.error("Failed to load processed resumes")
        return False
    
    # Load embeddings
    embeddings = load_from_pickle(str(embeddings_file))
    if embeddings is None:
        logging.error("Failed to load embeddings")
        return False
    
    # Load TF-IDF vectorizer
    tfidf_vectorizer = load_from_pickle(str(tfidf_vectorizer_file))
    if tfidf_vectorizer is None:
        logging.error("Failed to load TF-IDF vectorizer")
        return False
    
    # Load TF-IDF matrix
    tfidf_matrix = load_from_pickle(str(tfidf_matrix_file))
    if tfidf_matrix is None:
        logging.error("Failed to load TF-IDF matrix")
        return False
    
    # Get feature names from vectorizer
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    
    # Set current results directory
    current_results_dir = str(results_dir)
    
    logging.info(f"Successfully loaded all precomputed data")
    logging.info(f"DataFrame shape: {df.shape}")
    logging.info(f"Embeddings shape: {embeddings.shape}")
    logging.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    logging.info(f"Number of TF-IDF features: {len(tfidf_feature_names)}")
    
    return True

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
    weighted_text = text
    for term, score in zip(top_terms, top_scores):
        # Normalize score to a reasonable number of repetitions (1-5)
        repetitions = min(5, max(1, int(score * 10)))
        weighted_text += f" {term * repetitions}"
    
    return weighted_text

def match_text(text: str, top_n: int = 5) -> Dict:
    """Match text against TF-IDF weighted BGE embeddings."""
    global df, embeddings, tfidf_vectorizer, tfidf_matrix
    
    logging.info(f"Matching text against TF-IDF weighted BGE embeddings...")
    
    # Clean the input text
    text = clean_text(text)
    
    # Apply TF-IDF weighting to the query text
    weighted_text = apply_tfidf_weighting(text)
    
    # For now, we'll use TF-IDF similarity directly
    # Transform the weighted text using the TF-IDF vectorizer
    text_tfidf = tfidf_vectorizer.transform([weighted_text])
    
    # Calculate similarities using the TF-IDF vectors
    similarities = cosine_similarity(text_tfidf, tfidf_matrix)[0]
    
    # Get more candidates than needed to allow for deduplication
    top_indices = similarities.argsort()[-(top_n*3):][::-1]
    top_similarities = similarities[top_indices]
    
    # Get the corresponding rows from the dataframe
    top_matches = df.iloc[top_indices].copy()
    top_matches['similarity_percentage'] = (top_similarities * 100).round(2)
    
    # Apply proper title formatting
    top_matches['title'] = top_matches['title'].apply(lambda x: clean_standardize_title(x))
    
    # Normalize and deduplicate job titles
    top_matches = normalize_job_titles(top_matches)
    
    # Take only the top N after deduplication
    top_matches = top_matches.head(top_n)
    
    # Calculate average similarity score
    avg_similarity = top_matches['similarity_percentage'].mean()
    
    logging.info(f"Average similarity score: {avg_similarity:.2f}%")
    
    return {
        'matches': top_matches,
        'avg_similarity': avg_similarity
    }

def extract_text_from_file(file_path: str) -> str:
    """Extract text from various document formats."""
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == '.pdf':
            text = ""
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    page_text = reader.pages[page_num].extract_text()
                    if page_text:
                        text += page_text + " "
            
            if not text.strip():
                text = textract.process(file_path).decode('utf-8')
                
        elif file_ext in ['.docx', '.doc']:
            try:
                text = docx2txt.process(file_path)
            except:
                text = textract.process(file_path).decode('utf-8')
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        return clean_text(text)
        
    except Exception as e:
        logging.error(f"Error extracting text from {file_path}: {str(e)}")
        raise

# API endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the API is running and the model is loaded"""
    global df, embeddings, tfidf_vectorizer, tfidf_matrix
    
    model_loaded = all([
        df is not None,
        embeddings is not None,
        tfidf_vectorizer is not None,
        tfidf_matrix is not None
    ])
    
    return {
        "status": "ok",
        "version": "1.0.0",
        "model_loaded": model_loaded
    }

@app.post("/initialize")
async def initialize_model():
    """Initialize the model by loading precomputed data"""
    global df, embeddings, tfidf_vectorizer, tfidf_matrix
    
    if all([df is not None, embeddings is not None, tfidf_vectorizer is not None, tfidf_matrix is not None]):
        return {"message": "Model already initialized"}
    
    success = load_precomputed_data()
    
    if success:
        return {"message": "Model initialized successfully"}
    else:
        raise HTTPException(
            status_code=500,
            detail="Failed to initialize model. Check server logs for details."
        )

@app.post("/scan/text", response_model=ScanResponse)
async def scan_text(text: str = Form(...), top_n: int = Form(5)):
    """Scan resume text and match against job titles"""
    global df, embeddings, tfidf_vectorizer, tfidf_matrix
    
    if not all([df is not None, embeddings is not None, tfidf_vectorizer is not None, tfidf_matrix is not None]):
        success = load_precomputed_data()
        if not success:
            raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        results = match_text(text, top_n=top_n)
        
        matches = []
        for _, row in results['matches'].iterrows():
            matches.append({
                "title": row['title'],
                "similarity_percentage": row['similarity_percentage'],
                "embedding_text": row.get('embedding_text', '')
            })
        
        return {
            "matches": matches,
            "avg_similarity": results['avg_similarity']
        }
    except Exception as e:
        logging.error(f"Error scanning text: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/scan/file", response_model=ScanResponse)
async def scan_file(file: UploadFile = File(...), top_n: int = Form(5)):
    """Scan resume file and match against job titles"""
    global df, embeddings, tfidf_vectorizer, tfidf_matrix, current_cv_file
    
    if not all([df is not None, embeddings is not None, tfidf_vectorizer is not None, tfidf_matrix is not None]):
        success = load_precomputed_data()
        if not success:
            raise HTTPException(status_code=503, detail="Model not initialized")
    
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ['.pdf', '.docx', '.doc']:
        raise HTTPException(status_code=400, detail="Unsupported file format")
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
    temp_file_path = temp_file.name
    
    try:
        with temp_file:
            shutil.copyfileobj(file.file, temp_file)
        
        # Extract text
        resume_text = extract_text_from_file(temp_file_path)
        
        # Match against job titles
        results = match_text(resume_text, top_n=top_n)
        
        current_cv_file = file.filename
        
        matches = []
        for _, row in results['matches'].iterrows():
            matches.append({
                "title": row['title'],
                "similarity_percentage": row['similarity_percentage'],
                "embedding_text": row.get('embedding_text', '')
            })
        
        return {
            "matches": matches,
            "avg_similarity": results['avg_similarity']
        }
    except Exception as e:
        logging.error(f"Error scanning file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

@app.get("/debug/pickle-files")
async def debug_pickle_files():
    """Debug endpoint to check pickle files and their loading status"""
    import glob
    
    # Find all pickle files
    pickle_files = glob.glob(str(OUTPUT_DIR) + "/**/*.pkl", recursive=True)
    
    results = []
    for file_path in pickle_files:
        try:
            # Try to load the file
            with open(file_path, 'rb') as f:
                # Just read the first few bytes to check if it's a valid pickle
                header = f.read(10)
                is_valid_pickle = header.startswith(b'\x80\x03') or header.startswith(b'\x80\x04') or header.startswith(b'\x80\x05')
            
            # Get file size and modification time
            file_stat = os.stat(file_path)
            file_size = file_stat.st_size
            mod_time = time.ctime(file_stat.st_mtime)
            
            results.append({
                "file_path": file_path,
                "file_size": file_size,
                "modified": mod_time,
                "is_valid_pickle": is_valid_pickle,
                "status": "Found"
            })
        except Exception as e:
            results.append({
                "file_path": file_path,
                "error": str(e),
                "status": "Error"
            })
    
    # Also check the specific files we're looking for
    required_files = [
        str(OUTPUT_DIR / "processed_resumes.pkl"),
        str(OUTPUT_DIR / "tfidf_enhanced_1748065889/tfidf_bge_embeddings.pkl"),
        str(OUTPUT_DIR / "tfidf_enhanced_1748065889/tfidf_vectorizer.pkl"),
        str(OUTPUT_DIR / "tfidf_enhanced_1748065889/tfidf_matrix.pkl")
    ]
    
    for file_path in required_files:
        if file_path not in [r["file_path"] for r in results]:
            results.append({
                "file_path": file_path,
                "status": "Missing"
            })
    
    return {
        "pickle_files": results,
        "numpy_version": np.__version__,
        "python_version": sys.version,
        "output_dir": str(OUTPUT_DIR)
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    for directory in [INPUT_DIR, OUTPUT_DIR, CV_DIR]:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
    
    # Try to load data on startup
    if load_precomputed_data():
        logging.info("Data loaded successfully on startup")
    else:
        logging.warning("Failed to load data on startup")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
