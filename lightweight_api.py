from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from pickle_fix import safe_load_pickle
import os
import tempfile
import shutil
import logging
import time
import json
import traceback
import glob
import pickle
import numpy as np
from pathlib import Path
import sys
import re
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import docx2txt
import PyPDF2
import textract

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
RESULTS_DIR = OUTPUT_DIR / f"api_results_{int(time.time())}"

# Global variables
df = None
embeddings = None
tfidf_vectorizer = None
tfidf_matrix = None
current_cv_file = None
current_results_dir = str(RESULTS_DIR)

# Create necessary directories
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR / "evaluation", exist_ok=True)

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

# Utility functions
# Then replace the load_from_pickle function with:
def load_from_pickle(file_path: str) -> Any:
    """Load an object from a pickle file with version compatibility handling."""
    return safe_load_pickle(file_path)

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

def clean_text(text):
    """Clean text by removing quotes and exclamation marks."""
    if not text:
        return ""
    
    # Convert to string if not already
    text = str(text)
    
    # Remove single quotes, double quotes, and exclamation marks
    text = text.replace("'", "").replace('"', "").replace('!', "")
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def clean_standardize_title(title):
    """Clean and standardize job titles."""
    if not title:
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
    
    # Handle commas in titles - typically keep only the main role before the comma
    if ',' in title:
        title = title.split(',')[0].strip()
    
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
    
    for word in words:
        word_lower = word.lower()
        
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

def normalize_job_titles(matches_df):
    """Normalize job titles in the matches dataframe."""
    import pandas as pd
    
    if 'title' not in matches_df.columns:
        return matches_df
    
    # Clean up titles first
    matches_df['title'] = matches_df['title'].apply(clean_standardize_title)
    
    # Add a normalized title column for comparison
    matches_df['normalized_title'] = matches_df['title'].apply(lambda x: x.lower())
    
    # Group by normalized title and keep the one with highest similarity
    result = matches_df.loc[matches_df.groupby('normalized_title')['similarity_percentage'].idxmax()]
    
    # Drop the normalized_title column as it's no longer needed
    result = result.drop(columns=['normalized_title'])
    
    # Sort by similarity percentage in descending order
    result = result.sort_values('similarity_percentage', ascending=False)
    
    return result

def extract_text_from_file(file_path: str) -> str:
    """Extract text from various document formats (doc, docx, pdf)."""
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == '.pdf':
            text = _extract_from_pdf(file_path)
        elif file_ext in ['.docx', '.doc', '.docs']:
            text = _extract_from_doc(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        # Clean the extracted text
        text = clean_text(text)
        
        return text
        
    except Exception as e:
        logging.error(f"Error extracting text from {file_path}: {str(e)}")
        logging.error(traceback.format_exc())
        raise

def _extract_from_pdf(file_path: str) -> str:
    """Extract text from PDF file."""
    text = ""
    try:
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
            
    except Exception as e:
        logging.error(f"Error in PDF extraction: {str(e)}")
        logging.error(traceback.format_exc())
        # Try textract as a fallback
        try:
            text = textract.process(file_path).decode('utf-8')
            text = text.replace('\n', ' ')  # Replace newlines with spaces
        except Exception as e2:
            logging.error(f"Textract also failed: {str(e2)}")
            raise
            
    return text

def _extract_from_doc(file_path: str) -> str:
    """Extract text from DOC/DOCX file."""
    try:
        # Try docx2txt first (for .docx)
        text = docx2txt.process(file_path)
        text = text.replace('\n', ' ')  # Replace newlines with spaces
    except Exception as e:
        logging.error(f"docx2txt failed: {str(e)}")
        # Fall back to textract (handles .doc and other formats)
        try:
            text = textract.process(file_path).decode('utf-8')
            text = text.replace('\n', ' ')  # Replace newlines with spaces
        except Exception as e2:
            logging.error(f"Textract also failed: {str(e2)}")
            raise
    
    return text

def apply_tfidf_weighting(text, tfidf_vectorizer, tfidf_feature_names):
    """Apply TF-IDF weighting to text before embedding."""
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

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Split text into chunks with overlap to handle long texts that exceed token limits."""
    # Simple approximation: 4 characters per token on average
    chars_per_token = 4
    chunk_length = chunk_size * chars_per_token
    overlap_length = overlap * chars_per_token
    
    if len(text) <= chunk_length:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_length
        
        # If this is not the last chunk, try to break at word boundary
        if end < len(text):
            # Look for the last space within the chunk
            last_space = text.rfind(' ', start, end)
            if last_space > start:
                end = last_space
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = max(start + 1, end - overlap_length)
        
        # Prevent infinite loop
        if start >= len(text):
            break
    
    logging.info(f"Split text into {len(chunks)} chunks (chunk_size={chunk_size}, overlap={overlap})")
    return chunks

def match_text(text: str, top_n: int = 5) -> Dict:
    """Match text against TF-IDF weighted BGE embeddings."""
    global df, embeddings, tfidf_vectorizer, tfidf_matrix
    
    logging.info(f"Matching text against TF-IDF weighted BGE embeddings...")
    
    # Clean the input text
    text = clean_text(text)
    
    # Apply TF-IDF weighting to the query text
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    weighted_text = apply_tfidf_weighting(text, tfidf_vectorizer, tfidf_feature_names)
    
    # For the lightweight API, we'll use the average of the embeddings of the chunks
    # This is a simplified approach that doesn't require the full model
    chunks = chunk_text(weighted_text)
    
    # Get the TF-IDF vectors for each chunk
    chunk_tfidf_vectors = tfidf_vectorizer.transform(chunks)
    
    # Calculate the average TF-IDF vector
    if scipy.sparse.issparse(chunk_tfidf_vectors):
        avg_tfidf_vector = chunk_tfidf_vectors.mean(axis=0)
    else:
        avg_tfidf_vector = np.mean(chunk_tfidf_vectors, axis=0)
    
    # Calculate similarities using the TF-IDF vectors
    # This is a simplified approach that doesn't use the embeddings directly
    similarities = cosine_similarity(avg_tfidf_vector, tfidf_matrix)[0]
    
    # Get more candidates than needed to allow for deduplication
    top_indices = similarities.argsort()[-(top_n*3):][::-1]  # Get 3x needed to filter duplicates
    top_similarities = similarities[top_indices]
    
    # Get the corresponding rows from the dataframe
    import pandas as pd
    top_matches = df.iloc[top_indices].copy()
    top_matches['similarity_percentage'] = (top_similarities * 100).round(2)
    
    # Apply proper title formatting to ensure consistent display
    top_matches['title'] = top_matches['title'].apply(lambda x: clean_standardize_title(x))
    
    # Normalize and deduplicate job titles
    top_matches = normalize_job_titles(top_matches)
    
    # Take only the top N after deduplication
    top_matches = top_matches.head(top_n)
    
    # Calculate average similarity score
    avg_similarity = top_matches['similarity_percentage'].mean()
    
    logging.info(f"Average similarity score: {avg_similarity:.2f}%")
    
    # Return results
    return {
        'matches': top_matches,
        'avg_similarity': avg_similarity
    }

def process_resume_file(file_path: str, top_n: int = 5) -> Dict:
    """Process a resume file and match it against the database."""
    global current_cv_file, current_results_dir
    
    try:
        # Extract text from file
        resume_text = extract_text_from_file(file_path)
        logging.info(f"Extracted {len(resume_text)} characters from {file_path}")
        
        # Get base filename for output naming
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Save extracted text to file
        text_file = os.path.join(current_results_dir, f"{base_name}_extracted_text.txt")
        
        try:
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(resume_text)
            logging.info(f"Saved extracted text to {text_file}")
        except Exception as e:
            logging.error(f"Failed to save extracted text to {text_file}: {str(e)}")
        
        # Match against job titles
        results = match_text(resume_text, top_n)
        
        # Generate visualizations
        _generate_visualizations(file_path, results)
        
        return results
        
    except Exception as e:
        logging.error(f"Error processing resume file {file_path}: {str(e)}")
        logging.error(traceback.format_exc())
        return {
            'error': str(e),
            'file_path': file_path
        }

def _generate_visualizations(file_path, results):
    """Generate visualizations for results."""
    if 'error' in results:
        return
    
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
    
    chart_file = os.path.join(current_results_dir, output_filename)
    plt.savefig(chart_file)
    plt.close()
    
    logging.info(f"Generated visualization chart for {file_path}")

def load_precomputed_data():
    """Load precomputed data from pickle files."""
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

# API endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the API is running and the model is loaded"""
    return {
        "status": "ok",
        "version": "1.0.0",
        "model_loaded": True,  # We don't need the full model
        "preloaded_data": all([df is not None, embeddings is not None, 
                              tfidf_vectorizer is not None, tfidf_matrix is not None])
    }

@app.post("/initialize", response_model=dict)
async def initialize_model():
    """Initialize with preloaded data"""
    global df, embeddings, tfidf_vectorizer, tfidf_matrix
    
    if all([df is not None, embeddings is not None, tfidf_vectorizer is not None, tfidf_matrix is not None]):
        return {"message": "Data already loaded"}
    
    # Load precomputed data
    if load_precomputed_data():
        return {"message": "Data loaded successfully"}
    else:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to load precomputed data. Check server logs for details."}
        )

@app.post("/scan/text", response_model=ScanResponse)
async def scan_text(text: str = Form(...), top_n: int = Form(5)):
    """Scan resume text and match against job titles"""
    global df, embeddings, tfidf_vectorizer, tfidf_matrix
    
    if not all([df is not None, embeddings is not None, tfidf_vectorizer is not None, tfidf_matrix is not None]):
        if not load_precomputed_data():
            raise HTTPException(status_code=503, detail="Data not loaded. Call /initialize first.")
    
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
    global df, embeddings, tfidf_vectorizer, tfidf_matrix, current_cv_file, current_results_dir
    
    if not all([df is not None, embeddings is not None, tfidf_vectorizer is not None, tfidf_matrix is not None]):
        if not load_precomputed_data():
            raise HTTPException(status_code=503, detail="Data not loaded. Call /initialize first.")
    
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
        results = process_resume_file(temp_file_path, top_n=top_n)
        
        # Store the current CV file name for later use
        current_cv_file = file.filename
        
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
            "results_dir": current_results_dir,
            "input_dir_exists": os.path.exists(INPUT_DIR),
            "output_dir_exists": os.path.exists(OUTPUT_DIR),
            "cv_dir_exists": os.path.exists(CV_DIR),
            "pickle_files": pickle_files,
            "tfidf_dirs": tfidf_dirs,
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
    
    # Try to load precomputed data
    if load_precomputed_data():
        logging.info("Precomputed data loaded successfully on startup")
    else:
        logging.warning("Failed to load precomputed data on startup")
        logging.warning("Please call /initialize endpoint manually after ensuring data files are available")

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("lightweight_api:app", host="0.0.0.0", port=8000, reload=True)