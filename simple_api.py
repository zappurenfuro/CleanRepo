from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, ConfigDict
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
tfidf_vectorizer = None
tfidf_matrix = None
current_cv_file = None
current_results_dir = str(RESULTS_DIR)

# Create necessary directories
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR / "evaluation", exist_ok=True)

# Response models with proper configuration
class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    status: str
    version: str
    model_loaded: bool
    data_loaded: bool

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

def clean_standardize_title(title):
    """Clean and standardize job titles."""
    if not title or pd.isna(title):
        return "Unknown Title"
    
    # Convert to string and take only the first title if multiple exist
    title = str(title).split(';')[0].strip()
    
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
    
    # Handle commas in titles
    if ',' in title:
        title = title.split(',')[0].strip()
    
    # Capitalize words
    words = title.split()
    processed_words = []
    
    for word in words:
        processed_words.append(word.capitalize())
    
    # Join the words back together
    processed_title = ' '.join(processed_words)
    
    # Remove any double spaces
    processed_title = re.sub(r'\s+', ' ', processed_title).strip()
    
    return processed_title

def load_csv_data():
    """Load data directly from CSV files."""
    global df, tfidf_vectorizer, tfidf_matrix
    
    try:
        logging.info("Loading data from CSV files...")
        
        # Check if CSV files exist
        required_files = [
            '01_people.csv',
            '02_abilities.csv',
            '03_education.csv',
            '04_experience.csv',
            '05_person_skills.csv'
        ]
        
        missing_files = []
        for file in required_files:
            file_path = INPUT_DIR / file
            if not file_path.exists():
                missing_files.append(file)
        
        if missing_files:
            logging.error(f"Missing required CSV files: {', '.join(missing_files)}")
            return False
        
        # Load CSV files
        df1 = pd.read_csv(INPUT_DIR / '01_people.csv')
        df2 = pd.read_csv(INPUT_DIR / '02_abilities.csv')
        df3 = pd.read_csv(INPUT_DIR / '03_education.csv')
        df4 = pd.read_csv(INPUT_DIR / '04_experience.csv')
        df5 = pd.read_csv(INPUT_DIR / '05_person_skills.csv')
        
        # Filter by person_id
        df1 = df1[df1['person_id'] <= 54928]
        df2 = df2[df2['person_id'] <= 54928]
        df3 = df3[df3['person_id'] <= 54928]
        df4 = df4[df4['person_id'] <= 54928]
        df5 = df5[df5['person_id'] <= 54928]
        
        # Clean text in all dataframes
        logging.info("Cleaning text in all dataframes...")
        for df in [df1, df2, df3, df4, df5]:
            for col in df.select_dtypes(include=['object']).columns:
                if col != 'person_id':
                    df[col] = df[col].apply(clean_text)
        
        # Process title column
        if 'title' in df4.columns:
            df4['title'] = df4['title'].apply(lambda x: str(x).split(';')[0].strip() if pd.notna(x) else x)
        
        # Aggregate text by person
        logging.info("Aggregating text by person...")
        
        # For abilities
        df2_agg = df2.groupby('person_id').agg({
            'ability': lambda x: '; '.join(x.dropna().astype(str).unique())
        }).reset_index()
        
        # For education
        df3_agg = df3.groupby('person_id').agg({
            col: lambda x: '; '.join(x.dropna().astype(str).unique()) 
            for col in df3.columns if col != 'person_id'
        }).reset_index()
        
        # For experience
        df4_agg = df4.groupby('person_id').agg({
            col: lambda x: '; '.join(x.dropna().astype(str).unique()) 
            for col in df4.columns if col != 'person_id'
        }).reset_index()
        
        # For skills
        df5_agg = df5.groupby('person_id').agg({
            'skill': lambda x: '; '.join(x.dropna().astype(str).unique())
        }).reset_index()
        
        # Merge dataframes
        logging.info("Merging dataframes...")
        df = df1.merge(df2_agg, on='person_id', how='left')
        df = df.merge(df3_agg, on='person_id', how='left')
        df = df.merge(df4_agg, on='person_id', how='left')
        df = df.merge(df5_agg, on='person_id', how='left')
        
        # Clean and standardize job titles
        logging.info("Cleaning and standardizing job titles...")
        if 'title' in df.columns:
            df['title'] = df['title'].apply(clean_standardize_title)
        else:
            df['title'] = 'Unknown Title'
        
        # Fill missing values
        df['ability'] = df['ability'].fillna('Unknown ability')
        df['skill'] = df['skill'].fillna('Unknown skill')
        
        # Create text representation for TF-IDF
        df['embedding_text'] = df.apply(lambda row: " | ".join([
            clean_text(str(row.get('ability', ''))),
            clean_text(str(row.get('title', ''))) * 3,  # Repeat title 3 times for higher weight
            clean_text(str(row.get('skill', '')))
        ]), axis=1)
        
        # Create TF-IDF vectors
        logging.info("Creating TF-IDF vectors...")
        tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            min_df=5,
            max_df=0.85,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        corpus = df['embedding_text'].tolist()
        tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
        
        logging.info(f"Successfully loaded {len(df)} records")
        logging.info(f"Created TF-IDF matrix with shape {tfidf_matrix.shape}")
        
        return True
        
    except Exception as e:
        logging.error(f"Error loading CSV data: {str(e)}")
        logging.error(traceback.format_exc())
        return False

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

def match_text(text: str, top_n: int = 5) -> Dict:
    """Match text against TF-IDF vectors."""
    global df, tfidf_vectorizer, tfidf_matrix
    
    if df is None or tfidf_vectorizer is None or tfidf_matrix is None:
        raise ValueError("Data not loaded. Call load_csv_data() first.")
    
    # Clean the input text
    text = clean_text(text)
    
    # Transform the text to TF-IDF vector
    text_vector = tfidf_vectorizer.transform([text])
    
    # Calculate similarities
    similarities = cosine_similarity(text_vector, tfidf_matrix)[0]
    
    # Get top matches
    top_indices = similarities.argsort()[-(top_n*2):][::-1]
    top_similarities = similarities[top_indices]
    
    # Get the corresponding rows
    top_matches = df.iloc[top_indices].copy()
    top_matches['similarity_percentage'] = (top_similarities * 100).round(2)
    
    # Remove duplicates based on title
    top_matches = top_matches.drop_duplicates(subset=['title'], keep='first')
    
    # Take only the top N
    top_matches = top_matches.head(top_n)
    
    # Calculate average similarity
    avg_similarity = top_matches['similarity_percentage'].mean()
    
    return {
        'matches': top_matches,
        'avg_similarity': avg_similarity
    }

# API endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the API is running and data is loaded"""
    return {
        "status": "ok",
        "version": "1.0.0",
        "model_loaded": True,
        "data_loaded": df is not None
    }

@app.post("/initialize")
async def initialize_data():
    """Initialize by loading CSV data"""
    if df is not None:
        return {"message": "Data already loaded"}
    
    if load_csv_data():
        return {"message": "Data loaded successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to load data")

@app.post("/scan/text", response_model=ScanResponse)
async def scan_text(text: str = Form(...), top_n: int = Form(5)):
    """Scan resume text and match against job titles"""
    if df is None:
        if not load_csv_data():
            raise HTTPException(status_code=503, detail="Data not loaded")
    
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
    global current_cv_file
    
    if df is None:
        if not load_csv_data():
            raise HTTPException(status_code=503, detail="Data not loaded")
    
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

# Startup event
@app.on_event("startup")
async def startup_event():
    for directory in [INPUT_DIR, OUTPUT_DIR, CV_DIR, RESULTS_DIR]:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
    
    # Try to load data on startup
    if load_csv_data():
        logging.info("Data loaded successfully on startup")
    else:
        logging.warning("Failed to load data on startup")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("simple_api:app", host="0.0.0.0", port=8000, reload=True)