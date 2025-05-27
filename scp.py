#%%
import os
import sys
import numpy as np
import pandas as pd
import logging
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Tuple, Optional, Union, Any
import tempfile
import hashlib
import time
import gc
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import Dataset, DataLoader
import traceback
import re
from colorama import init, Fore, Style
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from collections import Counter
from functools import lru_cache
import datetime
import scipy.sparse
import pickle
import joblib

# Initialize colorama for colored terminal output
init(autoreset=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define ResumeDataset outside methods to make it picklable
class ResumeDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx]

def install_required_packages():
    """Auto-install required packages if missing"""
    required_packages = {
        'psutil': 'psutil',
        'umap-learn': 'umap',
        'docx2txt': 'docx2txt',
        'PyPDF2': 'PyPDF2',
        'textract': 'textract',
        'tqdm': 'tqdm',
        'colorama': 'colorama',
        'matplotlib': 'matplotlib',
        'scikit-learn': 'sklearn',
        'datasets': 'datasets',
        'accelerate': 'accelerate'
    }
    
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            logging.info(f"Installing {package_name}...")
            try:
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
                logging.info(f"Successfully installed {package_name}")
            except Exception as e:
                logging.error(f"Failed to install {package_name}: {str(e)}")
                if package_name in ['datasets', 'accelerate']:
                    logging.warning(f"Fine-tuning will be disabled due to missing '{package_name}' package")

# Install required packages
install_required_packages()

# Now import the packages
import psutil
import docx2txt
import PyPDF2
import textract

# Check if datasets is available for fine-tuning
try:
    import datasets
    FINE_TUNING_AVAILABLE = True
except ImportError:
    FINE_TUNING_AVAILABLE = False
    logging.warning("Fine-tuning functionality will be disabled (datasets package not available)")

def check_cuda_availability():
    """Check if CUDA is available and log information"""
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Using CPU.")

# Update the NumpyEncoder class to handle pandas DataFrames
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types and pandas DataFrames."""
    def default(self, obj):
        import pandas as pd
        
        if isinstance(obj, np.number):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, scipy.sparse.spmatrix):
            return {
                "_type": "sparse_matrix",
                "format": "csr" if isinstance(obj, scipy.sparse.csr_matrix) else "coo",
                "data": obj.data.tolist(),
                "indices": obj.indices.tolist(),
                "indptr": obj.indptr.tolist() if hasattr(obj, "indptr") else None,
                "shape": obj.shape
            }
        return json.JSONEncoder.default(self, obj)

def save_as_json(obj: Any, file_path: str, metadata: Optional[Dict] = None) -> bool:
    """
    Save any object as a JSON file with optional metadata.
    For scikit-learn objects, uses pickle/joblib instead.
    
    Args:
        obj: Object to save
        file_path: Path to save the JSON file
        metadata: Optional dictionary with metadata to include
        
    Returns:
        bool: True if saved successfully, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logging.info(f"Created directory: {directory}")
        
        # Check if object is from scikit-learn (like TfidfVectorizer)
        from sklearn.base import BaseEstimator
        if isinstance(obj, BaseEstimator):
            # For scikit-learn objects, use joblib which handles them efficiently
            pickle_path = file_path.replace('.json', '.pkl')
            joblib.dump(obj, pickle_path)
            
            # Save metadata separately as JSON if provided
            if metadata:
                meta_path = file_path.replace('.json', '_metadata.json')
                with open(meta_path, 'w') as f:
                    json.dump({
                        'metadata': metadata,
                        'created_at': str(datetime.datetime.now()),
                        'version': '1.0',
                        'pickle_file': os.path.basename(pickle_path)
                    }, f, indent=2)
                
            # Verify file was created
            if os.path.exists(pickle_path):
                file_size = os.path.getsize(pickle_path) / (1024 * 1024)  # Size in MB
                logging.info(f"Successfully saved scikit-learn object to pickle: {pickle_path} ({file_size:.2f} MB)")
                return True
            else:
                logging.error(f"Pickle file was not created: {pickle_path}")
                return False
        
        # Handle pandas DataFrame specifically
        import pandas as pd
        if isinstance(obj, pd.DataFrame):
            # For large DataFrames, convert to records format
            data_to_save = obj.to_dict(orient='records')
            if metadata:
                data_to_save = {
                    'data': data_to_save,
                    'metadata': metadata,
                    'created_at': str(datetime.datetime.now()),
                    'version': '1.0',
                    '_pandas_dataframe': True  # Flag to identify this was a DataFrame
                }
        else:
            # Prepare data with metadata if provided
            if metadata:
                data_to_save = {
                    'data': obj,
                    'metadata': metadata,
                    'created_at': str(datetime.datetime.now()),
                    'version': '1.0'
                }
            else:
                data_to_save = obj
        
        # Save as JSON
        with open(file_path, 'w') as f:
            json.dump(data_to_save, f, indent=2, cls=NumpyEncoder)
        
        # Verify file was created
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            logging.info(f"Successfully saved JSON file: {file_path} ({file_size:.2f} MB)")
            return True
        else:
            logging.error(f"JSON file was not created: {file_path}")
            return False
            
    except Exception as e:
        logging.error(f"Error saving JSON file {file_path}: {str(e)}")
        logging.error(traceback.format_exc())
        return False

def load_from_json(file_path: str) -> Any:
    """
    Load an object from a JSON file.
    For scikit-learn objects, loads from pickle/joblib instead.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        The loaded object or None if loading failed
    """
    try:
        # Check if this might be a scikit-learn object (look for pickle file)
        pickle_path = file_path.replace('.json', '.pkl')
        if os.path.exists(pickle_path):
            logging.info(f"Found pickle file for scikit-learn object: {pickle_path}")
            try:
                # Load the scikit-learn object using joblib
                obj = joblib.load(pickle_path)
                logging.info(f"Successfully loaded scikit-learn object from pickle")
                return obj
            except Exception as e:
                logging.error(f"Error loading pickle file {pickle_path}: {str(e)}")
                # Continue to try loading as JSON as fallback
        
        if not os.path.exists(file_path):
            # Check for metadata file
            meta_path = file_path.replace('.json', '_metadata.json')
            if os.path.exists(meta_path):
                logging.info(f"Found metadata file: {meta_path}")
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                if 'pickle_file' in metadata:
                    pickle_file = os.path.join(os.path.dirname(file_path), metadata['pickle_file'])
                    if os.path.exists(pickle_file):
                        obj = joblib.load(pickle_file)
                        logging.info(f"Successfully loaded scikit-learn object from pickle via metadata")
                        return obj
            
            logging.error(f"JSON file does not exist: {file_path}")
            return None
        
        # Load the file
        with open(file_path, 'r') as f:
            loaded_data = json.load(f)
        
        # Check if the loaded data has metadata
        if isinstance(loaded_data, dict) and 'data' in loaded_data and 'metadata' in loaded_data:
            logging.info(f"Loaded JSON file with metadata: {file_path}")
            # Log metadata if needed
            if 'created_at' in loaded_data:
                logging.info(f"File created at: {loaded_data['created_at']}")
            if 'version' in loaded_data:
                logging.info(f"File version: {loaded_data['version']}")
            
            # Check if this was originally a DataFrame
            if loaded_data.get('_pandas_dataframe', False):
                import pandas as pd
                return pd.DataFrame(loaded_data['data'])
            
            # Handle sparse matrices if present
            data = loaded_data['data']
            if isinstance(data, dict) and data.get("_type") == "sparse_matrix":
                format_type = data.get("format", "csr")
                if format_type == "csr":
                    matrix = scipy.sparse.csr_matrix(
                        (data["data"], data["indices"], data["indptr"]),
                        shape=data["shape"]
                    )
                else:  # coo format
                    matrix = scipy.sparse.coo_matrix(
                        (data["data"], (data["row"], data["col"])),
                        shape=data["shape"]
                    )
                return matrix
            
            return loaded_data['data']
        else:
            logging.info(f"Loaded JSON file: {file_path}")
            
            # Check if this is a sparse matrix
            if isinstance(loaded_data, dict) and loaded_data.get("_type") == "sparse_matrix":
                format_type = loaded_data.get("format", "csr")
                if format_type == "csr":
                    matrix = scipy.sparse.csr_matrix(
                        (loaded_data["data"], loaded_data["indices"], loaded_data["indptr"]),
                        shape=loaded_data["shape"]
                    )
                else:  # coo format
                    matrix = scipy.sparse.coo_matrix(
                        (loaded_data["data"], (loaded_data["row"], loaded_data["col"])),
                        shape=loaded_data["shape"]
                    )
                return matrix
            
            return loaded_data
            
    except Exception as e:
        logging.error(f"Error loading JSON file {file_path}: {str(e)}")
        logging.error(traceback.format_exc())
        return None
    
# Evaluation Metrics Class
class ResumeEvaluationMetrics:
    """
    Evaluation metrics for resume recommendation systems.
    Implements NDCG, Precision@k, Recall@k, MRR, MAP, ARI, NMI, and Purity.
    """
    
    def __init__(self):
        """Initialize the evaluation metrics calculator."""
        self.results = {}  # Store results by resume_id
        self.ground_truth = {}  # Store ground truth by resume_id
        self.metrics = {}  # Store calculated metrics
    
    def load_ground_truth(self, ground_truth_file: str) -> bool:
        """
        Load ground truth data from a CSV file.
        
        Args:
            ground_truth_file: Path to the ground truth CSV file
                Expected columns: resume_id, job_id, relevance
                
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            df = pd.read_csv(ground_truth_file)
            required_columns = ['resume_id', 'job_id', 'relevance']
            
            # Check if all required columns exist
            if not all(col in df.columns for col in required_columns):
                missing = [col for col in required_columns if col not in df.columns]
                print(f"Error: Ground truth file missing columns: {missing}")
                return False
            
            # Convert to dictionary for faster lookup
            for _, row in df.iterrows():
                resume_id = str(row['resume_id'])
                job_id = str(row['job_id'])
                relevance = float(row['relevance'])
                
                if resume_id not in self.ground_truth:
                    self.ground_truth[resume_id] = {}
                
                self.ground_truth[resume_id][job_id] = relevance
            
            print(f"✅ Loaded ground truth data for {len(self.ground_truth)} resumes")
            return True
            
        except Exception as e:
            print(f"❌ Error loading ground truth data: {str(e)}")
            return False
    
    def add_result(self, resume_id: str, matches: List[Dict[str, Any]]) -> None:
        """
        Add a matching result for evaluation.
        
        Args:
            resume_id: ID of the resume
            matches: List of dictionaries with job_id and similarity_score
                Each dict should have at least: {'job_id': str, 'similarity_score': float}
        """
        self.results[str(resume_id)] = matches
    
    def calculate_ndcg(self, k: int = 10) -> Dict[str, float]:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG) at k.
        
        Args:
            k: Number of top results to consider
            
        Returns:
            Dictionary with NDCG scores by resume_id and average
        """
        if not self.results or not self.ground_truth:
            print("❌ No results or ground truth data available for NDCG calculation")
            return {'average': 0.0}
        
        ndcg_scores = {}
        valid_resumes = 0
        
        for resume_id, matches in self.results.items():
            if resume_id not in self.ground_truth:
                continue
            
            # Get top k matches
            top_k_matches = matches[:k]
            
            # Create relevance array for DCG calculation
            relevance = []
            for match in top_k_matches:
                job_id = str(match['job_id'])
                # Find relevance in ground truth (default to 0 if not found)
                rel = self.ground_truth[resume_id].get(job_id, 0.0)
                relevance.append(rel)
            
            # Calculate DCG
            dcg = 0.0
            for i, rel in enumerate(relevance):
                # Use i+2 because log2(1) = 0
                dcg += rel / np.log2(i + 2)
            
            # Calculate ideal DCG (IDCG)
            # Get all relevance scores for this resume
            all_relevance = list(self.ground_truth[resume_id].values())
            # Sort in descending order
            all_relevance.sort(reverse=True)
            # Take top k
            ideal_relevance = all_relevance[:k]
            
            # Calculate IDCG
            idcg = 0.0
            for i, rel in enumerate(ideal_relevance):
                idcg += rel / np.log2(i + 2)
            
            # Calculate NDCG
            if idcg > 0:
                ndcg = dcg / idcg
                ndcg_scores[resume_id] = ndcg
                valid_resumes += 1
            else:
                ndcg_scores[resume_id] = 0.0
        
        # Calculate average NDCG
        if valid_resumes > 0:
            ndcg_scores['average'] = sum(v for k, v in ndcg_scores.items() 
                                        if k != 'average') / valid_resumes
        else:
            ndcg_scores['average'] = 0.0
        
        return ndcg_scores
    
    def calculate_precision_at_k(self, k: int = 10, relevance_threshold: float = 1.0) -> Dict[str, float]:
        """
        Calculate Precision@k for each resume.
        
        Args:
            k: Number of top results to consider
            relevance_threshold: Minimum relevance score to consider a match relevant
            
        Returns:
            Dictionary with Precision@k scores by resume_id and average
        """
        if not self.results or not self.ground_truth:
            print("❌ No results or ground truth data available for Precision calculation")
            return {'average': 0.0}
        
        precision_scores = {}
        valid_resumes = 0
        
        for resume_id, matches in self.results.items():
            if resume_id not in self.ground_truth:
                continue
            
            # Get top k matches
            top_k_matches = matches[:k]
            
            # Count relevant items in top k
            relevant_count = 0
            for match in top_k_matches:
                job_id = str(match['job_id'])
                if job_id in self.ground_truth[resume_id]:
                    if self.ground_truth[resume_id][job_id] >= relevance_threshold:
                        relevant_count += 1
            
            # Calculate precision
            if len(top_k_matches) > 0:
                precision = relevant_count / len(top_k_matches)
                precision_scores[resume_id] = precision
                valid_resumes += 1
            else:
                precision_scores[resume_id] = 0.0
        
        # Calculate average precision
        if valid_resumes > 0:
            precision_scores['average'] = sum(v for k, v in precision_scores.items() 
                                             if k != 'average') / valid_resumes
        else:
            precision_scores['average'] = 0.0
        
        return precision_scores
    
    def calculate_recall_at_k(self, k: int = 10, relevance_threshold: float = 1.0) -> Dict[str, float]:
        """
        Calculate Recall@k for each resume.
        
        Args:
            k: Number of top results to consider
            relevance_threshold: Minimum relevance score to consider a match relevant
            
        Returns:
            Dictionary with Recall@k scores by resume_id and average
        """
        if not self.results or not self.ground_truth:
            print("❌ No results or ground truth data available for Recall calculation")
            return {'average': 0.0}
        
        recall_scores = {}
        valid_resumes = 0
        
        for resume_id, matches in self.results.items():
            if resume_id not in self.ground_truth:
                continue
            
            # Get top k matches
            top_k_matches = matches[:k]
            
            # Count relevant items in ground truth
            relevant_in_gt = 0
            for job_id, relevance in self.ground_truth[resume_id].items():
                if relevance >= relevance_threshold:
                    relevant_in_gt += 1
            
            if relevant_in_gt == 0:
                recall_scores[resume_id] = 1.0  # Perfect recall if no relevant items
                valid_resumes += 1
                continue
            
            # Count relevant items in top k
            relevant_retrieved = 0
            for match in top_k_matches:
                job_id = str(match['job_id'])
                if job_id in self.ground_truth[resume_id]:
                    if self.ground_truth[resume_id][job_id] >= relevance_threshold:
                        relevant_retrieved += 1
            
            # Calculate recall
            recall = relevant_retrieved / relevant_in_gt
            recall_scores[resume_id] = recall
            valid_resumes += 1
        
        # Calculate average recall
        if valid_resumes > 0:
            recall_scores['average'] = sum(v for k, v in recall_scores.items() 
                                          if k != 'average') / valid_resumes
        else:
            recall_scores['average'] = 0.0
        
        return recall_scores
    
    def calculate_mrr(self, relevance_threshold: float = 1.0) -> Dict[str, float]:
        """
        Calculate Mean Reciprocal Rank (MRR) for each resume.
        
        Args:
            relevance_threshold: Minimum relevance score to consider a match relevant
            
        Returns:
            Dictionary with MRR scores by resume_id and average
        """
        if not self.results or not self.ground_truth:
            print("❌ No results or ground truth data available for MRR calculation")
            return {'average': 0.0}
        
        mrr_scores = {}
        valid_resumes = 0
        
        for resume_id, matches in self.results.items():
            if resume_id not in self.ground_truth:
                continue
            
            # Find the first relevant match
            for i, match in enumerate(matches):
                job_id = str(match['job_id'])
                if job_id in self.ground_truth[resume_id]:
                    if self.ground_truth[resume_id][job_id] >= relevance_threshold:
                        # Found first relevant match at rank i+1
                        mrr_scores[resume_id] = 1.0 / (i + 1)
                        break
            else:
                # No relevant match found
                mrr_scores[resume_id] = 0.0
            
            valid_resumes += 1
        
        # Calculate average MRR
        if valid_resumes > 0:
            mrr_scores['average'] = sum(v for k, v in mrr_scores.items() 
                                       if k != 'average') / valid_resumes
        else:
            mrr_scores['average'] = 0.0
        
        return mrr_scores
    
    def calculate_map(self, relevance_threshold: float = 1.0) -> Dict[str, float]:
        """
        Calculate Mean Average Precision (MAP) for each resume.
        
        Args:
            relevance_threshold: Minimum relevance score to consider a match relevant
            
        Returns:
            Dictionary with MAP scores by resume_id and average
        """
        if not self.results or not self.ground_truth:
            print("❌ No results or ground truth data available for MAP calculation")
            return {'average': 0.0}
        
        map_scores = {}
        valid_resumes = 0
        
        for resume_id, matches in self.results.items():
            if resume_id not in self.ground_truth:
                continue
            
            # Count total relevant items in ground truth
            relevant_in_gt = sum(1 for relevance in self.ground_truth[resume_id].values() 
                               if relevance >= relevance_threshold)
            
            if relevant_in_gt == 0:
                map_scores[resume_id] = 1.0  # Perfect MAP if no relevant items
                valid_resumes += 1
                continue
            
            # Calculate precision at each relevant item
            precision_sum = 0.0
            relevant_found = 0
            
            for i, match in enumerate(matches):
                job_id = str(match['job_id'])
                if job_id in self.ground_truth[resume_id]:
                    if self.ground_truth[resume_id][job_id] >= relevance_threshold:
                        relevant_found += 1
                        precision_at_k = relevant_found / (i + 1)
                        precision_sum += precision_at_k
            
            # Calculate average precision
            if relevant_found > 0:
                average_precision = precision_sum / relevant_in_gt
                map_scores[resume_id] = average_precision
            else:
                map_scores[resume_id] = 0.0
            
            valid_resumes += 1
        
        # Calculate MAP (mean of average precision)
        if valid_resumes > 0:
            map_scores['average'] = sum(v for k, v in map_scores.items() 
                                       if k != 'average') / valid_resumes
        else:
            map_scores['average'] = 0.0
        
        return map_scores
    
    def calculate_adjusted_rand_index(self, relevance_threshold: float = 1.0) -> Dict[str, float]:
        """
        Calculate Adjusted Rand Index (ARI) for clustering evaluation.
        
        Args:
            relevance_threshold: Minimum relevance score to consider a match relevant
            
        Returns:
            Dictionary with ARI scores by resume_id and average
        """
        if not self.results or not self.ground_truth:
            print("❌ No results or ground truth data available for ARI calculation")
            return {'average': 0.0}
        
        ari_scores = {}
        valid_resumes = 0
        
        for resume_id, matches in self.results.items():
            if resume_id not in self.ground_truth:
                continue
            
            # Create ground truth labels
            gt_labels = {}
            for job_id, relevance in self.ground_truth[resume_id].items():
                # Assign cluster based on relevance
                cluster = int(relevance) if relevance >= relevance_threshold else 0
                gt_labels[job_id] = cluster
            
            # Create predicted labels from matches
            pred_labels = {}
            for i, match in enumerate(matches):
                job_id = str(match['job_id'])
                # Assign cluster based on rank (top 5, next 5, etc.)
                cluster = i // 5 + 1  # Cluster 1 for top 5, Cluster 2 for next 5, etc.
                pred_labels[job_id] = cluster
            
            # Get common job IDs
            common_jobs = set(gt_labels.keys()).intersection(set(pred_labels.keys()))
            
            if len(common_jobs) < 2:
                # Not enough common jobs for meaningful ARI
                ari_scores[resume_id] = 0.0
                valid_resumes += 1
                continue
            
            # Extract labels for common jobs
            gt_array = [gt_labels[job_id] for job_id in common_jobs]
            pred_array = [pred_labels[job_id] for job_id in common_jobs]
            
            # Calculate ARI
            ari = adjusted_rand_score(gt_array, pred_array)
            ari_scores[resume_id] = ari
            valid_resumes += 1
        
        # Calculate average ARI
        if valid_resumes > 0:
            ari_scores['average'] = sum(v for k, v in ari_scores.items() 
                                       if k != 'average') / valid_resumes
        else:
            ari_scores['average'] = 0.0
        
        return ari_scores
    
    def calculate_normalized_mutual_info(self, relevance_threshold: float = 1.0) -> Dict[str, float]:
        """
        Calculate Normalized Mutual Information (NMI) for clustering evaluation.
        
        Args:
            relevance_threshold: Minimum relevance score to consider a match relevant
            
        Returns:
            Dictionary with NMI scores by resume_id and average
        """
        if not self.results or not self.ground_truth:
            print("❌ No results or ground truth data available for NMI calculation")
            return {'average': 0.0}
        
        nmi_scores = {}
        valid_resumes = 0
        
        for resume_id, matches in self.results.items():
            if resume_id not in self.ground_truth:
                continue
            
            # Create ground truth labels
            gt_labels = {}
            for job_id, relevance in self.ground_truth[resume_id].items():
                # Assign cluster based on relevance
                cluster = int(relevance) if relevance >= relevance_threshold else 0
                gt_labels[job_id] = cluster
            
            # Create predicted labels from matches
            pred_labels = {}
            for i, match in enumerate(matches):
                job_id = str(match['job_id'])
                # Assign cluster based on rank (top 5, next 5, etc.)
                cluster = i // 5 + 1  # Cluster 1 for top 5, Cluster 2 for next 5, etc.
                pred_labels[job_id] = cluster
            
            # Get common job IDs
            common_jobs = set(gt_labels.keys()).intersection(set(pred_labels.keys()))
            
            if len(common_jobs) < 2:
                # Not enough common jobs for meaningful NMI
                nmi_scores[resume_id] = 0.0
                valid_resumes += 1
                continue
            
            # Extract labels for common jobs
            gt_array = [gt_labels[job_id] for job_id in common_jobs]
            pred_array = [pred_labels[job_id] for job_id in common_jobs]
            
            # Calculate NMI
            nmi = normalized_mutual_info_score(gt_array, pred_array)
            nmi_scores[resume_id] = nmi
            valid_resumes += 1
        
        # Calculate average NMI
        if valid_resumes > 0:
            nmi_scores['average'] = sum(v for k, v in nmi_scores.items() 
                                       if k != 'average') / valid_resumes
        else:
            nmi_scores['average'] = 0.0
        
        return nmi_scores
    
    def calculate_purity_score(self, relevance_threshold: float = 1.0) -> Dict[str, float]:
        """
        Calculate Purity Score for clustering evaluation.
        
        Args:
            relevance_threshold: Minimum relevance score to consider a match relevant
            
        Returns:
            Dictionary with Purity scores by resume_id and average
        """
        if not self.results or not self.ground_truth:
            print("❌ No results or ground truth data available for Purity calculation")
            return {'average': 0.0}
        
        purity_scores = {}
        valid_resumes = 0
        
        for resume_id, matches in self.results.items():
            if resume_id not in self.ground_truth:
                continue
            
            # Create ground truth labels
            gt_labels = {}
            for job_id, relevance in self.ground_truth[resume_id].items():
                # Assign cluster based on relevance
                cluster = int(relevance) if relevance >= relevance_threshold else 0
                gt_labels[job_id] = cluster
            
            # Create predicted labels from matches
            pred_labels = {}
            pred_clusters = {}
            for i, match in enumerate(matches):
                job_id = str(match['job_id'])
                # Assign cluster based on rank (top 5, next 5, etc.)
                cluster = i // 5 + 1  # Cluster 1 for top 5, Cluster 2 for next 5, etc.
                pred_labels[job_id] = cluster
                
                if cluster not in pred_clusters:
                    pred_clusters[cluster] = []
                pred_clusters[cluster].append(job_id)
            
            # Calculate purity
            correct_assignments = 0
            total_items = 0
            
            for cluster, job_ids in pred_clusters.items():
                # Count ground truth labels in this cluster
                cluster_gt_labels = [gt_labels.get(job_id, -1) for job_id in job_ids if job_id in gt_labels]
                
                if cluster_gt_labels:
                    # Find the most common ground truth label in this cluster
                    most_common_label = Counter(cluster_gt_labels).most_common(1)[0][0]
                    # Count correct assignments (items with the most common label)
                    correct_in_cluster = sum(1 for label in cluster_gt_labels if label == most_common_label)
                    correct_assignments += correct_in_cluster
                    total_items += len(cluster_gt_labels)
            
            # Calculate purity
            if total_items > 0:
                purity = correct_assignments / total_items
                purity_scores[resume_id] = purity
            else:
                purity_scores[resume_id] = 0.0
            
            valid_resumes += 1
        
        # Calculate average purity
        if valid_resumes > 0:
            purity_scores['average'] = sum(v for k, v in purity_scores.items() 
                                          if k != 'average') / valid_resumes
        else:
            purity_scores['average'] = 0.0
        
        return purity_scores
    
    def evaluate(self, k_values: List[int] = [3, 5, 10], 
                relevance_threshold: float = 1.0) -> Dict[str, Any]:
        """
        Run all evaluation metrics.
        
        Args:
            k_values: List of k values to evaluate at
            relevance_threshold: Minimum relevance score to consider a match relevant
            
        Returns:
            Dictionary with all evaluation results
        """
        metrics = {}
        
        # Calculate metrics for each k value
        for k in k_values:
            metrics[f'ndcg@{k}'] = self.calculate_ndcg(k)
            metrics[f'precision@{k}'] = self.calculate_precision_at_k(k, relevance_threshold)
            metrics[f'recall@{k}'] = self.calculate_recall_at_k(k, relevance_threshold)
        
        # Calculate MRR
        metrics['mrr'] = self.calculate_mrr(relevance_threshold)
        
        # Calculate MAP
        metrics['map'] = self.calculate_map(relevance_threshold)
        
        # Calculate clustering metrics
        metrics['ari'] = self.calculate_adjusted_rand_index(relevance_threshold)
        metrics['nmi'] = self.calculate_normalized_mutual_info(relevance_threshold)
        metrics['purity'] = self.calculate_purity_score(relevance_threshold)
        
        # Store metrics
        self.metrics = metrics
        
        return metrics
    
    def print_evaluation_report(self, k_values: List[int] = [3, 5, 10],
                               relevance_threshold: float = 1.0) -> None:
        """
        Print a comprehensive evaluation report.
        
        Args:
            k_values: List of k values to evaluate at
            relevance_threshold: Minimum relevance score to consider a match relevant
        """
        # Run evaluation if not already done
        if not self.metrics:
            self.evaluate(k_values, relevance_threshold)
        
        print("\n" + "="*80)
        print("RESUME RECOMMENDATION EVALUATION REPORT")
        print("="*80)
        
        print(f"\nRelevance threshold: {relevance_threshold}")
        print(f"Number of resumes evaluated: {len(self.results)}")
        print(f"Number of resumes with ground truth: {len(self.ground_truth)}")
        
        # Print average metrics
        print("\nAVERAGE METRICS:")
        print("-" * 40)
        
        # Print NDCG
        for k in k_values:
            print(f"NDCG@{k}: {self.metrics[f'ndcg@{k}']['average']:.4f}")
        
        # Print Precision
        for k in k_values:
            print(f"Precision@{k}: {self.metrics[f'precision@{k}']['average']:.4f}")
        
        # Print Recall
        for k in k_values:
            print(f"Recall@{k}: {self.metrics[f'recall@{k}']['average']:.4f}")
        
        # Print MRR
        print(f"MRR: {self.metrics['mrr']['average']:.4f}")
        
        # Print MAP
        print(f"MAP: {self.metrics['map']['average']:.4f}")
        
        # Print clustering metrics
        print(f"Adjusted Rand Index: {self.metrics['ari']['average']:.4f}")
        print(f"Normalized Mutual Information: {self.metrics['nmi']['average']:.4f}")
        print(f"Purity Score: {self.metrics['purity']['average']:.4f}")
        
        print("\n" + "="*80)
    
    def plot_metrics(self, output_dir: str = None) -> None:
        """
        Plot evaluation metrics.
        
        Args:
            output_dir: Directory to save plots (optional)
        """
        if not self.metrics:
            print("❌ No metrics available for plotting. Run evaluate() first.")
            return
        
        # Create output directory if specified
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Get k values
        k_values = []
        for key in self.metrics.keys():
            if key.startswith('ndcg@'):
                k_values.append(int(key.split('@')[1]))
        
        # Plot NDCG, Precision, and Recall at different k values
        plt.figure(figsize=(10, 6))
        
        ndcg_values = [self.metrics[f'ndcg@{k}']['average'] for k in k_values]
        precision_values = [self.metrics[f'precision@{k}']['average'] for k in k_values]
        recall_values = [self.metrics[f'recall@{k}']['average'] for k in k_values]
        
        plt.plot(k_values, ndcg_values, 'o-', label='NDCG', linewidth=2)
        plt.plot(k_values, precision_values, 's-', label='Precision', linewidth=2)
        plt.plot(k_values, recall_values, '^-', label='Recall', linewidth=2)
        
        plt.title('Evaluation Metrics at Different k Values')
        plt.xlabel('k')
        plt.ylabel('Score')
        plt.xticks(k_values)
        plt.ylim(0, 1)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'metrics_by_k.png'))
            print(f"✅ Saved metrics plot to {os.path.join(output_dir, 'metrics_by_k.png')}")
        else:
            plt.show()
        
        plt.close()
        
        # Plot MRR, MAP, and clustering metrics
        plt.figure(figsize=(12, 6))
        metric_names = ['MRR', 'MAP', 'ARI', 'NMI', 'Purity']
        metric_values = [
            self.metrics['mrr']['average'],
            self.metrics['map']['average'],
            self.metrics['ari']['average'],
            self.metrics['nmi']['average'],
            self.metrics['purity']['average']
        ]
        
        plt.bar(metric_names, metric_values, color=['green', 'blue', 'orange', 'purple', 'red'])
        plt.title('Advanced Evaluation Metrics')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Add value labels on top of bars
        for i, v in enumerate(metric_values):
            plt.text(i, v + 0.02, f'{v:.4f}', ha='center')
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'advanced_metrics.png'))
            print(f"✅ Saved advanced metrics plot to {os.path.join(output_dir, 'advanced_metrics.png')}")
        else:
            plt.show()
        
        plt.close()
    
    def save_metrics(self, output_file: str) -> bool:
        """
        Save metrics to a JSON file.
        
        Args:
            output_file: Path to save the metrics
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        if not self.metrics:
            print("❌ No metrics available for saving. Run evaluate() first.")
            return False
        
        try:
            # Create directory if it doesn't exist
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Save as JSON with metadata
            metadata = {
                'timestamp': str(datetime.datetime.now()),
                'metrics_type': 'evaluation_metrics',
                'description': 'Resume recommendation system evaluation metrics'
            }
            
            # Save as JSON
            if save_as_json(self.metrics, output_file, metadata):
                print(f"✅ Saved metrics to JSON file: {output_file}")
                return True
            else:
                print(f"❌ Error saving metrics to {output_file}")
                return False
            
        except Exception as e:
            print(f"❌ Error saving metrics: {str(e)}")
            return False

class TFIDFEnhancedResumeScanner:
    """Enhanced ResumeScanner with TF-IDF weighted BGE embeddings and chunking support."""
    
    def __init__(self, input_folder: str, output_folder: str, cv_folder: str = None):
        """Initialize the ResumeScanner with input and output folders."""
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.cv_folder = cv_folder
        self.model = None
        self.fine_tuned_model = None  # Store fine-tuned model
        self.df = None
        self.embeddings = None  # TF-IDF weighted BGE embeddings
        self.tfidf_vectorizer = None  # TF-IDF vectorizer
        self.tfidf_matrix = None  # TF-IDF matrix for the corpus
        self.tfidf_feature_names = None  # Feature names from TF-IDF
        self.temp_dir = None
        self.results_saved = {}  # Track saved files to avoid duplicates
        self.evaluator = ResumeEvaluationMetrics()  # Initialize evaluator
        self.optimal_threshold = 1.0  # Default threshold that will be optimized
        
        # Chunking parameters
        self.chunk_size = 512  # Maximum tokens per chunk
        self.chunk_overlap = 50  # Overlap between chunks
        
        # Create output directory
        self.results_dir = os.path.join(self.output_folder, f"tfidf_enhanced_{int(time.time())}")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Create evaluation directory
        self.eval_dir = os.path.join(self.results_dir, "evaluation")
        os.makedirs(self.eval_dir, exist_ok=True)
        
        # Create models directory for saving fine-tuned models
        self.models_dir = os.path.join(self.results_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Check and create folders
        self.check_and_create_folders()
        
        # Set up RAM disk for temporary files
        self.setup_ram_disk()
        
        # Check for GPU availability and log system resources
        self.setup_device()
        
        # Load the embedding model
        self._load_model()
    
    def check_and_create_folders(self):
        """Check if input and output folders exist and create them if needed."""
        # Check input folder
        if not os.path.exists(self.input_folder):
            logging.warning(f"Input folder '{self.input_folder}' does not exist!")
            # Make absolute path if relative
            if not os.path.isabs(self.input_folder):
                abs_input = os.path.abspath(self.input_folder)
                logging.warning(f"Converting relative path to absolute: {abs_input}")
                self.input_folder = abs_input
            
            # Create the folder
            try:
                os.makedirs(self.input_folder, exist_ok=True)
                logging.info(f"Created input folder: {self.input_folder}")
            except Exception as e:
                logging.error(f"Failed to create input folder: {str(e)}")
        
        # Check output folder
        if not os.path.exists(self.output_folder):
            logging.warning(f"Output folder '{self.output_folder}' does not exist!")
            # Make absolute path if relative
            if not os.path.isabs(self.output_folder):
                abs_output = os.path.abspath(self.output_folder)
                logging.warning(f"Converting relative path to absolute: {abs_output}")
                self.output_folder = abs_output
            
            # Create the folder
            try:
                os.makedirs(self.output_folder, exist_ok=True)
                logging.info(f"Created output folder: {self.output_folder}")
            except Exception as e:
                logging.error(f"Failed to create output folder: {str(e)}")
        
        # Check CV folder if specified
        if self.cv_folder and not os.path.exists(self.cv_folder):
            logging.warning(f"CV folder '{self.cv_folder}' does not exist!")
            # Make absolute path if relative
            if not os.path.isabs(self.cv_folder):
                abs_cv = os.path.abspath(self.cv_folder)
                logging.warning(f"Converting relative path to absolute: {abs_cv}")
                self.cv_folder = abs_cv
            
            # Create the folder
            try:
                os.makedirs(self.cv_folder, exist_ok=True)
                logging.info(f"Created CV folder: {self.cv_folder}")
            except Exception as e:
                logging.error(f"Failed to create CV folder: {str(e)}")
        
        # Print folder paths for verification
        logging.info(f"Using input folder: {self.input_folder}")
        logging.info(f"Using output folder: {self.output_folder}")
        logging.info(f"Using results directory: {self.results_dir}")
        logging.info(f"Using evaluation directory: {self.eval_dir}")
        logging.info(f"Using models directory: {self.models_dir}")
        if self.cv_folder:
            logging.info(f"Using CV folder: {self.cv_folder}")
    
    def safe_save_file(self, file_path, save_function, *args, **kwargs):
        """Safely save a file with error handling."""
        try:
            # Create directory if it doesn't exist
            directory = os.path.dirname(file_path)
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                logging.info(f"Created directory: {directory}")
            
            # Save the file
            save_function(*args, **kwargs)
            
            # Verify file was created
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
                logging.info(f"Successfully saved file: {file_path} ({file_size:.2f} MB)")
                return True
            else:
                logging.error(f"File was not created: {file_path}")
                return False
        
        except Exception as e:
            logging.error(f"Error saving file {file_path}: {str(e)}")
            logging.error(traceback.format_exc())
            return False
    
    @lru_cache(maxsize=1000)
    def clean_text(self, text):
        """
        Clean text by removing quotes and exclamation marks.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
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
    def clean_title_text(self, title):
        """
        Minimal cleaning for job titles to preserve structure.
        
        Args:
            title: Job title to clean
            
        Returns:
            Cleaned job title
        """
        if not title or pd.isna(title):
            return ""
        
        # Convert to string
        title = str(title).strip()
        
        # Remove multiple spaces
        title = re.sub(r'\s+', ' ', title).strip()
        
        return title
    
    @lru_cache(maxsize=1000)
    def clean_standardize_title(self, title):
        """
        Clean and standardize job titles according to specific requirements.
        
        Args:
            title: Job title to clean and standardize
            
        Returns:
            Cleaned and standardized job title
        """
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
    def normalize_title_for_comparison(self, title):
        """
        Create a normalized version of a title for comparison and deduplication.
        
        Args:
            title: Job title to normalize
            
        Returns:
            Normalized title for comparison
        """
        if not title or pd.isna(title):
            return ""
        
        # First apply the standard cleaning
        title = self.clean_standardize_title(title)
        
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
    
    def normalize_job_titles(self, matches_df):
        """
        Normalize job titles in the matches dataframe to group similar titles,
        removing duplicates and near-duplicates, and ensure proper sorting.
        
        Args:
            matches_df: DataFrame containing job matches
            
        Returns:
            DataFrame with normalized titles and deduplicated entries, sorted by similarity
        """
        if 'title' not in matches_df.columns:
            return matches_df
        
        # Clean up titles first
        matches_df['title'] = matches_df['title'].apply(self.clean_standardize_title)
        
        # Add a normalized title column for comparison
        matches_df['normalized_title'] = matches_df['title'].apply(self.normalize_title_for_comparison)
        
        # Group by normalized title and keep the one with highest similarity
        result = matches_df.loc[matches_df.groupby('normalized_title')['similarity_percentage'].idxmax()]
        
        # Drop the normalized_title column as it's no longer needed
        result = result.drop(columns=['normalized_title'])
        
        # Sort by similarity percentage in descending order
        result = result.sort_values('similarity_percentage', ascending=False)
        
        return result
    
    def setup_device(self):
        """Set up device (CPU/GPU) and log system resources."""
        # Force CUDA device detection
        if torch.cuda.is_available():
            # Set the device explicitly
            torch.cuda.set_device(0)  # Use the first GPU
            self.device = torch.device('cuda:0')
            logging.info(f"Using GPU device: {torch.cuda.get_device_name(0)}")
        else:
            # Check if NVIDIA drivers are properly installed
            logging.warning("CUDA not available. Checking system configuration...")
            try:
                import subprocess
                nvidia_smi = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if nvidia_smi.returncode == 0:
                    logging.warning("nvidia-smi works but PyTorch can't detect CUDA. This might be a PyTorch installation issue.")
                    logging.warning("Try reinstalling PyTorch with CUDA support: pip install torch --upgrade --force-reinstall --index-url https://download.pytorch.org/whl/cu118")
                else:
                    logging.warning("nvidia-smi failed. NVIDIA drivers might not be installed or GPU is not detected.")
            except:
                logging.warning("Could not check nvidia-smi. NVIDIA drivers might not be installed.")
            
            self.device = torch.device('cpu')
            logging.info("Falling back to CPU")
        
        # Log system resources
        self.log_system_resources()
        
        # Set up mixed precision if available
        self.use_mixed_precision = False
        if self.device.type == 'cuda' and torch.cuda.is_available():
            if hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
                self.use_mixed_precision = True
                logging.info("Mixed precision is available and will be used")
                
            # Set optimal CUDA settings
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            logging.info("CUDA optimizations enabled")
    
    def log_system_resources(self):
        """Log available system resources."""
        # CPU info
        cpu_count = multiprocessing.cpu_count()
        
        # RAM info
        ram = psutil.virtual_memory()
        ram_total = ram.total / (1024 ** 3)  # GB
        ram_available = ram.available / (1024 ** 3)  # GB
        
        # GPU info
        gpu_info = "Not available"
        gpu_memory = 0
        if torch.cuda.is_available():
            gpu_info = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # GB
        
        logging.info(f"System resources:")
        logging.info(f"  CPU: {cpu_count} cores")
        logging.info(f"  RAM: {ram_total:.2f} GB total, {ram_available:.2f} GB available")
        if torch.cuda.is_available():
            logging.info(f"  GPU: {gpu_info} with {gpu_memory:.2f} GB memory")
        else:
            logging.info(f"  GPU: {gpu_info}")
    
    def setup_ram_disk(self, size_mb=1024):
        """Set up a RAM disk for temporary files."""
        self.temp_dir = tempfile.TemporaryDirectory()
        temp_path = self.temp_dir.name
        logging.info(f"Created RAM-based temporary directory at {temp_path}")
        return temp_path
    
    def _load_model(self):
        """Load the sentence transformer model."""
        logging.info("Loading embedding model (BAAI/bge-large-en-v1.5)...")
        
        # Force device selection before model loading
        if torch.cuda.is_available():
            # Explicitly set the device
            device_name = f"cuda:{torch.cuda.current_device()}"
            logging.info(f"Use pytorch device_name: {device_name}")
            self.model = SentenceTransformer('BAAI/bge-large-en-v1.5', device=device_name)
        else:
            logging.info(f"Use pytorch device_name: cpu")
            self.model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        
        # Optimize model for inference
        self.model.eval()
        
        # Verify model device
        model_device = next(self.model.parameters()).device
        logging.info(f"Model loaded on device: {model_device}")
        
        # Save model to JSON file
        model_file = os.path.join(self.models_dir, 'bge_model.json')
        metadata = {
            'model_name': 'BAAI/bge-large-en-v1.5',
            'device': str(model_device),
            'embedding_dim': 1024
        }
        
        # We don't actually save the model here as it's large and can be reloaded from HuggingFace
        # But we save the metadata for reference
        save_as_json(metadata, model_file)
        logging.info(f"Saved model metadata to {model_file}")
    
    def chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """
        Split text into chunks with overlap to handle long texts that exceed token limits.
        
        Args:
            text: Text to chunk
            chunk_size: Maximum tokens per chunk (defaults to self.chunk_size)
            overlap: Overlap between chunks in tokens (defaults to self.chunk_overlap)
            
        Returns:
            List of text chunks
        """
        if chunk_size is None:
            chunk_size = self.chunk_size
        if overlap is None:
            overlap = self.chunk_overlap
            
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
    
    def create_chunked_embeddings(self, texts: List[str], output_file: str, batch_size: int = 32) -> np.ndarray:
        """
        Create embeddings using chunking for long texts.
        
        Args:
            texts: List of texts to embed
            output_file: Path to save embeddings
            batch_size: Batch size for processing
            
        Returns:
            Array of embeddings
        """
        logging.info(f"Creating chunked embeddings for {len(texts)} texts...")
        
        # Process each text by chunking
        all_embeddings = []
        
        # Use fine-tuned model if available, otherwise use base model
        embedding_model = self.fine_tuned_model if self.fine_tuned_model is not None else self.model
        
        for text in tqdm(texts, desc="Processing texts with chunking"):
            # Split text into chunks
            chunks = self.chunk_text(text)
            
            # Get embeddings for each chunk
            chunk_embeddings = []
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                with torch.no_grad():
                    if self.use_mixed_precision and torch.cuda.is_available():
                        with torch.cuda.amp.autocast():
                            batch_embeddings = embedding_model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
                    else:
                        batch_embeddings = embedding_model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
                    chunk_embeddings.extend(batch_embeddings)
            
            # Average the chunk embeddings to get a single embedding for the text
            if chunk_embeddings:
                text_embedding = np.mean(chunk_embeddings, axis=0)
                # Normalize the averaged embedding
                text_embedding = text_embedding / np.linalg.norm(text_embedding)
                all_embeddings.append(text_embedding)
            else:
                # Fallback for empty text
                embedding_dim = 1024  # For BAAI/bge-large-en-v1.5
                all_embeddings.append(np.zeros(embedding_dim))
        
        # Convert to numpy array
        all_embeddings = np.array(all_embeddings)
        
        # Save embeddings with metadata
        metadata = {
            'created_at': str(datetime.datetime.now()),
            'model_name': 'BAAI/bge-large-en-v1.5',
            'embedding_dim': all_embeddings.shape[1],
            'num_embeddings': len(texts),
            'chunking_enabled': True,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap
        }
        
        if save_as_json(all_embeddings, output_file, metadata):
            logging.info(f"Created and saved chunked embeddings to {output_file}")
        else:
            logging.error(f"Failed to save chunked embeddings to {output_file}")
        
        return all_embeddings
    
    def optimize_relevance_threshold(self, k_values=[3, 5, 10]):
        """
        Find optimal relevance threshold by grid search.
        
        Args:
            k_values: List of k values to evaluate
            
        Returns:
            Optimal threshold and corresponding metrics
        """
        logging.info("Optimizing relevance threshold...")
        print(f"{Fore.BLUE}{'='*80}")
        print(f"{Fore.BLUE}🔍 OPTIMIZING RELEVANCE THRESHOLD")
        print(f"{Fore.BLUE}{'='*80}{Style.RESET_ALL}")
        
        # Define threshold range to test
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0]
        
        best_f1 = 0
        best_threshold = 1.0
        best_metrics = None
        
        # Test each threshold
        for threshold in thresholds:
            # Evaluate with this threshold
            metrics = self.evaluator.evaluate(k_values=k_values, relevance_threshold=threshold)
            
            # Calculate F1 score (harmonic mean of precision and recall)
            precision = metrics[f'precision@{k_values[1]}']['average']  # Use middle k value
            recall = metrics[f'recall@{k_values[1]}']['average']
            
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0
            
            logging.info(f"Threshold: {threshold}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
            print(f"Threshold: {threshold}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
            
            # Update best if improved
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_metrics = metrics
        
        # Save the optimal threshold
        self.optimal_threshold = best_threshold
        
        logging.info(f"Optimal threshold: {best_threshold} with F1: {best_f1:.4f}")
        print(f"{Fore.GREEN}✅ Optimal threshold: {best_threshold} with F1: {best_f1:.4f}{Style.RESET_ALL}")
        
        # Save threshold to file
        threshold_file = os.path.join(self.eval_dir, 'optimal_threshold.json')
        
        metadata = {
            'optimal_threshold': best_threshold,
            'f1_score': best_f1,
            'precision': metrics[f'precision@{k_values[1]}']['average'],
            'recall': metrics[f'recall@{k_values[1]}']['average'],
            'k_values': k_values
        }
        
        if save_as_json(metadata, threshold_file, metadata):
            logging.info(f"Saved optimal threshold to {threshold_file}")
        else:
            logging.error(f"Failed to save optimal threshold to {threshold_file}")
        
        return best_threshold, best_metrics
    
    def load_processed_data(self, file_path):
        """
        Load pre-processed data from CSV file.
        
        Args:
            file_path: Path to the processed CSV file
            
        Returns:
            DataFrame with processed data
        """
        try:
            logging.info(f"Loading pre-processed data from {file_path}...")
            start_time = time.time()
            
            # Load the CSV file
            self.df = pd.read_csv(file_path)
            
            # Convert columns to appropriate types
            if 'person_id' in self.df.columns:
                self.df['person_id'] = self.df['person_id'].astype('int32')
            
            # Convert categorical columns
            for col in ['ability', 'title', 'skill']:
                if col in self.df.columns:
                    self.df[col] = self.df[col].astype('category')
            
            # Verify the data has the required columns
            required_columns = ['person_id', 'embedding_text']
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            
            if missing_columns:
                raise ValueError(f"Pre-processed data is missing required columns: {', '.join(missing_columns)}")
            
            # Log loading time and data size
            elapsed = time.time() - start_time
            logging.info(f"Loaded {len(self.df)} records in {elapsed:.2f}s")
            
            # Optimize memory usage
            self.optimize_dataframe_memory()
            
            return self.df
            
        except Exception as e:
            logging.error(f"Error loading pre-processed data: {str(e)}")
            logging.error(traceback.format_exc())
            logging.warning("Will process data from scratch instead.")
            return None
    
    def load_data(self):
        """Load and process the resume datasets with memory optimization."""
        # Check if processed data already exists
        processed_file = os.path.join(self.output_folder, 'processed_resumes.csv')
        processed_json = os.path.join(self.output_folder, 'processed_resumes.json')
        
        # Try to load from JSON first (faster and preserves data types)
        if os.path.exists(processed_json):
            logging.info(f"Found existing processed JSON file: {processed_json}")
            
            # Check file size and modification time
            file_size_mb = os.path.getsize(processed_json) / (1024 * 1024)
            mod_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(processed_json)))
            
            logging.info(f"File size: {file_size_mb:.2f} MB, Last modified: {mod_time}")
            
            # Try to load the pre-processed data
            loaded_data = load_from_json(processed_json)
            if loaded_data is not None:
                self.df = loaded_data
                print(f"{Fore.GREEN}✅ Loaded pre-processed data from {processed_json}{Style.RESET_ALL}")
                return self.df
        
        # Fall back to CSV if JSON not available
        if os.path.exists(processed_file):
            logging.info(f"Found existing processed CSV file: {processed_file}")
            
            # Check file size and modification time
            file_size_mb = os.path.getsize(processed_file) / (1024 * 1024)
            mod_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(processed_file)))
            
            logging.info(f"File size: {file_size_mb:.2f} MB, Last modified: {mod_time}")
            
            # Try to load the pre-processed data
            loaded_df = self.load_processed_data(processed_file)
            
            if loaded_df is not None:
                print(f"{Fore.GREEN}✅ Loaded pre-processed data from {processed_file}{Style.RESET_ALL}")
                
                # Save as JSON for future use
                if save_as_json(self.df, processed_json, {'source': 'converted_from_csv'}):
                    logging.info(f"Saved dataframe to JSON for future use: {processed_json}")
                
                return loaded_df
        
        logging.info("Processing data from scratch...")
        
        # Check if CSV files exist in the input folder
        required_files = [
            '01_people.csv',
            '02_abilities.csv',
            '04_experience.csv',
            '05_person_skills.csv'
        ]
        
        missing_files = []
        for file in required_files:
            file_path = os.path.join(self.input_folder, file)
            if not os.path.exists(file_path):
                missing_files.append(file)
        
        if missing_files:
            error_msg = f"Missing required CSV files in input folder: {', '.join(missing_files)}"
            logging.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Define optimized dtypes to reduce memory usage
        dtypes = {
            'person_id': 'int32',
            'ability': 'category',
            'title': 'category',
            'skill': 'category'
        }
        
        # Determine optimal chunk size based on available RAM
        available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
        chunk_size = min(100000, max(10000, int(available_ram_gb * 20000)))  # Heuristic
        logging.info(f"Using chunk size of {chunk_size} based on {available_ram_gb:.2f} GB available RAM")
        
        try:
            # Load CSV files with chunking to reduce memory usage
            logging.info("Loading CSV files in chunks...")
            
            # Process each file in chunks
            df1_chunks = pd.read_csv(os.path.join(self.input_folder, '01_people.csv'), 
                                    dtype=dtypes, chunksize=chunk_size)
            df1 = pd.concat(list(df1_chunks))
            
            df2_chunks = pd.read_csv(os.path.join(self.input_folder, '02_abilities.csv'), 
                                    dtype=dtypes, chunksize=chunk_size)
            df2 = pd.concat(list(df2_chunks))
            
            df4_chunks = pd.read_csv(os.path.join(self.input_folder, '04_experience.csv'), 
                                    dtype=dtypes, chunksize=chunk_size)
            df4 = pd.concat(list(df4_chunks))
            
            df5_chunks = pd.read_csv(os.path.join(self.input_folder, '05_person_skills.csv'), 
                                    dtype=dtypes, chunksize=chunk_size)
            df5 = pd.concat(list(df5_chunks))
            
            # Clean text in all dataframes
            logging.info("Cleaning text in all dataframes...")
            for df in [df1, df2, df4, df5]:
                for col in df.select_dtypes(include=['object', 'category']).columns:
                    if col != 'person_id':  # Skip ID columns
                        df[col] = df[col].apply(self.clean_text)
            
            # Filter and clean data
            logging.info("Filtering and cleaning data...")
            df1 = self._filter_person(df1).drop(columns=['name', 'email', 'phone', 'linkedin'], errors='ignore')
            df2 = self._filter_person(df2)
            df4 = self._filter_person(df4).drop(columns=['firm', 'start_date', 'end_date', 'location'], errors='ignore')
            df5 = self._filter_person(df5)
            
            # Process title column to keep only the first title if multiple exist
            logging.info("Processing title column to keep only the first title...")
            if 'title' in df4.columns:
                df4['title'] = df4['title'].apply(lambda x: str(x).split(';')[0].strip() if pd.notna(x) else x)
            
            # Aggregate text by person
            logging.info("Aggregating text by person...")
            df2_agg = self._aggregate_text(df2)
            df4_agg = self._aggregate_text(df4)
            df5_agg = self._aggregate_text(df5)
            
            # Free memory after each step
            del df2
            gc.collect()
            
            # Merge dataframes incrementally to save memory
            logging.info("Merging dataframes...")
            self.df = df1.merge(df2_agg, on='person_id', how='left')
            del df1, df2_agg
            gc.collect()
            
            
            self.df = self.df.merge(df4_agg, on='person_id', how='left')
            del df4_agg
            gc.collect()
            
            self.df = self.df.merge(df5_agg, on='person_id', how='left')
            del df5_agg, df4, df5
            gc.collect()
            
            # Remove duplicate rows after merging, excluding person_id from the check
            logging.info("Removing duplicate rows based on content (excluding person_id)...")
            initial_rows = len(self.df)
            
            # Get all columns except person_id for duplicate checking
            content_columns = [col for col in self.df.columns if col != 'person_id']
            
            # Keep first occurrence of each unique content combination
            self.df = self.df.drop_duplicates(subset=content_columns, keep='first')
            
            removed_rows = initial_rows - len(self.df)
            logging.info(f"Removed {removed_rows} duplicate content rows ({removed_rows/initial_rows*100:.2f}% of data)")
            
            # Clean and standardize job titles according to requirements
            logging.info("Cleaning and standardizing job titles...")
            if 'title' in self.df.columns:
                # Apply the new title cleaning and standardization function
                self.df['title'] = self.df['title'].apply(self.clean_standardize_title)
            else:
                self.df['title'] = 'Unknown Title'
            
            # Fill missing values
            logging.info("Creating text representation...")
            self.df['ability'] = self.df['ability'].fillna('Unknown ability')
            self.df['skill'] = self.df['skill'].fillna('Unknown skill')
            
            # Create text representation for embeddings
            # Use ability, title, and skill for embeddings
            self.df['embedding_text'] = self.df.apply(lambda row: " | ".join([
                self.clean_text(str(row.get('ability', ''))),
                self.clean_title_text(str(row.get('title', ''))) * 3,  # Repeat title 3 times for higher weight
                self.clean_text(str(row.get('skill', '')))
            ]), axis=1)
            
            # Optimize memory usage
            self.optimize_dataframe_memory()
            
            # Save processed data with error handling
            def save_csv():
                self.df.to_csv(processed_file, index=False)
            
            if self.safe_save_file(processed_file, save_csv):
                logging.info(f"Processed {len(self.df)} resumes and saved to CSV.")
            else:
                logging.error(f"Failed to save processed data to {processed_file}")
            
            # Save as JSON for faster loading next time
            metadata = {
                'created_at': str(datetime.datetime.now()),
                'num_records': len(self.df),
                'columns': list(self.df.columns)
            }
            
            if save_as_json(self.df, processed_json, metadata):
                logging.info(f"Saved processed data to JSON file for faster loading.")
            
            return self.df
            
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            logging.error(traceback.format_exc())
            raise
    
    def optimize_dataframe_memory(self):
        """Optimize DataFrame memory usage."""
        start_mem = self.df.memory_usage(deep=True).sum() / (1024 * 1024)
        logging.info(f"DataFrame memory usage before optimization: {start_mem:.2f} MB")
        
        # Convert object types to categories where appropriate
        for col in self.df.select_dtypes(include=['object']).columns:
            if self.df[col].nunique() < 0.5 * len(self.df):
                self.df[col] = self.df[col].astype('category')
        
        # Downcast numeric columns
        for col in self.df.select_dtypes(include=['int']).columns:
            self.df[col] = pd.to_numeric(self.df[col], downcast='integer')
        
        for col in self.df.select_dtypes(include=['float']).columns:
            self.df[col] = pd.to_numeric(self.df[col], downcast='float')
        
        # Print memory usage
        end_mem = self.df.memory_usage(deep=True).sum() / (1024 * 1024)
        logging.info(f"DataFrame memory usage after optimization: {end_mem:.2f} MB")
        logging.info(f"Memory reduced by {100 * (start_mem - end_mem) / start_mem:.2f}%")
        
        return self.df
    
    def _filter_person(self, df):
        """Filter persons by ID."""
        if 'person_id' in df.columns:
            df['person_id'] = df['person_id'].astype('int32')
            return df[df['person_id'] <= 54928]
        return df
    
    def _aggregate_text(self, df, group_col='person_id'):
        """Aggregate text data by group column."""
        if df is None or df.empty:
            return pd.DataFrame()
        
        if group_col not in df.columns:
            logging.warning(f"Group column '{group_col}' not found in dataframe. Available columns: {list(df.columns)}")
            return df
        
        # Get all columns except the group column
        text_columns = [col for col in df.columns if col != group_col]
        
        if not text_columns:
            # If no text columns to aggregate, return unique group values
            return df[[group_col]].drop_duplicates().reset_index(drop=True)
        
        # Create aggregation dictionary for text columns only
        agg_dict = {}
        for col in text_columns:
            agg_dict[col] = lambda x: '; '.join(x.dropna().astype(str).unique())
        
        # Group by the group column and aggregate text columns
        try:
            result = df.groupby(group_col, as_index=False).agg(agg_dict)
            return result
        except Exception as e:
            logging.error(f"Error in _aggregate_text: {str(e)}")
            logging.error(f"DataFrame columns: {list(df.columns)}")
            logging.error(f"Group column: {group_col}")
            logging.error(f"Text columns: {text_columns}")
            raise
    
    def create_tfidf_vectors(self):
        """Create TF-IDF vectors for the corpus."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Check if TF-IDF vectors already exist
        tfidf_model_file = os.path.join(self.results_dir, 'tfidf_vectorizer.pkl')
        tfidf_matrix_file = os.path.join(self.results_dir, 'tfidf_matrix.json')
        
        # Try to load existing TF-IDF model and matrix
        if os.path.exists(tfidf_model_file) and os.path.exists(tfidf_matrix_file):
            logging.info(f"Loading existing TF-IDF model and matrix")
            
            try:
                # Load the vectorizer using joblib
                self.tfidf_vectorizer = joblib.load(tfidf_model_file)
                self.tfidf_matrix = load_from_json(tfidf_matrix_file)
                
                if self.tfidf_vectorizer is not None and self.tfidf_matrix is not None:
                    self.tfidf_feature_names = self.tfidf_vectorizer.get_feature_names_out()
                    logging.info(f"Loaded TF-IDF matrix with shape {self.tfidf_matrix.shape}")
                    return self.tfidf_matrix
            except Exception as e:
                logging.error(f"Error loading TF-IDF model or matrix: {str(e)}")
                logging.error(traceback.format_exc())
        
        logging.info("Creating TF-IDF vectors for the corpus...")
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,  # Limit features to reduce memory usage
            min_df=5,            # Ignore terms that appear in less than 5 documents
            max_df=0.85,         # Ignore terms that appear in more than 85% of documents
            stop_words='english',
            ngram_range=(1, 2)   # Use unigrams and bigrams
        )
        
        # Fit and transform the corpus
        corpus = self.df['embedding_text'].tolist()
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
        self.tfidf_feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        logging.info(f"Created TF-IDF matrix with shape {self.tfidf_matrix.shape}")
        
        # Save TF-IDF model using joblib (better for scikit-learn objects)
        tfidf_model_file = os.path.join(self.results_dir, 'tfidf_vectorizer.pkl')
        try:
            joblib.dump(self.tfidf_vectorizer, tfidf_model_file)
            logging.info(f"Saved TF-IDF vectorizer to {tfidf_model_file}")
        except Exception as e:
            logging.error(f"Error saving TF-IDF vectorizer: {str(e)}")
        
        # Save matrix with metadata
        metadata = {
            'created_at': str(datetime.datetime.now()),
            'corpus_size': len(corpus),
            'features': len(self.tfidf_feature_names),
            'matrix_shape': self.tfidf_matrix.shape
        }
        
        # Save matrix with the same metadata
        if save_as_json(self.tfidf_matrix, tfidf_matrix_file, metadata):
            logging.info(f"Saved TF-IDF matrix to {tfidf_matrix_file}")
        else:
            logging.error(f"Failed to save TF-IDF matrix to {tfidf_matrix_file}")
        
        return self.tfidf_matrix
    
    def apply_tfidf_weighting(self, text):
        """Apply TF-IDF weighting to text before embedding."""
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer not initialized. Call create_tfidf_vectors() first.")
        
        # Transform the text to TF-IDF vector
        text_tfidf = self.tfidf_vectorizer.transform([text])
        
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
        top_terms = [self.tfidf_feature_names[i] for i in top_indices]
        
        # Weight the original text by repeating important terms
        # The number of repetitions is proportional to the TF-IDF score
        weighted_text = text
        for term, score in zip(top_terms, top_scores):
            # Normalize score to a reasonable number of repetitions (1-5)
            repetitions = min(5, max(1, int(score * 10)))
            weighted_text += f" {term * repetitions}"
        
        return weighted_text
    
    def create_embeddings(self, batch_size=32, use_chunking=True):
        """Create TF-IDF weighted BGE embeddings with improved file finding and chunking support."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Create TF-IDF vectors first
        if self.tfidf_vectorizer is None:
            self.create_tfidf_vectors()
        
        # Check if embeddings already exist in JSON format
        embeddings_file = os.path.join(self.results_dir, 'tfidf_bge_embeddings.json')
        
        # Log the path we're checking
        logging.info(f"Checking for embeddings at: {embeddings_file}")
        
        # If not found in the default location, search for it in the output directory
        if not os.path.exists(embeddings_file):
            logging.info("Embeddings not found in default location, searching in output directory...")
            
            # Search for any tfidf_bge_embeddings.json file in the output directory
            embedding_files = []
            for root, dirs, files in os.walk(self.output_folder):
                for file in files:
                    if file == 'tfidf_bge_embeddings.json':
                        embedding_path = os.path.join(root, file)
                        embedding_files.append(embedding_path)
                        logging.info(f"Found embedding file: {embedding_path}")
            
            # If multiple files found, use the most recent one
            if embedding_files:
                embedding_files.sort(key=os.path.getmtime, reverse=True)
                embeddings_file = embedding_files[0]
                logging.info(f"Using most recent embedding file: {embeddings_file}")
        
        # Load embeddings if found
        if os.path.exists(embeddings_file):
            logging.info(f"Loading existing TF-IDF weighted BGE embeddings from {embeddings_file}")
            try:
                # Check file size and modification time
                file_size_mb = os.path.getsize(embeddings_file) / (1024 * 1024)
                mod_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(embeddings_file)))
                logging.info(f"Embeddings file size: {file_size_mb:.2f} MB, Last modified: {mod_time}")
                
                # Try to load the embeddings
                loaded_embeddings = load_from_json(embeddings_file)
                if loaded_embeddings is not None:
                    self.embeddings = loaded_embeddings
                    print(f"{Fore.GREEN}✅ Loaded TF-IDF weighted BGE embeddings from {embeddings_file}{Style.RESET_ALL}")
                    
                    # Verify the shape of the loaded embeddings
                    if isinstance(self.embeddings, np.ndarray):
                        logging.info(f"Loaded embeddings shape: {self.embeddings.shape}")
                    
                    return self.embeddings
                else:
                    logging.warning(f"Failed to load embeddings from {embeddings_file}, will create new embeddings")
            except Exception as e:
                logging.error(f"Error loading embeddings: {str(e)}")
                logging.error(traceback.format_exc())
                logging.warning("Will create new embeddings due to loading error")
        
        logging.info("Creating TF-IDF weighted BGE embeddings...")
        
        # Apply TF-IDF weighting to each text
        weighted_texts = []
        for text in tqdm(self.df['embedding_text'].tolist(), desc="Applying TF-IDF weighting"):
            weighted_texts.append(self.apply_tfidf_weighting(text))
        
        # Choose embedding creation method based on use_chunking parameter
        if use_chunking:
            logging.info("Using chunking approach for long texts...")
            self.embeddings = self.create_chunked_embeddings(weighted_texts, 
                                                            embeddings_file, 
                                                            batch_size=batch_size)
        else:
            logging.info("Using standard approach without chunking...")
            self.embeddings = self._create_embeddings(weighted_texts, 
                                                    embeddings_file, 
                                                    batch_size=batch_size)
        
        return self.embeddings
    
    def _create_embeddings(self, texts, output_file, batch_size=32):
        """Create embeddings using PyTorch DataLoader for efficient batching."""
        logging.info(f"Creating embeddings for {len(texts)} texts...")
        
        # Determine optimal batch size based on system resources
        if torch.cuda.is_available():
            # If GPU is available, adjust batch size based on GPU memory
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            optimal_batch_size = min(batch_size, max(4, int(gpu_memory_gb * 4)))  # Heuristic
            logging.info(f"Using GPU with {gpu_memory_gb:.2f} GB memory, batch size: {optimal_batch_size}")
        else:
            # If CPU only, adjust based on available RAM
            available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
            optimal_batch_size = min(batch_size, max(4, int(available_ram_gb * 2)))  # Heuristic
            logging.info(f"Using CPU with {available_ram_gb:.2f} GB available RAM, batch size: {optimal_batch_size}")
        
        # Determine optimal number of workers
        optimal_workers = min(multiprocessing.cpu_count(), 4)  # Limit to 4 workers
        
        logging.info(f"Using batch size: {optimal_batch_size}, workers: {optimal_workers}")
        
        # Create dataset and dataloader
        dataset = ResumeDataset(texts)
        
        dataloader = DataLoader(
            dataset, 
            batch_size=optimal_batch_size, 
            shuffle=False, 
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            pin_memory=torch.cuda.is_available()  # Only use pin_memory if GPU is available
        )
        
        # Pre-allocate output array in RAM
        embedding_dim = 1024  # For BAAI/bge-large-en-v1.5
        all_embeddings = np.zeros((len(texts), embedding_dim), dtype=np.float32)
        
        # Process batches
        start_time = time.time()
        start_idx = 0
        
        # Verify model is on correct device before batch processing
        model_device = next(self.model.parameters()).device
        logging.info(f"Model is on device: {model_device}")
        
        # Use fine-tuned model if available, otherwise use base model
        embedding_model = self.fine_tuned_model if self.fine_tuned_model is not None else self.model
        
        for i, batch in enumerate(tqdm(dataloader, desc="Creating embeddings")):
            # Use mixed precision if available
            if self.use_mixed_precision and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        batch_embeddings = embedding_model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
            else:
                with torch.no_grad():
                    batch_embeddings = embedding_model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
            
            # Store in pre-allocated array
            end_idx = start_idx + len(batch)
            all_embeddings[start_idx:end_idx] = batch_embeddings
            start_idx = end_idx
            
            # Log progress and clear memory periodically
            if (i + 1) % 20 == 0:
                elapsed = time.time() - start_time
                logging.info(f"Processed {end_idx}/{len(texts)} samples ({end_idx/len(texts)*100:.1f}%) in {elapsed:.1f}s")
                
                # Force garbage collection
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        
        # Save embeddings to JSON file with metadata
        metadata = {
            'created_at': str(datetime.datetime.now()),
            'model_name': 'BAAI/bge-large-en-v1.5',
            'embedding_dim': embedding_dim,
            'num_embeddings': len(texts),
            'batch_size_used': optimal_batch_size,
            'chunking_enabled': False
        }
        
        if save_as_json(all_embeddings, output_file, metadata):
            logging.info(f"Created and saved embeddings to {output_file} in {time.time() - start_time:.1f}s")
        else:
            logging.error(f"Failed to save embeddings to {output_file}")
        
        return all_embeddings
    
    def extract_text_from_file(self, file_path: str) -> str:
        """
        Extract text from various document formats (doc, docx, pdf).
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Extracted text as string
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == '.pdf':
                text = self._extract_from_pdf(file_path)
            elif file_ext in ['.docx', '.doc', '.docs']:
                text = self._extract_from_doc(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            # Clean the extracted text
            text = self.clean_text(text)
                
            # Debug: Print a hash of the extracted text to verify it's different
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
    
    def _extract_from_pdf(self, file_path: str) -> str:
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
    
    def _extract_from_doc(self, file_path: str) -> str:
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
    
    def match_text(self, text: str, top_n: int = 5, file_name: str = None, use_chunking: bool = True) -> Dict:
        """
        Match text against TF-IDF weighted BGE embeddings with chunking support.
        
        Args:
            text: Text to match
            top_n: Number of top matches to return
            file_name: Optional name of the file for output naming
            use_chunking: Whether to use chunking for long texts
            
        Returns:
            Dictionary with results
        """
        logging.info(f"Matching text against TF-IDF weighted BGE embeddings...")
        
        # Clean the input text
        text = self.clean_text(text)
        
        # Debug: Print a hash of the input text
        text_hash = hashlib.md5(text.encode()).hexdigest()
        logging.info(f"Matching text (hash: {text_hash[:8]}...)")
        
        # Check if embeddings exist
        if self.embeddings is None:
            logging.warning("Embeddings not found. Creating embeddings...")
            self.create_embeddings(use_chunking=use_chunking)
        
        # Apply TF-IDF weighting to the query text
        weighted_text = self.apply_tfidf_weighting(text)
        
        # Use fine-tuned model if available, otherwise use base model
        embedding_model = self.fine_tuned_model if self.fine_tuned_model is not None else self.model
        
        # Create embeddings for the weighted query text with chunking if enabled
        if use_chunking and len(weighted_text) > self.chunk_size * 4:  # Approximate character limit
            logging.info("Query text is long, using chunking...")
            chunks = self.chunk_text(weighted_text)
            
            # Get embeddings for each chunk
            chunk_embeddings = []
            with torch.no_grad():
                for chunk in chunks:
                    if self.use_mixed_precision and torch.cuda.is_available():
                        with torch.cuda.amp.autocast():
                            chunk_embedding = embedding_model.encode([chunk], normalize_embeddings=True)
                    else:
                        chunk_embedding = embedding_model.encode([chunk], normalize_embeddings=True)
                    chunk_embeddings.append(chunk_embedding[0])
            
            # Average the chunk embeddings
            if chunk_embeddings:
                query_embedding = np.mean(chunk_embeddings, axis=0)
                # Normalize the averaged embedding
                query_embedding = query_embedding / np.linalg.norm(query_embedding)
                query_embedding = query_embedding.reshape(1, -1)
            else:
                # Fallback for empty chunks
                embedding_dim = 1024  # For BAAI/bge-large-en-v1.5
                query_embedding = np.zeros((1, embedding_dim))
        else:
            # Standard embedding creation for short texts
            with torch.no_grad():
                if self.use_mixed_precision and torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        query_embedding = embedding_model.encode([weighted_text], normalize_embeddings=True)
                else:
                    query_embedding = embedding_model.encode([weighted_text], normalize_embeddings=True)
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get more candidates than needed to allow for deduplication
        top_indices = similarities.argsort()[-(top_n*3):][::-1]  # Get 3x needed to filter duplicates
        top_similarities = similarities[top_indices]
        
        # Get the corresponding rows from the dataframe
        top_matches = self.df.iloc[top_indices].copy()
        top_matches['similarity_percentage'] = (top_similarities * 100).round(2)
        
        # Apply proper title formatting to ensure consistent display
        top_matches['title'] = top_matches['title'].apply(lambda x: self.clean_standardize_title(x))
        
        # Normalize and deduplicate job titles
        top_matches = self.normalize_job_titles(top_matches)
        
        # Take only the top N after deduplication
        top_matches = top_matches.head(top_n)
        
        # Save results with unique filename if provided
        if file_name:
            base_name = os.path.splitext(os.path.basename(file_name))[0]
            
            # Save results
            results_file = os.path.join(self.results_dir, f"{base_name}_matches.json")
            def save_matches():
                save_as_json(top_matches.to_dict(orient='records'), results_file)
            
            if self.safe_save_file(results_file, save_matches):
                logging.info(f"Saved {len(top_matches)} matches to {results_file}")
            else:
                logging.error(f"Failed to save matches to {results_file}")
        
        # Calculate average similarity score
        avg_similarity = top_matches['similarity_percentage'].mean()
        
        logging.info(f"Average similarity score: {avg_similarity:.2f}%")
        
        # Return results
        return {
            'matches': top_matches,
            'avg_similarity': avg_similarity
        }
    
    def process_resume_file(self, file_path: str, top_n: int = 5, use_chunking: bool = True) -> Dict:
        """
        Process a resume file and match it against the database with chunking support.
        
        Args:
            file_path: Path to the resume file
            top_n: Number of top matches to return
            use_chunking: Whether to use chunking for long texts
            
        Returns:
            Dictionary with results
        """
        try:
            # Extract text from file
            resume_text = self.extract_text_from_file(file_path)
            logging.info(f"Extracted {len(resume_text)} characters from {file_path}")
            
            # Get base filename for output naming
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            
            # Save extracted text to file
            text_file = os.path.join(self.results_dir, f"{base_name}_extracted_text.txt")
            
            def save_text():
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(resume_text)
            
            if self.safe_save_file(text_file, save_text):
                logging.info(f"Saved extracted text to {text_file}")
            else:
                logging.error(f"Failed to save extracted text to {text_file}")
            
            # Print file processing header
            print(f"\n{Fore.MAGENTA}{'='*80}")
            print(f"{Fore.MAGENTA}📄 PROCESSING RESUME: {file_path}")
            print(f"{Fore.MAGENTA}{'='*80}{Style.RESET_ALL}")
            
            # Match against job titles with chunking support
            results = self.match_text(resume_text, top_n, file_path, use_chunking=use_chunking)
            
            # Print results in a formatted way
            self._print_results(file_path, results)
            
            # Generate visualizations
            self._generate_visualizations(file_path, results)
            
            return results
            
        except Exception as e:
            logging.error(f"Error processing resume file {file_path}: {str(e)}")
            logging.error(traceback.format_exc())
            return {
                'error': str(e),
                'file_path': file_path
            }
    
    def _print_results(self, file_path, results):
        """Print results in a formatted way."""
        if 'error' in results:
            print(f"{Fore.RED}❌ Error processing {file_path}: {results['error']}{Style.RESET_ALL}")
            return
        
        # Print results
        print(f"\n{Fore.CYAN}{'='*80}")
        print(f"{Fore.CYAN}🏆 TOP MATCHES FOR {os.path.basename(file_path)}")
        print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
        
        for _, row in results['matches'].iterrows():
            title = row['title']
            similarity = row['similarity_percentage']
            description = row.get('embedding_text', '')
            if description:
                description = description[:100] + "..." if len(description) > 100 else description
                print(f"{Fore.GREEN}🔹 {title} - {similarity:.2f}%{Style.RESET_ALL}")
                print(f"   {description}")
                print()
            else:
                print(f"{Fore.GREEN}🔹 {title} - {similarity:.2f}%{Style.RESET_ALL}")
                print()
        
        # Print summary
        print(f"\n{Fore.MAGENTA}{'='*80}")
        print(f"{Fore.MAGENTA}📊 SUMMARY")
        print(f"{Fore.MAGENTA}{'='*80}{Style.RESET_ALL}")
        
        print(f"Average similarity score: {results['avg_similarity']:.2f}%")
    
    def _generate_visualizations(self, file_path, results):
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
        
        chart_file = os.path.join(self.results_dir, output_filename)
        plt.savefig(chart_file)
        plt.close()
        
        logging.info(f"Generated visualization chart for {file_path}")
    
    def scan_cv_folder(self, folder_path=None, top_n=5, use_chunking=True):
        """
        Scan a folder for CV files and process each one with chunking support.
        
        Args:
            folder_path: Path to the folder containing CV files
            top_n: Number of top matches to return for each CV
            use_chunking: Whether to use chunking for long texts
            
        Returns:
            Dictionary with results for each CV
        """
        if folder_path is None:
            folder_path = self.cv_folder
        
        if folder_path is None:
            raise ValueError("CV folder path not specified")
        
        if not os.path.exists(folder_path):
            raise ValueError(f"CV folder path does not exist: {folder_path}")
        
        print(f"\n{Fore.BLUE}{'='*80}")
        print(f"{Fore.BLUE}📂 SCANNING CV FOLDER: {folder_path}")
        print(f"{Fore.BLUE}{'='*80}{Style.RESET_ALL}")
        
        # Get all CV files in the folder
        cv_files = []
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path) and file.lower().endswith(('.pdf', '.docx', '.doc')):
                cv_files.append(file_path)
        
        if not cv_files:
            print(f"{Fore.YELLOW}⚠️ No CV files found in folder: {folder_path}{Style.RESET_ALL}")
            return {}
        
        print(f"{Fore.GREEN}Found {len(cv_files)} CV files in folder{Style.RESET_ALL}")
        
        # Create a summary file
        summary_file = os.path.join(self.results_dir, 'summary.json')
        summary_data = []
        
        # Process each CV file
        results = {}
        for cv_file in cv_files:
            try:
                print(f"\n{Fore.BLUE}Processing CV file: {os.path.basename(cv_file)}{Style.RESET_ALL}")
                result = self.process_resume_file(cv_file, top_n, use_chunking=use_chunking)
                results[cv_file] = result
                
                # Add to summary data
                if 'error' not in result:
                    for _, match in result['matches'].iterrows():
                        summary_data.append({
                            'cv_file': os.path.basename(cv_file),
                            'match_title': match['title'],
                            'similarity': match['similarity_percentage'],
                            'match_details': match.get('embedding_text', '')[:200]
                        })
                
            except Exception as e:
                print(f"{Fore.RED}❌ Error processing {cv_file}: {str(e)}{Style.RESET_ALL}")
                results[cv_file] = {'error': str(e), 'file_path': cv_file}
        
        # Save summary to JSON
        if summary_data:
            def save_summary():
                save_as_json(summary_data, summary_file)
            
            if self.safe_save_file(summary_file, save_summary):
                print(f"{Fore.GREEN}✅ Saved summary to {summary_file}{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}❌ Failed to save summary to {summary_file}{Style.RESET_ALL}")
        
        # Generate overall comparison
        self._generate_overall_comparison(results)
        
        print(f"\n{Fore.GREEN}🎉 Completed processing {len(cv_files)} CV files{Style.RESET_ALL}")
        print(f"{Fore.GREEN}✅ Results saved to {self.results_dir}{Style.RESET_ALL}")
        
        return results

    def calculate_overall_summary(self, all_results):
        """
        Calculate overall summary statistics for all processed CVs.
        
        Args:
            all_results: Dictionary containing results for each CV
            
        Returns:
            Dictionary with overall summary statistics
        """
        # Initialize variables to track statistics
        total_cvs = len(all_results)
        avg_scores = []
        highest_individual_match = 0
        lowest_individual_match = 100
        highest_avg_score = 0
        lowest_avg_score = 100
        
        # Process each CV's results
        for cv_name, results in all_results.items():
            if 'error' in results:
                continue
                
            if 'avg_similarity' in results:
                avg_score = results['avg_similarity']
                avg_scores.append(avg_score)
                
                # Track highest and lowest average scores
                highest_avg_score = max(highest_avg_score, avg_score)
                lowest_avg_score = min(lowest_avg_score, avg_score)
            
            # Track highest and lowest individual match scores
            if 'matches' in results and len(results['matches']) > 0:
                for _, match in results['matches'].iterrows():
                    if 'similarity_percentage' in match:
                        score = match['similarity_percentage']
                        highest_individual_match = max(highest_individual_match, score)
                        lowest_individual_match = min(lowest_individual_match, score)
        
        # Calculate overall average
        overall_avg = sum(avg_scores) / len(avg_scores) if avg_scores else 0
        
        # Calculate standard deviation
        std_dev = 0
        if len(avg_scores) > 1:
            variance = sum((x - overall_avg) ** 2 for x in avg_scores) / len(avg_scores)
            std_dev = variance ** 0.5
        
        # Create summary dictionary
        summary = {
            'total_cvs': total_cvs,
            'average_similarity': round(overall_avg, 2),
            'highest_avg_similarity': round(highest_avg_score, 2),
            'lowest_avg_similarity': round(lowest_avg_score, 2),
            'highest_individual_match': round(highest_individual_match, 2),
            'lowest_individual_match': round(lowest_individual_match, 2),
            'standard_deviation': round(std_dev, 2)
        }
        
        return summary

    def display_overall_summary(self, summary):
        """
        Display overall summary statistics.
        
        Args:
            summary: Dictionary with overall summary statistics
        """
        print(f"\n{Fore.MAGENTA}{'='*80}")
        print(f"{Fore.MAGENTA}📊 OVERALL RESULTS")
        print(f"{Fore.MAGENTA}{'='*80}{Style.RESET_ALL}")
        
        print(f"Total CVs processed: {summary['total_cvs']}")
        print(f"Average similarity score: {summary['average_similarity']:.2f}%")
        print(f"Highest average CV similarity: {summary['highest_avg_similarity']:.2f}%")
        print(f"Lowest average CV similarity: {summary['lowest_avg_similarity']:.2f}%")
        print(f"Highest individual match: {summary['highest_individual_match']:.2f}%")
        print(f"Lowest individual match: {summary['lowest_individual_match']:.2f}%")
        print(f"Standard deviation: {summary['standard_deviation']:.2f}%")

    def _generate_overall_comparison(self, results):
        """Generate overall comparison of results across all CVs."""
        # Filter out results with errors
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not valid_results:
            logging.warning("No valid results to generate overall comparison")
            return
        
        # Calculate overall summary statistics
        summary = self.calculate_overall_summary(valid_results)
        
        # Save summary to JSON
        summary_file = os.path.join(self.results_dir, 'overall_summary.json')
        
        def save_summary():
            # Convert any NumPy types to native Python types
            json_safe_summary = {}
            for key, value in summary.items():
                # Convert numpy types to Python native types
                if isinstance(value, np.number):
                    json_safe_summary[key] = float(value)
                else:
                    json_safe_summary[key] = value
            
            save_as_json(json_safe_summary, summary_file)
        
        if self.safe_save_file(summary_file, save_summary):
            logging.info(f"Saved overall summary to {summary_file}")
        else:
            logging.error(f"Failed to save overall summary to {summary_file}")
        
        # Collect metrics for visualization
        avg_scores = []
        cv_names = []
        
        for file_path, result in valid_results.items():
            cv_names.append(os.path.basename(file_path))
            avg_scores.append(result['avg_similarity'])
        
        # Create visualizations
        
        # Line chart showing scores across all CVs
        plt.figure(figsize=(12, 8))
        
        x = range(len(cv_names))
        plt.plot(x, avg_scores, 'o-', color='#2ecc71', linewidth=2)
        
        plt.title('Similarity Scores Across All CVs')
        plt.xlabel('CV')
        plt.ylabel('Average Similarity Score (%)')
        plt.xticks(x, [name[:15] + '...' if len(name) > 15 else name for name in cv_names], rotation=45)
        plt.axhline(y=summary['average_similarity'], color='r', linestyle='--', 
                    label=f'Average: {summary["average_similarity"]:.2f}%')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save the chart
        line_file = os.path.join(self.results_dir, 'scores_across_cvs.png')
        plt.savefig(line_file)
        plt.close()
        
        # Display the summary
        self.display_overall_summary(summary)
        
    def create_sample_ground_truth(self, output_file, num_resumes=10, num_jobs_per_resume=20):
        """
        Create a sample ground truth file for testing.
        In a real scenario, this would be created by experts.
        
        Args:
            output_file: Path to save the ground truth file
            num_resumes: Number of resumes to create ground truth for
            num_jobs_per_resume: Number of jobs to evaluate per resume
        """
        data = []
        
        # Get job titles from the dataframe
        job_titles = self.df['title'].unique().tolist()
        
        # If there are too few job titles, repeat them
        while len(job_titles) < num_jobs_per_resume:
            job_titles = job_titles * 2
        
        # Limit to the required number
        job_titles = job_titles[:num_jobs_per_resume]
        
        # Create job_id for each job title
        job_ids = {f"job_{i}": title for i, title in enumerate(job_titles)}
        
        for i in range(1, num_resumes + 1):
            resume_id = f"resume_{i}"
            
            # Assign random relevance scores to jobs for this resume
            for job_id, title in job_ids.items():
                # Randomly assign relevance (0-3)
                # 0: Not relevant
                # 1: Somewhat relevant
                # 2: Relevant
                # 3: Highly relevant
                relevance = np.random.choice([0, 1, 2, 3], p=[0.4, 0.3, 0.2, 0.1])
                
                data.append({
                    'resume_id': resume_id,
                    'job_id': job_id,
                    'job_title': title,
                    'relevance': relevance
                })
        
        # Save to CSV
        df = pd.DataFrame(data)
        
        def save_ground_truth():
            df.to_csv(output_file, index=False)
        
        if self.safe_save_file(output_file, save_ground_truth):
            print(f"{Fore.GREEN}✅ Created sample ground truth file with {len(df)} entries at {output_file}{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}❌ Failed to create sample ground truth file{Style.RESET_ALL}")
        
        return job_ids
    
    def evaluate_model(self, ground_truth_file=None, k_values=[3, 5, 10], relevance_threshold=None):
        """
        Evaluate the model using the provided ground truth.
        
        Args:
            ground_truth_file: Path to ground truth CSV file
            k_values: List of k values to evaluate at
            relevance_threshold: Minimum relevance score to consider a match relevant
            
        Returns:
            Dictionary with evaluation results
        """
        # Create ground truth file if not provided
        if ground_truth_file is None:
            ground_truth_file = os.path.join(self.eval_dir, 'ground_truth.csv')
            job_ids = self.create_sample_ground_truth(ground_truth_file)
        else:
            # Load job IDs from ground truth file
            gt_df = pd.read_csv(ground_truth_file)
            job_ids = {row['job_id']: row['job_title'] for _, row in gt_df.iterrows() 
                      if 'job_id' in gt_df.columns and 'job_title' in gt_df.columns}
        
        # Load ground truth
        if not self.evaluator.load_ground_truth(ground_truth_file):
            print(f"{Fore.RED}❌ Failed to load ground truth data. Exiting evaluation.{Style.RESET_ALL}")
            return None
        
        # Create title to job_id mapping
        title_to_job_id = {}
        for job_id, title in job_ids.items():
            title_lower = title.lower()
            title_to_job_id[title_lower] = job_id
        
        # Process each resume in the ground truth
        for resume_id in self.evaluator.ground_truth.keys():
            # Create a mock result for this resume
            # In a real scenario, this would be the result of processing a real resume file
            
            # Get random text from the dataframe to simulate a resume
            random_idx = np.random.randint(0, len(self.df))
            resume_text = self.df.iloc[random_idx]['embedding_text']
            
            # Match against job titles
            result = self.match_text(resume_text, top_n=max(k_values))
            
            # Convert result to the format expected by evaluator
            matches = []
            for _, row in result['matches'].iterrows():
                title = row['title'].lower()
                
                # Find the closest matching job_id
                best_match_job_id = None
                best_match_score = 0
                
                for db_title, job_id in title_to_job_id.items():
                    # Simple matching score based on word overlap
                    title_words = set(title.split())
                    db_title_words = set(db_title.split())
                    overlap = len(title_words.intersection(db_title_words))
                    
                    if overlap > best_match_score:
                        best_match_score = overlap
                        best_match_job_id = job_id
                
                # If no match found, use a hash of the title
                if best_match_job_id is None:
                    best_match_job_id = f"job_{hash(title) % 10000}"
                
                matches.append({
                    'job_id': best_match_job_id,
                    'similarity_score': row['similarity_percentage'] / 100.0  # Convert to 0-1 scale
                })
            
            # Add to evaluator
            self.evaluator.add_result(resume_id, matches)
        
        # If relevance_threshold is None, find the optimal threshold
        if relevance_threshold is None:
            print(f"{Fore.YELLOW}No relevance threshold specified. Finding optimal threshold...{Style.RESET_ALL}")
            relevance_threshold, _ = self.optimize_relevance_threshold(k_values)
        
        # Run evaluation with the specified or optimal threshold
        metrics = self.evaluator.evaluate(k_values=k_values, relevance_threshold=relevance_threshold)
        
        # Print evaluation report
        self.evaluator.print_evaluation_report(k_values=k_values, relevance_threshold=relevance_threshold)
        
        # Generate plots
        self.evaluator.plot_metrics(self.eval_dir)
        
        # Save metrics to file
        metrics_file = os.path.join(self.eval_dir, 'evaluation_metrics.json')
        self.evaluator.save_metrics(metrics_file)
        
        return metrics
    
    def cleanup(self):
        """Clean up resources."""
        logging.info("Cleaning up resources...")
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Remove temporary directory
        if self.temp_dir:
            self.temp_dir.cleanup()
        
        # Force garbage collection
        gc.collect()
        
        logging.info("Cleanup complete")


# Main execution
if __name__ == "__main__":
    # Define folders with absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(script_dir, "input")
    output_folder = os.path.join(script_dir, "output")
    cv_folder = os.path.join(script_dir, "cv_dummy")
    
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"CV folder: {cv_folder}")
    
    # Initialize scanner with the CV folder
    scanner = TFIDFEnhancedResumeScanner(input_folder, output_folder, cv_folder)
    
    try:
        # Check if processed data already exists
        processed_file = os.path.join(output_folder, 'processed_resumes.csv')
        processed_json = os.path.join(output_folder, 'processed_resumes.json')
        
        if os.path.exists(processed_json):
            print(f"{Fore.GREEN}✅ Found existing processed JSON data{Style.RESET_ALL}")
            print(f"   - {processed_json}")
            print(f"{Fore.YELLOW}ℹ️ Using existing files to save processing time{Style.RESET_ALL}")
        elif os.path.exists(processed_file):
            print(f"{Fore.GREEN}✅ Found existing processed CSV data{Style.RESET_ALL}")
            print(f"   - {processed_file}")
            print(f"{Fore.YELLOW}ℹ️ Using existing files to save processing time{Style.RESET_ALL}")
        else:
            # Check if input files exist
            input_files_exist = all(
                os.path.exists(os.path.join(input_folder, f)) 
                for f in ['01_people.csv', '02_abilities.csv', '04_experience.csv', '05_person_skills.csv']
            )
            
            if not input_files_exist:
                print(f"{Fore.RED}❌ Input CSV files are missing. Please place them in: {input_folder}{Style.RESET_ALL}")
                sys.exit(1)
            
            logging.info("No existing processed data found. Processing from scratch...")
        
        # Load data (will use existing processed data if available)
        scanner.load_data()
        
        # Create TF-IDF vectors
        scanner.create_tfidf_vectors()
            
        # Create embeddings with chunking enabled
        scanner.create_embeddings(use_chunking=True)
        
        print(f"{Fore.GREEN}✅ Data and embeddings ready for processing{Style.RESET_ALL}")
        
        # Run evaluation with adaptive threshold selection
        print(f"\n{Fore.BLUE}{'='*80}")
        print(f"{Fore.BLUE}📊 RUNNING MODEL EVALUATION WITH ADAPTIVE THRESHOLD")
        print(f"{Fore.BLUE}{'='*80}{Style.RESET_ALL}")
        
        scanner.evaluate_model(k_values=[3, 5, 10], relevance_threshold=None)  # None triggers adaptive threshold selection
        
        # Scan the CV folder if it exists and contains files
        if os.path.exists(cv_folder) and any(f.lower().endswith(('.pdf', '.docx', '.doc')) for f in os.listdir(cv_folder)):
            results = scanner.scan_cv_folder(use_chunking=True)
            
            # Print a summary of the results
            print(f"\n{Fore.CYAN}{'='*80}")
            print(f"{Fore.CYAN}📊 PROCESSING COMPLETE")
            print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
            
            print(f"Check the results directory for detailed results and visualizations:")
            print(f"{scanner.results_dir}")
            
        else:
            print(f"{Fore.YELLOW}⚠️ CV folder is empty or doesn't exist. Skipping CV scanning.{Style.RESET_ALL}")
        
        print(f"\n{Fore.GREEN}🎉 All processing completed!{Style.RESET_ALL}")
        
    except FileNotFoundError as e:
        print(f"{Fore.RED}❌ File not found: {str(e)}{Style.RESET_ALL}")
        print(f"Please make sure all required CSV files are in the input folder: {input_folder}")
    except Exception as e:
        print(f"{Fore.RED}❌ Error during processing: {str(e)}{Style.RESET_ALL}")
        print(traceback.format_exc())
    
    finally:
        # Clean up resources
        scanner.cleanup()
# %%
