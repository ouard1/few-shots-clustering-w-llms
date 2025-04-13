import os
import json
import time
from datetime import datetime
from tqdm import tqdm
import pickle
from pathlib import Path
import dotenv

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment
from openai import OpenAI
import nltk
from sentence_transformers import SentenceTransformer
import pandas as pd
import matplotlib.pyplot as plt





dotenv.load_dotenv()        
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CACHE_DIR = "./cache"
RESULTS_DIR = "./results"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Helper function to get timestamp for filenames
def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# Function to save results
def save_results(result_dict, method_name):
    timestamp = get_timestamp()
    filename = f"{RESULTS_DIR}/{method_name}_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(result_dict, f, indent=2)
    print(f"Results saved to {filename}")
    return filename
    
def load_banking77_data(train_limit=3000, test_limit=1000):
    """
    Load Banking77 dataset with limits on number of examples
    """
    # Check if processed data exists in cache
    cache_file = f"{CACHE_DIR}/banking77_train{train_limit}_test{test_limit}.pkl"
    if os.path.exists(cache_file):
        print(f"Loading cached dataset from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    # Load the data from CSV files
    train_data = [] 
    with open('Datasets/BankingData/train.csv', 'r', encoding='utf-8') as f:
        next(f)  
        for i, line in enumerate(f):
            if i >= train_limit:
                break
            parts = line.strip().split(',', 1)
            if len(parts) == 2:
                text, category = parts
                train_data.append({"text": text, "category": category})
    
    test_data = []
    with open('Datasets/BankingData/test.csv', 'r', encoding='utf-8') as f:
        next(f)  # Skip header
        for i, line in enumerate(f):
            if i >= test_limit:
                break
            parts = line.strip().split(',', 1)
            if len(parts) == 2:
                text, category = parts
                test_data.append({"text": text, "category": category})
    
    # Load category mapping
    with open('Datasets/BankingData/categories.json', 'r') as f:
        categories = json.load(f)
    
    # Create category to index mapping
    category_to_idx = {category: idx for idx, category in enumerate(categories)}
    
    # Add numeric labels
    for item in train_data:
        item["label"] = category_to_idx[item["category"]]
    
    for item in test_data:
        item["label"] = category_to_idx[item["category"]]
    
    # Save to cache
    dataset = {
        "train": train_data,
        "test": test_data,
        "categories": categories,
        "category_to_idx": category_to_idx,
        "idx_to_category": {v: k for k, v in category_to_idx.items()}
    }
    
    with open(cache_file, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"Loaded {len(train_data)} train examples and {len(test_data)} test examples")
    print(f"Dataset saved to cache: {cache_file}")
    
    return dataset
def main():
    # Load dataset
    dataset = load_banking77_data(train_limit=3000, test_limit=1000)
    print(dataset)

    
if __name__ == "__main__":
    main()