"""
Text Clustering on Banking77 Dataset (V0)
------------------------------------------------
This script implements and compares several methods for unsupervised intent clustering on the Banking77 dataset, which consists of customer banking queries. The main functionalities include:

- Loading and preprocessing the Banking77 dataset (train/test splits)
- Generating and caching embeddings for queries using Sentence Transformers
- Baseline K-Means clustering for intent discovery
- LLM-based keyphrase expansion to improve clustering
- Pairwise constraint K-Means (PCKMeans) using LLM-generated must-link/cannot-link constraints
- LLM-based correction of low-confidence cluster assignments
- Evaluation of clustering results with accuracy, NMI, and ARI metrics
- Comparison and visualization of different clustering methods

The script is designed for research and experimentation with text clustering and intent discovery, leveraging both traditional clustering and large language models (OpenAI GPT-3.5-turbo) for enhanced performance. Results and intermediate data are cached for efficiency.

Usage: Run the script directly or with a specific task argument. It will prompt for which methods to run and save results/plots in the results directory.
"""
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
from sklearn.metrics.pairwise import euclidean_distances

from create_balanced_dataset import create_banking77_dataset




dotenv.load_dotenv()        
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CACHE_DIR = "./cache"
RESULTS_DIR = "./results"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_results(result_dict, method_name):
    timestamp = get_timestamp()
    filename = f"{RESULTS_DIR}/{method_name}_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(result_dict, f, indent=2)
    print(f"Results saved to {filename}")
    return filename
    
def load_banking77_data(train_limit=3000, test_limit=500):
    """
    Load Banking77 dataset with pandas for more robust CSV parsing
    """
    import pandas as pd
    
  
    cache_file = f"{CACHE_DIR}/banking77_train{train_limit}_test{test_limit}.pkl"
    if os.path.exists(cache_file):
        print(f"Loading cached dataset from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    
    data_dir = "datasets/banking77/"
    
    if not data_dir:
        raise FileNotFoundError("Could not find banking_data directory")
    
    print(f"Loading data from: {data_dir}")
    
    train_path = os.path.join(data_dir, 'banking77train.csv')
    try:
        train_df = pd.read_csv(train_path)
    except:
        train_df = pd.read_csv(train_path, header=None, names=['text', 'category'])
    
    test_path = os.path.join(data_dir, 'test.csv')
    try:
        test_df = pd.read_csv(test_path)
    except:
        test_df = pd.read_csv(test_path, header=None, names=['text', 'category'])
    
    train_df = train_df.head(train_limit)
    test_df = test_df.head(test_limit)
    
    categories_path = os.path.join(data_dir, 'categories.json')
    with open(categories_path, 'r') as f:
        categories = json.load(f)
    
    category_to_idx = {category: idx for idx, category in enumerate(categories)}
    
    train_data = []
    for _, row in train_df.iterrows():
        category = row['category'].strip() if isinstance(row['category'], str) else row['category']
        item = {"text": row['text'], "category": category}
        if category in category_to_idx:
            item["label"] = category_to_idx[category]
            train_data.append(item)
        else:
            print(f"Warning: Category not found in train: {category}")
    
    test_data = []
    for _, row in test_df.iterrows():
        category = row['category'].strip() if isinstance(row['category'], str) else row['category']
        item = {"text": row['text'], "category": category}
        if category in category_to_idx:
            item["label"] = category_to_idx[category]
            test_data.append(item)
        else:
            print(f"Warning: Category not found in test: {category}")
    
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
    
def embed_texts(texts, model_name="all-MiniLM-L6-v2", use_cache=True):
    """
    Embed a list of texts using SentenceTransformer
    """
    cache_key = f"{model_name}_" + str(hash("".join(texts[:5])))
    cache_file = f"{CACHE_DIR}/embeddings_{cache_key}.pkl"
    
    if use_cache and os.path.exists(cache_file):
        print(f"Loading cached embeddings from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    print(f"Embedding {len(texts)} texts with {model_name}...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)
    
    if use_cache:
        with open(cache_file, 'wb') as f:
            pickle.dump(embeddings, f)
    
    return embeddings

def get_dataset_embeddings(dataset, model_name="all-MiniLM-L6-v2", use_cache=True):
    """
    Get embeddings for the entire dataset
    """
    train_texts = [item["text"] for item in dataset["train"]]
    test_texts = [item["text"] for item in dataset["test"]]
    
    train_embeddings = embed_texts(train_texts, model_name, use_cache)
    test_embeddings = embed_texts(test_texts, model_name, use_cache)
    
    return {
        "train": train_embeddings,
        "test": test_embeddings
    }
def calculate_accuracy(y_true, y_pred, n_clusters):
    """
    Calculate clustering accuracy using the Hungarian algorithm
    """
    contingency_matrix = np.zeros((n_clusters, n_clusters))
    for i in range(len(y_true)):
        contingency_matrix[y_true[i], y_pred[i]] += 1
    
    # Use Hungarian algorithm to find the best alignment
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    
    # Create a mapping from cluster to true label
    cluster_to_label = {col_ind[i]: row_ind[i] for i in range(len(row_ind))}
    
    # Map predictions to true labels
    y_pred_aligned = np.array([cluster_to_label[pred] for pred in y_pred])
    
    accuracy = np.sum(y_pred_aligned == y_true) / len(y_true)
    return accuracy

def evaluate_clustering(y_true, y_pred, n_clusters):
    """
    Evaluate clustering with multiple metrics
    """
    # Calculate NMI
    nmi = normalized_mutual_info_score(y_true, y_pred)
    
    # Calculate ARI
    ari = adjusted_rand_score(y_true, y_pred)
    
    # Calculate accuracy
    acc = calculate_accuracy(y_true, y_pred, n_clusters)
    
    return {
        "accuracy": acc,
        "nmi": nmi,
        "ari": ari
    }
def run_kmeans_baseline(dataset, embeddings, n_clusters=77):
    """
    Run baseline K-Means clustering
    """
    print(f"Running K-Means with {n_clusters} clusters...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    train_clusters = kmeans.fit_predict(embeddings["train"])
    
    train_metrics = evaluate_clustering(
        [item["label"] for item in dataset["train"]], 
        train_clusters, 
        n_clusters
    )
    
    test_clusters = kmeans.predict(embeddings["test"])
    
    test_metrics = evaluate_clustering(
        [item["label"] for item in dataset["test"]], 
        test_clusters, 
        n_clusters
    )
    
    results = {
        "method": "kmeans_baseline",
        "n_clusters": n_clusters,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics
    }
    
    save_results(results, "kmeans_baseline")
    
    return results, kmeans, train_clusters, test_clusters
def generate_keyphrases_with_llm(texts, cache_dir=CACHE_DIR, batch_size=50):
    """
    Generate keyphrases for each text using LLM
    """
    keyphrases_dir = os.path.join(cache_dir, "keyphrases")
    os.makedirs(keyphrases_dir, exist_ok=True)
    
    cache_key = str(hash("".join(texts[:5])))
    cache_file = os.path.join(keyphrases_dir, f"keyphrases_{cache_key}.json")
    
    if os.path.exists(cache_file):
        print(f"Loading keyphrases from cache: {cache_file}")
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    system_prompt = """I am trying to cluster online banking queries based on whether they express the same intent. For each query, generate a comprehensive set of keyphrases that could describe its intent, as a JSON-formatted list."""
    
    demonstrations = [
        {
            "query": "How do I locate my card?",
            "keyphrases": ["card status", "card location", "card tracking"]
        },
        {
            "query": "I still have not received my new card, I ordered over a week ago.",
            "keyphrases": ["card arrival", "card delivery status", "delayed card"]
        },
        {
            "query": "When will my card get here?",
            "keyphrases": ["card arrival", "card delivery estimate", "card shipping"]
        },
        {
            "query": "Is there a way to know when my card will arrive?",
            "keyphrases": ["card delivery estimate", "card arrival", "card tracking"]
        }
    ]
    
    demo_text = "\n\nExamples:\n"
    for demo in demonstrations:
        demo_text += f"Query: {demo['query']}\nKeyphrases: {json.dumps(demo['keyphrases'])}\n\n"
    
    all_keyphrases = {}
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating keyphrases"):
        batch_texts = texts[i:i+batch_size]
        batch_keyphrases = {}
        
        for text in tqdm(batch_texts, desc="Batch progress", leave=False):
            if text in all_keyphrases:
                continue
                
            prompt = f"{system_prompt}{demo_text}Query: {text}\nKeyphrases:"
            
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"{demo_text}Query: {text}\nKeyphrases:"}
                    ],
                    temperature=0.2
                )

                keyphrases_text = response.choices[0].message.content.strip()
                
                try:
                    if keyphrases_text.startswith("[") and keyphrases_text.endswith("]"):
                        keyphrases = json.loads(keyphrases_text)
                    else:
                        import re
                        json_match = re.search(r'\[.*\]', keyphrases_text, re.DOTALL)
                        if json_match:
                            keyphrases = json.loads(json_match.group(0))
                        else:
                            keyphrases = [k.strip(' "\'') for k in keyphrases_text.split(',')]
                except Exception as e:
                    print(f"Error parsing keyphrases for '{text}': {e}")
                    print(f"Raw response: {keyphrases_text}")
                    keyphrases = [text]  
                
                batch_keyphrases[text] = keyphrases
    
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error generating keyphrases for '{text}': {e}")
                batch_keyphrases[text] = [text]  
        
        all_keyphrases.update(batch_keyphrases)
        
        with open(cache_file, 'w') as f:
            json.dump(all_keyphrases, f)
    
    return all_keyphrases

def run_keyphrase_clustering(dataset, base_embeddings, model_name="all-MiniLM-L6-v2", n_clusters=77):
    """
    Run clustering with LLM keyphrase expansion
    """
    print("Running LLM keyphrase expansion clustering...")
    
    train_texts = [item["text"] for item in dataset["train"]]
    test_texts = [item["text"] for item in dataset["test"]]
    
    train_keyphrases = generate_keyphrases_with_llm(train_texts)
    test_keyphrases = generate_keyphrases_with_llm(test_texts)
    
    train_keyphrase_texts = [" ".join(train_keyphrases[text]) for text in train_texts]
    test_keyphrase_texts = [" ".join(test_keyphrases[text]) for text in test_texts]
    
    train_keyphrase_embeddings = embed_texts(train_keyphrase_texts, model_name)
    test_keyphrase_embeddings = embed_texts(test_keyphrase_texts, model_name)
    
    train_combined = np.concatenate([base_embeddings["train"], train_keyphrase_embeddings], axis=1)
    test_combined = np.concatenate([base_embeddings["test"], test_keyphrase_embeddings], axis=1)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    train_clusters = kmeans.fit_predict(train_combined)
    
    train_metrics = evaluate_clustering(
        [item["label"] for item in dataset["train"]], 
        train_clusters, 
        n_clusters
    )
    
    test_clusters = kmeans.predict(test_combined)
    
    test_metrics = evaluate_clustering(
        [item["label"] for item in dataset["test"]], 
        test_clusters, 
        n_clusters
    )
    
    results = {
        "method": "keyphrase_clustering",
        "n_clusters": n_clusters,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics
    }
    

    save_results(results, "keyphrase_clustering")
    
    return results, kmeans, train_clusters, test_clusters, train_combined, test_combined

def generate_pairwise_constraints_with_llm(dataset, n_constraints=1000, cache_dir=CACHE_DIR):
    """
    Generate pairwise constraints using LLM
    
    Args:
        dataset: The dataset dictionary containing train and test data
        n_constraints: Number of constraints to generate
        cache_dir: Directory for caching results
        
    Returns:
        Dictionary with must-link and cannot-link constraints
    """
    # Create cache directory
    constraints_dir = os.path.join(cache_dir, "constraints")
    os.makedirs(constraints_dir, exist_ok=True)
    
    # Cache file path with dataset size info to avoid mismatches
    train_size = len(dataset["train"])
    cache_file = os.path.join(constraints_dir, f"constraints_{n_constraints}_size{train_size}.json")
    
    # Try to load from cache with error handling
    if os.path.exists(cache_file):
        print(f"Loading constraints from cache: {cache_file}")
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error loading constraints from cache: {e}")
            print("Cache file appears to be corrupted. Regenerating constraints...")
           
            os.remove(cache_file)
    
 
    train_data = dataset["train"]
    
    print(f"Number of training examples: {len(train_data)}")
    
    system_prompt = """I am trying to cluster online banking queries based on whether they express the same intent. I will provide you with two queries, and I need you to determine if they express the same intent. Answer with 'Yes' if they express the same intent, or 'No' if they express different intents."""
    
    demonstrations = [
        {
            "query1": "How do I locate my card?",
            "query2": "When will I get my new card?",
            "same_intent": "Yes",
            "explanation": "Both queries are about card arrival/delivery."
        },
        {
            "query1": "Why has my card been blocked?",
            "query2": "I need to change my PIN code",
            "same_intent": "No",
            "explanation": "The first query is about a blocked card, while the second is about changing a PIN."
        },
        {
            "query1": "I need to top up my account",
            "query2": "How do I add money to my account?",
            "same_intent": "Yes",
            "explanation": "Both queries are about adding funds to an account."
        },
        {
            "query1": "I've lost my card",
            "query2": "My card isn't working at ATMs",
            "same_intent": "No",
            "explanation": "The first query is about a lost card, while the second is about a card not working."
        }
    ]
    
    demo_text = "\n\nExamples:\n"
    for demo in demonstrations:
        demo_text += f"Query 1: {demo['query1']}\nQuery 2: {demo['query2']}\nSame Intent: {demo['same_intent']}\nExplanation: {demo['explanation']}\n\n"
    
    np.random.seed(42)
    
    # Strategy: Mix of close pairs (likely same intent) and random pairs (likely different intent)
    # This follows the Explore-Consolidate algorithm mentioned in the paper
    
    train_texts = [item["text"] for item in train_data]
    train_labels = [item["label"] for item in train_data]
    
    embeddings_file = os.path.join(cache_dir, f"train_embeddings_size{train_size}_for_constraints.pkl")
    if os.path.exists(embeddings_file):
        try:
            with open(embeddings_file, 'rb') as f:
                cached_embeddings = pickle.load(f)
            if len(cached_embeddings) != len(train_texts):
                print(f"Warning: Cached embeddings size ({len(cached_embeddings)}) doesn't match training data size ({len(train_texts)})")
                print("Regenerating embeddings...")
                train_embeddings = embed_texts(train_texts, "all-MiniLM-L6-v2")
                with open(embeddings_file, 'wb') as f:
                    pickle.dump(train_embeddings, f)
            else:
                train_embeddings = cached_embeddings
        except Exception as e:
            print(f"Error loading embeddings from cache: {e}")
            print("Regenerating embeddings...")
            train_embeddings = embed_texts(train_texts, "all-MiniLM-L6-v2")
            with open(embeddings_file, 'wb') as f:
                pickle.dump(train_embeddings, f)
    else:
        train_embeddings = embed_texts(train_texts, "all-MiniLM-L6-v2")
        try:
            with open(embeddings_file, 'wb') as f:
                pickle.dump(train_embeddings, f)
        except Exception as e:
            print(f"Error saving embeddings to cache: {e}")
    
    if len(train_embeddings) != len(train_texts):
        raise ValueError(f"Embeddings shape ({len(train_embeddings)}) doesn't match training data size ({len(train_texts)})")
    
    # Calculate pairwise distances
    from sklearn.metrics.pairwise import cosine_distances
    distances = cosine_distances(train_embeddings)
    
    # Generate candidate pairs
    candidate_pairs = []
    max_idx = len(train_texts) - 1
    
    # 1. Close pairs (likely same intent)
    for i in range(len(train_texts)):
        # Find the 5 closest neighbors (or fewer if not enough data)
        k_neighbors = min(5, len(train_texts) - 1)
        if k_neighbors <= 0:
            print("Warning: Not enough data points for finding neighbors")
            break
            
        closest_indices = np.argsort(distances[i])[1:k_neighbors+1]  
        for j in closest_indices:
            if i != j and j < len(train_texts):
                candidate_pairs.append((int(i), int(j), float(distances[i][j])))
    
    # 2. Random pairs (likely different intent)
    if max_idx > 0:  
        for _ in range(min(n_constraints // 2, 1000)):  
            i = np.random.randint(0, max_idx + 1)
            j = np.random.randint(0, max_idx + 1)
            if i != j:
                candidate_pairs.append((int(i), int(j), float(distances[i][j])))
    
    print(f"Generated {len(candidate_pairs)} candidate pairs")
    if len(candidate_pairs) == 0:
        print("Warning: No candidate pairs generated. Check your data.")
        return {"must_link": [], "cannot_link": []}
    
    # Sort by distance and take the first n_constraints
    candidate_pairs.sort(key=lambda x: x[2])
    n_to_use = min(n_constraints, len(candidate_pairs))
    selected_pairs = [(i, j) for i, j, _ in candidate_pairs[:n_to_use]]
    
    print(f"Selected {len(selected_pairs)} pairs for constraint generation")
    
    # Generate constraints using LLM
    constraints = {"must_link": [], "cannot_link": []}
    processed_count = 0
    
    for i, (idx1, idx2) in enumerate(tqdm(selected_pairs, desc="Generating constraints")):
        if idx1 >= len(train_texts) or idx2 >= len(train_texts):
            print(f"Warning: Skipping pair with invalid indices ({idx1}, {idx2}). Max valid index: {len(train_texts)-1}")
            continue
            
        text1 = train_texts[idx1]
        text2 = train_texts[idx2]
        
        try:
            prompt = f"{system_prompt}{demo_text}Query 1: {text1}\nQuery 2: {text2}\nSame Intent:"
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{demo_text}Query 1: {text1}\nQuery 2: {text2}\nSame Intent:"}
                ],
                temperature=0.1
            )
            
            result = response.choices[0].message.content.strip().lower()
            # Determine constraint type
            if "yes" in result:
                constraints["must_link"].append((int(idx1), int(idx2)))
            else:
                constraints["cannot_link"].append((int(idx1), int(idx2)))
            
            processed_count += 1
            
            time.sleep(0.5)
    
            if (processed_count % 5 == 0) or (i == len(selected_pairs) - 1):
                try:
                    serializable_constraints = {
                        "must_link": [[int(i), int(j)] for i, j in constraints["must_link"]],
                        "cannot_link": [[int(i), int(j)] for i, j in constraints["cannot_link"]]
                    }
                    
                    json_str = json.dumps(serializable_constraints)
                    
                    with open(cache_file, 'w') as f:
                        f.write(json_str)
                    
                    print(f"Saved {processed_count}/{n_to_use} constraints to cache")
                except Exception as e:
                    print(f"Error saving intermediate constraints to cache: {e}")
            
        except Exception as e:
            print(f"Error generating constraint for pair ({idx1}, {idx2}): {e}")
    
    try:
        serializable_constraints = {
            "must_link": [[int(i), int(j)] for i, j in constraints["must_link"]],
            "cannot_link": [[int(i), int(j)] for i, j in constraints["cannot_link"]]
        }
        
        with open(cache_file, 'w') as f:
            json.dump(serializable_constraints, f, indent=2)
        
        print(f"Successfully generated and saved {len(constraints['must_link'])} must-link and {len(constraints['cannot_link'])} cannot-link constraints")
        
        return serializable_constraints
    except Exception as e:
        print(f"Error saving final constraints to cache: {e}")
        
        return {
            "must_link": [[int(i), int(j)] for i, j in constraints["must_link"]],
            "cannot_link": [[int(i), int(j)] for i, j in constraints["cannot_link"]]
        }
 

def pckm_objective(X, centers, ml, cl, w):
    """
    Objective function for PCKMeans
    """
    from sklearn.metrics.pairwise import euclidean_distances
    print("calculating pckm objective")

    distances = euclidean_distances(X, centers)
    labels = np.argmin(distances, axis=1)
    kmeans_obj = np.sum(np.min(distances, axis=1))

    penalty = 0
    
    # Must-link constraints
    for i, j in ml:
        if labels[i] != labels[j]:
            penalty += w
    
    # Cannot-link constraints
    for i, j in cl:
        if labels[i] == labels[j]:
            penalty += w
    print(f"pckm objective: {kmeans_obj + penalty}")
    return kmeans_obj + penalty

def run_pckm(X, n_clusters, ml_constraints, cl_constraints, w=1.0, max_iter=100):
    """
    Run PCKMeans (Pairwise Constrained K-Means) with robust error handling
    """
    ml_constraints = [(int(i), int(j)) for i, j in ml_constraints if i < len(X) and j < len(X)]
    cl_constraints = [(int(i), int(j)) for i, j in cl_constraints if i < len(X) and j < len(X)]
    
    print(f"Starting PCKMeans with {len(ml_constraints)} must-link and {len(cl_constraints)} cannot-link constraints")
    
    # Initialize with k-means++
    from sklearn.cluster import kmeans_plusplus
    centers, _ = kmeans_plusplus(X, n_clusters, random_state=42)
    
    # Run PCKMeans
    best_labels = None
    best_objective = float('inf')
    
    for iteration in range(max_iter):
        print(f"PCKMeans iteration {iteration+1}/{max_iter}")
        
        # E-step: Assign points to clusters
        distances = euclidean_distances(X, centers)
        labels = np.argmin(distances, axis=1)
        
        changed = True
        max_inner_iterations = 100
        inner_iter = 0
        
        while changed and inner_iter < max_inner_iterations:
            inner_iter += 1
            changed = False
            changes_count = 0
            
            # Apply must-link constraints
            for i, j in ml_constraints:
                if labels[i] != labels[j]:
                    if distances[i, labels[i]] < distances[j, labels[j]]:
                        labels[j] = labels[i]
                    else:
                        labels[i] = labels[j]
                    changed = True
                    changes_count += 1
            
            # Apply cannot-link constraints
            for i, j in cl_constraints:
                if labels[i] == labels[j]:
                    dist_j = distances[j].copy()
                    dist_j[labels[j]] = float('inf')
                    if np.all(np.isinf(dist_j)):
                        new_cluster = np.random.choice([c for c in range(n_clusters) if c != labels[j]])
                    else:
                        new_cluster = np.argmin(dist_j)
                    
                    labels[j] = new_cluster
                    changed = True
                    changes_count += 1
            
            if inner_iter % 10 == 0 or changes_count > 0:
                print(f"  Inner iteration {inner_iter}: {changes_count} changes")
        
        # M-step: Update centers
        new_centers = np.zeros_like(centers)
        empty_clusters = 0
        
        for k in range(n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                new_centers[k] = np.mean(cluster_points, axis=0)
            else:
                empty_clusters += 1
                new_centers[k] = X[np.random.randint(0, X.shape[0])]
        
        current_objective = pckm_objective(X, new_centers, ml_constraints, cl_constraints, w)
        print(f"  Objective: {current_objective:.4f}, Empty clusters: {empty_clusters}")
        
        if current_objective < best_objective:
            best_objective = current_objective
            best_labels = labels.copy()
            print(f"  New best objective: {best_objective:.4f}")

        center_shift = np.sum(np.sqrt(np.sum((centers - new_centers) ** 2, axis=1)))
        print(f"  Center shift: {center_shift:.6f}")
        centers = new_centers
        
        if center_shift < 1e-4:
            print(f"Converged after {iteration+1} iterations")
            break
    
    return best_labels, centers
def run_pairwise_constraint_clustering(dataset, embeddings, n_clusters=77, n_constraints=1000, constraint_weight=1.0):
    """
    Run Pairwise Constraint K-Means clustering
    """
    print(f"Running PCKMeans with {n_constraints} constraints...")
    
    constraints = generate_pairwise_constraints_with_llm(dataset, n_constraints)
    
    # Extract must-link and cannot-link constraints
    ml_constraints = constraints["must_link"]
    cl_constraints = constraints["cannot_link"]
    
    print(f"Generated {len(ml_constraints)} must-link and {len(cl_constraints)} cannot-link constraints")
    
    # Run PCKMeans on training data
    train_labels, centers = run_pckm(
        embeddings["train"], 
        n_clusters, 
        ml_constraints, 
        cl_constraints, 
        w=constraint_weight
    )
    print(f"pckm objective: {pckm_objective}")
 
    train_metrics = evaluate_clustering(
        [item["label"] for item in dataset["train"]], 
        train_labels, 
        n_clusters
    )
    print(f"pckm objective: {pckm_objective}")

    distances = euclidean_distances(embeddings["test"], centers)
    test_labels = np.argmin(distances, axis=1)
    
    test_metrics = evaluate_clustering(
        [item["label"] for item in dataset["test"]], 
        test_labels, 
        n_clusters
    )
    
    results = {
        "method": "pckm",
        "n_clusters": n_clusters,
        "n_constraints": n_constraints,
        "constraint_weight": constraint_weight,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics
    }
    
    save_results(results, "pckm")
    
    return results, centers, train_labels, test_labels

def run_llm_correction(dataset, embeddings, kmeans_model, train_clusters, test_clusters, n_clusters=77, n_corrections=500):
    """
    Run LLM correction on low-confidence cluster assignments
    """
    print(f"Running LLM correction for {n_corrections} points...")
    
    # Identify low-confidence points in test set
    test_texts = [item["text"] for item in dataset["test"]]
    test_distances = kmeans_model.transform(embeddings["test"])
    
    # Calculate margin between closest and second-closest cluster
    sorted_distances = np.sort(test_distances, axis=1)
    margins = sorted_distances[:, 1] - sorted_distances[:, 0]
    
    # Get indices of points with lowest margins (lowest confidence)
    low_confidence_indices = np.argsort(margins)[:n_corrections]
    
    # System prompt for LLM
    system_prompt = """I am trying to cluster online banking queries based on their intent. I'll provide you with a query and some example queries from a cluster. Tell me if the query belongs to this cluster or not. Answer with 'Yes' if it belongs to the cluster, or 'No' if it doesn't."""
    
    corrections_dir = os.path.join(CACHE_DIR, "corrections")
    os.makedirs(corrections_dir, exist_ok=True)
    cache_file = os.path.join(corrections_dir, f"corrections_{n_corrections}.json")
    
    if os.path.exists(cache_file):
        print(f"Loading corrections from cache: {cache_file}")
        with open(cache_file, 'r') as f:
            corrected_clusters_raw = json.load(f)
            corrected_clusters = {int(k): v for k, v in corrected_clusters_raw.items()}
    else:
        # Get representative examples for each cluster
        cluster_representatives = {}
        for k in range(n_clusters):
            # Get indices of points in this cluster
            cluster_indices = np.where(train_clusters == k)[0]
            
            if len(cluster_indices) > 0:
                # Get the 3 points closest to centroid
                cluster_center = kmeans_model.cluster_centers_[k]
                distances_to_center = np.linalg.norm(embeddings["train"][cluster_indices] - cluster_center, axis=1)
                closest_indices = cluster_indices[np.argsort(distances_to_center)[:3]]
                
                # Get the text of these points
                cluster_representatives[k] = [dataset["train"][i]["text"] for i in closest_indices]
            else:
                cluster_representatives[k] = []
        
        # Correct clusters for low-confidence points
        corrected_clusters = {}
        
        for i, idx in enumerate(tqdm(low_confidence_indices, desc="Correcting clusters")):
         
            idx = int(idx)
            query = test_texts[idx]
            current_cluster = int(test_clusters[idx])  
            
            # Get the current cluster's representatives
            current_representatives = cluster_representatives[current_cluster]
            
            if not current_representatives:
                corrected_clusters[idx] = current_cluster
                continue
            
            # First, check if the current assignment is correct
            prompt = f"{system_prompt}\n\nQuery: {query}\n\nCluster examples:\n"
            for rep in current_representatives:
                prompt += f"- {rep}\n"
            prompt += "\nDoes the query belong to this cluster?"
            
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1
                )
            
                result = response.choices[0].message.content.strip().lower()
                
                # If current assignment is correct, keep it
                if "yes" in result:
                    corrected_clusters[idx] = current_cluster
                else:
                    # Find alternative clusters
                    # Get the 4 next closest clusters
                    sorted_cluster_indices = np.argsort(test_distances[idx])
                    alternative_clusters = sorted_cluster_indices[1:5]  # Skip the current cluster
                    
                    # Check each alternative
                    for alt_cluster in alternative_clusters:
                        alt_cluster = int(alt_cluster)
                        alt_representatives = cluster_representatives[alt_cluster]
                        
                        if not alt_representatives:
                            continue
                        
                        alt_prompt = f"{system_prompt}\n\nQuery: {query}\n\nCluster examples:\n"
                        for rep in alt_representatives:
                            alt_prompt += f"- {rep}\n"
                        alt_prompt += "\nDoes the query belong to this cluster?"
                        
                        alt_response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": alt_prompt}
                            ],
                            temperature=0.1
                        )
                        
                        alt_result = alt_response.choices[0].message.content.strip().lower()
                        
                        # If found a better cluster, assign to it
                        if "yes" in alt_result:
                            corrected_clusters[idx] = alt_cluster
                            break
                    else:
                        # If no better cluster found, keep the original
                        corrected_clusters[idx] = current_cluster
                time.sleep(0.5)
                
                if (i + 1) % 20 == 0:
                    serializable_clusters = {int(k): int(v) for k, v in corrected_clusters.items()}
                    with open(cache_file, 'w') as f:
                        json.dump(serializable_clusters, f)
                    
            except Exception as e:
                print(f"Error correcting cluster for point {idx}: {e}")
                corrected_clusters[idx] = current_cluster
        
        serializable_clusters = {int(k): int(v) for k, v in corrected_clusters.items()}
        with open(cache_file, 'w') as f:
            json.dump(serializable_clusters, f)

    corrected_test_clusters = test_clusters.copy()
    for idx, cluster in corrected_clusters.items():
        corrected_test_clusters[idx] = cluster
    
    original_metrics = evaluate_clustering(
        [item["label"] for item in dataset["test"]], 
        test_clusters, 
        n_clusters
    )
    
    corrected_metrics = evaluate_clustering(
        [item["label"] for item in dataset["test"]], 
        corrected_test_clusters, 
        n_clusters
    )
    
    results = {
        "method": "llm_correction",
        "n_clusters": n_clusters,
        "n_corrections": n_corrections,
        "original_metrics": original_metrics,
        "corrected_metrics": corrected_metrics,
        "improvement": {
            "accuracy": corrected_metrics["accuracy"] - original_metrics["accuracy"],
            "nmi": corrected_metrics["nmi"] - original_metrics["nmi"],
            "ari": corrected_metrics["ari"] - original_metrics["ari"]
        }
    }
    
    
    save_results(results, "llm_correction")
    
    return results, corrected_test_clusters
def main():
    models_dir = os.path.join(CACHE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    """ Load dataset
        create_banking77_dataset(
            input_csv_path="datasets/banking77/train.csv",
            output_dir="datasets/banking77",
            train_size=3000,
            test_size=1000
        )
        """
    dataset = load_banking77_data(train_limit=3000, test_limit=500)    
    
    embeddings = get_dataset_embeddings(dataset, model_name="all-MiniLM-L6-v2")
    kmeans_model_path = os.path.join(models_dir, "kmeans_model.pkl")
    kmeans_results_path = os.path.join(models_dir, "kmeans_results.pkl")
    keyphrase_model_path = os.path.join(models_dir, "keyphrase_model.pkl")
    keyphrase_results_path = os.path.join(models_dir, "keyphrase_results.pkl")
    pckm_model_path = os.path.join(models_dir, "pckm_model.pkl")
    pckm_results_path = os.path.join(models_dir, "pckm_results.pkl")
    
    # Run or load baseline K-Means
    if os.path.exists(kmeans_model_path) and os.path.exists(kmeans_results_path):
        print("Loading K-Means model and results from cache...")
        with open(kmeans_model_path, 'rb') as f:
            kmeans_data = pickle.load(f)
            kmeans_model = kmeans_data['model']
            train_clusters = kmeans_data['train_clusters']
            test_clusters = kmeans_data['test_clusters']
        
        with open(kmeans_results_path, 'rb') as f:
            baseline_results = pickle.load(f)
    else:
        print("Running K-Means clustering...")
        baseline_results, kmeans_model, train_clusters, test_clusters = run_kmeans_baseline(
            dataset, embeddings, n_clusters=77
        )
        
        # Save model and results
        kmeans_data = {
            'model': kmeans_model,
            'train_clusters': train_clusters,
            'test_clusters': test_clusters
        }
        with open(kmeans_model_path, 'wb') as f:
            pickle.dump(kmeans_data, f)
        
        with open(kmeans_results_path, 'wb') as f:
            pickle.dump(baseline_results, f)
    
    print("Baseline results:")
    print(f"Train - Accuracy: {baseline_results['train_metrics']['accuracy']:.4f}, "
          f"NMI: {baseline_results['train_metrics']['nmi']:.4f}, "
          f"ARI: {baseline_results['train_metrics']['ari']:.4f}")
    print(f"Test - Accuracy: {baseline_results['test_metrics']['accuracy']:.4f}, "
          f"NMI: {baseline_results['test_metrics']['nmi']:.4f}, "
          f"ARI: {baseline_results['test_metrics']['ari']:.4f}")
    
    # Run or load keyphrase expansion clustering
    if os.path.exists(keyphrase_model_path) and os.path.exists(keyphrase_results_path):
        print("Loading Keyphrase Expansion model and results from cache...")
        with open(keyphrase_model_path, 'rb') as f:
            keyphrase_data = pickle.load(f)
            keyphrase_model = keyphrase_data['model']
            keyphrase_train_clusters = keyphrase_data['train_clusters']
            keyphrase_test_clusters = keyphrase_data['test_clusters']
            train_combined = keyphrase_data['train_combined']
            test_combined = keyphrase_data['test_combined']
        
        with open(keyphrase_results_path, 'rb') as f:
            keyphrase_results = pickle.load(f)
    else:
        print("Running Keyphrase Expansion clustering...")
        keyphrase_results, keyphrase_model, keyphrase_train_clusters, keyphrase_test_clusters, train_combined, test_combined = run_keyphrase_clustering(
            dataset, embeddings, n_clusters=77
        )
        
        # Save model and results
        keyphrase_data = {
            'model': keyphrase_model,
            'train_clusters': keyphrase_train_clusters,
            'test_clusters': keyphrase_test_clusters,
            'train_combined': train_combined,
            'test_combined': test_combined
        }
        with open(keyphrase_model_path, 'wb') as f:
            pickle.dump(keyphrase_data, f)
        
        with open(keyphrase_results_path, 'wb') as f:
            pickle.dump(keyphrase_results, f)
    
    print("\nKeyphrase expansion results:")
    print(f"Train - Accuracy: {keyphrase_results['train_metrics']['accuracy']:.4f}, "
          f"NMI: {keyphrase_results['train_metrics']['nmi']:.4f}, "
          f"ARI: {keyphrase_results['train_metrics']['ari']:.4f}")
    print(f"Test - Accuracy: {keyphrase_results['test_metrics']['accuracy']:.4f}, "
          f"NMI: {keyphrase_results['test_metrics']['nmi']:.4f}, "
          f"ARI: {keyphrase_results['test_metrics']['ari']:.4f}")
    
    # Run or load pairwise constraint clustering
    if os.path.exists(pckm_model_path) and os.path.exists(pckm_results_path):
        print("Loading PCKMeans model and results from cache...")
        with open(pckm_model_path, 'rb') as f:
            pckm_data = pickle.load(f)
            pckm_centers = pckm_data['centers']
            pckm_train_clusters = pckm_data['train_clusters']
            pckm_test_clusters = pckm_data['test_clusters']
        
        with open(pckm_results_path, 'rb') as f:
            pckm_results = pickle.load(f)
    else:
        print("Running Pairwise Constraint K-Means clustering...")
        pckm_results, pckm_centers, pckm_train_clusters, pckm_test_clusters = run_pairwise_constraint_clustering(
            dataset, embeddings, n_clusters=77, n_constraints=1000
        )
        
        # Save model and results
        pckm_data = {
            'centers': pckm_centers,
            'train_clusters': pckm_train_clusters,
            'test_clusters': pckm_test_clusters
        }
        with open(pckm_model_path, 'wb') as f:
            pickle.dump(pckm_data, f)
        
        with open(pckm_results_path, 'wb') as f:
            pickle.dump(pckm_results, f)
    
    print("\nPairwise constraint clustering results:")
    print(f"Train - Accuracy: {pckm_results['train_metrics']['accuracy']:.4f}, "
          f"NMI: {pckm_results['train_metrics']['nmi']:.4f}, "
          f"ARI: {pckm_results['train_metrics']['ari']:.4f}")
    print(f"Test - Accuracy: {pckm_results['test_metrics']['accuracy']:.4f}, "
          f"NMI: {pckm_results['test_metrics']['nmi']:.4f}, "
          f"ARI: {pckm_results['test_metrics']['ari']:.4f}")
    
    # Run LLM correction
    # Ask which model to use for correction
    print("\nWhich model would you like to use for LLM correction?")
    print("1: K-Means Baseline")
    print("2: Keyphrase Expansion")
    print("3: Pairwise Constraint K-Means")
    
    choice = input("Enter your choice (1-3): ").strip()
    
    if choice == '1':
        print("Running LLM correction on K-Means Baseline model...")
        correction_results, corrected_test_clusters = run_llm_correction(
            dataset, embeddings, kmeans_model, train_clusters, test_clusters, 
            n_clusters=77, n_corrections=500
        )
        original_results = baseline_results
    elif choice == '2':
        print("Running LLM correction on Keyphrase Expansion model...")
        correction_results, corrected_test_clusters = run_llm_correction(
            dataset, {"train": train_combined, "test": test_combined}, 
            keyphrase_model, keyphrase_train_clusters, keyphrase_test_clusters, 
            n_clusters=77, n_corrections=500
        )
        original_results = keyphrase_results
    elif choice == '3':
        # For PCKM, we need to create a KMeans-like model object for compatibility
        from sklearn.cluster import KMeans
        pckm_kmeans = KMeans(n_clusters=77)
        pckm_kmeans.cluster_centers_ = pckm_centers
        
        print("Running LLM correction on Pairwise Constraint K-Means model...")
        correction_results, corrected_test_clusters = run_llm_correction(
            dataset, embeddings, pckm_kmeans, pckm_train_clusters, pckm_test_clusters, 
            n_clusters=77, n_corrections=500
        )
        original_results = pckm_results
    else:
        print("Invalid choice. Skipping LLM correction.")
        correction_results = None
        original_results = None
    
    if correction_results:
        print("\nLLM correction results:")
        print(f"Original - Accuracy: {correction_results['original_metrics']['accuracy']:.4f}, "
              f"NMI: {correction_results['original_metrics']['nmi']:.4f}, "
              f"ARI: {correction_results['original_metrics']['ari']:.4f}")
        print(f"Corrected - Accuracy: {correction_results['corrected_metrics']['accuracy']:.4f}, "
              f"NMI: {correction_results['corrected_metrics']['nmi']:.4f}, "
              f"ARI: {correction_results['corrected_metrics']['ari']:.4f}")
        print(f"Improvement - Accuracy: {correction_results['improvement']['accuracy']:.4f}, "
              f"NMI: {correction_results['improvement']['nmi']:.4f}, "
              f"ARI: {correction_results['improvement']['ari']:.4f}")
        
        # Save the correction results
        correction_path = os.path.join(models_dir, f"correction_results_model{choice}.pkl")
        with open(correction_path, 'wb') as f:
            pickle.dump({
                'correction_results': correction_results,
                'corrected_test_clusters': corrected_test_clusters,
                'original_model': choice
            }, f)
        print(f"Correction results saved to {correction_path}")
    
    if all([baseline_results, keyphrase_results, pckm_results, correction_results]):
        compare_all_methods(baseline_results, keyphrase_results, pckm_results, correction_results)

def run_specific_task(task_name):
    """
    Run a specific task in the pipeline
    
    Args:
        task_name: String identifier for the task ('kmeans', 'keyphrase', 'pckm', 'correction')
    """
    models_dir = os.path.join(CACHE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    dataset = load_banking77_data(train_limit=3000, test_limit=500)
    
    embeddings = get_dataset_embeddings(dataset, model_name="all-MiniLM-L6-v2")
    
    if task_name == 'kmeans':
        kmeans_model_path = os.path.join(models_dir, "kmeans_model.pkl")
        kmeans_results_path = os.path.join(models_dir, "kmeans_results.pkl")
        
        baseline_results, kmeans_model, train_clusters, test_clusters = run_kmeans_baseline(
            dataset, embeddings, n_clusters=77
        )
        
        # Save model and results
        kmeans_data = {
            'model': kmeans_model,
            'train_clusters': train_clusters,
            'test_clusters': test_clusters
        }
        with open(kmeans_model_path, 'wb') as f:
            pickle.dump(kmeans_data, f)
        
        with open(kmeans_results_path, 'wb') as f:
            pickle.dump(baseline_results, f)
        
        print("Baseline results:")
        print(f"Train - Accuracy: {baseline_results['train_metrics']['accuracy']:.4f}, "
              f"NMI: {baseline_results['train_metrics']['nmi']:.4f}, "
              f"ARI: {baseline_results['train_metrics']['ari']:.4f}")
        print(f"Test - Accuracy: {baseline_results['test_metrics']['accuracy']:.4f}, "
              f"NMI: {baseline_results['test_metrics']['nmi']:.4f}, "
              f"ARI: {baseline_results['test_metrics']['ari']:.4f}")
    
    elif task_name == 'keyphrase':
        # Run Keyphrase Expansion
        keyphrase_model_path = os.path.join(models_dir, "keyphrase_model.pkl")
        keyphrase_results_path = os.path.join(models_dir, "keyphrase_results.pkl")
        
        keyphrase_results, keyphrase_model, keyphrase_train_clusters, keyphrase_test_clusters, train_combined, test_combined = run_keyphrase_clustering(
            dataset, embeddings, n_clusters=77
        )
        
        # Save model and results
        keyphrase_data = {
            'model': keyphrase_model,
            'train_clusters': keyphrase_train_clusters,
            'test_clusters': keyphrase_test_clusters,
            'train_combined': train_combined,
            'test_combined': test_combined
        }
        with open(keyphrase_model_path, 'wb') as f:
            pickle.dump(keyphrase_data, f)
        
        with open(keyphrase_results_path, 'wb') as f:
            pickle.dump(keyphrase_results, f)
        
        print("Keyphrase expansion results:")
        print(f"Train - Accuracy: {keyphrase_results['train_metrics']['accuracy']:.4f}, "
              f"NMI: {keyphrase_results['train_metrics']['nmi']:.4f}, "
              f"ARI: {keyphrase_results['train_metrics']['ari']:.4f}")
        print(f"Test - Accuracy: {keyphrase_results['test_metrics']['accuracy']:.4f}, "
              f"NMI: {keyphrase_results['test_metrics']['nmi']:.4f}, "
              f"ARI: {keyphrase_results['test_metrics']['ari']:.4f}")
    
    elif task_name == 'pckm':
        # Run PCKMeans
        pckm_model_path = os.path.join(models_dir, "pckm_model.pkl")
        pckm_results_path = os.path.join(models_dir, "pckm_results.pkl")
        
        pckm_results, pckm_centers, pckm_train_clusters, pckm_test_clusters = run_pairwise_constraint_clustering(
            dataset, embeddings, n_clusters=77, n_constraints=1000
        )
        
        # Save model and results
        pckm_data = {
            'centers': pckm_centers,
            'train_clusters': pckm_train_clusters,
            'test_clusters': pckm_test_clusters
        }
        with open(pckm_model_path, 'wb') as f:
            pickle.dump(pckm_data, f)
        
        with open(pckm_results_path, 'wb') as f:
            pickle.dump(pckm_results, f)
        
        print("Pairwise constraint clustering results:")
        print(f"Train - Accuracy: {pckm_results['train_metrics']['accuracy']:.4f}, "
              f"NMI: {pckm_results['train_metrics']['nmi']:.4f}, "
              f"ARI: {pckm_results['train_metrics']['ari']:.4f}")
        print(f"Test - Accuracy: {pckm_results['test_metrics']['accuracy']:.4f}, "
              f"NMI: {pckm_results['test_metrics']['nmi']:.4f}, "
              f"ARI: {pckm_results['test_metrics']['ari']:.4f}")
    
    elif task_name == 'correction':
        # Run LLM correction on a specified model
        print("Which model would you like to use for LLM correction?")
        print("1: K-Means Baseline")
        print("2: Keyphrase Expansion")
        print("3: Pairwise Constraint K-Means")
        
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == '1':
            # Load K-Means model
            kmeans_model_path = os.path.join(models_dir, "kmeans_model.pkl")
            kmeans_results_path = os.path.join(models_dir, "kmeans_results.pkl")
            
            if not os.path.exists(kmeans_model_path) or not os.path.exists(kmeans_results_path):
                print("K-Means model not found. Please run K-Means first.")
                return
            
            with open(kmeans_model_path, 'rb') as f:
                kmeans_data = pickle.load(f)
                kmeans_model = kmeans_data['model']
                train_clusters = kmeans_data['train_clusters']
                test_clusters = kmeans_data['test_clusters']
            
            with open(kmeans_results_path, 'rb') as f:
                baseline_results = pickle.load(f)
            
            print("Running LLM correction on K-Means Baseline model...")
            correction_results, corrected_test_clusters = run_llm_correction(
                dataset, embeddings, kmeans_model, train_clusters, test_clusters, 
                n_clusters=77, n_corrections=500
            )
            original_results = baseline_results
        
        elif choice == '2':
            # Load Keyphrase model
            keyphrase_model_path = os.path.join(models_dir, "keyphrase_model.pkl")
            keyphrase_results_path = os.path.join(models_dir, "keyphrase_results.pkl")
            
            if not os.path.exists(keyphrase_model_path) or not os.path.exists(keyphrase_results_path):
                print("Keyphrase model not found. Please run Keyphrase expansion first.")
                return
            
            with open(keyphrase_model_path, 'rb') as f:
                keyphrase_data = pickle.load(f)
                keyphrase_model = keyphrase_data['model']
                keyphrase_train_clusters = keyphrase_data['train_clusters']
                keyphrase_test_clusters = keyphrase_data['test_clusters']
                train_combined = keyphrase_data['train_combined']
                test_combined = keyphrase_data['test_combined']
            
            with open(keyphrase_results_path, 'rb') as f:
                keyphrase_results = pickle.load(f)
            
            print("Running LLM correction on Keyphrase Expansion model...")
            correction_results, corrected_test_clusters = run_llm_correction(
                dataset, {"train": train_combined, "test": test_combined}, 
                keyphrase_model, keyphrase_train_clusters, keyphrase_test_clusters, 
                n_clusters=77, n_corrections=500
            )
            original_results = keyphrase_results
        
        elif choice == '3':
            # Load PCKMeans model
            pckm_model_path = os.path.join(models_dir, "pckm_model.pkl")
            pckm_results_path = os.path.join(models_dir, "pckm_results.pkl")
            
            if not os.path.exists(pckm_model_path) or not os.path.exists(pckm_results_path):
                print("PCKMeans model not found. Please run PCKMeans first.")
                return
            
            with open(pckm_model_path, 'rb') as f:
                pckm_data = pickle.load(f)
                pckm_centers = pckm_data['centers']
                pckm_train_clusters = pckm_data['train_clusters']
                pckm_test_clusters = pckm_data['test_clusters']
            
            with open(pckm_results_path, 'rb') as f:
                pckm_results = pickle.load(f)
            
            # For PCKM, we need to create a KMeans-like model object for compatibility
            from sklearn.cluster import KMeans
            pckm_kmeans = KMeans(n_clusters=77)
            pckm_kmeans.cluster_centers_ = pckm_centers
            
            print("Running LLM correction on Pairwise Constraint K-Means model...")
            correction_results, corrected_test_clusters = run_llm_correction(
                dataset, embeddings, pckm_kmeans, pckm_train_clusters, pckm_test_clusters, 
                n_clusters=77, n_corrections=500
            )
            original_results = pckm_results
        
        else:
            print("Invalid choice. Skipping LLM correction.")
            return
        
        print("\nLLM correction results:")
        print(f"Original - Accuracy: {correction_results['original_metrics']['accuracy']:.4f}, "
              f"NMI: {correction_results['original_metrics']['nmi']:.4f}, "
              f"ARI: {correction_results['original_metrics']['ari']:.4f}")
        print(f"Corrected - Accuracy: {correction_results['corrected_metrics']['accuracy']:.4f}, "
              f"NMI: {correction_results['corrected_metrics']['nmi']:.4f}, "
              f"ARI: {correction_results['corrected_metrics']['ari']:.4f}")
        print(f"Improvement - Accuracy: {correction_results['improvement']['accuracy']:.4f}, "
              f"NMI: {correction_results['improvement']['nmi']:.4f}, "
              f"ARI: {correction_results['improvement']['ari']:.4f}")
        
        # Save the correction results
        correction_path = os.path.join(models_dir, f"correction_results_model{choice}.pkl")
        with open(correction_path, 'wb') as f:
            pickle.dump({
                'correction_results': correction_results,
                'corrected_test_clusters': corrected_test_clusters,
                'original_model': choice
            }, f)
        print(f"Correction results saved to {correction_path}")
    
    else:
        print(f"Unknown task: {task_name}")
        print("Available tasks: 'kmeans', 'keyphrase', 'pckm', 'correction'")


def compare_all_methods(baseline_results, keyphrase_results, pckm_results, correction_results):
    """
    Compare all methods and create visualization
    """
    # Collect test metrics
    methods = ["Baseline K-Means", "Keyphrase Expansion", "Pairwise Constraint K-Means", "LLM Correction"]
    
    accuracy = [
        baseline_results["test_metrics"]["accuracy"],
        keyphrase_results["test_metrics"]["accuracy"],
        pckm_results["test_metrics"]["accuracy"],
        correction_results["corrected_metrics"]["accuracy"]
    ]
    
    nmi = [
        baseline_results["test_metrics"]["nmi"],
        keyphrase_results["test_metrics"]["nmi"],
        pckm_results["test_metrics"]["nmi"],
        correction_results["corrected_metrics"]["nmi"]
    ]
    
    ari = [
        baseline_results["test_metrics"]["ari"],
        keyphrase_results["test_metrics"]["ari"],
        pckm_results["test_metrics"]["ari"],
        correction_results["corrected_metrics"]["ari"]
    ]
    
    # Create comparison table
    comparison = {
        "Method": methods,
        "Accuracy": accuracy,
        "NMI": nmi,
        "ARI": ari
    }
    
    comparison_df = pd.DataFrame(comparison)
    
    # Save to file
    timestamp = get_timestamp()
    comparison_file = f"{RESULTS_DIR}/method_comparison_{timestamp}.csv"
    comparison_df.to_csv(comparison_file, index=False)
    print(f"\nComparison saved to {comparison_file}")
    
    # Create bar chart
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(methods))
    width = 0.25
    
    plt.bar(x - width, accuracy, width, label='Accuracy')
    plt.bar(x, nmi, width, label='NMI')
    plt.bar(x + width, ari, width, label='ARI')
    
    plt.xlabel('Method')
    plt.ylabel('Score')
    plt.title('Comparison of Clustering Methods on Banking77')
    plt.xticks(x, methods, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f"{RESULTS_DIR}/method_comparison_{timestamp}.png", dpi=300)
    print(f"Visualization saved to {RESULTS_DIR}/method_comparison_{timestamp}.png")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        run_specific_task(sys.argv[1])
    else:
        main()
    