"""
Entity Canonicalization on OPIEC Dataset (V0)
------------------------------------------------
This script implements multiple methods for entity canonicalization (clustering entity mentions that refer to the same real-world entity) on the OPIEC dataset, which is derived from Wikipedia. The main functionalities include:

- Loading and preprocessing OPIEC entity data (train/test splits)
- Generating and caching embeddings for entity mentions using Sentence Transformers
- Baseline K-Means clustering for entity canonicalization
- LLM-based keyphrase expansion to improve clustering
- Pairwise constraint K-Means (PCKMeans) using LLM-generated must-link/cannot-link constraints
- LLM-based correction of low-confidence cluster assignments
- Evaluation of clustering results with macro/micro/pairwise F1 metrics
- Comparison and visualization of different canonicalization methods

The script is designed for research and experimentation with entity canonicalization methods, leveraging both traditional clustering and large language models (OpenAI GPT-3.5-turbo) for enhanced performance. Results and intermediate data are cached for efficiency.

Usage: Run the script directly. It will prompt for optional steps (PCKMeans, LLM correction) and save results/plots in the results directory.
"""
import os
import json
import time
from datetime import datetime
from tqdm import tqdm
import pickle
from pathlib import Path
import re
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from scipy.optimize import linear_sum_assignment
import openai
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import pandas as pd
import dotenv
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('entity_canonicalization.log'),
        logging.StreamHandler()
    ]
)

# Load environment variables
dotenv.load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Constants
CACHE_DIR = "cache/embeddings"
RESULTS_DIR = "../results"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_timestamp():
    """Get current timestamp for file naming"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_results(result_dict, method_name):
    """Save results to a JSON file"""
    timestamp = get_timestamp()
    filename = f"{RESULTS_DIR}/{method_name}_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(result_dict, f, indent=2)
    print(f"Results saved to {filename}")
    return filename

def load_opiec_data(train_file: str, test_file: str) -> Dict:
    """Load and preprocess OPIEC dataset."""
    logging.info(f"Loading OPIEC data from {train_file} and {test_file}")
    
    with open(train_file, 'r') as f:
        train_data = json.load(f)
    with open(test_file, 'r') as f:
        test_data = json.load(f)
    
    # Process train data
    train_mentions = []
    train_entities = []
    for item in train_data:
        mention = item['mention']
        entity = item['entity']
        train_mentions.append(mention)
        train_entities.append(entity)
    
    # Process test data
    test_mentions = []
    test_entities = []
    for item in test_data:
        mention = item['mention']
        entity = item['entity']
        test_mentions.append(mention)
        test_entities.append(entity)
    
    return {
        'train': {
            'mentions': train_mentions,
            'entities': train_entities
        },
        'test': {
            'mentions': test_mentions,
            'entities': test_entities
        }
    }

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

def create_entity_embeddings(dataset, model_name="all-MiniLM-L6-v2", use_cache=True):
    """
    Create multi-view embeddings for entities
    """
    cache_key = f"opiec_{model_name}_{len(dataset['train'])}"
    cache_file = f"{CACHE_DIR}/opiec_embeddings_{cache_key}.pkl"
    
    if use_cache and os.path.exists(cache_file):
        print(f"Loading cached embeddings from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    # Extract texts for context view
    train_texts = [item["text"] for item in dataset["train"]]
    test_texts = [item["text"] for item in dataset["test"]]
    
    # Create context view embeddings
    train_context_embeddings = embed_texts(train_texts, model_name, use_cache)
    test_context_embeddings = embed_texts(test_texts, model_name, use_cache)
    
    # For fact view, use the surface form as a proxy
    train_surface_forms = [item["surface_form"] for item in dataset["train"]]
    test_surface_forms = [item["surface_form"] for item in dataset["test"]]
    
    train_fact_embeddings = embed_texts(train_surface_forms, model_name, use_cache)
    test_fact_embeddings = embed_texts(test_surface_forms, model_name, use_cache)
    
    # Combine views by concatenation
    train_combined = np.concatenate([train_context_embeddings, train_fact_embeddings], axis=1)
    test_combined = np.concatenate([test_context_embeddings, test_fact_embeddings], axis=1)
    
    embeddings = {
        "train": train_combined,
        "test": test_combined,
        "train_context": train_context_embeddings,
        "test_context": test_context_embeddings,
        "train_fact": train_fact_embeddings,
        "test_fact": test_fact_embeddings
    }
    
    # Save embeddings
    with open(cache_file, 'wb') as f:
        pickle.dump(embeddings, f)
    
    return embeddings

def evaluate_entity_clustering(y_true, y_pred, n_clusters):
    """
    Evaluate entity canonicalization with macro, micro, and pairwise metrics
    """
    # Convert cluster assignments to sets of points
    true_clusters = {}
    pred_clusters = {}
    
    for i in range(len(y_true)):
        if y_true[i] not in true_clusters:
            true_clusters[y_true[i]] = set()
        true_clusters[y_true[i]].add(i)
        
        if y_pred[i] not in pred_clusters:
            pred_clusters[y_pred[i]] = set()
        pred_clusters[y_pred[i]].add(i)
    
    # Calculate Macro Precision
    macro_precision = 0
    for cluster in pred_clusters.values():
        if len(cluster) == 0:
            continue
        
        # Find the true cluster with most overlap
        max_overlap = 0
        for true_cluster in true_clusters.values():
            overlap = len(cluster.intersection(true_cluster))
            max_overlap = max(max_overlap, overlap)
        
        precision = max_overlap / len(cluster)
        macro_precision += precision
    
    if len(pred_clusters) > 0:
        macro_precision /= len(pred_clusters)
    else:
        macro_precision = 0
    
    # Calculate Macro Recall
    macro_recall = 0
    for cluster in true_clusters.values():
        if len(cluster) == 0:
            continue
        
        # Find the predicted cluster with most overlap
        max_overlap = 0
        for pred_cluster in pred_clusters.values():
            overlap = len(cluster.intersection(pred_cluster))
            max_overlap = max(max_overlap, overlap)
        
        recall = max_overlap / len(cluster)
        macro_recall += recall
    
    if len(true_clusters) > 0:
        macro_recall /= len(true_clusters)
    else:
        macro_recall = 0
    
    # Calculate Micro Precision and Recall
    micro_correct = 0
    micro_total_pred = 0
    
    for cluster in pred_clusters.values():
        if len(cluster) == 0:
            continue
        
        # Find the true cluster with most overlap
        max_overlap = 0
        for true_cluster in true_clusters.values():
            overlap = len(cluster.intersection(true_cluster))
            max_overlap = max(max_overlap, overlap)
        
        micro_correct += max_overlap
        micro_total_pred += len(cluster)
    
    micro_precision = micro_correct / micro_total_pred if micro_total_pred > 0 else 0
    
    micro_correct = 0
    micro_total_true = 0
    
    for cluster in true_clusters.values():
        if len(cluster) == 0:
            continue
        
        # Find the predicted cluster with most overlap
        max_overlap = 0
        for pred_cluster in pred_clusters.values():
            overlap = len(cluster.intersection(pred_cluster))
            max_overlap = max(max_overlap, overlap)
        
        micro_correct += max_overlap
        micro_total_true += len(cluster)
    
    micro_recall = micro_correct / micro_total_true if micro_total_true > 0 else 0
    
    # Calculate Pairwise Precision and Recall
    true_pairs = set()
    pred_pairs = set()
    
    for cluster in true_clusters.values():
        for i in cluster:
            for j in cluster:
                if i < j:
                    true_pairs.add((i, j))
    
    for cluster in pred_clusters.values():
        for i in cluster:
            for j in cluster:
                if i < j:
                    pred_pairs.add((i, j))
    
    if len(pred_pairs) > 0:
        pair_precision = len(true_pairs.intersection(pred_pairs)) / len(pred_pairs)
    else:
        pair_precision = 0
    
    if len(true_pairs) > 0:
        pair_recall = len(true_pairs.intersection(pred_pairs)) / len(true_pairs)
    else:
        pair_recall = 0
    
    # Calculate F1 scores
    macro_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall) if (macro_precision + macro_recall) > 0 else 0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    pair_f1 = 2 * (pair_precision * pair_recall) / (pair_precision + pair_recall) if (pair_precision + pair_recall) > 0 else 0
    
    # Calculate average F1
    avg_f1 = (macro_f1 + micro_f1 + pair_f1) / 3
    
    return {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "pair_precision": pair_precision,
        "pair_recall": pair_recall,
        "pair_f1": pair_f1,
        "avg_f1": avg_f1
    }

def run_kmeans_baseline(dataset, embeddings, n_clusters=490):
    """
    Run baseline K-Means clustering for entity canonicalization
    """
    print(f"Running K-Means with {n_clusters} clusters...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    train_clusters = kmeans.fit_predict(embeddings["train"])
    
    train_metrics = evaluate_entity_clustering(
        [item["label"] for item in dataset["train"]], 
        train_clusters, 
        n_clusters
    )
    
    test_clusters = kmeans.predict(embeddings["test"])
    
    test_metrics = evaluate_entity_clustering(
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
    
    save_results(results, "entity_kmeans_baseline")
    
    return results, kmeans, train_clusters, test_clusters

def generate_entity_keyphrases(entities, contexts, cache_dir=CACHE_DIR, batch_size=50):
    """
    Generate keyphrases for entity canonicalization using LLM
    """
    keyphrases_dir = os.path.join(cache_dir, "entity_keyphrases")
    os.makedirs(keyphrases_dir, exist_ok=True)
    
    cache_key = str(hash("".join(entities[:5] + contexts[:5])))
    cache_file = os.path.join(keyphrases_dir, f"entity_keyphrases_{cache_key}.json")
    
    if os.path.exists(cache_file):
        print(f"Loading entity keyphrases from cache: {cache_file}")
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    system_prompt = """I am trying to cluster entity strings on Wikipedia according to the Wikipedia article title they refer to. 
    To help me with this, for a given entity name, please provide me with a comprehensive set of alternative names 
    that could refer to the same entity. Generate a comprehensive set of alternate entity names as a JSON-formatted list."""
    
    demonstrations = [
        {"entity": "fictional character", 
         "context": "Camille Raquin is a fictional character created by Émile Zola.",
         "keyphrases": ["fictional characters", "characters", "character"]},
        {"entity": "Catholicism",
         "context": "Years after Anne had herself converted, James avowed his Catholicism, which was a contributing factor to the Glorious Revolution.",
         "keyphrases": ["Catholic Church", "Roman Catholic", "Catholic"]},
        {"entity": "Wind",
         "context": "It was co-produced by Earth, Wind & Fire's keyboardist Larry Dunn.",
         "keyphrases": ["Earth & Fire", "Earth", "Wind & Fire"]},
        {"entity": "Elizabeth",
         "context": "He also performed at the London Palladium for Queen Elizabeth.",
         "keyphrases": ["Elizabeth II", "HM", "Queen Elizabeth"]}
    ]
    
    demo_text = "\n\n"
    for demo in demonstrations:
        demo_text += f"Entity: \"{demo['entity']}\"\n\nContext Sentences: {demo['context']}\n\nAlternate Entity Names: {json.dumps(demo['keyphrases'])}\n\n"
    
    all_keyphrases = {}
    
    for i in tqdm(range(0, len(entities), batch_size), desc="Generating entity keyphrases"):
        batch_entities = entities[i:i+batch_size]
        batch_contexts = contexts[i:i+batch_size]
        batch_keyphrases = {}
        
        for entity, context in tqdm(zip(batch_entities, batch_contexts), desc="Batch progress", leave=False):
            key = f"{entity}|{context}"
            if key in all_keyphrases:
                continue
                
            context_text = f"Entity: \"{entity}\"\n\nContext Sentences: {context}"
            
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"{demo_text}{context_text}\n\nAlternate Entity Names:"}
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
                    print(f"Error parsing keyphrases for '{entity}': {e}")
                    print(f"Raw response: {keyphrases_text}")
                    keyphrases = [entity]
                
                batch_keyphrases[key] = keyphrases
    
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error generating keyphrases for '{entity}': {e}")
                batch_keyphrases[key] = [entity]
        
        all_keyphrases.update(batch_keyphrases)
        
        # Save intermediate results
        with open(cache_file, 'w') as f:
            json.dump(all_keyphrases, f)
    
    return all_keyphrases

def run_entity_keyphrase_clustering(dataset, base_embeddings, model_name="all-MiniLM-L6-v2", n_clusters=490):
    """
    Run clustering with LLM keyphrase expansion for entity canonicalization
    """
    print("Running LLM keyphrase expansion for entity canonicalization...")
    
    train_entities = [item["surface_form"] for item in dataset["train"]]
    test_entities = [item["surface_form"] for item in dataset["test"]]
    
    train_contexts = [item["text"] for item in dataset["train"]]
    test_contexts = [item["text"] for item in dataset["test"]]
    
    # Generate keyphrases
    train_keyphrases_dict = generate_entity_keyphrases(train_entities, train_contexts)
    test_keyphrases_dict = generate_entity_keyphrases(test_entities, test_contexts)
    
    # Extract keyphrases for each example
    train_keyphrase_texts = []
    for entity, context in zip(train_entities, train_contexts):
        key = f"{entity}|{context}"
        keyphrases = train_keyphrases_dict.get(key, [entity])
        train_keyphrase_texts.append(" ".join(keyphrases))
    
    test_keyphrase_texts = []
    for entity, context in zip(test_entities, test_contexts):
        key = f"{entity}|{context}"
        keyphrases = test_keyphrases_dict.get(key, [entity])
        test_keyphrase_texts.append(" ".join(keyphrases))
    
    # Embed keyphrases
    train_keyphrase_embeddings = embed_texts(train_keyphrase_texts, model_name)
    test_keyphrase_embeddings = embed_texts(test_keyphrase_texts, model_name)
    
    # Combine with base embeddings
    train_combined = np.concatenate([base_embeddings["train"], train_keyphrase_embeddings], axis=1)
    test_combined = np.concatenate([base_embeddings["test"], test_keyphrase_embeddings], axis=1)
    
    # Run K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    train_clusters = kmeans.fit_predict(train_combined)
    
    # Calculate metrics
    train_metrics = evaluate_entity_clustering(
        [item["label"] for item in dataset["train"]], 
        train_clusters, 
        n_clusters
    )
    
    test_clusters = kmeans.predict(test_combined)
    
    test_metrics = evaluate_entity_clustering(
        [item["label"] for item in dataset["test"]], 
        test_clusters, 
        n_clusters
    )
    
    results = {
        "method": "entity_keyphrase_clustering",
        "n_clusters": n_clusters,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics
    }
    
    save_results(results, "entity_keyphrase_clustering")
    
    return results, kmeans, train_clusters, test_clusters, train_combined, test_combined

def generate_entity_pairwise_constraints(dataset, embeddings, n_constraints=1000, cache_dir=CACHE_DIR):
    """
    Generate pairwise constraints for entity canonicalization using LLM
    """
    constraints_dir = os.path.join(cache_dir, "entity_constraints")
    os.makedirs(constraints_dir, exist_ok=True)
    
    cache_key = f"entity_constraints_{n_constraints}_{len(dataset['train'])}"
    cache_file = os.path.join(constraints_dir, f"{cache_key}.json")
    
    if os.path.exists(cache_file):
        print(f"Loading entity constraints from cache: {cache_file}")
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    system_prompt = """You are tasked with clustering entity strings based on whether they refer to the same Wikipedia article.
    To do this, you will be given pairs of entity names and asked if their anchor text, if used separately to link to a 
    Wikipedia article, is likely referring to the same article. Entity names may be truncated, abbreviated, or ambiguous."""
    
    # Get entities and contexts
    train_entities = [item["surface_form"] for item in dataset["train"]]
    train_contexts = [item["text"] for item in dataset["train"]]
    
    # Calculate pairwise distances
    distances = cosine_distances(embeddings["train"])
    
    # Generate candidate pairs (closest pairs)
    candidate_pairs = []
    for i in range(len(train_entities)):
        # Find 5 closest neighbors
        closest_indices = np.argsort(distances[i])[1:6] 
        for j in closest_indices:
            if i != j:
                candidate_pairs.append((i, j, float(distances[i][j])))
    
    # Sort by distance and take top n_constraints
    candidate_pairs.sort(key=lambda x: x[2])
    selected_pairs = candidate_pairs[:n_constraints]
    
    # Generate constraints using LLM
    constraints = {"must_link": [], "cannot_link": []}
    
    for i, j, _ in tqdm(selected_pairs, desc="Generating entity constraints"):
        entity1 = train_entities[i]
        entity2 = train_entities[j]
        context1 = train_contexts[i]
        context2 = train_contexts[j]
        
        prompt = f"{system_prompt}\n\n1) {entity1}\nContext: {context1}\n\n2) {entity2}\nContext: {context2}\n\nGiven this context, would {entity1} and {entity2} link to the same entity's article on Wikipedia?"
        
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
            
            # Determine constraint type
            if "yes" in result:
                constraints["must_link"].append([int(i), int(j)])
            else:
                constraints["cannot_link"].append([int(i), int(j)])
            
            time.sleep(0.5)  # Avoid rate limiting
        
        except Exception as e:
            print(f"Error generating constraint for pair ({entity1}, {entity2}): {e}")
    
    # Save constraints
    with open(cache_file, 'w') as f:
        json.dump(constraints, f)
    
    return constraints

def run_pckm(X, n_clusters, ml_constraints, cl_constraints, w=1.0, max_iter=100):
    """
    Run PCKMeans (Pairwise Constrained K-Means) for entity canonicalization
    """
    # Initialize with k-means++
    from sklearn.cluster import kmeans_plusplus
    centers, _ = kmeans_plusplus(X, n_clusters, random_state=42)
    
    # Convert constraints to tuple format
    ml_constraints = [(int(i), int(j)) for i, j in ml_constraints]
    cl_constraints = [(int(i), int(j)) for i, j in cl_constraints]
    
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
        
        # Calculate objective function
        objective = 0
        for i in range(X.shape[0]):
            objective += np.sum((X[i] - new_centers[labels[i]]) ** 2)
        
        for i, j in ml_constraints:
            if labels[i] != labels[j]:
                objective += w
        
        for i, j in cl_constraints:
            if labels[i] == labels[j]:
                objective += w
        
        print(f"  Objective: {objective:.4f}, Empty clusters: {empty_clusters}")
        
        if objective < best_objective:
            best_objective = objective
            best_labels = labels.copy()
            print(f"  New best objective: {best_objective:.4f}")

        center_shift = np.sum(np.sqrt(np.sum((centers - new_centers) ** 2, axis=1)))
        print(f"  Center shift: {center_shift:.6f}")
        centers = new_centers
        
        if center_shift < 1e-4:
            print(f"Converged after {iteration+1} iterations")
            break
    
    return best_labels, centers

def run_entity_pairwise_constraint_clustering(dataset, embeddings, n_clusters=490, n_constraints=1000, constraint_weight=1.0):
    """
    Run Pairwise Constraint K-Means clustering for entity canonicalization
    """
    print(f"Running PCKMeans with {n_constraints} constraints for entity canonicalization...")
    
    # Generate constraints
    constraints = generate_entity_pairwise_constraints(dataset, embeddings, n_constraints)
    
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
    
    # Evaluate training results
    train_metrics = evaluate_entity_clustering(
        [item["label"] for item in dataset["train"]], 
        train_labels, 
        n_clusters
    )
    
    # Predict test clusters
    distances = euclidean_distances(embeddings["test"], centers)
    test_labels = np.argmin(distances, axis=1)
    
    # Evaluate test results
    test_metrics = evaluate_entity_clustering(
        [item["label"] for item in dataset["test"]], 
        test_labels, 
        n_clusters
    )
    
    results = {
        "method": "entity_pckm",
        "n_clusters": n_clusters,
        "n_constraints": n_constraints,
        "constraint_weight": constraint_weight,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics
    }
    
    save_results(results, "entity_pckm")
    
    return results, centers, train_labels, test_labels

def run_entity_llm_correction(dataset, embeddings, kmeans_model, train_clusters, test_clusters, n_clusters=490, n_corrections=500):
    """
    Run LLM correction on low-confidence entity cluster assignments
    """
    print(f"Running LLM correction for {n_corrections} entity points...")
    
    # Identify low-confidence points in test set
    test_entities = [item["surface_form"] for item in dataset["test"]]
    test_contexts = [item["text"] for item in dataset["test"]]
    test_distances = kmeans_model.transform(embeddings["test"])
    
    # Calculate margin between closest and second-closest cluster
    sorted_distances = np.sort(test_distances, axis=1)
    margins = sorted_distances[:, 1] - sorted_distances[:, 0]
    # Get indices of points with lowest margins (lowest confidence)
    low_confidence_indices = np.argsort(margins)[:n_corrections]
    
   
    system_prompt = """I am trying to cluster entity mentions based on whether they refer to the same real-world entity or Wikipedia article.
    I'll provide you with an entity mention and its context, along with some example entities from a cluster.
    Tell me if the entity belongs to this cluster or not. Answer with 'Yes' if it belongs to the cluster, or 'No' if it doesn't."""
    
    corrections_dir = os.path.join(CACHE_DIR, "entity_corrections")
    os.makedirs(corrections_dir, exist_ok=True)
    cache_file = os.path.join(corrections_dir, f"entity_corrections_{n_corrections}.json")
    
    if os.path.exists(cache_file):
        print(f"Loading entity corrections from cache: {cache_file}")
        with open(cache_file, 'r') as f:
            corrected_clusters_raw = json.load(f)
            corrected_clusters = {int(k): v for k, v in corrected_clusters_raw.items()}
    else:
      
        cluster_representatives = {}
        train_entities = [item["surface_form"] for item in dataset["train"]]
        train_contexts = [item["text"] for item in dataset["train"]]
        
        for k in range(n_clusters):
            
            cluster_indices = np.where(train_clusters == k)[0]
            
            if len(cluster_indices) > 0:
               
                cluster_center = kmeans_model.cluster_centers_[k]
                distances_to_center = np.linalg.norm(embeddings["train"][cluster_indices] - cluster_center, axis=1)
                closest_indices = cluster_indices[np.argsort(distances_to_center)[:3]]
                
               
                cluster_representatives[k] = [(train_entities[i], train_contexts[i]) for i in closest_indices]
            else:
                cluster_representatives[k] = []
        
     
        corrected_clusters = {}
        
        for i, idx in enumerate(tqdm(low_confidence_indices, desc="Correcting entity clusters")):
            idx = int(idx)
            entity = test_entities[idx]
            context = test_contexts[idx]
            current_cluster = int(test_clusters[idx])
            
         
            current_representatives = cluster_representatives[current_cluster]
            
            if not current_representatives:
                corrected_clusters[idx] = current_cluster
                continue
            
      
            prompt = f"{system_prompt}\n\nEntity: {entity}\nContext: {context}\n\nCluster examples:\n"
            for rep_entity, rep_context in current_representatives:
                prompt += f"- {rep_entity} (Context: {rep_context})\n"
            prompt += "\nDoes the entity belong to this cluster?"
            
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
                
               
                if "yes" in result:
                    corrected_clusters[idx] = current_cluster
                else:
                   
                    sorted_cluster_indices = np.argsort(test_distances[idx])
                    alternative_clusters = sorted_cluster_indices[1:5]  
                    
                    
                    for alt_cluster in alternative_clusters:
                        alt_cluster = int(alt_cluster)
                        alt_representatives = cluster_representatives[alt_cluster]
                        
                        if not alt_representatives:
                            continue
                        
                        alt_prompt = f"{system_prompt}\n\nEntity: {entity}\nContext: {context}\n\nCluster examples:\n"
                        for rep_entity, rep_context in alt_representatives:
                            alt_prompt += f"- {rep_entity} (Context: {rep_context})\n"
                        alt_prompt += "\nDoes the entity belong to this cluster?"
                        
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
                    serializable_clusters = {str(k): v for k, v in corrected_clusters.items()}
                    with open(cache_file, 'w') as f:
                        json.dump(serializable_clusters, f)
                    
            except Exception as e:
                print(f"Error correcting cluster for entity {entity}: {e}")
                corrected_clusters[idx] = current_cluster
        
        serializable_clusters = {str(k): v for k, v in corrected_clusters.items()}
        with open(cache_file, 'w') as f:
            json.dump(serializable_clusters, f)

    corrected_test_clusters = test_clusters.copy()
    for idx, cluster in corrected_clusters.items():
        idx = int(idx)
        if idx < len(corrected_test_clusters):
            corrected_test_clusters[idx] = cluster
    
    original_metrics = evaluate_entity_clustering(
        [item["label"] for item in dataset["test"]], 
        test_clusters, 
        n_clusters
    )
    
    corrected_metrics = evaluate_entity_clustering(
        [item["label"] for item in dataset["test"]], 
        corrected_test_clusters, 
        n_clusters
    )
    
    results = {
        "method": "entity_llm_correction",
        "n_clusters": n_clusters,
        "n_corrections": n_corrections,
        "original_metrics": original_metrics,
        "corrected_metrics": corrected_metrics,
        "improvement": {
            "macro_f1": corrected_metrics["macro_f1"] - original_metrics["macro_f1"],
            "micro_f1": corrected_metrics["micro_f1"] - original_metrics["micro_f1"],
            "pair_f1": corrected_metrics["pair_f1"] - original_metrics["pair_f1"],
            "avg_f1": corrected_metrics["avg_f1"] - original_metrics["avg_f1"]
        }
    }
    
    save_results(results, "entity_llm_correction")
    
    return results, corrected_test_clusters

def compare_entity_methods(baseline_results, keyphrase_results, pckm_results=None, correction_results=None):
    """
    Compare all entity canonicalization methods
    """
    methods = ["Baseline K-Means", "Keyphrase Expansion"]
    
    macro_f1 = [
        baseline_results["test_metrics"]["macro_f1"],
        keyphrase_results["test_metrics"]["macro_f1"]
    ]
    
    micro_f1 = [
        baseline_results["test_metrics"]["micro_f1"],
        keyphrase_results["test_metrics"]["micro_f1"]
    ]
    
    pair_f1 = [
        baseline_results["test_metrics"]["pair_f1"],
        keyphrase_results["test_metrics"]["pair_f1"]
    ]
    
    avg_f1 = [
        baseline_results["test_metrics"]["avg_f1"],
        keyphrase_results["test_metrics"]["avg_f1"]
    ]
    
    if pckm_results:
        methods.append("Pairwise Constraint K-Means")
        macro_f1.append(pckm_results["test_metrics"]["macro_f1"])
        micro_f1.append(pckm_results["test_metrics"]["micro_f1"])
        pair_f1.append(pckm_results["test_metrics"]["pair_f1"])
        avg_f1.append(pckm_results["test_metrics"]["avg_f1"])
    
    if correction_results:
        methods.append("LLM Correction")
        macro_f1.append(correction_results["corrected_metrics"]["macro_f1"])
        micro_f1.append(correction_results["corrected_metrics"]["micro_f1"])
        pair_f1.append(correction_results["corrected_metrics"]["pair_f1"])
        avg_f1.append(correction_results["corrected_metrics"]["avg_f1"])
    
   
    comparison = {
        "Method": methods,
        "Macro F1": macro_f1,
        "Micro F1": micro_f1,
        "Pair F1": pair_f1,
        "Avg F1": avg_f1
    }
    
    comparison_df = pd.DataFrame(comparison)
    

    timestamp = get_timestamp()
    comparison_file = f"{RESULTS_DIR}/entity_method_comparison_{timestamp}.csv"
    comparison_df.to_csv(comparison_file, index=False)
    print(f"\nComparison saved to {comparison_file}")
    
   
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(methods))
    width = 0.2
    
    plt.bar(x - 1.5*width, macro_f1, width, label='Macro F1')
    plt.bar(x - 0.5*width, micro_f1, width, label='Micro F1')
    plt.bar(x + 0.5*width, pair_f1, width, label='Pair F1')
    plt.bar(x + 1.5*width, avg_f1, width, label='Avg F1')
    
    plt.xlabel('Method')
    plt.ylabel('F1 Score')
    plt.title('Comparison of Entity Canonicalization Methods on OPIEC')
    plt.xticks(x, methods, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f"{RESULTS_DIR}/entity_method_comparison_{timestamp}.png", dpi=300)
    print(f"Visualization saved to {RESULTS_DIR}/entity_method_comparison_{timestamp}.png")

def main():
    """
    Main function to run entity canonicalization experiments
    """
    
    dataset = load_opiec_data(
        train_file="datasets/opiec/opiec_processed.json",
        test_file="datasets/opiec/opiec_processed_test.json"
    )
    
   
    n_clusters = min(490, len(set(dataset['train']['entities'])))
    print(f"Using {n_clusters} clusters for entity canonicalization")
    
 
    embeddings = create_entity_embeddings(dataset)
    
    
    baseline_results, kmeans_model, train_clusters, test_clusters = run_kmeans_baseline(
        dataset, embeddings, n_clusters=n_clusters
    )
    
    print("Baseline K-Means results:")
    print(f"Macro F1: {baseline_results['test_metrics']['macro_f1']:.4f}")
    print(f"Micro F1: {baseline_results['test_metrics']['micro_f1']:.4f}")
    print(f"Pair F1: {baseline_results['test_metrics']['pair_f1']:.4f}")
    print(f"Avg F1: {baseline_results['test_metrics']['avg_f1']:.4f}")
    
    
    keyphrase_results, keyphrase_model, keyphrase_train_clusters, keyphrase_test_clusters, train_combined, test_combined = run_entity_keyphrase_clustering(
        dataset, embeddings, n_clusters=n_clusters
    )
    
    print("\nKeyphrase Expansion results:")
    print(f"Macro F1: {keyphrase_results['test_metrics']['macro_f1']:.4f}")
    print(f"Micro F1: {keyphrase_results['test_metrics']['micro_f1']:.4f}")
    print(f"Pair F1: {keyphrase_results['test_metrics']['pair_f1']:.4f}")
    print(f"Avg F1: {keyphrase_results['test_metrics']['avg_f1']:.4f}")
    
   
    run_pckm = input("Run Pairwise Constraint K-Means? (y/n): ").strip().lower() == 'y'
    
    pckm_results = None
    if run_pckm:
        pckm_results, pckm_centers, pckm_train_clusters, pckm_test_clusters = run_entity_pairwise_constraint_clustering(
            dataset, embeddings, n_clusters=n_clusters, n_constraints=1000
        )
        
        print("\nPairwise Constraint K-Means results:")
        print(f"Macro F1: {pckm_results['test_metrics']['macro_f1']:.4f}")
        print(f"Micro F1: {pckm_results['test_metrics']['micro_f1']:.4f}")
        print(f"Pair F1: {pckm_results['test_metrics']['pair_f1']:.4f}")
        print(f"Avg F1: {pckm_results['test_metrics']['avg_f1']:.4f}")
    
   
    run_correction = input("Run LLM Correction? (y/n): ").strip().lower() == 'y'
    
    correction_results = None
    if run_correction:
    
        print("\nWhich model would you like to use for LLM correction?")
        print("1: K-Means Baseline")
        print("2: Keyphrase Expansion")
        if run_pckm:
            print("3: Pairwise Constraint K-Means")
        
        choice = input("Enter your choice: ").strip()
        
        if choice == '1':
            correction_results, corrected_test_clusters = run_entity_llm_correction(
                dataset, embeddings, kmeans_model, train_clusters, test_clusters, 
                n_clusters=n_clusters, n_corrections=500
            )
        elif choice == '2':
            correction_results, corrected_test_clusters = run_entity_llm_correction(
                dataset, {"train": train_combined, "test": test_combined}, 
                keyphrase_model, keyphrase_train_clusters, keyphrase_test_clusters, 
                n_clusters=n_clusters, n_corrections=500
            )
        elif choice == '3' and run_pckm:
            
            from sklearn.cluster import KMeans
            pckm_kmeans = KMeans(n_clusters=n_clusters)
            pckm_kmeans.cluster_centers_ = pckm_centers
            
            correction_results, corrected_test_clusters = run_entity_llm_correction(
                dataset, embeddings, pckm_kmeans, pckm_train_clusters, pckm_test_clusters, 
                n_clusters=n_clusters, n_corrections=500
            )
        
        if correction_results:
            print("\nLLM Correction results:")
            print(f"Original Macro F1: {correction_results['original_metrics']['macro_f1']:.4f}")
            print(f"Corrected Macro F1: {correction_results['corrected_metrics']['macro_f1']:.4f}")
            print(f"Original Micro F1: {correction_results['original_metrics']['micro_f1']:.4f}")
            print(f"Corrected Micro F1: {correction_results['corrected_metrics']['micro_f1']:.4f}")
            print(f"Original Pair F1: {correction_results['original_metrics']['pair_f1']:.4f}")
            print(f"Corrected Pair F1: {correction_results['corrected_metrics']['pair_f1']:.4f}")
            print(f"Original Avg F1: {correction_results['original_metrics']['avg_f1']:.4f}")
            print(f"Corrected Avg F1: {correction_results['corrected_metrics']['avg_f1']:.4f}")
    
   
    compare_entity_methods(baseline_results, keyphrase_results, pckm_results, correction_results)
    
    print("\nAll entity canonicalization experiments completed!")

if __name__ == "__main__":
    main()