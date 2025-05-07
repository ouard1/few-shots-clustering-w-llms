from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from mistralai import Mistral
from sentence_transformers import SentenceTransformer
import time
import chromadb
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from dotenv import load_dotenv

"""
Flask Backend for Replia: AI-Powered Customer Support

This application powers the backend of Replia's intelligent customer support platform. It performs the following key functions:

- Accepts user queries via a `/query` API endpoint (POST request).
- Extracts relevant keywords from the query using a language model (Mistral API).
- Embeds the query using a sentence transformer model (MiniLM).
- Classifies the query into customer support categories using both:
    - An llm-based classification.
    - A KMeans-like custom model with cosine similarity and precomputed centroids.
- Retrieves the most relevant support documents from a Chroma vector database.
- Generates a context-aware response to the user's query based on retrieved documents.

Technologies used:
- Flask (API framework)
- SentenceTransformer (for embeddings)
- Mistral API (for LLM interactions)
- ChromaDB (for vector similarity search)
- Scikit-learn (for cosine similarity)

"""

load_dotenv()

app = Flask(__name__)
CORS(app) 

MISTRAL_API_KEY_ROTATE_1 = os.getenv("MISTRAL_API_KEY_ROTATE_1")
MISTRAL_API_KEY_ROTATE_2 = os.getenv("MISTRAL_API_KEY_ROTATE_2")

minilm_emb_model = SentenceTransformer("all-MiniLM-L6-v2") 
mistral = "mistral-large-latest"

llm_client_rotate_1 = Mistral(api_key=MISTRAL_API_KEY_ROTATE_1)
llm_client_rotate_2 = Mistral(api_key=MISTRAL_API_KEY_ROTATE_2)


chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_collection(name="e-commerce_conversations_support")

centroids_balanced = np.load('centroids_minilm_balanced.npy')
df_mapping = pd.read_csv("cluster_to_issue_area_balanced.csv")
cluster_to_issue_area_balanced = dict(zip(df_mapping["cluster"], df_mapping["issue_area"]))


def extract_keywords(text):
    prompt = """
    You are an expert in extracting keywords from a sentence describing a user's query.
    Your task is to:
    - Analyze the following sentence carefully.
    - Extract the most relevant keywords that represent the user's real intention.
    - Focus only on keywords that would help classify the query into one of the following categories:
    'Login and Account', 'Order', 'Shopping', 'Cancellations and returns', 'Warranty', "Shipping'
    Instructions:
    - Be concise and clear.
    - Do not add any extra commentary or explanation.
    - Return only the keywords, separated by commas if multiple.

    Here are 6 examples:
    Example 1 :
    User Question : The customer wants to deactivate their account due to dissatisfaction with a purchased food processor.
    Keywords : deactivate account
    Example 2 :
    User Question: The customer wants to know why their electric kettle order is delayed at the nearest hub and has not been sent out for delivery.
    Keywords : delayed order, order not sent out for delivery 
    Example 3 :
    User Question: The customer wants to know if an account is needed to purchase a refrigerator, what additional information is required, and the return policy.
    Keywords : purchase a refrigerator, return policy 
    Example 4 : 
    User Question: The customer wants to cancel a recent order for a smartwatch.
    Keywords : cancel order
    Example 5 : 
    User Question: The customer wants to register their recently purchased air cooler with the brand CoolAir for warranty benefits.
    Keywords : register warranty benefits, recently purchased air cooler
    Example 6 : 
    User Question: The customer wants to know about the delivery charges for an External Hard Disk and the options for faster shipping.
    Keywords : delivery charges, faster shipping

    Please do the same with : 
    """
    prompt += f"{text}"
    chat_response = llm_client_rotate_1.chat.complete(
        model=mistral,
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
        ],
        temperature=0.5
    )
    return chat_response.choices[0].message.content


def get_embeddings(texts, model):
    embedding = model.encode(texts)
    return embedding


def cluster_by_llm(text):
    prompt = """
    You are an expert in clustering user's query to 6 domains : 'Login and Account', 'Order', 'Shopping', 'Cancellations and returns', 'Warranty', "Shipping'
    Instructions:
    - Be concise and clear.
    - Do not add any extra commentary or explanation.
    - Return only the predicted cluster

    """
    prompt += f"{text}"
    chat_response = llm_client_rotate_2.chat.complete(
        model=mistral,
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
        ],
        temperature=0.5
    )
    return chat_response.choices[0].message.content


def get_clusters(query, kmeans_balanced):
    cluster_llm = cluster_by_llm(query)
    predicted_cluster_balanced = kmeans_balanced.predict(query)[0]
    cluster_kmeans_balanced = cluster_to_issue_area_balanced[int(predicted_cluster_balanced)]
    return cluster_llm, cluster_kmeans_balanced


def get_documents(query, cluster_llm, cluster_kmeans):
    if cluster_llm == cluster_kmeans: 
        results = collection.query(
            query_texts=query,
            where={"issue_area": cluster_llm},
            n_results=3,
            include=["documents", "metadatas", "distances"]
        ) 
        documents_to_pass = results["documents"][0]
    else : 
        results_llm = collection.query(
            query_texts=query,
            where={"issue_area": cluster_llm},
            n_results=3,
            include=["documents", "metadatas", "distances"]
        ) 
        results_kmeans = collection.query(
            query_texts=query,
            where={"issue_area": cluster_kmeans},
            n_results=3,
            include=["documents", "metadatas", "distances"]
        ) 
        documents_to_pass = results_llm["documents"][0] + results_kmeans["documents"][0]
    return documents_to_pass


def get_response(query, documents): 
    system_prompt = (
        "You are an expert assistant in answering user queries related to 'Login and Account', 'Order', "
        "'Shopping', 'Cancellations and returns', 'Warranty', and 'Shipping'.\n\n"
        "You must carefully analyze the user's question and provide a clear, helpful answer based on the following past experiences (examples) provided.\n"
        "Use these documents to understand the user's need and the appropriate steps to assist.\n\n"
        "Documents:\n"
        f"{documents}\n\n"
        "Instructions:\n"
        "- These documents are just examples, do not answer exactly the same way (same objects)"
        "- Be concise and clear.\n"
        "- Do not add any extra commentary or explanation.\n"
        "- Respond directly to the user's question based on the given experiences.\n"
    )

    chat_response = llm_client_rotate_1.chat.complete(
        model=mistral,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        temperature=0.7
    )
    return chat_response.choices[0].message.content


def assign_clusters(embeddings, centroids):
    if len(embeddings.shape) == 1:
        embeddings = embeddings.reshape(1, -1)

    labels = []
    for emb in embeddings:
        similarities = cosine_similarity(emb.reshape(1, -1), centroids)
        labels.append(np.argmax(similarities))
    return labels


class KLLMMeans:
    def __init__(self, centroids, embedding_model):
        self.centroids = centroids
        self.embedding_model = embedding_model

    def predict(self, new_documents):
        new_embeddings = get_embeddings(new_documents, self.embedding_model)
        labels = assign_clusters(new_embeddings, self.centroids)
        return labels
    


@app.route('/query', methods=['POST'])
def process_query():
    data = request.get_json()
    query = data.get("query")

    if not query:
        return jsonify({"error": "Missing 'query' in request body"}), 400

    try:
        keywords = extract_keywords(query)
        full_query = f"{query}.. Keywords {keywords}"
        print(full_query)
        kllm_model_balanced = KLLMMeans(centroids_balanced, minilm_emb_model)
        cluster_llm, cluster_kmeans_balanced = get_clusters(full_query, kllm_model_balanced)
        print(f"cluster_llm: {cluster_llm}\n cluster_kmeans_balanced : {cluster_kmeans_balanced}")
        documents = get_documents(query, cluster_llm, cluster_kmeans_balanced)
        time.sleep(1)
        response = get_response(query, documents)
        return jsonify({"response": response})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)