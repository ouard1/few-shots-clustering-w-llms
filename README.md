# ProjectMLSD - Large Language Models Enable Few Shot Clustering

This repository implements the approach described in "Large Language Models Enable Few Shot Clustering" paper, demonstrating how LLMs can be leveraged for effective clustering tasks with minimal labeled data.

## Overview

The project explores how Large Language Models (LLMs) can be used to improve clustering performance in few-shot scenarios, where labeled data is scarce. We demonstrate this through multiple applications and datasets:

1. Text Clustering on Banking77 Dataset
2. Text Clustering on CLINC Dataset
3. Entity Canonicalization on OPIEC Dataset
4. Entity Canonicalization on Reverb Dataset
5. Customer Support Application (Practical Use Case)

## Methods

Our approach leverages LLMs to enable effective clustering with minimal labeled examples. Text clustering focuses on grouping similar queries by their underlying intent, applied to Banking77 and CLINC datasets for intent recognition. Entity canonicalization addresses the challenge of mapping different mentions of the same entity to a canonical form, implemented on OPIEC and Reverb datasets. Both tasks benefit from our few-shot LLM methods: keyphrase-based representation enhancement, LLM-guided constraint generation, hierarchical clustering with LLM validation, and cluster refinement using LLM-based correction. These techniques significantly outperform traditional clustering methods, especially in low-resource scenarios, and demonstrate practical applications in our customer support system.

For entity canonicalization specifically, we employ a multi-view approach that captures different aspects of entity information:

1. **Context View**: Captures the surrounding text that provides context about the entity
2. **Fact View**: Extracts factual information about the entity using LLM-based property extraction


These complementary views are combined to create a rich entity representation that helps overcome the challenges of ambiguity and variation in entity mentions, leading to more accurate canonicalization with minimal labeled examples.

## Key Contributions

- **Few-Shot Learning**: Leveraging LLMs to perform clustering with minimal labeled examples
- **Multi-Modal Clustering**: Combining text embeddings with LLM-generated features
- **Hierarchical Approaches**: Implementing multi-level clustering strategies
- **Constraint-Based Learning**: Using LLM-generated constraints to improve clustering quality
- **Real-World Application**: Implementation in a customer support system


## Project Components

### 1. Text Clustering (Banking77 Dataset)

Demonstrates few-shot clustering of banking customer queries using LLMs.

**Key Files:**
- `text-clustering-banking77/text-clustering-bank77-V2.py` - Latest version with Mistral AI integration
- `text-clustering-banking77/text-clustering-bank77-V1.py` - Version with OpenAI integration and hirarchical clustering
- `text-clustering-banking77/text-clustering-bank77-V0.py` - Initial implementation with OpenAO integration
- `text-clustering-banking77/create_balanced_dataset.py` - Dataset preparation utility

**Few-Shot Techniques:**
- LLM-based feature extraction
- Keyphrase expansion with minimal examples
- Hierarchical clustering with LLM guidance
- Constraint-based clustering using LLM-generated constraints
- Few-shot cluster assignment correction

### 2. Text Clustering (CLINC Dataset)

Implements few-shot clustering for general intent classification.

**Key Files:**
- `text_clustering-clinc/scripts/` - Implementation scripts
- `text_clustering-clinc/dataset/` - Dataset processing utilities

**Features:**
- Multi-domain intent clustering
- Cross-domain few-shot learning
- Dynamic intent discovery
- Intent hierarchy construction
- Performance evaluation metrics

### 3. Entity Canonicalization (OPIEC Dataset)

Shows how LLMs can improve entity clustering with limited supervision.

**Key Files:**
- `entity-canonicalization-opiec/entity-canonicalization-opiec-V1.py` - Enhanced few-shot clustering with hirarchical clustering with OpenAI integration
- `entity-canonicalization-opiec/entity-canonicalization-opiec-V0.py` - Initial implementation with OpenAI integration

**Few-Shot Approaches:**
- Multi-view entity representations
- LLM-guided entity clustering
- Few-shot constraint generation
- Hierarchical entity organization
- LLM-based cluster refinement

### 4. Entity Canonicalization (Reverb Dataset)


**Key Files:**
- `entity_canonicalization-reverb/scripts/` - Implementation scripts
- `entity_canonicalization-reverb/data/` - Dataset processing utilities

**Features:**
- Cross-dataset entity linking
- Domain-specific entity clustering
- Canonical form generation
- Performance benchmarking

### 5. Customer Support Application (Practical Use Case)

A real-world implementation of the few-shot clustering approach in a customer support system.

**Key Files:**
- `customer-support-app/index.html` - Main frontend HTML
- `customer-support-app/styles.css` - Frontend styling
- `customer-support-app/script.js` - Frontend JavaScript for chat functionality
- `customer-support-app/api/` - Backend implementation with LLM-powered clustering
- `customer-support-app/README.md` - Setup instructions

**Features:**
- Real-time query clustering
- Dynamic intent recognition
- Automated response generation


## Setup

### Environment Setup

Create and activate a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate on macOS/Linux
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### Install Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

### API Keys

Create a `.env` file in the root directory with your API keys:

```
MISTRAL_API_KEY=your_mistral_api_key_here
OPENAI_API_KEY=your_openai_api_key_here 
MISTRAL_API_KEY_ROTATE_1 = your_mistral_api_key_here
MISTRAL_API_KEY_ROTATE_2 = your_mistral_api_key_here
```



### Customer Support Application

Follow the instructions in `customer-support-page/README.md` for detailed setup, but in short:

```bash
# Start the backend
cd customer-support-page/api
pip install -r requirements.txt
python app.py

# In another terminal, serve the frontend
cd customer-support-page
python -m http.server
```

Then access the application at http://localhost:8000.

