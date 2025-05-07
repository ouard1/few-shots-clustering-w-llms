
"""
OPIEC Entity Canonicalization Preprocessing Script
--------------------------------------------------

This script processes the OPIEC dataset (Open Predicate Inference Extracted Corpus) 
to extract and structure entity information for canonicalization tasks.

Functionality:
- Loads OPIEC data from a pickle file
- Extracts subject and object entities with valid Wikipedia links
- Captures the surface form, entity type (subject/object), and sentence context
- Groups entities into clusters by their Wikipedia link
- Outputs the processed data in JSON format

Useful for:
- Preparing training or evaluation data for entity resolution, linking, or clustering models

Requirements:
- Input file must be a pickle containing a list of sentence-level data structures
- Each structure should include 'subject' and/or 'object' fields with optional 'w_link'

"""
import os
import json
import pickle
import tqdm
from collections import defaultdict

def process_opiec_for_canonicalization(input_file, output_file, max_items=1000):
    """
    Process OPIEC data for entity canonicalization
    
    Args:
        input_file: Path to OPIEC dataset file (pickle format)
        output_file: Path to save processed data as JSON
        max_items: Maximum number of items to process
    """
    print(f"Processing OPIEC data from {input_file}")
    
  
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    
   
    processed_data = {
        "entities": [],
        "clusters": defaultdict(list)
    }
    

    count = 0
    
    
    for item in tqdm.tqdm(data, total=min(len(data), max_items)):
        if count >= max_items:
            break
            
        try:
            
            sentence = item.get('src_sentences', '')
            
           
            if 'subject' in item and isinstance(item['subject'], list):
                for subject in item['subject']:
                    if 'w_link' in subject and subject['w_link']['wiki_link']:
                        entity_data = process_entity(subject, sentence, 'subject')
                        if entity_data:
                            processed_data["entities"].append(entity_data)
                            processed_data["clusters"][entity_data["wiki_link"]].append(len(processed_data["entities"]) - 1)
                            count += 1
                            if count >= max_items:
                                break
            
            
            if count < max_items and 'object' in item and isinstance(item['object'], list):
                for obj in item['object']:
                    if 'w_link' in obj and obj['w_link']['wiki_link']:
                        entity_data = process_entity(obj, sentence, 'object')
                        if entity_data:
                            processed_data["entities"].append(entity_data)
                            processed_data["clusters"][entity_data["wiki_link"]].append(len(processed_data["entities"]) - 1)
                            count += 1
                            if count >= max_items:
                                break
        
        except Exception as e:
            print(f"Error processing item: {e}")
    
    
    processed_data["clusters"] = dict(processed_data["clusters"])
    
 
    with open(output_file, 'w') as f:
        json.dump(processed_data, f, indent=2)
    
    print(f"Processed {count} entities from OPIEC data")
    print(f"Found {len(processed_data['clusters'])} unique entity clusters")
    print(f"Saved to {output_file}")
    
    return processed_data

def process_entity(entity_data, sentence, entity_type):
    """Process a single entity from OPIEC data"""
    if not entity_data or not isinstance(entity_data, dict):
        return None
        
   
    if 'word' in entity_data:
        surface_form = entity_data['word']
    else:
        return None
    
    
    wiki_link = None
    if 'w_link' in entity_data and entity_data['w_link']:
        wiki_link = entity_data['w_link'].get('wiki_link', '')
    
    if not wiki_link:
        return None
    
    
    entity_obj = {
        "surface_form": surface_form,
        "wiki_link": wiki_link,
        "context": sentence,
        "entity_type": entity_type
    }
    
    return entity_obj


input_file = '/Users/ouarda.boumansour/Desktop/ProjetMLSD/datasets/opiec/OPIEC59k_valid'
output_file = "opiec_processed.json"
#process_opiec_for_canonicalization(input_file, output_file, max_items=5000)

input_file = '/Users/ouarda.boumansour/Desktop/ProjetMLSD/datasets/opiec/OPIEC59k_test'
output_file = "opiec_processed_test.json"
#process_opiec_for_canonicalization(input_file, output_file, max_items=1000)

