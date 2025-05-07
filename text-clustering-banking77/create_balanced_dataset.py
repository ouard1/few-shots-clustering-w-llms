import json
import pandas as pd
import os
import random
from collections import Counter
import matplotlib.pyplot as plt

def create_balanced_dataset(input_csv_path, output_csv_path, train_size=3000, test_size=1000, min_samples_per_category=5, random_seed=42):
    """
    Create a balanced dataset from a CSV file, ensuring all categories are represented
    
    Args:
        input_csv_path: Path to the input CSV file
        output_csv_path: Path to save the balanced dataset
        train_size: Number of examples for training set
        test_size: Number of examples for test set
        min_samples_per_category: Minimum number of samples per category
        random_seed: Random seed for reproducibility
    
    Returns:
        train_df: Pandas DataFrame with the training data
        test_df: Pandas DataFrame with the test data
    """
    random.seed(random_seed)
    
    print(f"Loading data from {input_csv_path}...")
    
    try:
        df = pd.read_csv(input_csv_path)
    except Exception as e:
        print(f"Standard CSV reading failed: {e}")
        try:
            df = pd.read_csv(input_csv_path, quoting=3, escapechar='\\')
        except Exception as e:
            print(f"CSV reading with escape char failed: {e}")
            with open(input_csv_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            data = []
            for line in lines:
                if line.strip():
                    if ',' in line:
                        last_comma_index = line.rindex(',')
                        text = line[:last_comma_index].strip()
                        category = line[last_comma_index+1:].strip()
                        
                        if text.startswith('"') and text.endswith('"'):
                            text = text[1:-1]
                        
                        data.append({'text': text, 'category': category})
            
            df = pd.DataFrame(data)
    
    if 'text' not in df.columns or 'category' not in df.columns:
        if len(df.columns) == 2:
            df.columns = ['text', 'category']
        else:
            raise ValueError(f"CSV file doesn't have the expected format. Columns: {df.columns}")
    
    df['text'] = df['text'].str.strip()
    df['category'] = df['category'].str.strip()
    
    categories = df['category'].unique()
    print(f"Found {len(categories)} unique categories")
    
    base_samples_per_category = max(min_samples_per_category, train_size // len(categories))
    print(f"Base samples per category: {base_samples_per_category}")
    
    train_data = []
    test_data = []
    
    for category in categories:
        category_df = df[df['category'] == category]
        
        if len(category_df) <= base_samples_per_category:
            train_data.append(category_df)
            print(f"Category {category}: Using all {len(category_df)} samples for training")
        else:
            train_samples = category_df.sample(base_samples_per_category, random_state=random_seed)
            train_data.append(train_samples)
            
            remaining = category_df.drop(train_samples.index)
            test_samples_for_category = min(len(remaining), test_size // len(categories))
            if test_samples_for_category > 0:
                test_samples = remaining.sample(test_samples_for_category, random_state=random_seed)
                test_data.append(test_samples)
            
            print(f"Category {category}: {len(train_samples)} for training, {test_samples_for_category} for testing")
    
    train_df = pd.concat(train_data)
    
    if len(train_df) < train_size:
        remaining_df = df.drop(train_df.index)
        additional_samples = min(len(remaining_df), train_size - len(train_df))
        if additional_samples > 0:
            train_df = pd.concat([train_df, remaining_df.sample(additional_samples, random_state=random_seed)])
    
    if test_data:
        test_df = pd.concat(test_data)
    else:
        test_df = pd.DataFrame(columns=df.columns)
    
    remaining_for_test = df.drop(train_df.index)
    if len(test_df) < test_size and len(remaining_for_test) > 0:
        additional_test_samples = min(len(remaining_for_test), test_size - len(test_df))
        if additional_test_samples > 0:
            additional_test = remaining_for_test.sample(additional_test_samples, random_state=random_seed)
            test_df = pd.concat([test_df, additional_test])
    
    train_df = train_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    print(f"\nTraining set: {len(train_df)} examples")
    print(f"Testing set: {len(test_df)} examples")
    
    train_category_counts = Counter(train_df['category'])
    print(f"\nCategory distribution in training set:")
    for category, count in train_category_counts.most_common():
        print(f"  {category}: {count} examples")
    
    train_output_path = output_csv_path.replace('.csv', 'train.csv')
    test_output_path = output_csv_path.replace('.csv', 'test.csv')
    
    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)
    
    print(f"\nSaved training data to {train_output_path}")
    print(f"Saved testing data to {test_output_path}")
    
    plt.figure(figsize=(12, 8))
    
    top_categories = [category for category, _ in train_category_counts.most_common(20)]
    counts = [train_category_counts[category] for category in top_categories]
    
    plt.bar(range(len(top_categories)), counts)
    plt.xticks(range(len(top_categories)), top_categories, rotation=90)
    plt.title('Top 20 Categories in Training Set')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.tight_layout()
    
    plot_path = output_csv_path.replace('.csv', '_distribution.png')
    plt.savefig(plot_path)
    print(f"Saved category distribution plot to {plot_path}")
    
    return train_df, test_df

def create_banking77_dataset(input_csv_path, output_dir="datasets/banking77", train_size=3000, test_size=1000):
    """
    Create a balanced Banking77 dataset and save it in the format expected by the clustering code
    
    Args:
        input_csv_path: Path to the input CSV file
        output_dir: Directory to save the processed dataset
        train_size: Number of examples for training set
        test_size: Number of examples for test set
    """
    os.makedirs(output_dir, exist_ok=True)
    
    train_df, test_df = create_balanced_dataset(
        input_csv_path, 
        os.path.join(output_dir, "banking77.csv"),
        train_size=train_size,
        test_size=test_size
    )
    
    all_categories = sorted(list(set(train_df['category'].unique()) | set(test_df['category'].unique())))
    
    with open(os.path.join(output_dir, "categories.json"), 'w') as f:
        json.dump(all_categories, f, indent=2)
    
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    print(f"\nBanking77 dataset created successfully in {output_dir}")
    print(f"Total categories: {len(all_categories)}")
    print(f"Training examples: {len(train_df)}")
    print(f"Testing examples: {len(test_df)}")
    
    return train_df, test_df, all_categories