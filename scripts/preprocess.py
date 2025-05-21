#!/usr/bin/env python
# Data preprocessing utilities for transformer models

import os
import argparse
import json
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess data for transformer training")
    parser.add_argument(
        "--input_file", type=str, required=True,
        help="Path to input data file (CSV, JSON, TSV)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data",
        help="Directory to save processed datasets"
    )
    parser.add_argument(
        "--text_column", type=str, default="text",
        help="Column name containing the text data"
    )
    parser.add_argument(
        "--label_column", type=str, default="label",
        help="Column name containing the labels"
    )
    parser.add_argument(
        "--test_size", type=float, default=0.1,
        help="Proportion of data to use for validation"
    )
    parser.add_argument(
        "--tokenizer", type=str, default="distilbert-base-uncased",
        help="Tokenizer to use for analysis"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Max number of samples to use (useful for debugging)"
    )
    return parser.parse_args()

def load_data(file_path, text_column, label_column, max_samples=None):
    """Load data from various file formats"""
    print(f"📂 Loading data from {file_path}")
    
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext == ".tsv":
        df = pd.read_csv(file_path, sep="\t")
    elif ext == ".json" or ext == ".jsonl":
        with open(file_path, "r") as f:
            # Try to load as JSON lines first
            try:
                data = [json.loads(line) for line in f]
                df = pd.DataFrame(data)
            except json.JSONDecodeError:
                # If that fails, try loading as a single JSON object
                f.seek(0)
                data = json.load(f)
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                else:
                    # Handle case where JSON is a dict with keys as features
                    df = pd.DataFrame([data])
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    # Verify columns exist
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in data. Available columns: {df.columns.tolist()}")
    
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in data. Available columns: {df.columns.tolist()}")
    
    # Limit samples for debugging if requested
    if max_samples and max_samples < len(df):
        df = df.sample(max_samples, random_state=42)
    
    # Ensure text column is string
    df[text_column] = df[text_column].astype(str)
    
    print(f"📊 Loaded {len(df)} samples")
    return df

def analyze_text_lengths(df, text_column, tokenizer_name):
    """Analyze text lengths to help with max_length setting"""
    print(f"📏 Analyzing text lengths using {tokenizer_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Tokenize a sample of texts (max 1000 for speed)
    sample = df[text_column].sample(min(1000, len(df)), random_state=42)
    lengths = [len(tokenizer(text).input_ids) for text in sample]
    
    # Calculate statistics
    avg_length = sum(lengths) / len(lengths)
    p95_length = sorted(lengths)[int(len(lengths) * 0.95)]
    max_length = max(lengths)
    
    print(f"Average token length: {avg_length:.1f}")
    print(f"95th percentile length: {p95_length}")
    print(f"Maximum token length: {max_length}")
    
    return {
        "avg_length": avg_length,
        "p95_length": p95_length,
        "max_length": max_length
    }
    
def split_and_save_data(df, text_column, label_column, output_dir, test_size):
    """Split data into train/validation and save as JSON files"""
    print(f"✂️ Splitting data into train/validation sets ({test_size:.0%} validation)")
    
    # Create train/validation split
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df[label_column] if len(df[label_column].unique()) > 1 else None)
    
    print(f"📊 Train set: {len(train_df)} samples")
    print(f"📊 Validation set: {len(val_df)} samples")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as JSON files
    train_path = os.path.join(output_dir, "train.json")
    val_path = os.path.join(output_dir, "validation.json")
    
    train_df.to_json(train_path, orient="records", lines=True)
    val_df.to_json(val_path, orient="records", lines=True)
    
    print(f"💾 Train data saved to {train_path}")
    print(f"💾 Validation data saved to {val_path}")
    
    return train_path, val_path

def create_dataset_dict(train_path, val_path):
    """Create a HF DatasetDict from the saved files"""
    train_dataset = Dataset.from_json(train_path)
    val_dataset = Dataset.from_json(val_path)
    
    return DatasetDict({
        "train": train_dataset,
        "validation": val_dataset
    })

def main():
    # Parse arguments
    args = parse_args()
    
    # Load data
    df = load_data(args.input_file, args.text_column, args.label_column, args.max_samples)
    
    # Analyze text lengths
    length_stats = analyze_text_lengths(df, args.text_column, args.tokenizer)
    
    # Split and save data
    train_path, val_path = split_and_save_data(
        df, args.text_column, args.label_column, args.output_dir, args.test_size
    )
    
    # Create and verify dataset
    dataset = create_dataset_dict(train_path, val_path)
    print(f"✅ Successfully created dataset with {len(dataset['train'])} train and {len(dataset['validation'])} validation examples")
    
    # Print summary and recommendations
    print("\n📋 Data preprocessing complete!")
    print(f"📝 Recommended max_length setting: {int(length_stats['p95_length'])}")
    print(f"🏷️ Label distribution: {df[args.label_column].value_counts().to_dict()}")
    
if __name__ == "__main__":
    main() 