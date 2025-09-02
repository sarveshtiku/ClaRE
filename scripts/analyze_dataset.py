#!/usr/bin/env python3
"""
Script to analyze the compiled datasets.
"""

import sys
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def load_dataset(filepath: str) -> pd.DataFrame:
    """Load dataset from JSONL file."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

def analyze_dataset():
    """Analyze the compiled dataset."""
    data_dir = Path(__file__).parent.parent / "data"
    
    # Load datasets
    harmful_file = data_dir / "harmful" / "harmful_prompts.jsonl"
    benign_file = data_dir / "benign" / "benign_prompts.jsonl"
    
    if not harmful_file.exists() or not benign_file.exists():
        print("Dataset files not found. Please run the compilation first.")
        return
    
    print("Loading datasets...")
    harmful_df = load_dataset(harmful_file)
    benign_df = load_dataset(benign_file)
    
    print(f"Loaded {len(harmful_df)} harmful and {len(benign_df)} benign prompts")
    
    # Basic statistics
    print("\n Dataset Statistics:")
    print("=" * 40)
    print(f"Total prompts: {len(harmful_df) + len(benign_df):,}")
    print(f"Harmful prompts: {len(harmful_df):,} ({len(harmful_df)/(len(harmful_df)+len(benign_df))*100:.1f}%)")
    print(f"Benign prompts: {len(benign_df):,} ({len(benign_df)/(len(harmful_df)+len(benign_df))*100:.1f}%)")
    
    # Length analysis
    harmful_lengths = harmful_df['prompt'].str.len()
    benign_lengths = benign_df['prompt'].str.len()
    
    print(f"\n📏 Prompt Length Analysis:")
    print("=" * 40)
    print(f"Harmful prompts - Avg: {harmful_lengths.mean():.0f}, Min: {harmful_lengths.min()}, Max: {harmful_lengths.max()}")
    print(f"Benign prompts - Avg: {benign_lengths.mean():.0f}, Min: {benign_lengths.min()}, Max: {benign_lengths.max()}")
    
    # Source analysis
    print(f"\nSource Analysis:")
    print("=" * 40)
    print("Harmful prompts by source:")
    print(harmful_df['source'].value_counts())
    print("\nBenign prompts by source:")
    print(benign_df['source'].value_counts())
    
    # Sample prompts
    print(f"\nSample Harmful Prompts:")
    print("=" * 40)
    for i, prompt in enumerate(harmful_df['prompt'].head(3)):
        print(f"{i+1}. {prompt[:100]}...")
    
    print(f"\nSample Benign Prompts:")
    print("=" * 40)
    for i, prompt in enumerate(benign_df['prompt'].head(3)):
        print(f"{i+1}. {prompt[:100]}...")

if __name__ == "__main__":
    analyze_dataset()
