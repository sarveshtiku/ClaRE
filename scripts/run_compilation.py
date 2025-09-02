#!/usr/bin/env python3
"""
Script to run the ClaRE data compilation process.
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_compiler import DatasetCompiler

def main():
    """Run the data compilation process."""
    print("Starting ClaRE Data Compilation...")
    print("=" * 50)
    
    # Initialize compiler
    compiler = DatasetCompiler()
    
    try:
        # Download and process Anthropic HH dataset
        print("Downloading Anthropic HH dataset...")
        raw_path = compiler.download_anthropic_hh_dataset()
        
        if raw_path:
            print("Dataset downloaded successfully!")
            print("Processing dataset...")
            
            harmful_prompts, benign_prompts = compiler.process_anthropic_hh_dataset(raw_path)
            
            print("Saving datasets...")
            compiler.save_datasets(harmful_prompts, benign_prompts)
            
            print("Generating statistics...")
            stats = compiler.generate_statistics(harmful_prompts, benign_prompts)
            
            print("\n" + "=" * 50)
            print("Data compilation completed successfully!")
            print("=" * 50)
            print(f"Final Statistics:")
            print(f"   • Harmful prompts: {stats['harmful_prompts']:,}")
            print(f"   • Benign prompts: {stats['benign_prompts']:,}")
            print(f"   • Total prompts: {stats['total_prompts']:,}")
            print(f"   • Harmful ratio: {stats['harmful_ratio']:.1%}")
            print(f"   • Benign ratio: {stats['benign_ratio']:.1%}")
            print(f"   • Avg prompt length: {stats['quality_metrics']['avg_prompt_length']:.0f} chars")
            print("\n Output files saved in the 'data/' directory")
            
        else:
            print("Failed to download Anthropic dataset")
            return 1
            
    except Exception as e:
        print(f"Error during compilation: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
