#!/usr/bin/env python3
"""
ClaRE Data Compiler
Compiles and processes datasets containing harmful and benign prompts.
"""

import os
import json
import jsonlines
import pandas as pd
import requests
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
import logging

from config import (
    DATASET_SOURCES, HARMFUL_INDICATORS, BENIGN_INDICATORS, 
    HARMFUL_PATTERNS, MIN_RESPONSE_LENGTH, MAX_RESPONSE_LENGTH
)
from utils import (
    clean_text, extract_prompt_from_conversation, generate_prompt_id,
    validate_prompt_data, deduplicate_prompts, split_dataset,
    save_jsonl, load_jsonl, create_data_summary
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetCompiler:
    """Main class for compiling harmful and benign prompt datasets."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.data_dir / "raw").mkdir(exist_ok=True)
        (self.data_dir / "processed").mkdir(exist_ok=True)
        (self.data_dir / "harmful").mkdir(exist_ok=True)
        (self.data_dir / "benign").mkdir(exist_ok=True)
        
    def download_anthropic_hh_dataset(self) -> str:
        """Download the Anthropic HH dataset from Hugging Face."""
        try:
            from datasets import load_dataset
            
            logger.info("Downloading Anthropic HH dataset...")
            dataset = load_dataset("Anthropic/hh-rlhf")
            
            # Save the dataset
            raw_path = self.data_dir / "raw" / "anthropic_hh"
            raw_path.mkdir(exist_ok=True)
            
            # Save train and test splits
            for split in ['train', 'test']:
                if split in dataset:
                    split_data = dataset[split]
                    output_file = raw_path / f"{split}.jsonl"
                    
                    with jsonlines.open(output_file, 'w') as writer:
                        for item in tqdm(split_data, desc=f"Saving {split} split"):
                            writer.write(item)
                    
                    logger.info(f"Saved {len(split_data)} items to {output_file}")
            
            return str(raw_path)
            
        except Exception as e:
            logger.error(f"Error downloading Anthropic dataset: {e}")
            return None
    
    def process_anthropic_hh_dataset(self, raw_path: str) -> Tuple[List[Dict], List[Dict]]:
        """Process the Anthropic HH dataset to separate harmful and benign prompts."""
        harmful_prompts = []
        benign_prompts = []
        
        raw_path = Path(raw_path)
        
        for split_file in ['train.jsonl', 'test.jsonl']:
            file_path = raw_path / split_file
            if not file_path.exists():
                continue
                
            logger.info(f"Processing {split_file}...")
            
            with jsonlines.open(file_path) as reader:
                for item in tqdm(reader, desc=f"Processing {split_file}"):
                    # Extract the prompt from the chosen/rejected responses
                    chosen = item.get('chosen', '')
                    rejected = item.get('rejected', '')
                    
                    # Extract the actual prompt (before the assistant response)
                    prompt_chosen = self._extract_prompt(chosen)
                    prompt_rejected = self._extract_prompt(rejected)
                    
                    # Classify based on the response content
                    if self._is_harmful_response(rejected):
                        harmful_prompts.append({
                            'prompt': prompt_rejected,
                            'response': rejected,
                            'label': 'harmful',
                            'source': 'anthropic_hh',
                            'split': split_file.replace('.jsonl', ''),
                            'original_item': item
                        })
                    
                    if self._is_benign_response(chosen):
                        benign_prompts.append({
                            'prompt': prompt_chosen,
                            'response': chosen,
                            'label': 'benign',
                            'source': 'anthropic_hh',
                            'split': split_file.replace('.jsonl', ''),
                            'original_item': item
                        })
        
        logger.info(f"Extracted {len(harmful_prompts)} harmful prompts and {len(benign_prompts)} benign prompts")
        return harmful_prompts, benign_prompts
    
    def _extract_prompt(self, text: str) -> str:
        """Extract the human prompt from the assistant response."""
        return extract_prompt_from_conversation(text)
    
    def _is_harmful_response(self, response: str) -> bool:
        """Determine if a response is harmful based on content analysis."""
        response_lower = response.lower()
        
        # Check for harmful indicators
        for indicator in HARMFUL_INDICATORS:
            if indicator.lower() in response_lower:
                return True
        
        # Additional checks for harmful content patterns
        for pattern in HARMFUL_PATTERNS:
            if pattern in response_lower:
                return True
        
        return False
    
    def _is_benign_response(self, response: str) -> bool:
        """Determine if a response is benign (helpful and harmless)."""
        # If it's not harmful and provides useful information, consider it benign
        if not self._is_harmful_response(response):
            response_lower = response.lower()
            
            # Check for positive indicators
            for indicator in BENIGN_INDICATORS:
                if indicator in response_lower:
                    return True
            
            # If response is substantial and not harmful, consider it benign
            if MIN_RESPONSE_LENGTH <= len(response.strip()) <= MAX_RESPONSE_LENGTH:
                return True
        
        return False
    
    def save_datasets(self, harmful_prompts: List[Dict], benign_prompts: List[Dict]):
        """Save the processed datasets to files."""
        # Validate and clean data
        harmful_prompts = [p for p in harmful_prompts if validate_prompt_data(p)]
        benign_prompts = [p for p in benign_prompts if validate_prompt_data(p)]
        
        # Deduplicate prompts
        harmful_prompts = deduplicate_prompts(harmful_prompts)
        benign_prompts = deduplicate_prompts(benign_prompts)
        
        # Add unique IDs
        for prompt in harmful_prompts:
            prompt['id'] = generate_prompt_id(prompt['prompt'], prompt['source'])
        for prompt in benign_prompts:
            prompt['id'] = generate_prompt_id(prompt['prompt'], prompt['source'])
        
        # Save harmful prompts
        harmful_file = self.data_dir / "harmful" / "harmful_prompts.jsonl"
        save_jsonl(harmful_prompts, str(harmful_file))
        
        # Save benign prompts
        benign_file = self.data_dir / "benign" / "benign_prompts.jsonl"
        save_jsonl(benign_prompts, str(benign_file))
        
        # Save as CSV for easy analysis
        harmful_df = pd.DataFrame(harmful_prompts)
        benign_df = pd.DataFrame(benign_prompts)
        
        harmful_df.to_csv(self.data_dir / "harmful" / "harmful_prompts.csv", index=False)
        benign_df.to_csv(self.data_dir / "benign" / "benign_prompts.csv", index=False)
        
        # Save combined dataset
        combined_df = pd.concat([harmful_df, benign_df], ignore_index=True)
        combined_df.to_csv(self.data_dir / "processed" / "combined_dataset.csv", index=False)
        
        # Create train/val/test splits
        all_prompts = harmful_prompts + benign_prompts
        splits = split_dataset(all_prompts)
        
        for split_name, split_data in splits.items():
            split_file = self.data_dir / "processed" / f"{split_name}.jsonl"
            save_jsonl(split_data, str(split_file))
            
            split_df = pd.DataFrame(split_data)
            split_df.to_csv(self.data_dir / "processed" / f"{split_name}.csv", index=False)
        
        logger.info(f"Saved datasets:")
        logger.info(f"  - Harmful prompts: {len(harmful_prompts)} items")
        logger.info(f"  - Benign prompts: {len(benign_prompts)} items")
        logger.info(f"  - Combined dataset: {len(combined_df)} items")
        logger.info(f"  - Train/Val/Test splits created")
    
    def generate_statistics(self, harmful_prompts: List[Dict], benign_prompts: List[Dict]):
        """Generate statistics about the compiled datasets."""
        all_prompts = harmful_prompts + benign_prompts
        summary = create_data_summary(all_prompts)
        
        # Additional detailed statistics
        stats = {
            **summary,
            'dataset_info': {
                'sources': list(set(p['source'] for p in all_prompts)),
                'splits': list(set(p.get('split', 'unknown') for p in all_prompts)),
                'creation_date': pd.Timestamp.now().isoformat()
            },
            'quality_metrics': {
                'avg_prompt_length': summary['average_prompt_length'],
                'min_prompt_length': min(len(p['prompt']) for p in all_prompts) if all_prompts else 0,
                'max_prompt_length': max(len(p['prompt']) for p in all_prompts) if all_prompts else 0,
                'duplicates_removed': len(harmful_prompts + benign_prompts) - len(all_prompts)
            }
        }
        
        # Save statistics
        stats_file = self.data_dir / "processed" / "dataset_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info("Dataset Statistics:")
        logger.info(f"  - Total harmful prompts: {stats['harmful_prompts']}")
        logger.info(f"  - Total benign prompts: {stats['benign_prompts']}")
        logger.info(f"  - Total prompts: {stats['total_prompts']}")
        logger.info(f"  - Harmful ratio: {stats['harmful_ratio']:.2%}")
        logger.info(f"  - Benign ratio: {stats['benign_ratio']:.2%}")
        logger.info(f"  - Average prompt length: {stats['quality_metrics']['avg_prompt_length']:.1f} characters")
        logger.info(f"  - Sources: {', '.join(stats['dataset_info']['sources'])}")
        
        return stats

def main():
    """Main function to run the data compilation process."""
    logger.info("Starting ClaRE Data Compilation...")
    
    compiler = DatasetCompiler()
    
    # Download and process Anthropic HH dataset
    raw_path = compiler.download_anthropic_hh_dataset()
    if raw_path:
        harmful_prompts, benign_prompts = compiler.process_anthropic_hh_dataset(raw_path)
        
        # Save datasets
        compiler.save_datasets(harmful_prompts, benign_prompts)
        
        # Generate statistics
        compiler.generate_statistics(harmful_prompts, benign_prompts)
        
        logger.info("Data compilation completed successfully!")
    else:
        logger.error("Failed to download Anthropic dataset")

if __name__ == "__main__":
    main()
