#!/usr/bin/env python3
"""
Enhanced ClaRE Data Compiler for Multiple Datasets
Compiles and processes multiple datasets containing harmful and benign prompts.
"""

import os
import json
import jsonlines
import pandas as pd
import requests
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from tqdm import tqdm
import logging

from config import DATASET_SOURCES, HARMFUL_INDICATORS, BENIGN_INDICATORS, HARMFUL_PATTERNS, MIN_RESPONSE_LENGTH, MAX_RESPONSE_LENGTH
from utils import (
    clean_text, extract_prompt_from_conversation, generate_prompt_id,
    validate_prompt_data, deduplicate_prompts, split_dataset,
    save_jsonl, load_jsonl, create_data_summary
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiDatasetCompiler:
    """Enhanced class for compiling harmful and benign prompt datasets from multiple sources."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.data_dir / "raw").mkdir(exist_ok=True)
        (self.data_dir / "processed").mkdir(exist_ok=True)
        (self.data_dir / "harmful").mkdir(exist_ok=True)
        (self.data_dir / "benign").mkdir(exist_ok=True)
        
        self.all_harmful_prompts = []
        self.all_benign_prompts = []
    
    def download_dataset(self, dataset_key: str) -> Optional[str]:
        """Download a specific dataset from Hugging Face."""
        try:
            from datasets import load_dataset
            
            dataset_info = DATASET_SOURCES[dataset_key]
            huggingface_id = dataset_info['huggingface_id']
            
            logger.info(f"Downloading {dataset_info['name']}...")
            dataset = load_dataset(huggingface_id)
            
            # Save the dataset
            raw_path = self.data_dir / "raw" / dataset_key
            raw_path.mkdir(exist_ok=True)
            
            # Save all splits
            for split_name, split_data in dataset.items():
                output_file = raw_path / f"{split_name}.jsonl"
                
                with jsonlines.open(output_file, 'w') as writer:
                    for item in tqdm(split_data, desc=f"Saving {split_name} split"):
                        writer.write(item)
                
                logger.info(f"Saved {len(split_data)} items to {output_file}")
            
            return str(raw_path)
            
        except Exception as e:
            logger.error(f"Error downloading {dataset_key}: {e}")
            return None
    
    def process_anthropic_hh_dataset(self, raw_path: str) -> Tuple[List[Dict], List[Dict]]:
        """Process the Anthropic HH dataset (already implemented)."""
        from data_compiler import DatasetCompiler
        compiler = DatasetCompiler(str(self.data_dir))
        return compiler.process_anthropic_hh_dataset(raw_path)
    
    def process_allenai_dataset(self, raw_path: str) -> Tuple[List[Dict], List[Dict]]:
        """Process AllenAI real-toxicity-prompts dataset."""
        harmful_prompts = []
        benign_prompts = []
        
        raw_path = Path(raw_path)
        
        for split_file in raw_path.glob("*.jsonl"):
            logger.info(f"Processing {split_file.name}...")
            
            with jsonlines.open(split_file) as reader:
                for item in tqdm(reader, desc=f"Processing {split_file.name}"):
                    # Extract prompt from the text field
                    prompt = item.get('text', '').strip()
                    if not prompt:
                        continue
                    
                    # Check toxicity score if available
                    toxicity = item.get('toxicity', 0)
                    
                    if toxicity > 0.5:  # High toxicity threshold
                        harmful_prompts.append({
                            'prompt': clean_text(prompt),
                            'label': 'harmful',
                            'source': 'allenai_harmful',
                            'split': split_file.stem,
                            'toxicity_score': toxicity,
                            'original_item': item
                        })
                    else:
                        benign_prompts.append({
                            'prompt': clean_text(prompt),
                            'label': 'benign',
                            'source': 'allenai_harmful',
                            'split': split_file.stem,
                            'toxicity_score': toxicity,
                            'original_item': item
                        })
        
        logger.info(f"Extracted {len(harmful_prompts)} harmful and {len(benign_prompts)} benign prompts from AllenAI")
        return harmful_prompts, benign_prompts
    
    def process_onepaneai_dataset(self, raw_path: str) -> Tuple[List[Dict], List[Dict]]:
        """Process OnepaneAI harmful prompts dataset."""
        harmful_prompts = []
        benign_prompts = []
        
        raw_path = Path(raw_path)
        
        for split_file in raw_path.glob("*.jsonl"):
            logger.info(f"Processing {split_file.name}...")
            
            with jsonlines.open(split_file) as reader:
                for item in tqdm(reader, desc=f"Processing {split_file.name}"):
                    # Extract prompt
                    prompt = item.get('prompt', item.get('text', '')).strip()
                    if not prompt:
                        continue
                    
                    # Check if marked as harmful
                    is_harmful = item.get('is_harmful', True)  # Default to harmful for this dataset
                    category = item.get('category', 'unknown')
                    
                    if is_harmful:
                        harmful_prompts.append({
                            'prompt': clean_text(prompt),
                            'label': 'harmful',
                            'source': 'onepaneai_harmful',
                            'split': split_file.stem,
                            'category': category,
                            'original_item': item
                        })
                    else:
                        benign_prompts.append({
                            'prompt': clean_text(prompt),
                            'label': 'benign',
                            'source': 'onepaneai_harmful',
                            'split': split_file.stem,
                            'category': category,
                            'original_item': item
                        })
        
        logger.info(f"Extracted {len(harmful_prompts)} harmful and {len(benign_prompts)} benign prompts from OnepaneAI")
        return harmful_prompts, benign_prompts
    
    def process_nask_dataset(self, raw_path: str) -> Tuple[List[Dict], List[Dict]]:
        """Process NASK-PIB harmful prompts dataset."""
        harmful_prompts = []
        benign_prompts = []
        
        raw_path = Path(raw_path)
        
        for split_file in raw_path.glob("*.jsonl"):
            logger.info(f"Processing {split_file.name}...")
            
            with jsonlines.open(split_file) as reader:
                for item in tqdm(reader, desc=f"Processing {split_file.name}"):
                    # Extract prompt
                    prompt = item.get('prompt', item.get('text', '')).strip()
                    if not prompt:
                        continue
                    
                    # This dataset is specifically for harmful prompts
                    category = item.get('category', 'unknown')
                    subcategory = item.get('subcategory', 'unknown')
                    
                    harmful_prompts.append({
                        'prompt': clean_text(prompt),
                        'label': 'harmful',
                        'source': 'nask_harmful',
                        'split': split_file.stem,
                        'category': category,
                        'subcategory': subcategory,
                        'original_item': item
                    })
        
        logger.info(f"Extracted {len(harmful_prompts)} harmful prompts from NASK-PIB")
        return harmful_prompts, benign_prompts
    
    def process_hex_phi_dataset(self, raw_path: str) -> Tuple[List[Dict], List[Dict]]:
        """Process HEx-PHI harmful instructions dataset."""
        harmful_prompts = []
        benign_prompts = []
        
        raw_path = Path(raw_path)
        
        for split_file in raw_path.glob("*.jsonl"):
            logger.info(f"Processing {split_file.name}...")
            
            with jsonlines.open(split_file) as reader:
                for item in tqdm(reader, desc=f"Processing {split_file.name}"):
                    # Extract instruction
                    instruction = item.get('instruction', item.get('prompt', '')).strip()
                    if not instruction:
                        continue
                    
                    # This dataset is specifically for harmful instructions
                    category = item.get('category', 'unknown')
                    
                    harmful_prompts.append({
                        'prompt': clean_text(instruction),
                        'label': 'harmful',
                        'source': 'hex_phi_harmful',
                        'split': split_file.stem,
                        'category': category,
                        'original_item': item
                    })
        
        logger.info(f"Extracted {len(harmful_prompts)} harmful prompts from HEx-PHI")
        return harmful_prompts, benign_prompts
    
    def process_guychuk_dataset(self, raw_path: str) -> Tuple[List[Dict], List[Dict]]:
        """Process Guychuk benign-malicious classification dataset."""
        harmful_prompts = []
        benign_prompts = []
        
        raw_path = Path(raw_path)
        
        for split_file in raw_path.glob("*.jsonl"):
            logger.info(f"Processing {split_file.name}...")
            
            with jsonlines.open(split_file) as reader:
                for item in tqdm(reader, desc=f"Processing {split_file.name}"):
                    # Extract prompt
                    prompt = item.get('prompt', item.get('text', '')).strip()
                    if not prompt:
                        continue
                    
                    # Check label
                    label = item.get('label', item.get('is_malicious', False))
                    
                    if label == 1 or label == 'malicious' or label is True:
                        harmful_prompts.append({
                            'prompt': clean_text(prompt),
                            'label': 'harmful',
                            'source': 'guychuk_harmful',
                            'split': split_file.stem,
                            'original_item': item
                        })
                    else:
                        benign_prompts.append({
                            'prompt': clean_text(prompt),
                            'label': 'benign',
                            'source': 'guychuk_harmful',
                            'split': split_file.stem,
                            'original_item': item
                        })
        
        logger.info(f"Extracted {len(harmful_prompts)} harmful and {len(benign_prompts)} benign prompts from Guychuk")
        return harmful_prompts, benign_prompts
    
    def process_openassistant_dataset(self, raw_path: str) -> Tuple[List[Dict], List[Dict]]:
        """Process OpenAssistant conversations dataset."""
        harmful_prompts = []
        benign_prompts = []
        
        raw_path = Path(raw_path)
        
        for split_file in raw_path.glob("*.jsonl"):
            logger.info(f"Processing {split_file.name}...")
            
            with jsonlines.open(split_file) as reader:
                for item in tqdm(reader, desc=f"Processing {split_file.name}"):
                    # Extract text from conversation
                    text = item.get('text', '').strip()
                    if not text:
                        continue
                    
                    # OpenAssistant conversations are generally benign
                    # Extract the human part if it's a conversation
                    if "Human:" in text:
                        prompt = extract_prompt_from_conversation(text)
                    else:
                        prompt = text
                    
                    if prompt and len(prompt.strip()) > 10:
                        benign_prompts.append({
                            'prompt': clean_text(prompt),
                            'label': 'benign',
                            'source': 'openassistant_benign',
                            'split': split_file.stem,
                            'original_item': item
                        })
        
        logger.info(f"Extracted {len(benign_prompts)} benign prompts from OpenAssistant")
        return harmful_prompts, benign_prompts
    
    def process_alpaca_dataset(self, raw_path: str) -> Tuple[List[Dict], List[Dict]]:
        """Process Stanford Alpaca dataset."""
        harmful_prompts = []
        benign_prompts = []
        
        raw_path = Path(raw_path)
        
        for split_file in raw_path.glob("*.jsonl"):
            logger.info(f"Processing {split_file.name}...")
            
            with jsonlines.open(split_file) as reader:
                for item in tqdm(reader, desc=f"Processing {split_file.name}"):
                    # Extract instruction
                    instruction = item.get('instruction', '').strip()
                    if not instruction:
                        continue
                    
                    # Alpaca instructions are generally benign
                    benign_prompts.append({
                        'prompt': clean_text(instruction),
                        'label': 'benign',
                        'source': 'alpaca_benign',
                        'split': split_file.stem,
                        'original_item': item
                    })
        
        logger.info(f"Extracted {len(benign_prompts)} benign prompts from Alpaca")
        return harmful_prompts, benign_prompts
    
    def process_dolly_dataset(self, raw_path: str) -> Tuple[List[Dict], List[Dict]]:
        """Process Databricks Dolly dataset."""
        harmful_prompts = []
        benign_prompts = []
        
        raw_path = Path(raw_path)
        
        for split_file in raw_path.glob("*.jsonl"):
            logger.info(f"Processing {split_file.name}...")
            
            with jsonlines.open(split_file) as reader:
                for item in tqdm(reader, desc=f"Processing {split_file.name}"):
                    # Extract instruction
                    instruction = item.get('instruction', '').strip()
                    if not instruction:
                        continue
                    
                    # Dolly instructions are generally benign
                    benign_prompts.append({
                        'prompt': clean_text(instruction),
                        'label': 'benign',
                        'source': 'dolly_benign',
                        'split': split_file.stem,
                        'category': item.get('category', 'unknown'),
                        'original_item': item
                    })
        
        logger.info(f"Extracted {len(benign_prompts)} benign prompts from Dolly")
        return harmful_prompts, benign_prompts
    
    def process_dataset(self, dataset_key: str) -> bool:
        """Process a specific dataset."""
        dataset_info = DATASET_SOURCES[dataset_key]
        
        # Download dataset
        raw_path = self.download_dataset(dataset_key)
        if not raw_path:
            logger.error(f"Failed to download {dataset_key}")
            return False
        
        # Process based on dataset type
        try:
            if dataset_key == 'anthropic_hh':
                harmful_prompts, benign_prompts = self.process_anthropic_hh_dataset(raw_path)
            elif dataset_key == 'allenai_harmful':
                harmful_prompts, benign_prompts = self.process_allenai_dataset(raw_path)
            elif dataset_key == 'onepaneai_harmful':
                harmful_prompts, benign_prompts = self.process_onepaneai_dataset(raw_path)
            elif dataset_key == 'nask_harmful':
                harmful_prompts, benign_prompts = self.process_nask_dataset(raw_path)
            elif dataset_key == 'hex_phi_harmful':
                harmful_prompts, benign_prompts = self.process_hex_phi_dataset(raw_path)
            elif dataset_key == 'guychuk_harmful':
                harmful_prompts, benign_prompts = self.process_guychuk_dataset(raw_path)
            elif dataset_key == 'openassistant_benign':
                harmful_prompts, benign_prompts = self.process_openassistant_dataset(raw_path)
            elif dataset_key == 'alpaca_benign':
                harmful_prompts, benign_prompts = self.process_alpaca_dataset(raw_path)
            elif dataset_key == 'dolly_benign':
                harmful_prompts, benign_prompts = self.process_dolly_dataset(raw_path)
            else:
                logger.error(f"Unknown dataset: {dataset_key}")
                return False
            
            # Add to global collections
            self.all_harmful_prompts.extend(harmful_prompts)
            self.all_benign_prompts.extend(benign_prompts)
            
            logger.info(f"Successfully processed {dataset_key}: {len(harmful_prompts)} harmful, {len(benign_prompts)} benign")
            return True
            
        except Exception as e:
            logger.error(f"Error processing {dataset_key}: {e}")
            return False
    
    def compile_all_datasets(self, exclude_anthropic: bool = True) -> Tuple[List[Dict], List[Dict]]:
        """Compile all datasets except Anthropic (already done)."""
        logger.info("Starting compilation of all datasets...")
        
        datasets_to_process = list(DATASET_SOURCES.keys())
        if exclude_anthropic:
            datasets_to_process.remove('anthropic_hh')
        
        successful_datasets = []
        failed_datasets = []
        
        for dataset_key in datasets_to_process:
            logger.info(f"Processing {dataset_key}...")
            if self.process_dataset(dataset_key):
                successful_datasets.append(dataset_key)
            else:
                failed_datasets.append(dataset_key)
        
        logger.info(f"Successfully processed: {successful_datasets}")
        if failed_datasets:
            logger.warning(f"Failed to process: {failed_datasets}")
        
        return self.all_harmful_prompts, self.all_benign_prompts
    
    def save_compiled_datasets(self, harmful_prompts: List[Dict], benign_prompts: List[Dict]):
        """Save the compiled datasets with enhanced processing."""
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
        harmful_file = self.data_dir / "harmful" / "all_harmful_prompts.jsonl"
        save_jsonl(harmful_prompts, str(harmful_file))
        
        # Save benign prompts
        benign_file = self.data_dir / "benign" / "all_benign_prompts.jsonl"
        save_jsonl(benign_prompts, str(benign_file))
        
        # Save as CSV
        harmful_df = pd.DataFrame(harmful_prompts)
        benign_df = pd.DataFrame(benign_prompts)
        
        harmful_df.to_csv(self.data_dir / "harmful" / "all_harmful_prompts.csv", index=False)
        benign_df.to_csv(self.data_dir / "benign" / "all_benign_prompts.csv", index=False)
        
        # Save combined dataset
        combined_df = pd.concat([harmful_df, benign_df], ignore_index=True)
        combined_df.to_csv(self.data_dir / "processed" / "all_combined_dataset.csv", index=False)
        
        # Create train/val/test splits
        all_prompts = harmful_prompts + benign_prompts
        splits = split_dataset(all_prompts)
        
        for split_name, split_data in splits.items():
            split_file = self.data_dir / "processed" / f"all_{split_name}.jsonl"
            save_jsonl(split_data, str(split_file))
            
            split_df = pd.DataFrame(split_data)
            split_df.to_csv(self.data_dir / "processed" / f"all_{split_name}.csv", index=False)
        
        logger.info(f"Saved compiled datasets:")
        logger.info(f"  - Harmful prompts: {len(harmful_prompts)} items")
        logger.info(f"  - Benign prompts: {len(benign_prompts)} items")
        logger.info(f"  - Combined dataset: {len(combined_df)} items")
        logger.info(f"  - Train/Val/Test splits created")
    
    def generate_comprehensive_statistics(self, harmful_prompts: List[Dict], benign_prompts: List[Dict]):
        """Generate comprehensive statistics for all datasets."""
        all_prompts = harmful_prompts + benign_prompts
        summary = create_data_summary(all_prompts)
        
        # Source breakdown
        source_stats = {}
        for prompt in all_prompts:
            source = prompt['source']
            if source not in source_stats:
                source_stats[source] = {'harmful': 0, 'benign': 0}
            source_stats[source][prompt['label']] += 1
        
        # Enhanced statistics
        stats = {
            **summary,
            'source_breakdown': source_stats,
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
        stats_file = self.data_dir / "processed" / "comprehensive_dataset_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info("Comprehensive Dataset Statistics:")
        logger.info(f"  - Total harmful prompts: {stats['harmful_prompts']}")
        logger.info(f"  - Total benign prompts: {stats['benign_prompts']}")
        logger.info(f"  - Total prompts: {stats['total_prompts']}")
        logger.info(f"  - Harmful ratio: {stats['harmful_ratio']:.2%}")
        logger.info(f"  - Benign ratio: {stats['benign_ratio']:.2%}")
        logger.info(f"  - Average prompt length: {stats['quality_metrics']['avg_prompt_length']:.1f} characters")
        logger.info(f"  - Sources: {', '.join(stats['dataset_info']['sources'])}")
        
        # Print source breakdown
        logger.info("Source Breakdown:")
        for source, counts in source_stats.items():
            logger.info(f"  - {source}: {counts['harmful']} harmful, {counts['benign']} benign")
        
        return stats

def main():
    """Main function to run the comprehensive data compilation process."""
    logger.info("Starting Comprehensive ClaRE Data Compilation...")
    
    compiler = MultiDatasetCompiler()
    
    try:
        # Compile all datasets except Anthropic (already done)
        harmful_prompts, benign_prompts = compiler.compile_all_datasets(exclude_anthropic=True)
        
        # Save compiled datasets
        compiler.save_compiled_datasets(harmful_prompts, benign_prompts)
        
        # Generate comprehensive statistics
        compiler.generate_comprehensive_statistics(harmful_prompts, benign_prompts)
        
        logger.info("Comprehensive data compilation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during comprehensive compilation: {e}")

if __name__ == "__main__":
    main()
