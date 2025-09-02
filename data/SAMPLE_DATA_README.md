# Sample Dataset Files

This directory contains sample files from the full ClaRE dataset for demonstration and testing purposes.

## Sample Files Available

### Harmful Prompts
- `sample_harmful_prompts.csv` - 2,000 sample harmful prompts
- `sample_harmful_prompts.jsonl` - Same data in JSONL format

### Benign Prompts  
- `sample_benign_prompts.csv` - 2,000 sample benign prompts
- `sample_benign_prompts.jsonl` - Same data in JSONL format

### Combined Dataset
- `sample_combined_dataset.csv` - 4,000 total prompts (2k harmful + 2k benign)

### Train/Val/Test Splits
- `sample_train.csv` - Training data (80% of sample)
- `sample_val.csv` - Validation data (10% of sample)  
- `sample_test.csv` - Test data (10% of sample)

## Full Dataset

The complete dataset contains **618,966 prompts** (236,358 harmful + 382,608 benign) from 6 major sources. To access the full dataset, run:

```bash
python3 scripts/run_comprehensive_compilation.py
```

This will download and compile all datasets locally.

## Usage

```python
import pandas as pd

# Load sample data
harmful_sample = pd.read_csv('data/harmful/sample_harmful_prompts.csv')
benign_sample = pd.read_csv('data/benign/sample_benign_prompts.csv')
combined_sample = pd.read_csv('data/processed/sample_combined_dataset.csv')

print(f"Sample: {len(harmful_sample)} harmful, {len(benign_sample)} benign")
```

## Data Structure

Each row contains:
- `prompt`: The actual prompt text
- `label`: 'harmful' or 'benign'
- `source`: Dataset source identifier
- `id`: Unique identifier
- Additional metadata varies by source
