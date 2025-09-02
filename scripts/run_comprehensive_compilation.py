#!/usr/bin/env python3
"""
Script to run the comprehensive ClaRE data compilation process.
This compiles ALL datasets from the list provided by the user.
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from multi_dataset_compiler import MultiDatasetCompiler

def main():
    """Run the comprehensive data compilation process."""
    print("🚀 Starting Comprehensive ClaRE Data Compilation...")
    print("=" * 60)
    print("This will compile ALL datasets from your list:")
    print("📋 Harmful Datasets:")
    print("   • AllenAI Real Toxicity Prompts (~160k conversations)")
    print("   • OnepaneAI Adversarial Prompts (~120k prompts)")
    print("   • NASK-PIB Harmful Prompts (428 manually reviewed)")
    print("   • HEx-PHI Harmful Instructions (330 instructions)")
    print("   • Guychuk Benign-Malicious Classification (464k prompts)")
    print("")
    print("📋 Benign Datasets:")
    print("   • OpenAssistant Conversations (88k conversations)")
    print("   • Stanford Alpaca (52k instructions)")
    print("   • Databricks Dolly 15k (15k instructions)")
    print("")
    print("ℹ️  Note: Anthropic HH dataset already compiled (skipping)")
    print("=" * 60)
    
    # Initialize compiler
    compiler = MultiDatasetCompiler()
    
    try:
        # Compile all datasets except Anthropic (already done)
        print("📥 Starting dataset compilation...")
        harmful_prompts, benign_prompts = compiler.compile_all_datasets(exclude_anthropic=True)
        
        print("💾 Saving compiled datasets...")
        compiler.save_compiled_datasets(harmful_prompts, benign_prompts)
        
        print("📊 Generating comprehensive statistics...")
        stats = compiler.generate_comprehensive_statistics(harmful_prompts, benign_prompts)
        
        print("\n" + "=" * 60)
        print("✅ Comprehensive data compilation completed successfully!")
        print("=" * 60)
        print(f"📈 Final Statistics:")
        print(f"   • Total Harmful prompts: {stats['harmful_prompts']:,}")
        print(f"   • Total Benign prompts: {stats['benign_prompts']:,}")
        print(f"   • Total prompts: {stats['total_prompts']:,}")
        print(f"   • Harmful ratio: {stats['harmful_ratio']:.1%}")
        print(f"   • Benign ratio: {stats['benign_ratio']:.1%}")
        print(f"   • Avg prompt length: {stats['quality_metrics']['avg_prompt_length']:.0f} chars")
        print(f"   • Sources: {len(stats['dataset_info']['sources'])} datasets")
        print("")
        print("📁 Output files saved in the 'data/' directory:")
        print("   • data/harmful/all_harmful_prompts.jsonl")
        print("   • data/benign/all_benign_prompts.jsonl")
        print("   • data/processed/all_combined_dataset.csv")
        print("   • data/processed/all_train.jsonl")
        print("   • data/processed/all_val.jsonl")
        print("   • data/processed/all_test.jsonl")
        
        # Show source breakdown
        print("\n📊 Source Breakdown:")
        for source, counts in stats['source_breakdown'].items():
            total = counts['harmful'] + counts['benign']
            print(f"   • {source}: {total:,} total ({counts['harmful']:,} harmful, {counts['benign']:,} benign)")
        
    except Exception as e:
        print(f"❌ Error during compilation: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
