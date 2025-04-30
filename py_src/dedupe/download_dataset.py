#!/usr/bin/env python

import argparse
import os
from datasets import load_dataset, get_dataset_split_names

def download_dataset(args):
    """
    Download a dataset from Hugging Face or another source.
    
    Args:
        args: Command line arguments containing dataset specifications
    """
    print(f"Downloading dataset: {args.dataset_name}")
    
    # Create download directory if specified
    if args.download_dir:
        os.makedirs(args.download_dir, exist_ok=True)
    
    _ = load_dataset(
        args.dataset_name,
        cache_dir=args.download_dir,
        num_proc=args.num_proc
    )
    
    download_location = args.download_dir if args.download_dir else "default HuggingFace cache directory"
    print(f"Dataset downloaded to {download_location}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a dataset from Hugging Face.")
    parser.add_argument("--dataset-name", type=str, required=True, 
                        help="Name of the dataset on Hugging Face")
    parser.add_argument("--download-dir", type=str, default=None,
                        help="Directory to download the dataset to (optional)")
    parser.add_argument("--num-proc", type=int, default=2,
                        help="Number of processes to use for downloading")
    
    args = parser.parse_args()
    download_dataset(args)