#!/usr/bin/env python

import argparse
import os
import logging
import sys
from datasets import load_dataset
from tqdm import tqdm

def setup_logging(verbose=False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger('slimpajama_downloader')

def download_slimpajama(args, logger):
    """
    Download the SlimPajama dataset from Hugging Face.
    
    Args:
        args: Command line arguments
        logger: Logger instance
    """
    logger.info(f"Downloading SlimPajama dataset")
    if args.download_dir:
        logger.info(f"Using custom download directory: {args.download_dir}")
        os.makedirs(args.download_dir, exist_ok=True)
    else:
        logger.info("Using default HuggingFace cache directory")
    
    # Define chunk ranges
    chunks = args.chunks if args.chunks else list(range(1, 11))
    logger.info(f"Will download chunks: {chunks}")
    
    for chunk in chunks:
        try:
            logger.info(f"==== Downloading chunk {chunk} ====")
            
            # Build file patterns
            if args.start_idx is not None and args.end_idx is not None:
                # Download specific file range
                files = [f"train/chunk{chunk}/example_train_{i}.jsonl.zst" 
                        for i in range(args.start_idx, args.end_idx + 1)]
                logger.info(f"Downloading files {args.start_idx} to {args.end_idx} from chunk {chunk}")
            else:
                # Download all files in the chunk
                files = [f"train/chunk{chunk}/example_train_*.jsonl.zst"]
                logger.info(f"Downloading all files from chunk {chunk}")
            
            # Download the data
            _ = load_dataset(
                "cerebras/SlimPajama-627B",
                data_files=files,
                split="train",
                cache_dir=args.download_dir,
                num_proc=args.num_proc
            )
            
            logger.info(f"Successfully downloaded chunk {chunk}")
            
        except Exception as e:
            logger.error(f"Error downloading chunk {chunk}: {e}")
            if args.stop_on_error:
                logger.error("Stopping due to --stop-on-error flag")
                sys.exit(1)
    
    logger.info("SlimPajama download complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SlimPajama dataset from Hugging Face.")
    parser.add_argument("--download-dir", type=str, default=None,
                        help="Directory to download the dataset to (uses HF default if not specified)")
    parser.add_argument("--chunks", type=int, nargs="+", default=None,
                        help="Specific chunks to download (1-10, downloads all if not specified)")
    parser.add_argument("--start-idx", type=int, default=None,
                        help="Starting file index within each chunk")
    parser.add_argument("--end-idx", type=int, default=None,
                        help="Ending file index within each chunk")
    parser.add_argument("--num-proc", type=int, default=2,
                        help="Number of processes to use for downloading")
    parser.add_argument("--stop-on-error", action="store_true",
                        help="Stop if an error occurs during download")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose (debug) logging")
    
    args = parser.parse_args()
    
    # Validate arguments
    if (args.start_idx is None) != (args.end_idx is None):
        parser.error("Both --start-idx and --end-idx must be provided together")
    
    # Setup logging
    logger = setup_logging(verbose=args.verbose)
    
    # Download the dataset
    download_slimpajama(args, logger)