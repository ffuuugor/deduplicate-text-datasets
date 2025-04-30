#!/usr/bin/env python

import os
import glob
import struct
import numpy as np
import re
import sys
import traceback
import logging
from transformers import GPT2Tokenizer
from datasets import load_dataset
from tqdm import tqdm
import argparse


def setup_logging(verbose=False):
    """
    Set up logging configuration.
    
    Args:
        verbose: Whether to enable verbose (DEBUG) logging
    """
    # Configure the root logger
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Return the logger
    return logging.getLogger('slimpajama_processor')


def discover_slimpajama_structure(dataset_dir, logger):
    """
    Discover the structure of SlimPajama dataset by analyzing the directory contents.
    
    Args:
        dataset_dir: Path to the SlimPajama dataset directory
        logger: Logger instance
        
    Returns:
        dict: Information about chunks and files per chunk
        
    Raises:
        ValueError: If dataset directory doesn't match the expected structure
    """
    logger.info(f"Scanning dataset directory: {dataset_dir}")
    
    # Check if dataset_dir exists
    if not os.path.exists(dataset_dir):
        raise ValueError(f"Dataset directory does not exist: {dataset_dir}")
    
    # Enforce strict expected structure: {dataset_dir}/train/chunk{i}/example_train_{j}.jsonl.zst
    train_dir = os.path.join(dataset_dir, "train")
    if not os.path.exists(train_dir):
        raise ValueError(
            f"Train directory not found at {train_dir}. The SlimPajama dataset must follow "
            f"the structure: {dataset_dir}/train/chunk{{i}}/example_train_{{j}}.jsonl.zst"
        )
    
    # Find all chunk directories
    chunk_pattern = os.path.join(train_dir, "chunk*")
    chunk_dirs = sorted(glob.glob(chunk_pattern))
    
    if not chunk_dirs:
        raise ValueError(
            f"No chunk directories found in {train_dir}. The SlimPajama dataset must contain "
            f"directories named 'chunk{{i}}' inside the train directory."
        )
    
    # Process each chunk directory
    chunks = {}
    invalid_chunks = []
    
    for chunk_dir in chunk_dirs:
        # Extract chunk number from directory name
        chunk_num_match = re.search(r'chunk(\d+)', os.path.basename(chunk_dir))
        if not chunk_num_match:
            invalid_chunks.append((chunk_dir, "Directory name does not match pattern 'chunk{i}'"))
            continue
        chunk_num = int(chunk_num_match.group(1))
        
        # Find all jsonl.zst files in this chunk
        file_pattern = os.path.join(chunk_dir, "example_train_*.jsonl.zst")
        files = sorted(glob.glob(file_pattern))
        
        if not files:
            invalid_chunks.append((chunk_dir, "No example_train_*.jsonl.zst files found"))
            continue
        
        # Verify all files follow the expected naming pattern but don't skip the entire chunk
        valid_files = []
        invalid_files = []
        for file_path in files:
            filename = os.path.basename(file_path)
            if re.match(r'example_train_\d+\.jsonl\.zst', filename):
                valid_files.append(file_path)
            else:
                invalid_files.append(filename)
        
        if invalid_files:
            logger.warning(f"Chunk {chunk_num}: Found {len(invalid_files)} files with invalid names. These will be ignored.")
            logger.warning(f"First few invalid files: {invalid_files[:5]}")
        
        if not valid_files:
            invalid_chunks.append((chunk_dir, "No valid example_train_*.jsonl.zst files found"))
            continue
            
        # Use only valid files
        files = valid_files
        
        # Store chunk information
        chunks[chunk_num] = {
            'path': chunk_dir,
            'files': files,
            'file_count': len(files)
        }
    
    # Report invalid chunks if any
    if invalid_chunks:
        error_msg = "Dataset structure validation failed:\n"
        for chunk_dir, reason in invalid_chunks:
            error_msg += f"  - {chunk_dir}: {reason}\n"
        error_msg += "\nExpected structure: {dataset_dir}/train/chunk{i}/example_train_{j}.jsonl.zst"
        raise ValueError(error_msg)
    
    if not chunks:
        raise ValueError(
            "No valid chunks found in the dataset directory. "
            "The SlimPajama dataset must follow the structure: "
            "{dataset_dir}/train/chunk{i}/example_train_{j}.jsonl.zst"
        )
    
    # Print summary of found chunks
    logger.info(f"Found {len(chunks)} chunks in SlimPajama dataset:")
    total_files = 0
    for chunk_num, info in sorted(chunks.items()):
        logger.info(f"  Chunk {chunk_num}: {info['file_count']} files")
        total_files += info['file_count']
    logger.info(f"Total files: {total_files}")
    
    return chunks


def process_slimpajama_part(args, part, chunks, logger):
    """
    Process a specific part of the SlimPajama dataset.
    
    Args:
        args: Command line arguments
        part: Part number to process (0-indexed)
        chunks: Dictionary of chunk information
        logger: Logger instance
    """
    try:
        logger.info(f"\n===== Processing part {part+1}/{args.parts} =====")
        
        # Get chunks and files for this part
        from scripts.slimpajama_utils import get_chunks_for_part
        
        chunk_assignments = get_chunks_for_part(
            part=part,
            n_parts=args.parts,
            chunks=chunks,
            logger=logger
        )
        
        if not chunk_assignments:
            logger.warning(f"No chunks assigned for part {part+1}")
            return False
        
        # Collect all files from assigned chunks
        file_paths = []
        chunk_info = []
        
        for chunk_num, file_indices in chunk_assignments.items():
            chunk_files = chunks[chunk_num]['files']
            idx_low, idx_high = file_indices
            part_files = chunk_files[idx_low:idx_high]
            
            if part_files:
                file_paths.extend(part_files)
                chunk_info.append((chunk_num, idx_low, idx_high, len(chunk_files)))
        
        if not file_paths:
            logger.warning(f"No files allocated for part {part+1}")
            return False
        
        # Log chunk assignment info
        for chunk_num, idx_low, idx_high, total in chunk_info:
            logger.info(f"Processing files {idx_low} to {idx_high-1} (of {total}) from chunk {chunk_num}")
        
        logger.info(f"Total: {len(file_paths)} files across {len(chunk_info)} chunks")
        
        # Check if output file already exists
        split = "train"
        save_dir = args.save_dir
        dataset_name = f"{args.name}_{part}_of_{args.parts}"
        output_path = os.path.join(save_dir, dataset_name + "." + split)
        size_path = os.path.join(save_dir, dataset_name + "." + split + ".size")
        
        if os.path.exists(output_path) and os.path.exists(size_path) and not args.force:
            logger.info(f"Output files already exist and --force not specified, skipping:")
            logger.info(f"  {output_path}")
            logger.info(f"  {size_path}")
            return True
        
        # Load dataset from files
        logger.info(f"Loading dataset from {len(file_paths)} files...")
        ds = load_dataset(
            "json", 
            data_files=file_paths,
            split="train"
        )
        logger.info(f"Loaded dataset with {len(ds)} examples")
        
        # Setup separators
        pre_sep = b"\xff\xff"  # Default separator bytes
        post_sep = b""
        
        # Initialize UID for separators
        UID = 0
        
        def sep():
            nonlocal UID
            UID += 1
            return pre_sep + struct.pack("<I", UID) + post_sep
        
        def tok(x):
            try:
                if args.tokenize:
                    # Handle case where text field might have a different name
                    text_field = "text"
                    if text_field not in x:
                        available_fields = list(x.keys())
                        if not available_fields:
                            return {"bytes": b""}
                        text_field = available_fields[0]
                        
                    out = tokenizer.encode(x[text_field])
                    out = np.array(out, dtype=np.uint16).view(np.uint8).tobytes()
                    return {"bytes": out}
                else:
                    # Just return the text as bytes
                    return {"bytes": x["text"].encode("utf-8") if "text" in x else b""}
            except Exception as e:
                logger.error(f"Error tokenizing example: {e}")
                return {"bytes": b""}
        
        # Create output directory if needed
        os.makedirs(save_dir, exist_ok=True)
        
        # Open output file
        logger.info(f"Writing to {output_path}")
        fout = open(output_path, "wb")
        
        # Process in slices to avoid memory issues
        slice_size = args.slice_size
        
        sizes = [0]
        total_processed = 0
        
        # Process the dataset in chunks
        for i in tqdm(range(0, len(ds), slice_size)):
            # Select a slice of the dataset
            ds_slice = ds.select(range(i, min(i + slice_size, len(ds))))
            
            # Apply tokenization or text conversion
            ds_slice = ds_slice.map(
                tok,
                num_proc=args.num_proc,
                remove_columns=ds.column_names,
                desc=f"Processing examples {i}-{min(i + slice_size, len(ds))}"
            )
            
            # Write data to output file
            for text in ds_slice["bytes"]:
                next_line = sep() + text
                fout.write(next_line)
                sizes.append(sizes[-1] + len(next_line))
                total_processed += 1
        
        fout.close()
        
        # Write size metadata
        with open(size_path, "wb") as f:
            f.write(np.array(sizes, dtype=np.uint64).tobytes())
            
        logger.info(f"Successfully processed part {part+1}")
        logger.info(f"Processed {total_processed} examples")
        logger.info(f"Output saved to {output_path}")
        logger.info(f"Size metadata saved to {size_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error processing part {part+1}:")
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process SlimPajama dataset into parts for deduplication.")
    parser.add_argument("--dataset-dir", type=str, required=True,
                        help="Directory containing the SlimPajama dataset")
    parser.add_argument("--save-dir", type=str, required=True,
                        help="Directory to save processed output")
    parser.add_argument("--name", type=str, required=True,
                        help="Base name for output files")
    parser.add_argument("--parts", type=int, required=True,
                        help="Number of parts to split the dataset into")
    parser.add_argument("--process-part", type=int,
                        help="Specific part to process (0-indexed)")
    parser.add_argument("--tokenize", action="store_true",
                        help="Whether to tokenize text with GPT-2 tokenizer")
    parser.add_argument("--num-proc", type=int, default=64,
                        help="Number of processes for parallel mapping")
    parser.add_argument("--slice-size", type=int, default=1_000_000,
                        help="Number of examples to process at once")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output files")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose (debug) logging")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(verbose=args.verbose)
    
    # Setup tokenizer if needed
    if args.tokenize:
        logger.info("Loading GPT-2 tokenizer...")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Discover dataset structure
    logger.info(f"Analyzing SlimPajama dataset structure in {args.dataset_dir}...")
    chunks = discover_slimpajama_structure(args.dataset_dir, logger)
    
    # Log chunk and part information
    logger.info(f"Processing {args.parts} parts with {len(chunks)} chunks")
    
    # Process specific part or all parts
    if args.process_part is not None:
        if args.process_part < 0 or args.process_part >= args.parts:
            raise ValueError(f"Part number must be between 0 and {args.parts-1}")
        success = process_slimpajama_part(args, args.process_part, chunks, logger)
        if not success:
            logger.error(f"Failed to process part {args.process_part}")
            sys.exit(1)
    else:
        # Process all parts
        failed_parts = []
        for part in range(args.parts):
            success = process_slimpajama_part(args, part, chunks, logger)
            if not success:
                failed_parts.append(part)
                logger.warning(f"Failed to process part {part}, continuing with next part")
        
        if failed_parts:
            logger.warning(f"All parts processed. {len(failed_parts)} parts failed: {failed_parts}")
            sys.exit(1)
        else:
            logger.info("All parts processed successfully")