#!/usr/bin/env python

import math


def get_chunks_for_part(part, n_parts, chunks, logger=None):
    """
    Determine which chunks and files to use for a specific part.
    
    This function supports having both more parts than chunks and fewer parts than chunks:
    - If n_parts > n_chunks: Multiple parts will be assigned to each chunk
    - If n_parts <= n_chunks: Multiple chunks will be assigned to each part
    
    Args:
        part: Part number to process (0-indexed)
        n_parts: Total number of parts
        chunks: Dictionary of chunk information {chunk_num: {'files': [...], ...}}
        logger: Optional logger for debug information
        
    Returns:
        dict: {chunk_num: (start_idx, end_idx), ...} - Mapping of chunk numbers to file index ranges to process
    """
    if part < 0 or part >= n_parts:
        if logger:
            logger.error(f"Part {part} is out of range (0-{n_parts-1})")
        return {}
    
    # Get chunk numbers sorted
    chunk_nums = sorted(chunks.keys())
    n_chunks = len(chunk_nums)
    
    if n_chunks == 0:
        if logger:
            logger.error("No chunks available")
        return {}
    
    # Case 1: More parts than chunks
    if n_parts > n_chunks:
        # Ensure all parts get assigned by cycling through chunks
        chunk_index = part % n_chunks
        chunk_num = chunk_nums[chunk_index]
        chunk_files = chunks[chunk_num]['files']
        chunk_file_count = len(chunk_files)
        
        # Calculate how many parts should process this chunk
        parts_for_this_chunk = math.ceil(n_parts / n_chunks)
        
        # Calculate which segment of the chunk this part should process
        segment = (part // n_chunks) % parts_for_this_chunk
        
        # Calculate file range
        chunk_idx_low = segment * chunk_file_count // parts_for_this_chunk
        chunk_idx_high = (segment + 1) * chunk_file_count // parts_for_this_chunk
        
        return {chunk_num: (chunk_idx_low, chunk_idx_high)}
    
    # Case 2: Exactly the same number of parts as chunks
    elif n_parts == n_chunks:
        # Each part gets exactly one chunk
        chunk_num = chunk_nums[part]
        chunk_files = chunks[chunk_num]['files']
        chunk_file_count = len(chunk_files)
        
        return {chunk_num: (0, chunk_file_count)}
    
    # Case 3: Fewer parts than chunks (multiple chunks per part)
    else:
        # Calculate chunks per part (use ceiling division for distribution)
        chunks_per_part = math.ceil(n_chunks / n_parts)
        
        # Calculate which chunks belong to this part
        start_chunk_idx = part * chunks_per_part
        end_chunk_idx = min((part + 1) * chunks_per_part, n_chunks)
        
        if start_chunk_idx >= n_chunks:
            if logger:
                logger.warning(f"Part {part+1} has no chunks assigned")
            return {}
        
        # Assign chunks to this part
        chunk_assignments = {}
        
        for i in range(start_chunk_idx, end_chunk_idx):
            chunk_num = chunk_nums[i]
            chunk_files = chunks[chunk_num]['files']
            chunk_file_count = len(chunk_files)
            
            # Process all files in this chunk
            chunk_assignments[chunk_num] = (0, chunk_file_count)
        
        return chunk_assignments