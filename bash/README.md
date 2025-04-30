# Bash Scripts for Text Deduplication

This directory contains bash scripts for text dataset deduplication.

## Available Scripts

- `all_count_occurrences.sh`: Count occurrences of patterns across datasets
- `all_self_similar.sh`: Find similar content in datasets
- `all_suffix_arrays.sh`: Generate suffix arrays for efficient text search
- `deduplicate_single_file.sh`: Deduplicate a single text file
- `run_pipeline.sh`: Run the full deduplication pipeline

## Usage

Most scripts expect to be run from the project root directory. For example:

```bash
# Running from project root
bash/run_pipeline.sh path/to/dataset

# Or with full permissions
chmod +x bash/*.sh
./bash/run_pipeline.sh path/to/dataset
```

See individual script comments for specific usage instructions.