# Python Deduplication Tools

This directory contains Python code for text dataset deduplication tools.

## Project Structure

- `dedupe/`: Core deduplication functionality
- `near_duplicates/`: Tools for detecting near-duplicate content
- `tests/`: Unit tests for the Python code

## Running Tests

To run all tests:

```bash
# From the project root directory:
python -m unittest discover py_src/tests

# Or from the py_src directory:
python -m unittest discover tests
```

To run a specific test file:

```bash
# For example, to run the slimpajama_utils tests:
python -m py_src.tests.test_slimpajama_utils
```

## Development

When modifying or extending the code, please:

1. Add unit tests for new functionality
2. Ensure tests pass before submitting changes
3. Follow the existing code style conventions