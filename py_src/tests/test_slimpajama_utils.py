#!/usr/bin/env python

import unittest
from py_src.dedupe.slimpajama_utils import get_chunks_for_part


class TestSlimPajamaUtils(unittest.TestCase):
    
    def setUp(self):
        # Create a sample chunks dictionary for testing
        self.chunks = {
            1: {'files': ['file1.jsonl.zst', 'file2.jsonl.zst', 'file3.jsonl.zst']},
            2: {'files': ['file4.jsonl.zst', 'file5.jsonl.zst', 'file6.jsonl.zst', 'file7.jsonl.zst']},
            3: {'files': ['file8.jsonl.zst', 'file9.jsonl.zst']}
        }
    
    def test_invalid_inputs(self):
        # Test invalid part number
        self.assertEqual(get_chunks_for_part(-1, 5, self.chunks), {})
        self.assertEqual(get_chunks_for_part(5, 5, self.chunks), {})
        
        # Test empty chunks
        self.assertEqual(get_chunks_for_part(0, 5, {}), {})
    
    def test_more_parts_than_chunks(self):
        """Test when there are more parts than chunks (multiple parts per chunk)"""
        # 9 parts, 3 chunks = 3 parts per chunk
        
        # Cycling through chunks with 3 parts per chunk
        # Part 0: Chunk 1, segment 0/3
        self.assertEqual(get_chunks_for_part(0, 9, self.chunks), {1: (0, 1)})  # 1st file
        
        # Part 1: Chunk 2, segment 0/3
        self.assertEqual(get_chunks_for_part(1, 9, self.chunks), {2: (0, 1)})  # 1st file
        
        # Part 2: Chunk 3, segment 0/3
        self.assertEqual(get_chunks_for_part(2, 9, self.chunks), {3: (0, 0)})  # Empty or 1st file
        
        # Part 3: Chunk 1, segment 1/3
        self.assertEqual(get_chunks_for_part(3, 9, self.chunks), {1: (1, 2)})  # 2nd file
        
        # Part 4: Chunk 2, segment 1/3
        self.assertEqual(get_chunks_for_part(4, 9, self.chunks), {2: (1, 2)})  # 2nd file
        
        # Part 5: Chunk 3, segment 1/3
        self.assertEqual(get_chunks_for_part(5, 9, self.chunks), {3: (0, 1)})  # 1st file
        
        # Part 6: Chunk 1, segment 2/3
        self.assertEqual(get_chunks_for_part(6, 9, self.chunks), {1: (2, 3)})  # 3rd file
        
        # Part 7: Chunk 2, segment 2/3
        self.assertEqual(get_chunks_for_part(7, 9, self.chunks), {2: (2, 4)})  # 3rd-4th file
        
        # Part 8: Chunk 3, segment 2/3
        self.assertEqual(get_chunks_for_part(8, 9, self.chunks), {3: (1, 2)})  # 2nd file
    
    def test_uneven_parts_per_chunk(self):
        """Test when parts don't divide evenly among chunks"""
        # 10 parts, 3 chunks = 4 parts per chunk (rounded up)
        
        # Cycling through chunks with 4 parts per chunk
        # Part 0: Chunk 1, segment 0/4
        self.assertEqual(get_chunks_for_part(0, 10, self.chunks), {1: (0, 0)})
        
        # Part 1: Chunk 2, segment 0/4
        self.assertEqual(get_chunks_for_part(1, 10, self.chunks), {2: (0, 1)})
        
        # Part 2: Chunk 3, segment 0/4
        self.assertEqual(get_chunks_for_part(2, 10, self.chunks), {3: (0, 0)})
        
        # Part 3: Chunk 1, segment 1/4
        self.assertEqual(get_chunks_for_part(3, 10, self.chunks), {1: (0, 1)})
        
        # Part 4: Chunk 2, segment 1/4
        self.assertEqual(get_chunks_for_part(4, 10, self.chunks), {2: (1, 2)})
        
        # Part 5: Chunk 3, segment 1/4
        self.assertEqual(get_chunks_for_part(5, 10, self.chunks), {3: (0, 1)})
        
        # Part 6: Chunk 1, segment 2/4
        self.assertEqual(get_chunks_for_part(6, 10, self.chunks), {1: (1, 2)})
        
        # Part 7: Chunk 2, segment 2/4
        self.assertEqual(get_chunks_for_part(7, 10, self.chunks), {2: (2, 3)})
        
        # Part 8: Chunk 3, segment 2/4
        self.assertEqual(get_chunks_for_part(8, 10, self.chunks), {3: (1, 1)})
        
        # Part 9: Chunk 1, segment 3/4
        self.assertEqual(get_chunks_for_part(9, 10, self.chunks), {1: (2, 3)})
    
    def test_fewer_parts_than_chunks(self):
        """Test when there are fewer parts than chunks (multiple chunks per part)"""
        # 2 parts, 3 chunks
        
        # First part gets 2 chunks (ceiling division)
        part0 = get_chunks_for_part(0, 2, self.chunks)
        self.assertEqual(len(part0), 2)
        self.assertEqual(part0[1], (0, 3))  # All files from chunk 1
        self.assertEqual(part0[2], (0, 4))  # All files from chunk 2
        
        # Second part gets 1 chunk
        part1 = get_chunks_for_part(1, 2, self.chunks)
        self.assertEqual(len(part1), 1)
        self.assertEqual(part1[3], (0, 2))  # All files from chunk 3
    
    def test_exactly_one_chunk_per_part(self):
        """Test when number of parts equals number of chunks"""
        # 3 parts, 3 chunks
        
        # Each part gets exactly one chunk
        self.assertEqual(get_chunks_for_part(0, 3, self.chunks), {1: (0, 3)})  # All files from chunk 1
        self.assertEqual(get_chunks_for_part(1, 3, self.chunks), {2: (0, 4)})  # All files from chunk 2
        self.assertEqual(get_chunks_for_part(2, 3, self.chunks), {3: (0, 2)})  # All files from chunk 3


if __name__ == '__main__':
    unittest.main()