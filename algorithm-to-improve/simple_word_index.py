#!/usr/bin/env python3
"""
Simple Word Index with timsort and vergesort
Asymptotic Complexity Analysis:
- HashMap operations: O(1) average per word
- Timsort per word list: O(k log k) where k = positions per word
- Vergesort lexicographic ordering: O(n log n) where n = unique words
- Overall: O(m + Σ(k_i log k_i) + n log n) where m = total words
"""

import sys
import time
import os
from collections import defaultdict
import numpy as np
import vergesort_py


def create_word_index(text_file):
    """Create word index using simple hashmap"""
    # Initialize hashmap
    word_positions = defaultdict(list)
    
    with open(text_file, 'r', encoding='utf-8') as f:
        position = 0
        for line_num, line in enumerate(f, 1):
            words = line.strip().split()
            for word_pos, word in enumerate(words):
                # Clean word and convert to lowercase, skip non-encodable chars
                clean_word = ''.join(c.lower() for c in word if c.isalnum() and ord(c) < 128)
                if clean_word:
                    word_positions[clean_word].append(position)
                position += 1
    
    return word_positions


def sort_positions_timsort(word_positions):
    """Sort position lists using Python's timsort (built-in sorted())"""
    for word in word_positions:
        word_positions[word] = sorted(word_positions[word])
    return word_positions


def sort_dictionary_vergesort(word_positions):
    """Sort dictionary lexicographically using vergesort"""

    words = list(word_positions.keys())

    # Since we filter to ASCII-only in create_word_index, this should work
    # Convert to numpy array for vergesort using byte string type
    max_word_len = max(len(word) for word in words) if words else 1
    words_array = np.array(words, dtype=f'S{max_word_len + 1}')  # Byte strings
    
    # Use vergesort to sort the array directly
    vergesort_py.sort(words_array)
    
    # Create ordered dictionary based on sorted array
    sorted_dict = {}
    for word_bytes in words_array:
        word = word_bytes.decode('ascii')  # Use ASCII since we filtered non-ASCII
        sorted_dict[word] = word_positions[word]
    
    return sorted_dict



def sort_dictionary_numpy(word_positions):
    """Sort dictionary lexicographically using numpy.sort for comparison"""
    
    words = list(word_positions.keys())
    
    # Use numpy sort
    sorted_words = np.sort(words)
    
    # Create ordered dictionary
    sorted_dict = {}
    for word in sorted_words:
        sorted_dict[word] = word_positions[word]
    
    return sorted_dict


def save_word_index(word_positions, output_file):
    """Save word index to CSV file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("word,positions\n")
        for word, positions in word_positions.items():
            positions_str = ';'.join(map(str, positions))
            f.write(f"{word},{positions_str}\n")


def process_file(input_file, use_numpy=False):
    """Process single file and return processing time"""
    start_time = time.time()
    
    # Create word index
    word_positions = create_word_index(input_file)
    
    # Sort positions using timsort
    word_positions = sort_positions_timsort(word_positions)
    
    # Sort dictionary lexicographically
    if use_numpy:
        word_positions = sort_dictionary_numpy(word_positions)
        sort_method = "numpy"
    else:
        word_positions = sort_dictionary_vergesort(word_positions)
        sort_method = "vergesort"
    
    # Generate output filename
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = f"output/{base_name}_{sort_method}_index.csv"
    
    # Save results
    save_word_index(word_positions, output_file)
    
    end_time = time.time()
    return end_time - start_time, len(word_positions)


def main():
    """
    Main function with asymptotic complexity analysis:
    - HashMap initialization: O(1) average per word
    - Word processing: O(m) where m = total words in file
    - Position sorting per word: O(k_i log k_i) where k_i = positions for word i
    - Lexicographic sorting: O(n log n) where n = unique words
    - Total complexity: O(m + Σ(k_i log k_i) + n log n)
    """
    
    input_dir = "input"
    output_dir = "output"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get input files
    input_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    
    if not input_files:
        print("No .txt files found in input directory")
        return
    
    # Check if comparison mode is requested
    compare_mode = len(sys.argv) > 1 and sys.argv[1].lower() == "compare"
    
    for input_file in input_files:
        file_path = os.path.join(input_dir, input_file)
        print(f"\nProcessing: {input_file}")
        
        if compare_mode:
            # Run with numpy sort
            numpy_time, word_count = process_file(file_path, use_numpy=True)
            print(f"  Numpy sort time: {numpy_time:.4f}s ({word_count} unique words)")
            
            # Run with vergesort
            verge_time, word_count = process_file(file_path, use_numpy=False)
            print(f"  Vergesort time: {verge_time:.4f}s ({word_count} unique words)")
            
            # Calculate speedup
            speedup = numpy_time / verge_time if verge_time > 0 else float('inf')
            print(f"  Speedup: {speedup:.2f}x")
        else:
            # Run only with vergesort
            process_time, word_count = process_file(file_path, use_numpy=False)
            print(f"  Processing time: {process_time:.4f}s ({word_count} unique words)")


if __name__ == "__main__":
    main()