import os
import csv
import json
import numpy as np
from collections import Counter
from spiral import ronin
import pandas as pd

def calculate_word_frequencies(words):
    """
    Calculate normalized and log-transformed word frequencies from a series of words.
    
    Parameters:
    words (pd.Series): Series containing words
    
    Returns:
    dict: Dictionary of normalized and log-transformed word frequencies
    """
    # Convert all words to lowercase for consistent counting
    words = words.str.lower()
    # Calculate raw frequencies
    raw_counts = Counter(words)
    total_words = sum(raw_counts.values())
    
    # Normalize counts and apply log transformation
    word_frequencies = {word: np.log1p(count / total_words) for word, count in raw_counts.items()}
    return word_frequencies

def process_csv_files(input_dir, output_file):
    words_list = []

    # Iterate over all .csv files in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".csv"):
            file_path = os.path.join(input_dir, file_name)
            
            with open(file_path, 'r') as csv_file:
                reader = csv.reader(csv_file, delimiter=' ')
                
                for row in reader:
                    # Second item is the identifier
                    if len(row) > 1:
                        identifier = row[1]
                        
                        # Split the identifier into words using ronin
                        words = ronin.split(identifier)
                        
                        # Collect words into a list
                        words_list.extend(words)
    
    # Convert words to a Pandas Series for frequency calculation
    words_series = pd.Series(words_list)
    
    # Calculate normalized and log-transformed frequencies
    word_frequencies = calculate_word_frequencies(words_series)
    
    # Save the word frequencies dictionary to a JSON file
    with open(output_file, 'w') as json_file:
        json.dump(word_frequencies, json_file, indent=4)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process CSV files to calculate normalized word frequencies.")
    parser.add_argument("input_dir", help="Directory containing .csv files")
    parser.add_argument("output_file", help="Output JSON file to save word frequencies")

    args = parser.parse_args()
    process_csv_files(args.input_dir, args.output_file)