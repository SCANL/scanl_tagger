import os
from collections import defaultdict
from spiral import ronin
import json

# Define the directory containing the CSV files
directory_path = "../../identifier_lists"

# Function to process each CSV file and count word occurrences and context occurrences
def process_csv_files(directory):
    word_count = defaultdict(int)
    context_count = defaultdict(int)
    
    # Loop through each file in the directory
    for filename in os.listdir(directory):
        print("Working on: ", filename)
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            
            # Open and read each file
            with open(file_path, 'r') as file:
                for line in file:
                    # Split the line by spaces
                    columns = line.strip().split()
                    
                    # Extract the variable_name and context columns
                    variable_name = columns[1]
                    context = columns[2]
                    
                    # Apply the ronin.split function to get the words in the variable name
                    split_identifier = ronin.split(variable_name)
                    
                    # Count each word in the split identifier
                    for word in split_identifier:
                        word_count[word.lower()] += 1
                    
                    # Count the context occurrence
                    context_count[context] += 1
    
    return word_count, context_count

import json

def save_word_count(word_count, output_file):
    """
    Saves the word_count dictionary to a JSON file.

    Parameters:
        word_count (dict): The dictionary containing word counts.
        output_file (str): The path to the output JSON file.
    """
    try:
        with open(output_file, 'w') as f:
            json.dump(word_count, f, indent=4)
        print(f"Word counts successfully saved to {output_file}")
    except Exception as e:
        print(f"Failed to save word counts: {e}")

if __name__ == "__main__":
    word_count, context_count = process_csv_files(directory_path)
    save_word_count(word_count, "word_count.json")