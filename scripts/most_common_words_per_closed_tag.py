import csv
from collections import defaultdict

def count_pos_in_csv(file_path):
    # Dictionary to store the count of each word for each part of speech
    pos_counts = defaultdict(lambda: defaultdict(int))

    # Open the CSV file
    with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        # Check if required columns are present
        if 'SPLIT IDENTIFIER' not in reader.fieldnames or 'GRAMMAR PATTERN' not in reader.fieldnames:
            raise ValueError("CSV file must contain 'SPLIT IDENTIFIER' and 'GRAMMAR PATTERN' columns.")

        # Process each row in the CSV
        for row in reader:
            split_identifier = row['SPLIT IDENTIFIER'].split()
            grammar_pattern = row['GRAMMAR PATTERN'].split()

            # Ensure both lists have the same length
            if len(split_identifier) != len(grammar_pattern):
                print(split_identifier, " ", grammar_pattern)
                raise ValueError("Mismatch in lengths of 'SPLIT IDENTIFIER' and 'GRAMMAR PATTERN' in row.")

            # Count the occurrences
            for word, pos in zip(split_identifier, grammar_pattern):
                pos_counts[word.lower()][pos] += 1

    return pos_counts

def print_pos_counts(pos_counts):
    for word, pos_dict in pos_counts.items():
        for pos, count in pos_dict.items():
            print(f"{word},{pos},{count}")

# Example usage:
file_path = 'tagger_data_513.csv'
pos_counts = count_pos_in_csv(file_path)
print_pos_counts(pos_counts)