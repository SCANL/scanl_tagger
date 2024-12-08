import csv
import argparse
import sys
import os

# Mapping from language to file extension
LANGUAGE_EXTENSIONS = {
    'C++': '.cpp',
    'C': '.c',
    'Java': '.java',
    'C#': '.cs'
}

def extract_file_extension(file_path, language):
    # Get file extension if present
    ext = os.path.splitext(file_path)[1]
    file_name = os.path.basename(file_path)
    # If no valid extension, use LANGUAGE column to infer the file extension
    if ext == '.xml_identifiers':
        return file_name.replace(".xml_identifiers", language)
    return file_name

def extract_system_name(file_path):
    # Extract system name based on .git suffix
    parts = file_path.split('/')
    for part in parts:
        if part.endswith('.git'):
            return part
    return 'unknown'

def process_row(row, row_id):
    split_identifier = row['SPLIT_IDENTIFIER'].lower().split()
    grammar_pattern = row['GRAMMAR_PATTERN'].split()
    num_words = len(split_identifier)
    context = row['CONTEXT']
    
    # Extract file extension and system name
    file_name = row['FILE']
    language = row['LANGUAGE']
    file_extension = extract_file_extension(file_name, language)
    system_name = extract_system_name(file_name)

    # Map context to a number based on the rules
    context_number = 0
    if context == "ATTRIBUTE":
        context_number = 1
    elif context == "CLASS":
        context_number = 2
    elif context == "DECLARATION":
        context_number = 3
    elif context == "FUNCTION":
        context_number = 4
    elif context == "PARAMETER":
        context_number = 5

    new_rows = []
    for i, (word, correct_tag) in enumerate(zip(split_identifier, grammar_pattern)):
        new_row = row.copy()
        new_row['IDENTIFIER_ID'] = row_id
        new_row['POSITION'] = i
        new_row['WORD'] = word
        new_row['CORRECT_TAG'] = correct_tag
        new_row['CONTEXT_NUMBER'] = context_number
        new_row['FILE_EXTENSION'] = file_extension
        new_row['SYSTEM_NAME'] = system_name

        if i == 0:
            new_row['NORMALIZED_POSITION'] = 0
        elif i == num_words - 1:
            new_row['NORMALIZED_POSITION'] = 2
        else:
            new_row['NORMALIZED_POSITION'] = 1

        new_rows.append(new_row)
    return new_rows

def main(input_file, output_file):
    with open(input_file, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames + ['IDENTIFIER_ID', 'POSITION', 'NORMALIZED_POSITION', 'CONTEXT_NUMBER', 'WORD', 'CORRECT_TAG', 'FILE_EXTENSION', 'SYSTEM_NAME']
        rows = []
        for idx, row in enumerate(reader):
            new_rows = process_row(row, row_id=idx)
            rows.extend(new_rows)

    with open(output_file, 'w', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process CSV file and split columns.')
    parser.add_argument('input_file', type=str, help='Path to the input CSV file')
    parser.add_argument('output_file', type=str, help='Path to the output CSV file')
    args = parser.parse_args()

    main(args.input_file, args.output_file)