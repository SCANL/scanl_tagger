import csv
import argparse
import sys

def process_row(row, row_id):
    split_identifier = row['SPLIT_IDENTIFIER'].lower().split()
    grammar_pattern = row['GRAMMAR_PATTERN'].split()
    num_words = len(split_identifier)
    context = row['CONTEXT']
    
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

        if i == 0:
            new_row['NORMALIZED_POSITION'] = 0
        elif i == num_words - 1:
            new_row['NORMALIZED_POSITION'] = 2
        else:
            new_row['NORMALIZED_POSITION'] = 1

        new_rows.append(new_row)
    return new_rows

def main(input_file):
    with open(input_file, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames + ['IDENTIFIER_ID', 'POSITION', 'NORMALIZED_POSITION', 'CONTEXT_NUMBER', 'WORD', 'CORRECT_TAG']
        rows = []
        for idx, row in enumerate(reader):
            new_rows = process_row(row, row_id=idx)
            rows.extend(new_rows)
        

    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process CSV file and split columns.')
    parser.add_argument('input_file', type=str, help='Path to the input CSV file')
    args = parser.parse_args()

    main(args.input_file)