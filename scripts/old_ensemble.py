import pandas as pd
import requests
import csv
import sys
import time

def get_context_from_number(context_number):
    """
    Map context numbers to context strings.
    """
    context_map = {
        0: "DEFAULT",
        1: "ATTRIBUTE",
        2: "CLASS",
        3: "DECLARATION",
        4: "FUNCTION",
        5: "PARAMETER"
    }
    return context_map.get(context_number, "DEFAULT")

def tag_with_http_request(identifier, ctx, target_word):
    """
    Tag the full identifier using an HTTP request and extract the tag for the target word.
    """
    if pd.isna(identifier) or pd.isna(target_word):
        return 'Invalid'

    identifier = str(identifier).strip()
    target_word = str(target_word).strip()

    if not identifier or not target_word:
        return 'Invalid'

    try:
        # Encode the identifier and context
        encoded_identifier = "_".join(identifier.split()).lower()
        context_str = get_context_from_number(ctx)

        # Adjust identifier for FUNCTION context
        if context_str == 'FUNCTION':
            encoded_identifier += '()'

        # Construct the HTTP GET request URL
        url = f'http://127.0.0.1:5000/int/{encoded_identifier}/{context_str}'

        # Send the GET request
        response = requests.get(url)

        if response.status_code == 200:
            # Parse the response text (assumed format: "word|tag,word|tag,...")
            tagged_data = response.text.strip().split(',')

            # Search for the target word's tag
            for entry in tagged_data:
                word, tag = entry.split('|')
                if word.lower() == target_word.lower():
                    return tag

            # If the target word is not found
            print("NOT FOUND: ", tagged_data, target_word)
            return 'NOT_FOUND'
        else:
            print(f"HTTP error for {identifier}: {response.status_code}")
            return 'ERROR'

    except requests.RequestException as e:
        print(f"Request error for {identifier}: {e}")
        return 'ERROR'

def process_csv(input_file, output_file=None):
    """
    Process the CSV file and extract tags for each WORD.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)

        # Ensure required columns exist
        if 'SPLIT_IDENTIFIER' not in df.columns or 'CONTEXT_NUMBER' not in df.columns or 'WORD' not in df.columns:
            raise ValueError("CSV must contain 'SPLIT_IDENTIFIER', 'CONTEXT_NUMBER', and 'WORD' columns")

        # Process each row and collect results
        print("Processing identifiers...")
        start_time = time.time()
        results = df.apply(
            lambda row: tag_with_http_request(row['SPLIT_IDENTIFIER'], row['CONTEXT_NUMBER'], row['WORD']),
            axis=1
        )
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Total processing time: {processing_time:.2f} seconds")

        # Save results to CSV
        output_df = pd.DataFrame({'WORD': df['WORD'], 'TAG': results})
        output_csv = output_file or 'ensemble_annotated_output.csv'
        output_df.to_csv(output_csv, index=False, header=False)
        print(f"Results saved to {output_csv}")

    except Exception as e:
        print(f"Error processing CSV: {e}")

if __name__ == "__main__":
    # Parse command-line arguments
    input_file = sys.argv[1] if len(sys.argv) > 1 else '../output/X_validation.csv'
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    process_csv(input_file, output_file)
