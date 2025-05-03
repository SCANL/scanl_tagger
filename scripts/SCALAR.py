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
        return 'Invalid', []

    identifier = str(identifier).strip()
    target_word = str(target_word).strip()

    if not identifier or not target_word:
        return 'Invalid', []

    try:
        # Encode the identifier and context
        encoded_identifier = "_".join(identifier.split()).lower()
        context_str = get_context_from_number(ctx)

        # Construct the HTTP GET request URL
        url = f'http://127.0.0.1:5000/{encoded_identifier}/{context_str}'

        # Send the GET request
        response = requests.get(url)

        if response.status_code == 200:
            # Parse the JSON response from the server
            tagged_data = response.json()  # Assuming listener now returns JSON

            # Find the PoS tag for the target word
            tag_for_word = 'NOT_FOUND'
            pos_sequence = [
                f"{entry['word']}:{entry['pos_tag']}" for entry in tagged_data
            ]
            for entry in tagged_data:
                if entry['word'].lower() == target_word.lower():
                    tag_for_word = entry['pos_tag']
                    break

            return tag_for_word, pos_sequence
        else:
            print(f"HTTP error for {identifier}: {response.status_code}")
            return 'ERROR', []

    except requests.RequestException as e:
        print(f"Request error for {identifier}: {e}")
        return 'ERROR', []


def process_csv(input_file, output_file=None):
    """
    Process the CSV file and extract tags for each WORD, including the full PoS sequence.
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

        tags = []
        pos_sequences = []
        for _, row in df.iterrows():
            tag, pos_sequence = tag_with_http_request(row['SPLIT_IDENTIFIER'], row['CONTEXT_NUMBER'], row['WORD'])
            tags.append(tag)
            pos_sequences.append(" ".join(pos_sequence))

        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Total processing time: {processing_time:.2f} seconds")

        # Save results to CSV
        output_df = pd.DataFrame({
            'WORD': df['WORD'],
            'TAG': tags,
            'POS_SEQUENCE': pos_sequences
        })
        output_csv = output_file or 'scalar_annotated_output_with_pos.csv'
        output_df.to_csv(output_csv, index=False, header=True)
        print(f"Results saved to {output_csv}")

    except Exception as e:
        print(f"Error processing CSV: {e}")


if __name__ == "__main__":
    # Parse command-line arguments
    input_file = sys.argv[1] if len(sys.argv) > 1 else '../output/X_test.csv'
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    process_csv(input_file, output_file)
