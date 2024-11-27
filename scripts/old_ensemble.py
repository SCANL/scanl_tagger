import pandas as pd
import requests
import csv
import sys
import urllib.parse

def get_context_from_number(context_number):
    context_map = {
        0: "DEFAULT",
        1: "ATTRIBUTE",
        2: "CLASS",
        3: "DECLARATION",
        4: "FUNCTION",
        5: "PARAMETER"
    }
    return context_map.get(context_number, "DEFAULT")

def tag_with_http_request(identifier, ctx):
    # Handle potential NaN or non-string values
    if pd.isna(identifier):
        return ''
   
    identifier = str(identifier).strip()
    if not identifier:
        return ''
   
    try:
        # URL encode the identifier and context
        encoded_identifier = "_".join(identifier.split())
        
        # Convert context number to string context
        context_str = get_context_from_number(ctx)

        print(encoded_identifier, " in ", context_str)
        if context_str == 'FUNCTION':
            encoded_identifier = encoded_identifier+'()'
        # Construct the GET request URL
        url = f'http://127.0.0.1:5000/int/{encoded_identifier}/{context_str}'
       
        # Send GET request to local server
        response = requests.get(url)
       
        # Check if request was successful
        if response.status_code == 200:
            # Print and return the response text
            result = response.text.strip()
            print(result)
            return result
        else:
            print(f"Error with request for {identifier}: HTTP {response.status_code}")
            return ''
   
    except requests.RequestException as e:
        print(f"Request error for {identifier}: {str(e)}")
        return ''

def process_csv(input_file):
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
       
        # Verify the required columns exist
        if 'SPLIT_IDENTIFIER' not in df.columns or 'CONTEXT_NUMBER' not in df.columns:
            raise ValueError("CSV file must contain 'SPLIT_IDENTIFIER' and 'CONTEXT' columns")
       
        # Process identifiers and collect results
        results = []
        for index, row in df.iterrows():
            result = tag_with_http_request(row['SPLIT_IDENTIFIER'], row['CONTEXT_NUMBER'])
            results.append(result)
       
        # Add results to DataFrame
        df['HTTP_POS'] = results
       
        # Optional: Save results to CSV
        output_file = 'output_with_http_pos.csv'
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
       
    except Exception as e:
        print(f"Error processing CSV: {str(e)}")

if __name__ == "__main__":
    # Allow input file to be specified as command-line argument
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = '../validation_sets/X_validation.csv'  # Default file
   
    process_csv(input_file)