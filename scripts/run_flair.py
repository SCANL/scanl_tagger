import pandas as pd
from flair.data import Sentence
from flair.models import SequenceTagger
import csv
import logging

logging.getLogger('flair').setLevel(logging.ERROR)

def tag_with_flair(identifier):
    # Handle potential NaN or non-string values
    if pd.isna(identifier):
        return ''
    
    identifier = str(identifier).strip()
    if not identifier:
        return ''
    
    
    # Load the English POS tagger (this will be cached after first use)
    tagger = SequenceTagger.load('pos')
    
    # Create a Flair sentence from the identifier
    sentence = Sentence(identifier)
    
    # Predict POS tags
    tagger.predict(sentence)
    
    # Extract POS tags using the correct attribute
    pos_tags = [token.tag for token in sentence]
    
    # Return the POS tag (since we're processing one word at a time, 
    # we'll usually have just one tag)
    return ' '.join(pos_tags)

def process_csv(input_file):
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        
        # Verify the required column exists
        if 'SPLIT_IDENTIFIER' not in df.columns:
            raise ValueError("CSV file must contain a 'SPLIT_IDENTIFIER' column")
        
        # Add a new column for Flair POS tags
        print("Processing identifiers...")
        df['FLAIR_POS'] = df['SPLIT_IDENTIFIER'].apply(tag_with_flair)
        
        # Save the results to a new CSV file
        output_file = 'output_with_flair_pos.csv'
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        
        # Print example results
        print("\nExample results:")
        print(df[['SPLIT_IDENTIFIER', 'FLAIR_POS']].head())
        
    except Exception as e:
        print(f"Error processing CSV: {str(e)}")
if __name__ == "__main__":
    # Process the CSV file
    input_file = '../output/X_validation.csv'  # Replace with your actual file name
    process_csv(input_file)