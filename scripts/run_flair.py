import pandas as pd
from flair.data import Sentence
from flair.models import SequenceTagger
import csv
import logging, time

logging.getLogger('flair').setLevel(logging.ERROR)

def convert_tag(flair_tag):
    tag_mapping = {
        'CC': 'CJ', 'CD': 'D', 'DT': 'DT', 'FW': 'N', 'IN': 'P', 'JJ': 'NM', 'JJR': 'NM',
        'JJS': 'NM', 'LS': 'N', 'MD': 'V', 'NN': 'N', 'NNP': 'N', 'NNPS': 'NPL',
        'NNS': 'NPL', 'PRP': 'PR', 'PRPS': 'PR', 'RB': 'VM', 'RBR': 'VM', 'RP': 'VM',
        'SYM': 'N', 'TO': 'P', 'VB': 'V', 'VBD': 'NM', 'VBG': 'NM', 'VBN': 'NM',
        'VBP': 'V', 'VBZ': 'V'
    }
    return tag_mapping.get(flair_tag, 'UNKNOWN')

def tag_with_flair(full_identifier, target_word):
    # Handle potential NaN or non-string values
    if pd.isna(full_identifier) or pd.isna(target_word):
        return '', ''
    
    full_identifier = str(full_identifier.lower()).strip()
    target_word = str(target_word.lower()).strip()
    
    if not full_identifier or not target_word:
        return '', ''
    
    # Load the English POS tagger (this will be cached after first use)
    tagger = SequenceTagger.load('pos')
    
    # Create a Flair sentence from the full identifier
    sentence = Sentence(full_identifier.lower())
    print("Handle: ", full_identifier)
    # Predict POS tags
    tagger.predict(sentence)
    
    # Find the tag for the target word
    for token in sentence:
        if token.text == target_word:
            flair_tag = token.tag
            custom_tag = convert_tag(flair_tag)
            return flair_tag, custom_tag
    
    # Return a default value if the word isn't found (optional)
    return 'NOT_FOUND', 'UNKNOWN'

def process_csv(input_file):
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        
        # Verify the required columns exist
        required_columns = ['SPLIT_IDENTIFIER', 'WORD']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"CSV file must contain a '{col}' column")
        
        start_time = time.time()
        # Annotate each row
        print("Processing identifiers...")
        df[['FLAIR_TAG', 'CUSTOM_TAG']] = df.apply(lambda row: tag_with_flair(row['SPLIT_IDENTIFIER'], row['WORD']), axis=1, result_type='expand')
        
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Total processing time: {processing_time:.2f} seconds")

        # Save the results to a new CSV file
        output_file = 'annotated_output.csv'
        df[['WORD', 'FLAIR_TAG', 'CUSTOM_TAG']].to_csv(output_file, index=False, header=True)
        print(f"Results saved to {output_file}")
        
        # Print example results
        print("\nExample results:")
        print(df[['WORD', 'FLAIR_TAG', 'CUSTOM_TAG']].head())
        
    except Exception as e:
        print(f"Error processing CSV: {str(e)}")

if __name__ == "__main__":
    # Process the CSV file
    input_file = '../validation_sets/X_validation.csv'  # Replace with your actual file name
    process_csv(input_file)