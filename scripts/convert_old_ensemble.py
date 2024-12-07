import csv
import sys

def parse_pos_record(record):
    """
    Parse a part of speech record and return a list of (word, pos) tuples.
    
    Args:
        record (str): Input record in the format 'word|POS,word|POS,...'
    
    Returns:
        list: A list of tuples containing (word, part of speech)
    """
    # Remove quotes from the record if present
    record = record.strip('"')
    
    # Split the record into individual word|POS components
    components = record.split(',')
    
    # Parse each component into word and part of speech
    parsed_words = []
    for component in components:
        # Split each component by the '|' separator
        parts = component.split('|')
        
        # Some components might not have a POS, so we'll handle that
        if len(parts) == 2:
            word, pos = parts
            parsed_words.append((word, pos))
        elif len(parts) == 1:
            # If no POS is specified, use 'Unknown'
            parsed_words.append((parts[0], 'Unknown'))
    
    return parsed_words

def format_output(parsed_words):
    """
    Format the parsed words into a readable output.
    
    Args:
        parsed_words (list): List of (word, pos) tuples
    
    Returns:
        str: Formatted string with each word on a new line
    """
    # Find the maximum length of words for aligned formatting
    max_word_length = max(len(word) for word, _ in parsed_words)
    
    # Create formatted output
    formatted_lines = [
        f"{word},{pos}" 
        for word, pos in parsed_words
    ]
    
    return '\n'.join(formatted_lines)

def process_csv(input_file, output_file=None):
    """
    Process a CSV file with POS records.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str, optional): Path to the output file
    """
    # Prepare output destination
    output = []
    
    # Read the CSV file
    with open(input_file, 'r', newline='', encoding='utf-8') as csvfile:
        # Skip the first column header row
        next(csvfile, None)
        
        # Read each record
        for row in csvfile:
            # Remove newline and strip whitespace
            record = row.strip()
            
            try:
                # Parse the record
                parsed_words = parse_pos_record(record)
                
                # Format the parsed words
                formatted_record = format_output(parsed_words)
                
                # Add formatted record to output
                output.append(f"{formatted_record}\n")
            
            except Exception as e:
                output.append(f"Error processing record: {e}\n")
    
    # Output handling
    if output_file:
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(output)
    else:
        # Print to console
        print(''.join(output))

def main():
    # Check command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python pos_parser.py <input_csv_file> [output_file]")
        sys.exit(1)
    
    # Input file is required
    input_file = sys.argv[1]
    
    # Output file is optional
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Process the CSV
    process_csv(input_file, output_file)

if __name__ == "__main__":
    main()