import os
from pathlib import Path
from huggingface_hub import hf_hub_download
import requests, nltk

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_CHUNK_SIZE = 32768  # 32 KB

def create_directory_if_not_exists(directory_path):
    """
    Create a directory if it does not already exist.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def download_files():
    
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('universal_tagset')
    nltk.download('punkt_tab')
    nltk.download("words")
    
    files = [
        # "target_vecs.txt",
        # "token_vecs.txt"
    ]
    
    # Define the directory name and create the full path
    directory_name = "code2vec"
    directory_path = os.path.join(SCRIPT_DIR, directory_name)
    
    # Create the directory if it does not exist
    create_directory_if_not_exists(directory_path)
    
    for file_name in files:
        # Specify the destination path for the downloaded file
        destination = os.path.join(directory_path, file_name)
        
        # Check if the file already exists, and skip download if it does
        if not os.path.exists(destination):
            try:
                # Download from Hugging Face
                downloaded_path = hf_hub_download(
                    repo_id="sourceslicer/token_vecs",
                    filename=file_name,
                    repo_type="dataset",  # Add this if files are stored as a dataset
                    local_dir=directory_path,
                    local_dir_use_symlinks=False
                )
                print(f"Resource '{file_name}' downloaded and saved to '{downloaded_path}'.")
            except Exception as e:
                print(f"Error downloading '{file_name}': {e}")
        else:
            print(f"File '{destination}' already exists. Skipping download.")