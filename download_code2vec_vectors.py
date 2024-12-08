import os
import requests
from pathlib import Path

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_CHUNK_SIZE = 32768  # 32 KB

def create_directory_if_not_exists(directory_path):
    """
    Create a directory if it does not already exist.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def get_confirm_token(response):
    """
    Check for a confirmation token in the Google Drive download warning page.
    """
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    """
    Write the content of the response to a file in chunks.
    """
    total_size = int(response.headers.get("content-length", 0))
    bytes_written = 0

    with open(destination, "wb") as output_file:
        for chunk in response.iter_content(DOWNLOAD_CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                output_file.write(chunk)
                bytes_written += len(chunk)
                # Calculate and print download progress
                progress = (bytes_written / total_size) * 100 if total_size > 0 else 0
                print(f"Downloading: {progress:.2f}% complete", end="\r")

    print("\nDownload complete!")

def download_file_from_google_drive(file_id, destination):
    """
    Download a file from Google Drive and save it to the specified destination.
    Handles Google Drive virus scan warnings for large files.
    """
    URL = "https://drive.google.com/uc?export=download"

    with requests.Session() as session:
        # Initial request to get the file with potential warning page
        response = session.get(URL, params={'id': file_id}, stream=True)
        
        # Look for the confirm token to bypass the virus scan warning
        token = get_confirm_token(response)
        if token:
            # Retry download with the confirm token
            response = session.get(URL, params={'id': file_id, 'confirm': token}, stream=True)
        
        # If redirected, get the actual download URL from the page
        if "html" in response.headers.get('Content-Type', ''):
            download_url = response.url.replace("export=download", "export=download&confirm=t")
            response = session.get(download_url, stream=True)

        response.raise_for_status()  # Raise an exception for bad status codes
        save_response_content(response, destination)

def download_files():
    file_urls = [
        ("target_vecs.txt", "1L2d84J2_3rJxBZjcp9cBdMZlQJHrsspX"),
        ("token_vecs.txt", "1Um82N1ACiuIi2kz1XB0y2OYSZJiBTAVM")
    ]
    
    # Define the directory name and create the full path
    directory_name = "code2vec"
    directory_path = os.path.join(SCRIPT_DIR, directory_name)
    
    # Create the directory if it does not exist
    create_directory_if_not_exists(directory_path)
    
    for file_name, file_id in file_urls:
        # Specify the destination path for the downloaded file
        destination = os.path.join(directory_path, file_name)
        
        # Check if the file already exists, and skip download if it does
        if not os.path.exists(destination):
            # Download the resource from the Google Drive URL
            try:
                download_file_from_google_drive(file_id, destination)
                print(f"Resource '{file_name}' downloaded and saved to '{destination}'.")
            except requests.exceptions.RequestException as e:
                print(f"Error downloading '{file_name}': {e}")
        else:
            print(f"File '{destination}' already exists. Skipping download.")