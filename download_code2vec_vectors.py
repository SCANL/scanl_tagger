import os
import requests
from pathlib import Path

DOWNLOAD_CHUNK_SIZE = 32768  # 32 KB

def create_directory_if_not_exists(directory_name):
    """
    Create a directory if it does not already exist.
    """
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

def download_file_from_google_drive(file_id, destination):
    """
    Download a file from Google Drive and save it to the specified destination.
    """
    URL = "https://drive.google.com/uc?export=download&id=" + file_id
    session = requests.Session()

    response = session.get(URL, stream=True)
    token = None

    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value

    total_size = int(response.headers.get("content-length", 0))
    bytes_written = 0

    with open(destination, "wb") as output_file:
        for chunk in response.iter_content(DOWNLOAD_CHUNK_SIZE):
            output_file.write(chunk)
            bytes_written += len(chunk)
            # Calculate and print download progress
            progress = (bytes_written / total_size) * 100
            print(f"Downloading: {progress:.2f}% complete", end="\r")
    
    print("\nDownload complete!")

def download_files():
    file_urls = [
        ("target_vecs.txt", "https://drive.google.com/uc?export=download&id=1L2d84J2_3rJxBZjcp9cBdMZlQJHrsspX&confirm=t&uuid=6d4e113f-c16a-4a7b-8abd-35a9df2b7949&at=AB6BwCA99NAayMkvrcdwE58EPSua:1696902245928"),
        ("token_vecs.txt", "https://drive.google.com/uc?export=download&id=1Um82N1ACiuIi2kz1XB0y2OYSZJiBTAVM&confirm=t&uuid=2a103628-f879-4d95-94c6-5c37ea5c2582&at=AB6BwCB-W6bp6_00wizE5fbmNG-k:1696902427614")
    ]
    for file_name, url in file_urls:
        # Extract the file ID from the URL
        file_id = url.split("&id=")[-1]

        # Define the directory name
        directory_name = "code2vec"

        # Create the directory if it does not exist
        create_directory_if_not_exists(directory_name)

        # Specify the destination path for the downloaded file
        destination = os.path.join(directory_name, file_name)

        # Check if the file already exists, and skip download if it does
        if not os.path.exists(destination):
            # Download the resource from the Google Drive URL
            download_file_from_google_drive(file_id, destination)

            print(f"Resource '{file_name}' downloaded and saved to '{destination}' in '{directory_name}' directory.")
        else:
            print(f"File '{destination}' already exists. Skipping download.")