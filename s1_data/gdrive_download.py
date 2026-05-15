import gdown
import os

def download_from_drive(file_id: str, filename: str, folder: str = "data"):
    """
    Downloads a file from Google Drive using its file ID.
    Skips download if the file already exists.
    """
    os.makedirs(folder, exist_ok=True)

    local_path = os.path.join(folder, filename)

    url = f"https://drive.google.com/uc?id={file_id}"

    if not os.path.exists(local_path):
        print(f"Downloading to {local_path}...")
        gdown.download(url, local_path, quiet=False)
    else:
        print(f"File already exists: {local_path}")


