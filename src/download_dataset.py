import os
import time
from pathlib import Path
import requests
import zipfile

DATASET_URL = "https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip"
SAVE_DIR = Path("data")
ZIP_PATH = SAVE_DIR / "NLM-MontgomeryCXRSet.zip"
EXTRACT_PATH = SAVE_DIR / "extracted"

def download_file(url, save_path, max_retries=10, timeout=45):
    """
    Downloads a file with auto-retries, chunking, and resume/retry on error.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = save_path.with_suffix(".tmp")
    
    print(f"Downloading dataset from: {url}")
    print(f"Saving to: {save_path}")
    
    for attempt in range(1, max_retries + 1):
        try:
            start_time = time.time()
            # If the temp file exists, we can try to get its size and resume if supported, 
            # but NIH Open-i servers may not reliably support range headers. 
            # We will perform a clean, chunk-by-chunk download from the beginning with a longer timeout.
            response = requests.get(url, stream=True, timeout=timeout)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024 * 1024  # 1 MB
            
            print(f"Total file size: {total_size / (1024*1024):.2f} MB")
            print(f"Attempt {attempt}/{max_retries} started...")
            
            downloaded = 0
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        percent = (downloaded / total_size) * 100 if total_size else 0
                        elapsed = time.time() - start_time
                        speed = downloaded / (1024 * 1024) / elapsed if elapsed > 0 else 0
                        print(f"\rProgress: {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({percent:.1f}%) | Speed: {speed:.2f} MB/s", end="")
            
            print(f"\nDownload attempt {attempt} succeeded!")
            
            # Rename temp file to final zip name
            if save_path.exists():
                os.remove(save_path)
            os.rename(temp_path, save_path)
            return True
            
        except (requests.exceptions.RequestException, Exception) as e:
            print(f"\n[WARNING] Attempt {attempt} failed: {e}")
            if attempt == max_retries:
                print("[ERROR] Max retries reached. Download failed.")
                return False
            print("Waiting 5 seconds before retrying...")
            time.sleep(5)
            
    return False

def extract_zip(zip_path, extract_dir):
    """
    Extracts the downloaded zip file and removes it.
    """
    zip_path = Path(zip_path)
    extract_dir = Path(extract_dir)
    
    # Target folder checking
    target_folder = extract_dir / "MontgomerySet"
    if target_folder.exists():
        print(f"Extracted folder already exists at: {target_folder}. Skipping extraction.")
        return True
        
    print(f"Extracting {zip_path} to {extract_dir}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print("Extraction completed successfully.")
        
        # Remove the zip file to reclaim disk space
        os.remove(zip_path)
        print(f"Cleaned up zip file: {zip_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Extraction failed: {e}")
        return False

def main():
    # 1. Download
    success = download_file(DATASET_URL, ZIP_PATH)
    if not success:
        print("[ERROR] Could not retrieve the NIH dataset. Please check your network connection.")
        return False
        
    # 2. Extract
    success_extract = extract_zip(ZIP_PATH, EXTRACT_PATH)
    if not success_extract:
        print("[ERROR] Failed to extract the dataset.")
        return False
        
    print("[SUCCESS] Real Tuberculosis chest X-ray dataset is downloaded and extracted successfully!")
    return True

if __name__ == "__main__":
    main()
