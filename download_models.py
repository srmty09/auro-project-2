#!/usr/bin/env python3
"""
Model Download Script for Odia-English Translation System

This script helps users download the pretrained model weights.
Since the model files are large (>1GB), they are hosted separately.
"""

import os
import sys
import requests
from pathlib import Path
import hashlib
from tqdm import tqdm

# Model information
MODELS = {
    "best_translation_model.pt": {
        "url": "https://github.com/YOUR_USERNAME/odia-english-translation/releases/download/v1.0/best_translation_model.pt",
        "size": "1.2GB",
        "description": "Main BERT-based translation model",
        "md5": "06ac61d28cc84ba2618aabf7be43a38b"
    },
    "best_model_directory.tar.gz": {
        "url": "https://github.com/YOUR_USERNAME/odia-english-translation/releases/download/v1.0/best_model_directory.tar.gz", 
        "size": "867MB",
        "description": "PyTorch saved model directory (compressed)",
        "md5": "cc3bf117a391fa844e446940d0f793cb"
    }
}

def calculate_md5(filepath):
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def download_file(url, filepath, expected_md5=None):
    """Download a file with progress bar."""
    print(f"Downloading {filepath}...")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as f, tqdm(
        desc=filepath,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            pbar.update(size)
    
    # Verify MD5 if provided
    if expected_md5:
        print("Verifying file integrity...")
        actual_md5 = calculate_md5(filepath)
        if actual_md5 != expected_md5:
            print(f"❌ MD5 mismatch! Expected: {expected_md5}, Got: {actual_md5}")
            return False
        print("✅ File integrity verified")
    
    return True

def extract_if_needed(filepath):
    """Extract compressed files if needed."""
    if filepath.endswith('.tar.gz'):
        import tarfile
        print(f"Extracting {filepath}...")
        with tarfile.open(filepath, 'r:gz') as tar:
            tar.extractall(path='.')
        print("✅ Extraction complete")
        
        # Optionally remove the compressed file
        response = input("Remove compressed file? (y/N): ")
        if response.lower() == 'y':
            os.remove(filepath)
            print(f"Removed {filepath}")

def main():
    """Main download function."""
    print("🚀 Odia-English Translation Model Downloader")
    print("=" * 50)
    
    # Check available models
    print("Available models:")
    for i, (name, info) in enumerate(MODELS.items(), 1):
        print(f"{i}. {name} ({info['size']}) - {info['description']}")
    
    print("\nOptions:")
    print("1. Download specific model")
    print("2. Download all models")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        # Download specific model
        model_choice = input("Enter model number: ").strip()
        try:
            model_idx = int(model_choice) - 1
            model_name = list(MODELS.keys())[model_idx]
            model_info = MODELS[model_name]
            
            if download_file(model_info["url"], model_name, model_info.get("md5")):
                print(f"✅ Successfully downloaded {model_name}")
                extract_if_needed(model_name)
            else:
                print(f"❌ Failed to download {model_name}")
                
        except (ValueError, IndexError):
            print("❌ Invalid model number")
            
    elif choice == "2":
        # Download all models
        for model_name, model_info in MODELS.items():
            if download_file(model_info["url"], model_name, model_info.get("md5")):
                print(f"✅ Successfully downloaded {model_name}")
                extract_if_needed(model_name)
            else:
                print(f"❌ Failed to download {model_name}")
                
    elif choice == "3":
        print("Goodbye!")
        sys.exit(0)
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    # Check if models already exist
    existing_models = []
    for model_name in MODELS.keys():
        if os.path.exists(model_name):
            existing_models.append(model_name)
    
    if existing_models:
        print("⚠️  Found existing models:")
        for model in existing_models:
            print(f"  - {model}")
        
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled.")
            sys.exit(0)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Download cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
