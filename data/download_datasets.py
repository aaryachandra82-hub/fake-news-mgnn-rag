"""
Dataset Downloader for Multimodal Fake News Detection
Handles metadata loading via `datasets` and large image download via `huggingface_hub`.
"""

import os
from datasets import load_dataset
from huggingface_hub import snapshot_download, login

# ==========================================
# Step 1: Authentication (If required for gated datasets)
# ==========================================
# Run `hf auth` in your terminal to authenticate with your Hugging Face account.

# Set target directory
DATASET_DIR = "./raw"
os.makedirs(DATASET_DIR, exist_ok=True)


def download_mmfakebench():
    """
    Downloads MMFakeBench from Hugging Face.
    """
    print("\n--- Downloading MMFakeBench Dataset ---")

    # 1. Load dataset metadata (JSON/Parquet annotations)
    try:
        dataset = load_dataset("liuxuannan/MMFakeBench", cache_dir=os.path.join(DATASET_DIR, "mmfakebench"))
        print(f"[Success] MMFakeBench Metadata Loaded:")
        print(dataset)
    except Exception as e:
        print(f"[Info] Standard load failed or requires full repo snapshot: {e}")

    # 2. Download full repository including image archives
    repo_path = snapshot_download(
        repo_id="liuxuannan/MMFakeBench",
        repo_type="dataset",
        local_dir=os.path.join(DATASET_DIR, "mmfakebench_raw"),
        max_workers=4
    )
    print(f"[Success] Full MMFakeBench files downloaded to: {repo_path}")

def download_miragenews():
    """
    Downloads MiRAGeNews dataset from Hugging Face (`anson-huang/mirage-news`).
    Contains 15,000 real and AI-generated image-caption pairs (Midjourney, DALL-E 3, SDXL).
    """
    print("\n--- Downloading MiRAGeNews Dataset ---")
    dataset = load_dataset("anson-huang/mirage-news", cache_dir=os.path.join(DATASET_DIR, "miragenews"))
    print(f"[Success] Loaded MiRAGeNews Summary:")
    print(dataset)
    return dataset


if __name__ == "__main__":
    print("Starting Dataset Download Workflow...")

    # Run downloads
    download_mmfakebench()
    download_miragenews()

    print("\nAll datasets downloaded successfully! Check the `./raw` directory.")
