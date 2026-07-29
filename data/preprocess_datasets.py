"""
Dataset Preprocessor and Inspector
Extracts zip archives for MMFakeBench images and inspects sample records
for both MiRAGeNews and MMFakeBench.
"""

import os
import json
import zipfile
from datasets import load_from_disk, load_dataset

# ==========================================
# Dynamic Path Resolution
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(SCRIPT_DIR, "raw")

MMFAKEBENCH_DIR = os.path.join(RAW_DIR, "mmfakebench_raw")
MIRAGENEWS_DIR = os.path.join(RAW_DIR, "miragenews")


def extract_mmfakebench_images():
    """
    Extracts MMFakeBench image zip files if they haven't been extracted yet.
    """
    print("\n--- Unzipping MMFakeBench Image Archives ---")

    zip_files = [
        ("MMFakeBench_test.zip", "images_test"),
        ("MMFakeBench_val.zip", "images_val")
    ]

    for zip_filename, target_subfolder in zip_files:
        zip_path = os.path.join(MMFAKEBENCH_DIR, zip_filename)
        extract_path = os.path.join(MMFAKEBENCH_DIR, target_subfolder)

        if not os.path.exists(zip_path):
            print(f"[Skip] {zip_filename} not found in {MMFAKEBENCH_DIR}")
            continue

        if os.path.exists(extract_path) and len(os.listdir(extract_path)) > 0:
            print(f"[Info] {target_subfolder} already extracted at: {extract_path}")
            continue

        print(f"[Extracting] {zip_filename} -> {extract_path}...")
        os.makedirs(extract_path, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"[Success] Extracted {zip_filename} successfully!")


def inspect_mmfakebench():
    """
    Inspects sample annotations from MMFakeBench JSON metadata.
    """
    print("\n--- Inspecting MMFakeBench Annotations ---")
    json_path = os.path.join(MMFAKEBENCH_DIR, "MMFakeBench_val.json")

    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"[Success] Loaded MMFakeBench Validation Set ({len(data)} items)")
        if len(data) > 0:
            print("\nSample Item Structure:")
            sample = data[0] if isinstance(data, list) else list(data.values())[0]
            print(json.dumps(sample, indent=2)[:500] + "\n...")
    else:
        print(f"[Notice] {json_path} not found. Check if Hugging Face CLI login succeeded.")


def inspect_miragenews():
    """
    Loads and inspects the MiRAGeNews Arrow dataset.
    """
    print("\n--- Inspecting MiRAGeNews Dataset ---")
    try:
        dataset = load_dataset("anson-huang/mirage-news", cache_dir=MIRAGENEWS_DIR)
        print("[Success] MiRAGeNews Loaded:")
        print(dataset)
        sample = dataset['train'][0]
        print("\nSample Keys:", list(sample.keys()))
        print(f"Sample Headline: {sample.get('caption', sample.get('text', 'N/A'))}")
    except Exception as e:
        print(f"[Error] Failed to load MiRAGeNews: {e}")


if __name__ == "__main__":
    print(f"Target Raw Folder: {RAW_DIR}")

    # 1. Unzip MMFakeBench Images
    extract_mmfakebench_images()

    # 2. Inspect Annotations
    inspect_mmfakebench()
    inspect_miragenews()

    print("\nPre-processing & Verification Complete!")
