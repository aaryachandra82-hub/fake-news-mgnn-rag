"""
Custom PyTorch Dataset for loading MMFakeBench multimodal news pairs.
Handles text captions, image loading, and label encoding.
"""

import os
import json
from typing import Optional, Dict, Any, List, Union
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms

try:
    from datasets import load_dataset, Dataset as HFDataset
except ImportError:
    load_dataset = None
    HFDataset = None


def get_default_transform() -> transforms.Compose:
    """
    Standard image transformations for Vision Transformer (ViT) and CLIP vision backbones.
    Resizes images to 224x224 and normalizes with ImageNet statistics.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])


class MMFakeBenchDataset(Dataset):
    """
    Dataset loader for MMFakeBench multimodal news pairs.
    Handles cross-modal misalignment and contextual manipulation samples.
    """
    def __init__(
        self,
        json_path: str,
        images_dir: str,
        transform: Optional[Any] = None
    ):
        self.images_dir = images_dir
        self.transform = transform or get_default_transform()

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"MMFakeBench annotation file not found at: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            self.samples = list(data.values())
        else:
            self.samples = data

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_image_path(self, item: Dict[str, Any]) -> Optional[str]:
        rel_path = item.get("image_path") or item.get("image") or item.get("img")
        if not rel_path:
            return None

        full_path = os.path.join(self.images_dir, rel_path)
        if os.path.exists(full_path):
            return full_path

        filename = os.path.basename(rel_path)
        for root, _, files in os.walk(self.images_dir):
            if filename in files:
                return os.path.join(root, filename)

        return None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]

        caption = item.get("caption") or item.get("headline") or item.get("text") or ""

        label_raw = item.get("label", item.get("annotation", 0))
        if isinstance(label_raw, str):
            label = 1 if "fake" in label_raw.lower() else 0
        else:
            label = int(label_raw)

        img_path = self._resolve_image_path(item)
        if img_path and os.path.exists(img_path):
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception:
                image = Image.new("RGB", (224, 224), color=0)
        else:
            image = Image.new("RGB", (224, 224), color=0)

        image_tensor = self.transform(image)

        return {
            "id": f"mmfakebench_{item.get('id', idx)}",
            "text": caption,
            "image": image_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "dataset_origin": "mmfakebench"
        }


class MiRAGeNewsDataset(Dataset):
    """
    Dataset loader for MiRAGeNews deepfake & synthetic news pairs.
    Handles Midjourney/DALL-E images and AI-generated headlines.
    """
    def __init__(
        self,
        cache_dir: str,
        split: str = "validation",
        transform: Optional[Any] = None
    ):
        self.transform = transform or get_default_transform()
        self.split = split
        self.samples = []

        if load_dataset is None:
            raise ImportError("huggingface datasets library is required. Install via `uv add datasets`.")

        try:
            # Load dataset from local cache directory
            hf_data = load_dataset("anson-huang/mirage-news", cache_dir=cache_dir)
            if self.split in hf_data:
                self.samples = hf_data[self.split]
            else:
                available_splits = list(hf_data.keys())
                self.samples = hf_data[available_splits[0]]
                print(f"[Notice] Split '{split}' not found in MiRAGeNews. Fallback to '{available_splits[0]}'.")
        except Exception as e:
            print(f"[Warning] Failed to load MiRAGeNews from cache directory ({cache_dir}): {e}")
            self.samples = []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]

        caption = item.get("caption") or item.get("headline") or item.get("text") or ""

        # Parse AI-generated vs Real label
        label_raw = item.get("label", item.get("is_ai", item.get("annotation", 0)))
        if isinstance(label_raw, str):
            label = 1 if ("fake" in label_raw.lower() or "ai" in label_raw.lower()) else 0
        elif isinstance(label_raw, bool):
            label = 1 if label_raw else 0
        else:
            label = int(label_raw)

        # Process image
        raw_image = item.get("image") or item.get("img")
        if isinstance(raw_image, Image.Image):
            image = raw_image.convert("RGB")
        elif isinstance(raw_image, str) and os.path.exists(raw_image):
            try:
                image = Image.open(raw_image).convert("RGB")
            except Exception:
                image = Image.new("RGB", (224, 224), color=0)
        else:
            image = Image.new("RGB", (224, 224), color=0)

        image_tensor = self.transform(image)

        return {
            "id": f"miragenews_{item.get('id', idx)}",
            "text": caption,
            "image": image_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "dataset_origin": "miragenews"
        }


class UnifiedMultimodalDataset(Dataset):
    """
    Unified PyTorch Dataset combining MMFakeBench and MiRAGeNews samples.
    Ensures standard key output across both cross-modal and deepfake benchmark datasets.
    """
    def __init__(
        self,
        mmfakebench_json: Optional[str] = None,
        mmfakebench_images: Optional[str] = None,
        miragenews_dir: Optional[str] = None,
        miragenews_split: str = "validation",
        dataset_source: str = "combined",
        transform: Optional[Any] = None
    ):
        self.dataset_source = dataset_source.lower()
        self.datasets: List[Dataset] = []

        # Load MMFakeBench if requested
        if self.dataset_source in ["combined", "mmfakebench"] and mmfakebench_json and mmfakebench_images:
            if os.path.exists(mmfakebench_json):
                self.datasets.append(MMFakeBenchDataset(
                    json_path=mmfakebench_json,
                    images_dir=mmfakebench_images,
                    transform=transform
                ))

        # Load MiRAGeNews if requested
        if self.dataset_source in ["combined", "miragenews"] and miragenews_dir:
            if os.path.exists(miragenews_dir):
                self.datasets.append(MiRAGeNewsDataset(
                    cache_dir=miragenews_dir,
                    split=miragenews_split,
                    transform=transform
                ))

        if not self.datasets:
            print("[Warning] UnifiedMultimodalDataset initialized with 0 valid sub-datasets.")
            self.concat_dataset = ConcatDataset([])
        else:
            self.concat_dataset = ConcatDataset(self.datasets)

    def __len__(self) -> int:
        return len(self.concat_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.concat_dataset[idx]


def get_multimodal_dataloader(
    mmfakebench_json: Optional[str] = None,
    mmfakebench_images: Optional[str] = None,
    miragenews_dir: Optional[str] = None,
    miragenews_split: str = "validation",
    dataset_source: str = "combined",
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 2
) -> DataLoader:
    """
    Factory function to instantiate a PyTorch DataLoader for multimodal news.
    """
    dataset = UnifiedMultimodalDataset(
        mmfakebench_json=mmfakebench_json,
        mmfakebench_images=mmfakebench_images,
        miragenews_dir=miragenews_dir,
        miragenews_split=miragenews_split,
        dataset_source=dataset_source
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )


if __name__ == "__main__":
    print("Testing Unified Multimodal Dataset Pipeline...")

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../"))

    val_json = os.path.join(REPO_ROOT, "data/raw/mmfakebench_raw/MMFakeBench_val.json")
    val_images = os.path.join(REPO_ROOT, "data/raw/mmfakebench_raw/images_val")
    mirage_dir = os.path.join(REPO_ROOT, "data/raw/miragenews")

    dataset = UnifiedMultimodalDataset(
        mmfakebench_json=val_json,
        mmfakebench_images=val_images,
        miragenews_dir=mirage_dir,
        dataset_source="combined"
    )

    print(f"\n[Success] Unified Dataset Initialized!")
    print(f"Total samples loaded: {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]
        print("\nSample Output Structure:")
        print(f"  - Sample ID: {sample['id']}")
        print(f"  - Dataset Origin: {sample['dataset_origin']}")
        print(f"  - Text Length: {len(sample['text'])} chars")
        print(f"  - Image Tensor Shape: {sample['image'].shape}")
        print(f"  - Label Tensor: {sample['label']}")
