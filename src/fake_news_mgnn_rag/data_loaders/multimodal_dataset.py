"""
Custom PyTorch Dataset for loading MMFakeBench multimodal news pairs.
Handles text captions, image loading, and label encoding.
"""

import os
import json

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class MMFakeBenchDataset(Dataset):
    """
    Custom PyTorch Dataset for loading MMFakeBench multimodal news pairs.
    Handles text captions, image loading, and label encoding.
    """

    def __init__(self, json_path, images_dir, transform=None):
        """
        Args:
            json_path (str): Path to the annotation JSON file
                (e.g., MMFakeBench_val.json).
            images_dir (str): Root directory containing unzipped images
                (e.g., images_val/).
            transform (callable, optional): torchvision transforms for
                image preprocessing.
        """
        self.images_dir = images_dir
        self.transform = transform or self._default_transform()

        if not os.path.exists(json_path):
            raise FileNotFoundError(
                f"Annotation file not found at: {json_path}"
            )

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Standardize dictionary vs. list JSON annotations
        if isinstance(data, dict):
            self.samples = list(data.values())
        else:
            self.samples = data

    def _default_transform(self):
        """
        Default image preprocessing matching standard
        Vision Transformers (ViT / CLIP).
        """
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def _resolve_image_path(self, item):
        """
        Helper method to locate the image across nested folder structures.
        """
        rel_path = (
            item.get("image_path")
            or item.get("image")
            or item.get("img")
        )

        if not rel_path:
            return None

        # Check relative to root images_dir
        full_path = os.path.join(self.images_dir, rel_path)

        if os.path.exists(full_path):
            return full_path

        # Search recursively within images_dir if path format varies
        filename = os.path.basename(rel_path)

        for root, _, files in os.walk(self.images_dir):
            if filename in files:
                return os.path.join(root, filename)

        return None

    def __getitem__(self, idx):
        item = self.samples[idx]

        # Extract text/headline
        caption = (
            item.get("caption")
            or item.get("headline")
            or item.get("text")
            or ""
        )

        # Parse label (1 for Fake, 0 for Real)
        label_raw = item.get("label", item.get("annotation", 0))

        if isinstance(label_raw, str):
            label = 1 if "fake" in label_raw.lower() else 0
        else:
            label = int(label_raw)

        # Load image
        img_path = self._resolve_image_path(item)

        if img_path and os.path.exists(img_path):
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception:
                image = Image.new(
                    "RGB",
                    (224, 224),
                    color=0,
                )
        else:
            # Fallback blank image if file is missing
            image = Image.new(
                "RGB",
                (224, 224),
                color=0,
            )

        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = image

        return {
            "id": str(item.get("id", idx)),
            "text": caption,
            "image": image_tensor,
            "label": torch.tensor(label, dtype=torch.long),
        }


def get_multimodal_dataloader(
    json_path,
    images_dir,
    batch_size=16,
    shuffle=True,
    num_workers=2,
):
    """
    Factory function to create a PyTorch DataLoader.
    """
    dataset = MMFakeBenchDataset(
        json_path=json_path,
        images_dir=images_dir,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    print("Testing MMFakeBench DataLoader...")

    # Path resolution relative to repository root
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.abspath(
        os.path.join(SCRIPT_DIR, "../../../")
    )

    val_json = os.path.join(
        REPO_ROOT,
        "data/raw/mmfakebench_raw/MMFakeBench_val.json",
    )

    val_images = os.path.join(
        REPO_ROOT,
        "data/raw/mmfakebench_raw/images_val",
    )

    if os.path.exists(val_json) and os.path.exists(val_images):
        loader = get_multimodal_dataloader(
            val_json,
            val_images,
            batch_size=4,
            shuffle=False,
        )

        batch = next(iter(loader))

        print("\n[Success] Batch loaded successfully!")
        print(f"Batch keys: {list(batch.keys())}")
        print(f"Text batch shape (count): {len(batch['text'])}")
        print(f"Image tensor batch shape: {batch['image'].shape}")
        print(f"Labels batch tensor: {batch['label']}")

    else:
        print(
            "[Notice] Missing validation dataset files. "
            "Verify paths at:\n"
            f"{val_json}\n"
            f"{val_images}"
        )
