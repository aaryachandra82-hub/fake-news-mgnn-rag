import os
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from fake_news_mgnn_rag.data_loaders.multimodal_dataset import get_multimodal_dataloader
from fake_news_mgnn_rag.models.baselines.image_baseline import ImageBaselineClassifier
from fake_news_mgnn_rag.models.baselines.text_baseline import TextBaselineClassifier

def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline models")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--model", type=str, choices=["image", "text"], default="image", help="Which baseline to train")
    return parser.parse_args()

def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize Dataset
    print("Initializing DataLoader...")
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    val_json = os.path.join(SCRIPT_DIR, "data/raw/mmfakebench_raw/MMFakeBench_val.json")
    val_images = os.path.join(SCRIPT_DIR, "data/raw/mmfakebench_raw/images_val")
    mirage_dir = os.path.join(SCRIPT_DIR, "data/raw/miragenews")

    dataloader = get_multimodal_dataloader(
        mmfakebench_json=val_json if os.path.exists(val_json) else None,
        mmfakebench_images=val_images if os.path.exists(val_images) else None,
        miragenews_dir=mirage_dir if os.path.exists(mirage_dir) else None,
        dataset_source="combined",
        batch_size=args.batch_size,
        shuffle=True
    )
    
    if len(dataloader.dataset) == 0:
        print("Warning: Dataset is empty. Make sure datasets are downloaded to data/raw/.")
        return

    print(f"Dataset size: {len(dataloader.dataset)} samples across {len(dataloader)} batches.")

    # Initialize Model
    if args.model == "image":
        model = ImageBaselineClassifier().to(device)
    else:
        model = TextBaselineClassifier().to(device)

    # Loss and Optimizer
    # Since output is a single logit, we use BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)

    print(f"\nStarting training for {args.epochs} epochs with {args.model} baseline...")
    
    # Training Loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        correct_preds = 0
        total_samples = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in progress_bar:
            # Prepare Labels
            # The labels are currently torch.long of shape [batch_size].
            # BCEWithLogitsLoss expects torch.float of shape [batch_size, 1].
            labels = batch["label"].to(device, dtype=torch.float).unsqueeze(1)
            
            # Forward Pass
            if args.model == "image":
                pixel_values = batch["image"].to(device)
                logits = model(pixel_values)
            else:
                texts = batch["text"]
                # Texts don't need to be moved to device here, TextEncoder handles it internally via tokenizer
                # Note: Assuming TextEncoder internal logic handles device placement
                logits = model(texts)
                
            # Compute Loss
            loss = criterion(logits, labels)
            
            # Backward and Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track Metrics
            total_loss += loss.item()
            
            # Predictions (logits > 0 corresponds to prob > 0.5)
            preds = (logits > 0).float()
            correct_preds += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct_preds/total_samples:.4f}"})
            
            # Break early after 5 batches for quick testing script
            if total_samples >= args.batch_size * 5:
                print("Stopping early for quick test. Remove this break in full training script.")
                break
                
        epoch_loss = total_loss / (total_samples / args.batch_size)
        epoch_acc = correct_preds / total_samples
        print(f"Epoch {epoch+1} Summary: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}")

if __name__ == "__main__":
    main()
