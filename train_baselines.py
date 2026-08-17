"""
Training script for text and image baseline classifiers.
Trains on MiRAGeNews train split, evaluates on validation,
and tests across all 5 official MiRAGeNews test sets.
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from fake_news_mgnn_rag.data_loaders.multimodal_dataset import get_multimodal_dataloader
from fake_news_mgnn_rag.models.baselines.text_baseline import TextBaselineClassifier
from fake_news_mgnn_rag.models.baselines.image_baseline import ImageBaselineClassifier

# ============================================================
# PATHS — local data locations
# ============================================================
MMFAKEBENCH_VAL_JSON   = "C:/mldata/mmfakebench_raw/MMFakeBench_val.json"
MMFAKEBENCH_VAL_IMAGES = "C:/mldata/mmfakebench_raw/images_val"
MIRAGENEWS_DIR         = "C:/mldata/miragenews"
CHECKPOINT_DIR         = "checkpoints"

# All 5 official MiRAGeNews test splits
TEST_SPLITS = [
    ("test1_nyt_mj",    "NYT + Midjourney"),
    ("test2_bbc_dalle", "BBC + DALL-E 3"),
    ("test3_cnn_dalle", "CNN + DALL-E 3"),
    ("test4_bbc_sdxl",  "BBC + SDXL"),
    ("test5_cnn_sdxl",  "CNN + SDXL"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline classifiers")
    parser.add_argument("--epochs",     type=int,   default=3,      help="Number of training epochs")
    parser.add_argument("--batch_size", type=int,   default=8,      help="Batch size")
    parser.add_argument("--lr",         type=float, default=1e-4,   help="Learning rate")
    parser.add_argument("--max_length", type=int,   default=512,    help="Max token length for text")
    parser.add_argument("--model",      type=str,   default="text",
                        choices=["text", "image"],  help="Which baseline to train")
    return parser.parse_args()


# ============================================================
# Evaluation function
# ============================================================
@torch.no_grad()
def evaluate(model, loader, criterion, device, args, split_name=""):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []

    for batch in tqdm(loader, desc=f"Evaluating {split_name}", leave=False):
        labels = batch["label"].to(device, dtype=torch.float).unsqueeze(1)

        if args.model == "text":
            logits = model(batch["text"], max_length=args.max_length)
        else:
            logits = model(batch["image"].to(device))

        loss = criterion(logits, labels)
        total_loss += loss.item()

        preds = (logits > 0).long().squeeze(1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(batch["label"].tolist())

    avg_loss = total_loss / len(loader)
    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec  = recall_score(all_labels, all_preds, zero_division=0)
    f1   = f1_score(all_labels, all_preds, zero_division=0)
    cm   = confusion_matrix(all_labels, all_preds)

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm
    }


# ============================================================
# Main training loop
# ============================================================
def main():
    args = parse_args()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Training DataLoader (MiRAGeNews train split) ──────────
    print("\nLoading training data...")
    train_loader = get_multimodal_dataloader(
        miragenews_dir=MIRAGENEWS_DIR,
        miragenews_split="train",
        dataset_source="miragenews",
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0         # 0 is safer on Windows
    )
    print(f"Train samples : {len(train_loader.dataset)}")

    # ── Validation DataLoader ──────────────────────────────────
    print("Loading validation data...")
    val_loader = get_multimodal_dataloader(
        miragenews_dir=MIRAGENEWS_DIR,
        miragenews_split="validation",
        dataset_source="miragenews",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    print(f"Val samples   : {len(val_loader.dataset)}")

    # ── Model ──────────────────────────────────────────────────
    print(f"\nInitializing {args.model} baseline model...")
    if args.model == "text":
        model = TextBaselineClassifier(freeze_encoder=False).to(device)
    else:
        model = ImageBaselineClassifier().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # ── Training ───────────────────────────────────────────────
    best_val_f1 = 0.0
    best_checkpoint_path = os.path.join(
        CHECKPOINT_DIR, f"best_{args.model}_baseline.pt"
    )

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        all_labels = []
        all_preds  = []

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{args.epochs}"
        )

        for batch in progress:
            labels = batch["label"].to(device, dtype=torch.float).unsqueeze(1)

            if args.model == "text":
                logits = model(batch["text"], max_length=args.max_length)
            else:
                logits = model(batch["image"].to(device))

            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = (logits > 0).long().squeeze(1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(batch["label"].tolist())

            progress.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc":  f"{accuracy_score(all_labels, all_preds):.4f}"
            })

        # Epoch summary
        train_f1 = f1_score(all_labels, all_preds, zero_division=0)
        print(f"\nEpoch {epoch+1} Train — "
              f"Loss: {total_loss/len(train_loader):.4f} | "
              f"F1: {train_f1:.4f}")

        # ── Validation ─────────────────────────────────────────
        val_metrics = evaluate(
            model, val_loader, criterion, device, args, "Validation"
        )
        print(f"Epoch {epoch+1} Val   — "
              f"Loss: {val_metrics['loss']:.4f} | "
              f"Acc: {val_metrics['accuracy']:.4f} | "
              f"F1: {val_metrics['f1']:.4f}")

        # Save best model based on validation F1
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"✅ New best model saved (Val F1: {best_val_f1:.4f})")

    # ── Final evaluation on all 5 test sets ───────────────────
    print(f"\nLoading best model from {best_checkpoint_path}...")
    model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))

    print("\n" + "="*65)
    print(f"{'TEST SET':<25} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    print("="*65)

    all_test_labels = []
    all_test_preds  = []

    for split_key, split_name in TEST_SPLITS:
        test_loader = get_multimodal_dataloader(
            miragenews_dir=MIRAGENEWS_DIR,
            miragenews_split=split_key,
            dataset_source="miragenews",
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0
        )

        metrics = evaluate(
            model, test_loader, criterion, device, args, split_name
        )

        print(f"{split_name:<25} "
              f"{metrics['accuracy']:>8.4f} "
              f"{metrics['precision']:>8.4f} "
              f"{metrics['recall']:>8.4f} "
              f"{metrics['f1']:>8.4f}")

        all_test_labels.extend([])
        all_test_preds.extend([])

    print("="*65)
    print(f"\nBest Validation F1: {best_val_f1:.4f}")
    print(f"Checkpoint saved  : {best_checkpoint_path}")


if __name__ == "__main__":
    main()