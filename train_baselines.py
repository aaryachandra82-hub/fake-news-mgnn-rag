import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# Ensure src/ is on the Python path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.fake_news_mgnn_rag.data_loaders.multimodal_dataset import get_multimodal_dataloader
from src.fake_news_mgnn_rag.models.text_baseline import TextBaselineClassifier

# Dynamic default path relative to repository root
DEFAULT_MIRAGENEWS_DIR = os.path.join(REPO_ROOT, "data", "miragenews")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Baseline Models for Fake News Detection")
    parser.add_argument(
        "--miragenews_dir",
        type=str,
        default=DEFAULT_MIRAGENEWS_DIR,
        help="Path to the MiRAGeNews dataset directory",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training/eval")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of DataLoader worker processes")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=os.path.join(REPO_ROOT, "checkpoints"),
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/deberta-v3-large",
        help="Pretrained HuggingFace transformer model name",
    )
    return parser.parse_args()


def evaluate(model, dataloader, device, criterion):
    """Evaluates the model on a given dataset and returns metrics along with raw labels/preds."""
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item() * input_ids.size(0)
            preds = torch.argmax(logits, dim=-1)

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "labels": all_labels,
        "predictions": all_preds,
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"MiRAGeNews directory: {args.miragenews_dir}")
    print(f"DataLoader workers: {args.num_workers}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(args.checkpoint_dir, "best_text_baseline.pt")

    # Load DataLoaders
    print("\nLoading training data...")
    train_loader = get_multimodal_dataloader(
        miragenews_dir=args.miragenews_dir,
        split="train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    print(f"Train samples : {len(train_loader.dataset)}")

    print("Loading validation data...")
    val_loader = get_multimodal_dataloader(
        miragenews_dir=args.miragenews_dir,
        split="val",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    print(f"Val samples   : {len(val_loader.dataset)}")

    # Initialize Model, Optimizer, and Loss Function
    print("\nInitializing text baseline model...")
    model = TextBaselineClassifier(model_name=args.model_name, num_classes=2)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0  # Safe default to prevent checkpoint load crashes

    # Training Loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        all_labels = []
        all_preds = []

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(logits, dim=-1).cpu().tolist()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

            # FIX 1: Indented inside the loop so tqdm updates live per batch
            train_acc = accuracy_score(all_labels, all_preds)
            progress.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{train_acc:.4f}"
            })

        # Validation Step
        val_metrics = evaluate(model, val_loader, device, criterion)
        print(
            f"\n[Epoch {epoch + 1}] Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['f1']:.4f}"
        )

        # Save Best Model Checkpoint
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f" Saved new best model checkpoint to {best_checkpoint_path}")

    # Final Test Evaluation
    print("\n" + "=" * 50)
    print("Running Final Evaluation on Test Sets...")
    print("=" * 50)

    if os.path.exists(best_checkpoint_path):
        model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
        print("Loaded best checkpoint for testing.")

    test_splits = ["test"]
    all_test_labels = []
    all_test_preds = []

    for split in test_splits:
        test_loader = get_multimodal_dataloader(
            miragenews_dir=args.miragenews_dir,
            split=split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )

        test_metrics = evaluate(model, test_loader, device, criterion)
        print(
            f"Test Split [{split}] -> Loss: {test_metrics['loss']:.4f} | "
            f"Acc: {test_metrics['accuracy']:.4f} | F1: {test_metrics['f1']:.4f}"
        )

        # FIX 2: Correctly aggregating test labels and predictions across splits
        all_test_labels.extend(test_metrics["labels"])
        all_test_preds.extend(test_metrics["predictions"])

    if all_test_labels:
        print("\nOverall Classification Report:")
        print(classification_report(all_test_labels, all_test_preds, digits=4))


if __name__ == "__main__":
    main()