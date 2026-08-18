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

from fake_news_mgnn_rag.data_loaders.multimodal_dataset import (
    get_multimodal_dataloader
)
from fake_news_mgnn_rag.models.baselines.text_baseline import (
    TextBaselineClassifier
)
from fake_news_mgnn_rag.models.baselines.image_baseline import (
    ImageBaselineClassifier
)


# ============================================================
# PATHS
# ============================================================

# 🔧 FIX 1:
# Do NOT hardcode C:/mldata/... paths.
# These defaults are relative to the repository.

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_DATA_DIR = os.path.join(
    SCRIPT_DIR,
    "data"
)

DEFAULT_MIRAGENEWS_DIR = os.path.join(
    DEFAULT_DATA_DIR,
    "miragenews"
)

DEFAULT_MMFAKEBENCH_VAL_JSON = os.path.join(
    DEFAULT_DATA_DIR,
    "raw",
    "mmfakebench_raw",
    "MMFakeBench_val.json"
)

DEFAULT_MMFAKEBENCH_VAL_IMAGES = os.path.join(
    DEFAULT_DATA_DIR,
    "raw",
    "mmfakebench_raw",
    "images_val"
)

CHECKPOINT_DIR = os.path.join(
    SCRIPT_DIR,
    "checkpoints"
)


# ============================================================
# TEST SPLITS
# ============================================================

TEST_SPLITS = [
    ("test1_nyt_mj", "NYT + Midjourney"),
    ("test2_bbc_dalle", "BBC + DALL-E 3"),
    ("test3_cnn_dalle", "CNN + DALL-E 3"),
    ("test4_bbc_sdxl", "BBC + SDXL"),
    ("test5_cnn_sdxl", "CNN + SDXL"),
]


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train baseline classifiers"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Max token length for text"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="text",
        choices=["text", "image"],
        help="Which baseline to train"
    )

    # 🔧 FIX 1:
    # Dataset paths are now configurable.
    # This keeps the GitHub project portable while allowing
    # each user to provide their own local dataset location.

    parser.add_argument(
        "--miragenews_dir",
        type=str,
        default=DEFAULT_MIRAGENEWS_DIR,
        help="Path to the MiRAGeNews dataset/cache directory"
    )

    parser.add_argument(
        "--mmfakebench_val_json",
        type=str,
        default=DEFAULT_MMFAKEBENCH_VAL_JSON,
        help="Path to MMFakeBench validation JSON"
    )

    parser.add_argument(
        "--mmfakebench_val_images",
        type=str,
        default=DEFAULT_MMFAKEBENCH_VAL_IMAGES,
        help="Path to MMFakeBench validation images"
    )

    # 🔧 FIX 3:
    # num_workers is now configurable instead of hardcoded to 0.

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of DataLoader workers"
    )

    return parser.parse_args()


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    args,
    split_name=""
):
    model.eval()

    total_loss = 0.0

    all_labels = []
    all_preds = []

    for batch in tqdm(
        loader,
        desc=f"Evaluating {split_name}",
        leave=False
    ):
        labels = batch["label"].to(
            device,
            dtype=torch.float
        ).unsqueeze(1)

        if args.model == "text":
            logits = model(
                batch["text"],
                max_length=args.max_length
            )
        else:
            logits = model(
                batch["image"].to(device)
            )

        loss = criterion(
            logits,
            labels
        )

        total_loss += loss.item()

        preds = (
            (logits > 0)
            .long()
            .squeeze(1)
            .cpu()
            .tolist()
        )

        all_preds.extend(preds)
        all_labels.extend(
            batch["label"].tolist()
        )

    avg_loss = total_loss / len(loader)

    acc = accuracy_score(
        all_labels,
        all_preds
    )

    prec = precision_score(
        all_labels,
        all_preds,
        zero_division=0
    )

    rec = recall_score(
        all_labels,
        all_preds,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_preds,
        zero_division=0
    )

    cm = confusion_matrix(
        all_labels,
        all_preds
    )

    # 🔧 FIX 2:
    # Previously evaluate() returned only metrics.
    # The final test loop therefore had no actual predictions
    # to aggregate.
    #
    # We now return the labels and predictions as well.

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,

        # 🔧 FIX 2
        "labels": all_labels,
        "predictions": all_preds
    }


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

def main():

    args = parse_args()

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )

    print(
        f"MiRAGeNews directory: "
        f"{args.miragenews_dir}"
    )

    print(
        f"DataLoader workers: "
        f"{args.num_workers}"
    )


    # ========================================================
    # TRAINING DATA
    # ========================================================

    print(
        "\nLoading training data..."
    )

    train_loader = get_multimodal_dataloader(
        miragenews_dir=args.miragenews_dir,
        miragenews_split="train",
        dataset_source="miragenews",
        batch_size=args.batch_size,
        shuffle=True,

        # 🔧 FIX 3
        num_workers=args.num_workers
    )

    print(
        f"Train samples : "
        f"{len(train_loader.dataset)}"
    )


    # ========================================================
    # VALIDATION DATA
    # ========================================================

    print(
        "Loading validation data..."
    )

    val_loader = get_multimodal_dataloader(
        miragenews_dir=args.miragenews_dir,
        miragenews_split="validation",
        dataset_source="miragenews",
        batch_size=args.batch_size,
        shuffle=False,

        # 🔧 FIX 3
        num_workers=args.num_workers
    )

    print(
        f"Val samples   : "
        f"{len(val_loader.dataset)}"
    )


    # ========================================================
    # MODEL
    # ========================================================

    print(
        f"\nInitializing "
        f"{args.model} baseline model..."
    )

    if args.model == "text":

        model = TextBaselineClassifier(
            freeze_encoder=False
        ).to(device)

    else:

        model = ImageBaselineClassifier().to(device)


    criterion = nn.BCEWithLogitsLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr
    )


    # ========================================================
    # TRAINING
    # ========================================================

    best_val_f1 = 0.0

    best_checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"best_{args.model}_baseline.pt"
    )


    for epoch in range(args.epochs):

        model.train()

        total_loss = 0.0

        all_labels = []
        all_preds = []

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs}"
        )


        for batch in progress:

            labels = batch["label"].to(
                device,
                dtype=torch.float
            ).unsqueeze(1)


            if args.model == "text":

                logits = model(
                    batch["text"],
                    max_length=args.max_length
                )

            else:

                logits = model(
                    batch["image"].to(device)
                )


            loss = criterion(
                logits,
                labels
            )


            optimizer.zero_grad()

            loss.backward()

            optimizer.step()


            total_loss += loss.item()


            preds = (
                (logits > 0)
                .long()
                .squeeze(1)
                .cpu()
                .tolist()
            )

            all_preds.extend(preds)

            all_labels.extend(
                batch["label"].tolist()
            )


            train_acc = accuracy_score(all_labels, all_preds)
        progress.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{train_acc:.4f}"
        })


        # ====================================================
        # EPOCH SUMMARY
        # ====================================================

        train_f1 = f1_score(
            all_labels,
            all_preds,
            zero_division=0
        )

        print(
            f"\nEpoch {epoch + 1} Train — "
            f"Loss: "
            f"{total_loss / len(train_loader):.4f} | "
            f"F1: {train_f1:.4f}"
        )


        # ====================================================
        # VALIDATION
        # ====================================================

        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            args,
            "Validation"
        )


        print(
            f"Epoch {epoch + 1} Val   — "
            f"Loss: {val_metrics['loss']:.4f} | "
            f"Acc: {val_metrics['accuracy']:.4f} | "
            f"F1: {val_metrics['f1']:.4f}"
        )


        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        if val_metrics["f1"] > best_val_f1:

            best_val_f1 = val_metrics["f1"]

            torch.save(
                model.state_dict(),
                best_checkpoint_path
            )

            print(
                f"✅ New best model saved "
                f"(Val F1: {best_val_f1:.4f})"
            )


    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    print(
        f"\nLoading best model from "
        f"{best_checkpoint_path}..."
    )

    model.load_state_dict(
        torch.load(
            best_checkpoint_path,
            map_location=device
        )
    )


    # ========================================================
    # FINAL TEST EVALUATION
    # ========================================================

    print(
        "\n" + "=" * 65
    )

    print(
        f"{'TEST SET':<25}"
        f"{'Acc':>8}"
        f"{'Prec':>8}"
        f"{'Rec':>8}"
        f"{'F1':>8}"
    )

    print(
        "=" * 65
    )


    # 🔧 FIX 2:
    # These lists will now actually receive the labels
    # and predictions from every test dataset.

    all_test_labels = []
    all_test_preds = []


    for split_key, split_name in TEST_SPLITS:

        test_loader = get_multimodal_dataloader(
            miragenews_dir=args.miragenews_dir,
            miragenews_split=split_key,
            dataset_source="miragenews",
            batch_size=args.batch_size,
            shuffle=False,

            # 🔧 FIX 3
            num_workers=args.num_workers
        )


        metrics = evaluate(
            model,
            test_loader,
            criterion,
            device,
            args,
            split_name
        )


        print(
            f"{split_name:<25}"
            f"{metrics['accuracy']:>8.4f}"
            f"{metrics['precision']:>8.4f}"
            f"{metrics['recall']:>8.4f}"
            f"{metrics['f1']:>8.4f}"
        )


        # 🔧 FIX 2:
        # BEFORE:
        #
        # all_test_labels.extend([])
        # all_test_preds.extend([])
        #
        # That added nothing.
        #
        # NOW:
        # Add the real labels and predictions.

        all_test_labels.extend(
            metrics["labels"]
        )

        all_test_preds.extend(
            metrics["predictions"]
        )


    # ========================================================
    # OVERALL TEST METRICS
    # ========================================================

    overall_accuracy = accuracy_score(
        all_test_labels,
        all_test_preds
    )

    overall_precision = precision_score(
        all_test_labels,
        all_test_preds,
        zero_division=0
    )

    overall_recall = recall_score(
        all_test_labels,
        all_test_preds,
        zero_division=0
    )

    overall_f1 = f1_score(
        all_test_labels,
        all_test_preds,
        zero_division=0
    )


    print(
        "=" * 65
    )

    print(
        f"{'OVERALL TEST':<25}"
        f"{overall_accuracy:>8.4f}"
        f"{overall_precision:>8.4f}"
        f"{overall_recall:>8.4f}"
        f"{overall_f1:>8.4f}"
    )

    print(
        "=" * 65
    )


    print(
        f"\nBest Validation F1: "
        f"{best_val_f1:.4f}"
    )

    print(
        f"Checkpoint saved  : "
        f"{best_checkpoint_path}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()