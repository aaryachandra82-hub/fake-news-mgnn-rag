# Agent Guidelines for Fake News MGNN RAG

When working on this repository, please follow these guidelines:

## 1. Path Management
- **Never use hardcoded absolute paths** (e.g., `C:/mldata/...`).
- Always use dynamic paths relative to the repository root by utilizing `os.path.dirname(os.path.abspath(__file__))` to construct `SCRIPT_DIR` or `REPO_ROOT`.
- Expose important paths as CLI arguments via `argparse` with sensible dynamic defaults.

## 2. Multi-Processing
- Provide a `--num_workers` argument in scripts handling `DataLoader`s.
- Always default `--num_workers` to `0` to ensure safe, cross-platform execution (specifically for Windows compatibility).

## 3. Training & Evaluation
- Ensure `tqdm` progress bars update live within the batch loops by correctly indenting `set_postfix`.
- Aggregate test evaluation metrics properly. Do not extend global metric lists with empty arrays. Always aggregate predictions and labels returned from evaluation functions to compute cross-dataset / cross-split metrics.
- Always implement a safe default for tracking the best model metric (e.g., `best_val_f1 = -1.0` instead of `0.0`) to guarantee that at least one checkpoint is saved.
- Ensure proper use of `torch.no_grad()` and `.eval()` during validation and testing phases.

## 4. General Style
- Keep `argparse` menus clean by avoiding unused or dead arguments.
- Maintain clear and concise documentation for model pipelines.
