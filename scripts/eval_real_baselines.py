import sys
import os
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.append(os.path.abspath("src"))
from fake_news_mgnn_rag.models.baselines.text_baseline import TextBaselineClassifier
from fake_news_mgnn_rag.models.baselines.image_baseline import ImageBaselineClassifier
from fake_news_mgnn_rag.data_loaders.multimodal_dataset import get_default_transform

def get_optimal_device():
    """Dynamically selects the best compute backend (CUDA, ROCm/HIP, MPS, or CPU)."""
    if torch.cuda.is_available():
        # Handles both NVIDIA CUDA and AMD ROCm/HIP if PyTorch is compiled for it
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def evaluate_real_sample(image_path: str, claim_text: str):
    device = get_optimal_device()
    print(f"--- Running Inference on Device: {device} ---")

    # 1. Initialize Baselines
    text_model = TextBaselineClassifier().to(device)
    image_model = ImageBaselineClassifier().to(device)

    # Set to evaluation mode to disable dropout
    text_model.eval()
    image_model.eval()

    print(f"\n[Input Claim]: {claim_text}")
    print(f"[Input Image]: {image_path}")

    # 2. Evaluate Text Baseline
    with torch.no_grad():
        text_logits = text_model([claim_text])
        text_fake_prob = torch.sigmoid(text_logits).item()
        text_real_prob = 1.0 - text_fake_prob
        text_pred = 1 if text_fake_prob > 0.5 else 0

    print(f"\n-> Text Baseline Prediction: {'Fake' if text_pred == 1 else 'Real'}")
    print(f"   Confidence: Real: {text_real_prob:.2%}, Fake: {text_fake_prob:.2%}")

    # 3. Evaluate Image Baseline
    if not os.path.exists(image_path):
        print("\n-> Image Baseline Skipped: Image path not found.")
        return

    raw_image = Image.open(image_path).convert("RGB")
    transform = get_default_transform()
    image_tensor = transform(raw_image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_logits = image_model(image_tensor)
        image_fake_prob = torch.sigmoid(image_logits).item()
        image_real_prob = 1.0 - image_fake_prob
        image_pred = 1 if image_fake_prob > 0.5 else 0

    print(f"\n-> Image Baseline Prediction: {'Fake' if image_pred == 1 else 'Real'}")
    print(f"   Confidence: Real: {image_real_prob:.2%}, Fake: {image_fake_prob:.2%}")

if __name__ == "__main__":
    # Supply a real-world test case here
    sample_text = "Breaking: Giant meteor seen crashing into the ocean off the coast of Miami."
    sample_image = "data/test_image.jpg"

    evaluate_real_sample(sample_image, sample_text)
