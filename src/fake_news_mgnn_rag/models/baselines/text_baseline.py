"""
Unimodal baseline classifier using only text features.
"""

import torch
import torch.nn as nn
from typing import List

from fake_news_mgnn_rag.encoders.text_encoder import TextEncoder


class TextBaselineClassifier(nn.Module):
    """
    A simple baseline model that classifies text into Real (0) or Fake (1).
    Uses TextEncoder (DeBERTa by default) for feature extraction and an MLP head.
    """
    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-large",
        hidden_dim: int = 256,
        dropout_rate: float = 0.2,
        freeze_encoder: bool = False
    ):
        super().__init__()

        # Text feature extractor
        self.encoder = TextEncoder(model_name=model_name, freeze=freeze_encoder)

        # Classification Head (MLP)
        # Input dim is from encoder's hidden_size (usually 1024 for deberta-v3-large)
        input_dim = self.encoder.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1)  # 1 output for binary classification (logits)
        )

    def train(self, mode: bool = True):
        """
        Override train mode to properly propagate to the encoder.
        """
        super().train(mode)
        self.encoder.train(mode)
        return self

    def forward(self, texts: List[str], max_length: int = 512) -> torch.Tensor:
        """
        Args:
            texts: List of strings (e.g., news headlines or captions).
            max_length: Max sequence length for tokenization.

        Returns:
            torch.Tensor of shape [batch_size, 1] containing raw logits.
            Use sigmoid(logits) >= 0.5 for binary prediction.
            0 = Real, 1 = Fake.
        """
        # [batch_size, hidden_size]
        features = self.encoder(texts, max_length=max_length)

        # [batch_size, 1]
        logits = self.classifier(features)

        return logits

    @torch.no_grad()
    def predict_proba(self, texts: List[str], max_length: int = 512) -> torch.Tensor:
        """
        Return probability of FAKE for each input text.

        Output range:
            0.0 → REAL
            1.0 → FAKE

        Args:
            texts: List of strings.
            max_length: Max sequence length for tokenization.

        Returns:
            torch.Tensor of shape [batch_size, 1] with probabilities.
        """
        self.eval()
        logits = self.forward(texts, max_length=max_length)
        return torch.sigmoid(logits)

    @torch.no_grad()
    def predict(self, texts: List[str], max_length: int = 512) -> torch.Tensor:
        """
        Return binary predictions.

        0 = REAL
        1 = FAKE

        Args:
            texts: List of strings.
            max_length: Max sequence length for tokenization.

        Returns:
            torch.Tensor of shape [batch_size, 1] with 0 or 1 values.
        """
        probabilities = self.predict_proba(texts, max_length=max_length)
        return (probabilities >= 0.5).long()