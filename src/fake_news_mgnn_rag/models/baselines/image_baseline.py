"""
Unimodal baseline classifier using only image features.
"""

import torch
import torch.nn as nn

from fake_news_mgnn_rag.encoders.vision_encoder import VisionEncoder


class ImageBaselineClassifier(nn.Module):
    """
    A simple baseline model that classifies images into Fake (0) or Real (1).
    Uses VisionEncoder (ViT by default) for feature extraction and an MLP head.
    """
    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        hidden_dim: int = 256,
        dropout_rate: float = 0.2,
        freeze_encoder: bool = False
    ):
        super().__init__()
        
        # Image feature extractor
        self.encoder = VisionEncoder(model_name=model_name, freeze=freeze_encoder)
        
        # Classification Head (MLP)
        # Input dim is from encoder's hidden_size (usually 768 for vit-base)
        input_dim = self.encoder.hidden_size
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1) # 1 output for binary classification (logits)
        )

    def train(self, mode: bool = True):
        """
        Override train mode to properly propagate to the encoder.
        """
        super().train(mode)
        self.encoder.train(mode)
        return self

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: torch.Tensor of shape [batch_size, channels, height, width]
                          normalized using ImageNet stats.

        Returns:
            torch.Tensor of shape [batch_size, 1] containing raw logits.
        """
        # [batch_size, hidden_size]
        features = self.encoder(pixel_values)
        
        # [batch_size, 1]
        logits = self.classifier(features)
        
        return logits
