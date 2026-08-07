import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Optional

class VisionEncoder(nn.Module):
    """
    Vision Encoder using ViT for MGNN node feature extraction.
    Defaults to 'google/vit-base-patch16-224' as it captures global semantic context well.
    Expects pre-normalized image tensors of shape [B, C, H, W] directly from the dataloader.
    """
    def __init__(
        self, 
        model_name: str = "google/vit-base-patch16-224",
        projection_dim: Optional[int] = None,
        freeze: bool = False
    ):
        super().__init__()
        self.model_name = model_name
        
        # Load vision model
        self.model = AutoModel.from_pretrained(model_name)
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
                
        # Hidden size is usually 768 for vit-base
        self.hidden_size = self.model.config.hidden_size
        
        # Optional projection layer if we want to align modalities to a specific dimension
        if projection_dim is not None:
            self.projection = nn.Linear(self.hidden_size, projection_dim)
        else:
            self.projection = None

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encodes a batch of image tensors into embedding vectors.
        
        Args:
            pixel_values: torch.Tensor of shape [batch_size, channels, height, width].
                          Should be normalized using ImageNet stats.
            
        Returns:
            torch.Tensor of shape [batch_size, output_dim]. 
            Output dim is projection_dim if set, otherwise hidden_size.
        """
        if not torch.is_grad_enabled() or not self.model.training:
            self.model.eval()

        # Forward pass through ViT
        # For HF ViT models, pixel_values should be passed directly
        outputs = self.model(pixel_values=pixel_values)
        
        # We can use the pooler_output (CLS token passed through a dense layer)
        # or just the CLS token representation from last_hidden_state.
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            image_embeddings = outputs.pooler_output
        else:
            # Fallback to CLS token if pooler_output isn't available
            image_embeddings = outputs.last_hidden_state[:, 0, :]
            
        if self.projection is not None:
            image_embeddings = self.projection(image_embeddings)
            
        return image_embeddings
