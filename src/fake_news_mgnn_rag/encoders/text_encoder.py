import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List, Optional

class TextEncoder(nn.Module):
    """
    Text Encoder using DeBERTa (or other LLM backbones) for MGNN node feature extraction.
    Defaults to 'microsoft/deberta-v3-large' as it captures contextual nuances well.
    """
    def __init__(
        self, 
        model_name: str = "microsoft/deberta-v3-large",
        projection_dim: Optional[int] = None,
        freeze: bool = False
    ):
        super().__init__()
        self.model_name = model_name
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
                
        # Hidden size is usually 1024 for deberta-v3-large, 768 for base models
        self.hidden_size = self.model.config.hidden_size
        
        # Optional projection layer if we want to align modalities to a specific dimension
        if projection_dim is not None:
            self.projection = nn.Linear(self.hidden_size, projection_dim)
        else:
            self.projection = None

    def forward(self, texts: List[str], max_length: int = 512) -> torch.Tensor:
        """
        Encodes a list of strings into a batch of embedding tensors.
        
        Args:
            texts: List of strings (e.g., captions, headlines, claims).
            max_length: Maximum sequence length for tokenization.
            
        Returns:
            torch.Tensor of shape [batch_size, output_dim]. 
            Output dim is projection_dim if set, otherwise hidden_size.
        """
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Move to the same device as the model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward pass
        outputs = self.model(**inputs)
        
        # Mean pooling: average over the non-padded tokens
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]
        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        
        if self.projection is not None:
            pooled_output = self.projection(pooled_output)
            
        return pooled_output
