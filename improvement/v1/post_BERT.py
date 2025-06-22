import torch
import torch.nn as nn
from ldm.modules.encoders.modules import BERTEmbedder
from ldm.modules.encoders.modules import BERTTokenizer

class AdapterLayer(nn.Module):
    def __init__(self, hidden_size=768, adapter_dim=64):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, adapter_dim),
            nn.ReLU(),
            nn.Linear(adapter_dim, hidden_size)
        )

    def forward(self, x):
        return x + self.adapter(x)  # residual connection

class AdapterBERTEmbedder(BERTEmbedder):
    def __init__(self, adapter_dim=64, **kwargs):
        super().__init__(**kwargs)
        hidden_size = kwargs.get("n_embed", 768)
        self.adapter = AdapterLayer(hidden_size, adapter_dim)
        self.max_length = 77

        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False)

    def forward(self, text, embedding_manager=None, **kwargs):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text)  # [B, T]
        else:
            tokens = text

        tokens = tokens.to(self.device)

        def embed_with_adapter(input_tokens):
            x = self.transformer.token_emb(input_tokens)       # [B, T, H]
            x = self.adapter(x)                                # [B, T, H]
            x = self.transformer.pos_emb(x)                    # [B, T, H]
            return x

        z = self.transformer(tokens, return_embeddings=True, embedding_manager=embedding_manager,
                             custom_emb_fn=embed_with_adapter)
        return z
