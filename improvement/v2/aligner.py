import torch
import torch.nn as nn
import csv
from ldm.modules.encoders.pre_BERT import MultilingualTextEmbedder
from ldm.modules.encoders.modules import BERTEmbedder

def pad_to_77(x, pad_value=0.0):
    if x.dim() == 2:
        x = x.unsqueeze(0)
    b, l, d = x.shape
    if l >= 77:
        return x[:, :77, :]
    pad_len = 77 - l
    pad = torch.zeros(b, pad_len, d, device=x.device, dtype=x.dtype)
    return torch.cat([x, pad], dim=1)

class Aligner(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.TransformerEncoderLayer(d_model=1280, nhead=8)

    def forward(self, x):  # x: [L, 1280] or [1, L, 1280]
        if x.dim() == 2:
            x = x.unsqueeze(0)  # → [1, L, 1280]

        x = x.permute(1, 0, 2)  # → [L, 1, 1280]
        x = self.transformer(x)  # → [L, 1, 1280]
        x = x.permute(1, 0, 2)  # → [1, L, 1280]

        x = pad_to_77(x)  # → [1, 77, 1280]
        return x

def train_and_save_aligner(save_path="aligner.pt"):
    mbert = MultilingualTextEmbedder().cuda()
    bert = BERTEmbedder(n_embed=1280, n_layer=32).cuda()
    aligner = Aligner().cuda()

    # Load word pairs
    pairs = []
    pairs_dir = "/home/elicer/textual_inversion_own_code_v3/improvement/v2/pairs.csv"
    with open(pairs_dir, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append((row["ko"], row["en"]))
    print(f"Loaded {len(pairs)} word pairs.")

    ko_embs, en_embs = {}, {}
    with torch.no_grad():
        for ko, en in pairs:
            ko_emb = mbert.encode_tokens(ko).detach()  # [T, 1280]
            print(ko_emb.shape)
            en_emb = bert.encode([en])[0].detach() # [77, 1280]
            ko_embs[ko] = ko_emb
            en_embs[en] = en_emb

    optimizer = torch.optim.Adam(aligner.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    for epoch in range(1000):
        for ko, en in pairs:
            pred = aligner(ko_emb) # [1, 77, 1280]
            loss = loss_fn(pred[0], en_embs[en])  # [77, 1280] vs [77, 1280]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 100 == 0:
            print(f"[Epoch {epoch}] Loss: {loss.item():.4f}")

    torch.save(aligner.state_dict(), save_path)
    print(f"✅ Aligner saved to {save_path}")
