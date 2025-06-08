import torch
import torch.nn as nn
import csv

pairs = []
with open("pairs.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        pairs.append((row["ko"], row["en"]))
print(f"Loaded {len(pairs)} word pairs.")


from ldm.modules.encoders.multilingual_clip import MultilingualTextEmbedder  # ✅ 추가
from ldm.modules.encoders.modules import BERTEmbedder  # ✅ 추가


mbert = MultilingualTextEmbedder().cuda()  # returns (B, 1280)
bert = BERTEmbedder(n_embed=1280, n_layer=32).cuda()


class Aligner(nn.Module):
    def __init__(self, in_dim=1280, out_dim=1280):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.proj(x)



aligner = Aligner().cuda()
optimizer = torch.optim.Adam(aligner.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

aligner = Aligner().cuda()
optimizer = torch.optim.Adam(aligner.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

# ✅ [1] 임베딩 캐싱 (단 한 번만)
ko_embs = {}
en_embs = {}
with torch.no_grad():
    for ko, en in pairs:
        ko_embs[ko] = mbert([ko])[0].detach()
        en_embs[en] = bert.encode([en])[0].detach()

# ✅ [2] 학습 루프 (빠르고 가볍게)
for epoch in range(1000):
    for ko, en in pairs:
        pred = aligner(ko_embs[ko])
        loss = loss_fn(pred, en_embs[en])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 100 == 0:
        print(f"[Epoch {epoch}] Loss: {loss.item():.4f}")

# ✅ 저장
torch.save(aligner.state_dict(), "aligner.pt")
print("✅ Aligner saved to aligner.pt")
