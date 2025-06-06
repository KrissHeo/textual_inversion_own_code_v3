# .pt 파일에 어떤 placeholder 가 저장되어 있는지 확인
import torch


ckpt_path = "/home/elicer/textual_inversion/merged/neon_mug.pt"
# 파일 불러오기
ckpt = torch.load(ckpt_path, map_location="cpu" )

# 저장된 placeholder 이름 확인
placeholders = ckpt.get("string_to_param", {}).keys()

print("📌 학습된 placeholder 목록:")
for p in placeholders:
    print(f"{p} 존재")

print(ckpt['string_to_param']['*'])  # 값이 거의 0인지, 제대로 학습됐는지 확인
print(ckpt['string_to_param']['^'])  # 값이 거의 0인지, 제대로 학습됐는지 확인
