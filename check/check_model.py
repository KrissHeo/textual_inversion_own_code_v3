# .pt 파일에 어떤 placeholder 가 저장되어 있는지 확인
import torch


ckpt_path = "/home/elicer/textual_inversion_own_code_v2/textual_inversion_own_code_v2/models/ldm/text2img-large/model.ckpt"
model = torch.load(ckpt_path)
print(model['state_dict'].keys())