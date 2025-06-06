import os
import torch
import torchvision
from PIL import Image


def save_images(images: dict, save_dir: str, step: int, prefix: str = "train", max_images: int = 4, clamp: bool = True):
    """
    이미지 딕셔너리를 저장하는 유틸 함수

    Args:
        images (dict): {"key": tensor(B, C, H, W)} 형태
        save_dir (str): 저장할 root 디렉토리
        step (int): 현재 학습 step (파일명 구분용)
        prefix (str): "train" or "val"
        max_images (int): 최대 저장할 이미지 수
        clamp (bool): 이미지 값을 [-1, 1]에서 [0, 1]로 변환할지 여부
    """
    save_root = os.path.join(save_dir, "images", prefix)
    os.makedirs(save_root, exist_ok=True)

    for k, v in images.items():
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu()
            if clamp:
                v = torch.clamp(v, -1., 1.)
            v = (v + 1.0) / 2.0  # [-1, 1] -> [0, 1]

            grid = torchvision.utils.make_grid(v[:max_images], nrow=4)
            ndarr = grid.mul(255).byte().permute(1, 2, 0).numpy()
            filename = f"{k}_step{step:06}.jpg"
            path = os.path.join(save_root, filename)
            Image.fromarray(ndarr).save(path)


def maybe_log_images(model, batch, step: int, save_dir: str, log_freq: int = 500):
    """
    일정 step마다 log_images()를 호출하여 이미지 저장
    model은 log_images(batch) 메서드를 갖고 있어야 함
    """
    if (step % log_freq == 0) and hasattr(model, "log_images"):
        model.eval()
        with torch.no_grad():
            images = model.log_images(batch, split="train")
        save_images(images, save_dir=save_dir, step=step, prefix="train")
        model.train()
