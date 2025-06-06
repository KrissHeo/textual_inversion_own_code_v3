import importlib
import torch

def get_obj_from_str(string, reload=False):

    """
    문자열로부터 클래스나 함수 객체를 가져오는 유틸 함수
    """

    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config, **kwargs):

    """
    config 파일(dict 구조)로부터 클래스 인스턴스를 생성하는 함수

    config: {"target": "모듈.클래스이름", "params": {...}} 형식의 딕셔너리
    → 해당 클래스를 params로 초기화하여 인스턴스화

    kwargs: 외부에서 추가 전달할 인자
    """

    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()), **kwargs)


def load_model_from_config(config, ckpt, verbose=False):

    """
    체크포인트 로드 + 모델 인스턴스화하는 함수

    config: 모델 설정이 담긴 OmegaConf 객체
    ckpt: 저장된 .ckpt 파일 경로
    verbose: 누락된 키 / 예기치 않은 키 출력 여부

    1. 모델 클래스 인스턴스화
    2. state_dict 불러오기
    3. state_dict를 모델에 로드
    4. 누락/예상외 키 출력
    """

    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    config.model.params.ckpt_path = ckpt
    model = instantiate_from_config(config.model)
    missing, unexpected= model.load_state_dict(sd, strict=False)
    if len(missing) > 0 and verbose:
        print("missing keys:")
        print(missing)
    if len(unexpected) > 0 and verbose:
        print("unexpected keys:")
        print(unexpected)

    return model
