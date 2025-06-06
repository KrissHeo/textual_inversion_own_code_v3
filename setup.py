import os, re
from omegaconf import OmegaConf

class SetupCallback(Callback):

    """
    학습 전에 로그 디렉토리/체크포인트/config 파일을 자동으로 생성/저장하는 콜백 클래스
    """

    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_keyboard_interrupt(self, trainer, pl_module):
        # Ctrl+C 로 종료시키면, last.ckpt(마지막 checkpoint) 저장
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def on_pretrain_routine_start(self, trainer, pl_module):
        # 학습 전에 log/config 디렉토리 생성 및 config 파일 저장
        if trainer.global_rank == 0:
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            # config 확인 및 저장
            print("Project config:")
            print(OmegaConf.to_yaml(self.config))
            yaml_path = os.path.join(self.cfgdir, f"{self.now}-project.yaml")
            OmegaConf.save(self.config, yaml_path)

            with open(yaml_path, "r", encoding="utf-8") as f:
                content = f.read()

            # placeholder_strings에 특수문자(e.g. ^)가 있을 경우 YAML이 깨지지 않도록 문자열에 따옴표를 감싸주는 함수
            def replace_placeholder_line(match):
                # match = pattern 과 비슷하게 생겼다고 찾은 yaml 파일의 일부분
                prefix = match.group(1)             # 'placeholder_strings:\n  ' 등 줄 앞부분 (들여쓰기 포함)
                list_marker_part = match.group(2)   # '-'와 그 뒤의 공백 (예: '- ')
                value = match.group(3).strip()      # 실제 placeholder 값 (예: ^, * 등 )

                # 값이 따옴표로 감싸져 있지 않다면 작은따옴표로 감쌈
                if value and not (value.startswith("'") and value.endswith("'")) and \
                   not (value.startswith('"') and value.endswith('"')):
                    quoted_value = f"'{value}'"
                    # * => '*'
                else:
                # 이미 감싸져 있거나 빈 문자열인 경우 그대로 사용
                    quoted_value = value

                # 완성된 YAML 줄을 재구성하여 반환
                return f"{prefix}{list_marker_part}{quoted_value}\n"

        #yaml 파일에서 placeholder_strings: - * 라고 적힌 부분을 찾음 = pattern
            pattern = r"(^\s*placeholder_strings:\s*\n\s*)(\-\s*)(.*)$"

            # pattern 찾은 거에 변환해주는 함수 적용
            new_content = re.sub(pattern, replace_placeholder_line, content, flags=re.MULTILINE)

            # placeholder 수정한거 다시 yaml에 저장
            with open(yaml_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            # lightning 설정도 별도 YAML로 저장
            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            # 없어도 될듯?
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass
