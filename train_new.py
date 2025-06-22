import os, sys, glob, datetime
import numpy as np
from omegaconf import OmegaConf
import argparse
from load import load_model_from_config, instantiate_from_config
from make_log_new import maybe_log_images  # <-- 이미지 로그 유틸 추가
import pytz


def get_parser(**parser_kwargs):
    
    """
    학습 및 실험 세팅을 위한 argparse 파서 정의
    ===============================
    Argument Parser 인자 요약
    ===============================

    [기본 구성]
    --base, -b
    설명: 사용할 config YAML 파일 목록
    예시: --base configs/my_config.yaml

    --train, -t
    설명: 학습 모드 실행 여부
    예시: -t

    --actual_resume
    설명: pretrained 모델 checkpoint 경로
    예시: --actual_resume model.ckpt

    --name, -n
    설명: 로그 디렉토리 이름
    예시: --name neon

    --postfix, -f
    설명 : name과 유사하지만, 같은 실험에서 버전 또는 상태 추가
    예시 : --postfix _v2

    --gpus
    설명: 사용할 GPU ID 지정 (쉼표로 구분)
    예시: --gpus 0,1

    --data_root
    설명: 학습 데이터 이미지 폴더 경로
    예시: --data_root dataset/neon

    --init_word
    설명: placeholder 토큰 초기화에 사용할 단어
    예시: --init_word painting

    --placeholder_string
    설명: 새 개념을 나타낼 특수 토큰
    예시: --placeholder_string '*'
    """
    
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

        # 기본 실험 설정

    parser = argparse.ArgumentParser(**parser_kwargs)

    # === 기본 실험 설정 ===
    parser.add_argument("-n", "--name", type=str, const=True, default="", nargs="?", help="postfix for logdir")
    parser.add_argument("-p", "--project", help="name of new or path to existing project")
    parser.add_argument("-f", "--postfix", type=str, default="", help="post-postfix for default name")
    parser.add_argument("-l", "--logdir", type=str, default="logs", help="directory for logging dat shit")

    # === resume 및 config ===
    parser.add_argument("-r", "--resume", type=str, const=True, default="", nargs="?", help="resume from logdir or checkpoint in logdir")
    parser.add_argument("-b", "--base", nargs="*", metavar="base_config.yaml", default=list(), help="paths to base configs. Loaded from left-to-right. Parameters can be overwritten or added with command-line options of the form `--key value`.")
    parser.add_argument("--actual_resume", type=str, required=True, help="Path to model to actually resume from")

    # === 학습 및 디버깅 플래그 ===
    parser.add_argument("-t", "--train", type=str2bool, const=True, default=False, nargs="?", help="train")
    parser.add_argument("-d", "--debug", type=str2bool, nargs="?", const=True, default=False, help="enable post-mortem debugging")

    # === Textual Inversion 관련 인자 ===
    parser.add_argument("--placeholder_string", type=str, help="Placeholder string which will be used to denote the concept in future prompts. Overwrites the config options.")
    parser.add_argument("--init_word", type=str, help="Word to use as source for initial token embedding")
    parser.add_argument("--embedding_manager_ckpt", type=str, default="", help="Initialize embedding manager from a checkpoint")

    # === 데이터 설정 ===
    parser.add_argument("--data_root", type=str, required=True, help="Path to directory with training images")

    # === seed 및 학습률 설정 ===
    parser.add_argument("-s", "--seed", type=int, default=23, help="seed for seed_everything")
    parser.add_argument("--scale_lr", type=str2bool, nargs="?", const=True, default=True, help="scale base-lr by ngpu * batch_size * n_accumulate")

    return parser

def nondefault_trainer_args(opt):

    """
    사용자가 Trainer에 넘긴 인자 중 기본값과 다른 인자만 추려 반환하는 함수
    PyTorch Lightning의 Trainer는 많은 인자를 지원하는데, 그중 실제로 사용자가 지정한 인자만 따로 추출
    이 함수를 통해 config에 쓸 필요 없는 default 값들은 걸러내고, 명시적으로 설정된 값만 trainer_config에 반영할 수 있음.
    """

    parser = argparse.ArgumentParser()
    args = parser.parse_args([])
    return sorted( param for param in vars(args) if getattr(opt, param) != getattr(args, param) )


if __name__ == "__main__":
    
    """
    Textual Inversion을 위한 학습 스크립트 main 진입점
    - argparse를 통해 사용자 설정을 받아 로그 디렉토리, config, resume 등을 구성
    - 모델 및 학습 설정을 초기화하고, Lightning trainer로 학습 실행
    """

    # 현재 시간(한국 시간 바탕)으로 logdir 이름 설정
    now = datetime.datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y-%m-%d-%H-%M-%S")

    sys.path.append(os.getcwd())

    # CLI parser 정의 및 PyTorch Lightning trainer 인자 추가
    parser = get_parser()

    # CLI parsing
    opt, unknown = parser.parse_known_args()

    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError(f"Cannot find {opt.resume}")

        if os.path.isfile(opt.resume):
            logdir = os.path.dirname(os.path.dirname(opt.resume))
            ckpt = opt.resume
        else:
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        nowname = os.path.basename(logdir)
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            name = "_" + os.path.splitext(os.path.basename(opt.base[0]))[0]
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    os.makedirs(ckptdir, exist_ok=True)
    os.makedirs(cfgdir, exist_ok=True)

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    config.model.params.personalization_config.params.embedding_manager_ckpt = opt.embedding_manager_ckpt
    if opt.placeholder_string:
        config.model.params.personalization_config.params.placeholder_strings = [opt.placeholder_string]
    if opt.init_word:
        config.model.params.personalization_config.params.initializer_words[0] = opt.init_word

    config.data.params.train.params.data_root = opt.data_root
    config.data.params.validation.params.data_root = opt.data_root

    model = load_model_from_config(config, opt.actual_resume).cuda()
    train_dataset = instantiate_from_config(config.data.params.train)
    val_dataset = instantiate_from_config(config.data.params.validation)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.data.params.batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.data.params.batch_size, shuffle=False, num_workers=4)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.model.base_learning_rate)

    bs = config.data.params.batch_size
    ngpu = torch.cuda.device_count()
    accumulate_grad_batches = 1
    if opt.scale_lr:
        model.learning_rate = accumulate_grad_batches * ngpu * bs * config.model.base_learning_rate
    else:
        model.learning_rate = config.model.base_learning_rate

    # === config 저장 ===
    for i, cfg_path in enumerate(opt.base):
        dst_path = os.path.join(cfgdir, f"{i:03d}_base.yaml")
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        with open(dst_path, "w") as f:
            f.write(OmegaConf.to_yaml(OmegaConf.load(cfg_path)))

    # CLI 인자로 override된 config도 따로 저장
    with open(os.path.join(cfgdir, "merged.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(config))

    global_step = 0
    num_epochs = config.get("num_epochs", 100)

    print("Entered training loop")

    if opt.train:
        model.train()
        for epoch in range(num_epochs):
            print(f"Epoch {epoch}")

            for batch in train_loader:
                optimizer.zero_grad()
                loss = model.training_step(batch, global_step)
                loss.backward()
                optimizer.step()

                if global_step % 100 == 0:
                    print(f"[Epoch {epoch} | Step {global_step}] Loss: {loss.item():.4f}")

                if global_step % 1000 == 0:
                    model.embedding_manager.save(os.path.join(ckptdir, f"step_{global_step}.pt"))
                    
                # 이미지 로그 추가
                maybe_log_images(model, batch, step=global_step, save_dir=logdir, log_freq=500)

                global_step += 1