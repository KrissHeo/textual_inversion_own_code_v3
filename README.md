# Textual_inversion own code + improvement

## 파일별 설명

setup.py : 학습 전 디렉토리 생성, 체크포인트 저장, YAML 파일 저장/수정 등을 통해 학습 환경을 자동 세팅하는 콜백입니다. \
load.py : 문자열로 모듈·클래스를 로드하고, 설정 파일과 체크포인트로 모델 객체를 생성해주는 유틸입니다. \
make_log_new.py : 학습 도중 생성된 이미지(예: 학습 샘플)를 저장하고 일정 스텝마다 로깅하는 유틸입니다. \
train_new.py : 학습 전체를 관리하는 메인 스크립트로, 데이터 로드, 모델 로드, 학습 루프, 체크포인트 저장, 로깅 등을 모두 담당합니다. \
txt2img.py : 원래 Latent Diffusion 모델과 학습된 embedding을 통해 prompt로 이미지를 생성하는 스크립트입니다. \

## 폴더별 설명

check : init_word, model 내부, placehoder 적용 여부 등을 확인할 수 있는 utility 폴더입니다. \
configs : train에 사용할 .yaml파일을 모아뒀습니다. 원 논문에서 사용했던 config, improvement에 사용했던 config 모두 모아놨습니다. \
dataset : 이미지 데이터셋입니다. \
ldm : latent diffusion 관련 코드 + Text Encoder, Embedding Manager 모두 들어있는 폴더입니다. \
logs : 학습이 시작하면 dataset, placeholder 등의 데이터가 추가된 config를 생성하고, 학습 중간중간 checkpoint, image log를 저장해두는 폴더입니다. 학습에 진행한 예시가 들어있습니다. \
models : pre-trained ldm이 보관되어 있습니다. \
outputs : txt2img_new.py로 sampling한 이미지를 보관하는 output_dir 입니다. \

## improvement 관련 설명
improvement/v1/post_BERT.py : BERT 임베더 위에 Adapter Layer + Transformer를 추가해 원래 임베딩 성능을 개선하는 모듈입니다. \
improvement/v2/pre_BERT.py : 다국어 Transformer 모델로 문장을 임베딩함. \
improvement/v2/aligner.py : 한국어-영어 임베딩 쌍으로 학습된 Aligner 모듈로, 다른 언어로 추출된 문장 임베딩도 원래 영어 임베딩 공간과 정렬해주는 역할을 함. \

train_new_v1.py : v1 실험에 맞춰서 학습해볼 수 있는 train 파일입니다. \
train_new_v2.py : v2 실험에 맞춰서 학습해볼 수 있는 train 파일입니다. \
txt2img_v1.py : v1 설정에 맞춰서 sampling 해보는 파일입니다.  \
txt2img_v2.py : v2 설정에 맞춰서 sampling 해보는 파일입니다. \

## 동작 방법
아래 명령어로 학습 스크립트 실행 가능: \


python train_new.py \
    --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml \
    -t \
    --actual_resume models/ldm/text2img-large/model.ckpt \
    --name {name} \
    --gpus 0 \
    --data_root dataset/{data} \
    --init_word {"init_word"} \
    --placeholder_string '*' \

아래 명령어로 txt2img 스크립트 실행 가능: \

python3 txt2img.py \
    --embedding_path {embedding_path} \
    --ckpt_path models/ldm/text2img-large/model.ckpt \
    --prompt "{prompt}" \
    --ddim_steps 200 \
    --outdir outputs/neon_v1 \
    --scale 7.5 \
    --n_samples 4 \
    --n_iter 1 \
    --ddim_eta 0.0

