# EpiPose: 강체 도킹 포즈 가중을 통한 항체 특이적 에피토프 예측

결합 전 항원과 항체가 주어지면, EpiPose는 항체가 어느 항원 잔기에 결합하는지
예측한다. 강체 FFT 도킹으로 후보 포즈를 만들고, 모델은 각 포즈를 얼마나 신뢰할지
학습한다.

포즈 토큰은 서로 주의를 주고받아 같은 항원의 다른 후보들과 비교되며, 항원 잔기
토큰에도 주의를 준다. 모델은 포즈마다 스칼라 하나를 내놓고, 그 값들의 softmax가
포즈-잔기 접촉 행렬에 가중치로 곱해져 에피토프 맵이 된다.

보조 손실은 항원 표면의 결합 전후 구조 변화를 쓴다. 학습 항원은 두 형태가 모두
있어 잔기별 변화를 잴 수 있고, 포즈가 덮는 잔기들의 평균 변화로 포즈 점수의 순위를
매기면 결합 시 열리는 자리를 덮는 포즈를 선호하도록 배운다. 이 값은 결합 구조가
필요하므로 학습 target으로만 쓰이고 추론에는 들어가지 않는다.

## 결과

테스트셋 98개 항원-항체 unit, seed 3개.

| | AUROC | AUPRC | MCC | F1 | F1@30 |
|---|---|---|---|---|---|
| 보조 손실 없음 | 0.8439 ± 0.005 | 0.2480 ± 0.021 | 0.2580 ± 0.039 | 0.2909 ± 0.043 | 0.2958 ± 0.016 |
| **EpiPose** | **0.8486 ± 0.003** | **0.2571 ± 0.009** | **0.2801 ± 0.017** | **0.3137 ± 0.015** | **0.3030 ± 0.008** |

판정 임계값은 validation에서 고르고 test에 적용한다. F1@30은 unit마다 점수가 가장
높은 30개 잔기를 뽑으므로 임계값이 필요 없다.

## 환경

`environment.yml`에 고정된 버전으로 개발했다.

- Python 3.13, CUDA 13.0, torch 2.11.0+cu130
- numpy 1.26, scipy 1.17, scikit-learn 1.8, pandas 3.0

엄격한 요구사항은 아니고 최근 버전의 torch, numpy, scikit-learn이면 동작한다.
배포 feature로 추론하면 CPU에서 몇 분이면 끝나고, 학습에는 GPU가 필요하다.

```bash
conda env create --file environment.yml
conda activate epipose
```

torch 빌드가 드라이버와 맞지 않으면 <https://pytorch.org>를 따라 따로 설치한다.

## 데이터

저장소에는 코드, 설정, split 표, 체크포인트 세 개가 들어 있다. 용량이 큰 데이터는
release 아카이브로 따로 배포하며 두 단계로 나뉜다. **빠른 경로**는 함께 배포하는
중간 산출물 feature로 보고 수치를 재현하고, **전체 경로**는 도킹부터 다시 돌리는 데
필요한 구조까지 포함한다. 모든 아카이브는 저장소 기준 상대 경로를 담고 있으므로
저장소 루트에서 풀면 된다.

| 단계 | 아카이브 | 크기 | 풀리는 위치 | 필요한 작업 |
|---|---|---|---|---|
| 빠른 | `epipose_features_core.zip` | 57.9 MB | `data/features/{pose_rep,residue}.pkl` | `eval`, `predict` |
| 빠른 | `epipose_features_train.zip` | 139.0 MB | `data/features/{pose_multi_apo,delta_target}.pkl` | `train` |
| 전체 | `epipose_structures_cif.zip` | 58.2 MB | `data/structures/complex_cif/` | 보조 target 생성 |
| 전체 | `epipose_structures_dock.zip` | 307.2 MB | `data/structures/dock_fft/` | 도킹, 포즈 feature |
| 전체 | `epipose_structures_contacts.zip` | 18.2 MB | `data/structures/{consasa,apoc}/` | 포즈 접촉 행렬 |

체크섬은 아카이브 옆의 `SHA256SUMS`에 있다.

### 빠른 경로: 보고 수치 재현

아카이브 두 개, 풀면 324 MB. 평가는 CPU에서 몇 분이면 끝난다.

```bash
unzip epipose_features_core.zip -d .        # eval, predict
unzip epipose_features_train.zip -d .       # 학습하려면 추가

python main.py eval --config configs/predict.yaml \
                    --ckpt checkpoints/epipose_seed0.pt
```

`eval`과 `predict`에는 `epipose_features_core.zip` 하나면 된다.
`pose_multi_apo.pkl`과 `delta_target.pkl`은 있을 때만 읽으므로, 없이 학습하면
대표 apo 구조 하나와 주 손실만 쓰는 설정이 되고 이는 보고한 설정이 아니다.

### 전체 경로: 도킹 파이프라인 재실행

아카이브 다섯 개, 풀면 860 MB. 빠른 경로가 내려받는 feature pickle을 직접
다시 만드는 경로이고 GPU가 필요하다. [도킹 파이프라인 재실행](#도킹-파이프라인-재실행)
절을 따른다.

```bash
for z in epipose_features_core.zip epipose_features_train.zip \
         epipose_structures_cif.zip epipose_structures_dock.zip \
         epipose_structures_contacts.zip; do unzip -q "$z" -d .; done
```

## 폴더 구성

```
epipose/
├── main.py                 train / eval / predict 진입점
├── epipose/
│   ├── config.py           yaml 설정, 명령줄로 덮어쓰기
│   ├── data.py             feature 적재와 배치 구성
│   ├── model.py            네트워크와 포즈 점수 -> 맵
│   ├── losses.py           포즈 손실과 보조 순위 손실
│   ├── metrics.py          AUROC, AUPRC, MCC, F1, F1@30
│   └── runner.py           학습, 평가, 예측
├── docking/
│   ├── run_pipeline.py     도킹과 feature 추출 진입점
│   ├── residue_feature.py  항원 잔기 feature
│   ├── delta_target.py     보조 학습 target
│   └── _vendor/            원본 연구 코드, 경로만 외부화
├── configs/
│   ├── train.yaml          학습
│   ├── train_no_aux.yaml   보조 손실 없이 학습
│   └── predict.yaml        평가와 예측
├── data/
│   ├── assets/             파이프라인이 읽는 표 (저장소에 포함)
│   ├── features/           포즈와 잔기 feature (빠른 경로 아카이브)
│   └── structures/         도킹 입력, 결과, 구조 (전체 경로 아카이브)
└── checkpoints/            epipose_seed{0,1,2}.pt (저장소에 포함)
```

## 체크포인트 평가

```bash
python main.py eval --config configs/predict.yaml \
                    --ckpt checkpoints/epipose_seed0.pt
```

## 맵 예측

```bash
python main.py predict --config configs/predict.yaml \
                       --ckpt checkpoints/epipose_seed0.pt \
                       --split test --out pred.npz
```

`pred.npz`에는 unit별 에피토프 맵(`map_<gid>`)과 포즈 점수(`pose_logit_<gid>`)가
들어 있다.

## 학습

```bash
python main.py train --config configs/train.yaml --gpu 0

# 보조 손실 없이
python main.py train --config configs/train_no_aux.yaml --gpu 0
```

yaml의 모든 키는 명령줄로 덮어쓸 수 있다. 각 실행은
`runs/<tag>/{config.yaml, best.pt, result.json}`을 남긴다.

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `use_delta_rank` | `true` | 보조 순위 손실 사용 여부 |
| `lambda_delta` | `0.125` | 보조 손실 가중치 |
| `rank_margin` | `0.1` | 순위 손실에 들어갈 쌍의 최소 target 차이 |
| `multi_apo` | `true` | 매 스텝 unit의 apo 구조를 다시 뽑는다 |
| `top_poses` | `500` | unit당 유지할 포즈 수 (도킹 점수 기준) |
| `map_tau` | `1.0` | 포즈 점수 softmax 온도 |

## 도킹 파이프라인 재실행

전체 경로 아카이브 세 개를 먼저 푼다. 그러면 `data/structures/`에 입력이 갖춰져
아래 단계가 배포본 그대로 돌아간다. 경로는
`docking/_vendor/paths.py`에서 정하고 `EPIPOSE_WORK`, `EPIPOSE_DOCK_OUT`,
`EPIPOSE_CONSASA`, `EPIPOSE_APOC`, `EPIPOSE_RAW`로 바꿀 수 있다.

```
data/structures/dock_fft/     항원·항체 원자 텐서, 도킹 결과
data/structures/consasa/      포즈별 접촉 행렬
data/structures/apoc/         추가 apo 구조의 접촉 행렬
data/structures/complex_cif/  결합 구조, 보조 target 생성용
```

```bash
export EPIPOSE_WORK=/path/to/scratch

# 1. 복합체 cif -> 원자 텐서 (배포본에 없는 항원일 때만 필요)
EPIPOSE_RAW=/path/to/complex/cif python docking/run_pipeline.py tensor --nproc 16

# 2. FFT 도킹
python docking/run_pipeline.py dock --units units.txt --gpu 0 --keep 500

# 3. 포즈 feature
python docking/run_pipeline.py pose --gpu 0 --out data/features/pose_rep.pkl

# 4. 항원 잔기 feature
python docking/run_pipeline.py residue --out data/features/residue.pkl
```

보조 target은 학습에만 필요하다.

```bash
python docking/run_pipeline.py delta --out data/features/delta_target.pkl
```

`docking/_vendor/`는 원본 연구 스크립트다. 코드에 박혀 있던 경로를
`_vendor/paths.py`로 옮기고 이 저장소가 쓰지 않는 모델의 코드를 제거했으며, 포즈
복원과 점수 계산 경로는 그대로다. `residue_feature.py`와 `delta_target.py`는 새로
작성했고 배포 feature를 그대로 재현한다.

## 데이터 형식

`data/features/`에는 pickle 네 개가 있고 모두 `{"name": [...], "feat": {...}}`
구조다.

| 파일 | 키 | 내용 |
|---|---|---|
| `pose_rep.pkl` | `gid` | `F` (P, 33) 포즈 feature, `C` 비트로 압축한 접촉 행렬, `y` 에피토프 라벨, `q` 포즈 품질 |
| `pose_multi_apo.pkl` | `(gid, k)` | 같은 항원의 다른 apo 구조에 대한 같은 내용 |
| `residue.pkl` | `gid` | (nres, 9) prop, kl, id, rsa, kd, protrusion, dens6, dens10, q_abs |
| `delta_target.pkl` | `gid` | (nres, 4) 보조 target, 학습 unit만 |

`q = |C ∩ y| / |C|`는 포즈의 접촉 중 실제 에피토프인 비율이고 주 손실의 target이다.

`data/assets/epipose_split.csv`는 unit 목록이다.

| 열 | 의미 |
|---|---|
| `gid` | 항원-항체 unit 식별자 |
| `split` | train, val, test |
| `use` | 사용 가능한 unit인지 |
| `scored` | test unit이 보고 지표에 포함되는지 |
| `apo_code` | 이 항원에 쓴 apo 구조의 PDB 엔트리 |
| `cluster` | 서열 클러스터, split을 이 경계로 나눴다 |
| `seq_len` | canonical 항원 서열 길이 |
| `epitope` | 비트 문자열로 표현한 에피토프 라벨 (참조용) |
