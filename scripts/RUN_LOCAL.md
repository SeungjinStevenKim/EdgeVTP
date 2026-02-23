# 로컬 실행 가이드 (cluster 바쁠 때)

cluster 대신 로컬에서 실행할 때 사용합니다. 결과 형식은 cluster job과 동일합니다.

## 실행 방법

```bash
# 기본 config (pedestrian_unified.yaml)
./scripts/run_local.sh

# preset config 지정
./scripts/run_local.sh configs/pedestrian/local/train_original_10m.yaml
./scripts/run_local.sh configs/pedestrian/local/inference_original_10m.yaml
./scripts/run_local.sh configs/pedestrian/local/train_kan_15m.yaml
./scripts/run_local.sh configs/pedestrian/local/inference_kan_15m.yaml
```

## 출력 형식 (cluster와 동일)

- **로그:** `experiments/pedestrian/slurm/local_<timestamp>.out`
- **Training 결과:** `experiments/pedestrian/train/<run_name>/` (eth.pt, hotel.pt, ...)
- **Inference 결과:** `experiments/pedestrian/inference/<run_name>/inference_results.txt`

Inference 시 출력 예:
```
Average ADE:  1.065
Average FDE:  1.917
Method | 1.2s | 2.4s | 3.6s | 4.8s | ADE | FDE | Params(K)
```

## Preset config 목록

| Config | 설명 |
|--------|------|
| train_original_10m | Original 10m 학습 |
| inference_original_10m | Original 10m inference (train_original_10m_v2 필요) |
| train_kan_10m | KAN 10m 학습 |
| inference_kan_10m | KAN 10m inference (train_kan_10m_v1 필요) |
| train_kan_15m | KAN 15m 학습 |
| inference_kan_15m | KAN 15m inference (train_kan_15m_v1 필요) |
| train_mamba_10m | Mamba 10m 학습 |
| inference_mamba_10m | Mamba 10m inference (train_mamba_10m_v1 필요) |

## 전체 벤치마크 (4개 모델 train + inference)

```bash
# SLURM cluster에서 실행 (GPU 필요)
sbatch scripts/run_all_benchmark.slurm

# 로컬에서 실행 (GPU 있는 머신)
./scripts/run_all_benchmark.sh
```

실행 순서: Original 10m → KAN 10m → KAN 15m → Mamba 10m (각각 train 후 inference)
