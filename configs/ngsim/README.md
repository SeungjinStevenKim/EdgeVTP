# NGSIM Configs

## Data 주의

**train_1k와 test_1k가 동일하면** (tr0=ts0, ...) data leakage. test_1k를 hold-out 데이터로 교체 필요.

## Data Options

| Folder       | 설명           | 파일 수   |
|-------------|----------------|------------|
| train_1k1   | 소량 (빠른 테스트) | 1개 each  |
| train_1k    | 중간           | 6개 each  |
| **train/val/test** | **전체 NGSIM** | 다수       |

## 전체 NGSIM으로 학습

1. **데이터 다운로드**: [README 링크](https://drive.google.com/file/d/1BnhGtGgiafV6LP9rnIJA6b5Lp4GxmPeB/view?usp=share_link)에서 preprocessed 데이터 다운로드
2. **압축 해제**: `datasets/ngsim/` 아래에 `train/`, `val/`, `test/` 폴더가 있어야 함
3. **실행**:
   ```bash
   sbatch scripts/run_ngsim_full_25m.slurm
   ```

## Config 파일

- `train_ngsim_25m.yaml` / `inference_ngsim_25m.yaml` — train_1k1 (소량)
- `train_ngsim_full_25m.yaml` / `inference_ngsim_full_25m.yaml` — train/val/test (전체)

공통: 25m radius filter, data augmentation (train만)

### augment_mode (YAML flag)

| 값 | 설명 |
|----|------|
| `full` | rotation + flip + scale (pedestrian, 기본값) |
| `scale_only` | scale만 (vehicle/highway용, rotation/flip 비활성화) |

- Pedestrian: `augment: true` → `full` (기본)
- NGSIM: `augment: true`, `augment_mode: scale_only`
