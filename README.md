# Retain–Reconstruct Integration

It implements corruption-aware feature reconstruction and sample-level retain/rebuild gating for Food101, MVSA, NYU Depth, and CMU-MOSI. 

## Method

For a corrupted modality, the system:
1. reconstructs its clean representation from the observed representation and
   the surviving modalities;
2. predicts a sample-specific retention coefficient `alpha` without receiving
   the corruption severity;
3. forms `alpha * observed + (1 - alpha) * reconstructed`; and
4. combines the hybrid feature with the surviving modalities for the task.

Each batch first updates reconstructors with sample and distribution-moment losses. It then detaches reconstructed
features and updates the fusion network and sample gates using task loss.

### Compact parameter budgets
| Dataset | Fusion width | Reconstructor width | Gate width | Trainable parameters |
|---|---:|---:|---:|---:|
| MVSA | 86 | 136 | 108 | 1,036,553 |
| Food101 | 92 | 136 | 112 | 1,074,479 |
| NYUD | 75 | 170 | 84 | 1,039,262 |

## Baselines
- `static_fusion`: fixed modality fusion weights.
- `oracle_fusion_weights`: searches modality fusion weights using test labels.

The oracle method is upper-bound diagnostic. 

## Installation

Python 3.10 or newer is required.

```bash
cd Retain_Reconstruct_Integration
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For development:

```bash
pip install -e '.[dev]'
pytest
```

PyTorch/CUDA may need to be installed using the command recommended for the
target machine before installing the remaining dependencies.

## Configuration

Every dataset YAML is merged over `configs/default.yaml`. Dataset paths use
environment variables rather than machine-specific absolute paths:

```bash
export FOOD101_ROOT=/path/to/Food101
export MVSA_ROOT=/path/to/MVSA_Single
export NYUD_ROOT=/path/to/nyu_dataset
export MOSI_ROOT=/path/to/MOSI
```

Any setting can be overridden from a script:

```bash
python scripts/train.py --config configs/food101.yaml \
  --set training.epochs=20 --set device=cuda
```

## Feature bundle

The training/evaluation CLIs consume encoded features so raw corruption and
expensive pretrained encoding can be cached once. The
configured `data.feature_bundle` is a PyTorch file with this schema:

```python
{
    "train": {
        "features": {
            "image": clean_image_features,       # [N, D_image]
            "text": clean_text_features,         # [N, D_text]
        },
        "targets": train_targets,                 # [N]
        "corruptions": {
            "image": {0.0: image_0, 0.1: image_01, ...},
            "text": {0.0: text_0, 0.1: text_01, ...},
        },
    },
    "test": {
        "features": {"image": ..., "text": ...},
        "targets": test_targets,
        "corruptions": {
            "image": {0.0: ..., 0.1: ..., ...},
            "text": {0.0: ..., 0.1: ..., ...},
        },
    },
}
```

Use the configured modality names for the other datasets:

- Food101/MVSA: `image`, `text`
- NYUD: `rgb`, `depth`
- MOSI: `visual`, `audio`, `text`

The dataset adapters in `rri.data`, pretrained wrappers in `rri.models.encoders`,
and deterministic operators in `rri.corruptions` provide the reusable pieces
for producing these caches from the original datasets. Keeping feature creation
separate prevents repeated encoder downloads and makes corruption comparisons
use identical encoded inputs.

## Commands

Train and create `best.pt`, `last.pt`, and `train_history.csv`:

```bash
python scripts/train.py --config configs/food101.yaml
```

Evaluate the clean test split:

```bash
python scripts/evaluate.py --config configs/food101.yaml
```

Evaluate static fusion and the proposed sample gate over corruptions:

```bash
python scripts/evaluate_corruptions.py --config configs/food101.yaml
```


Results are stored in long format with dataset, split, method, corrupted
modality, corruption, severity, seed, metric, value, and checkpoint columns.

## Checkpoints and generated results

Checkpoint and result directories are included.
`best.pt` and `last.pt` appear only after training. 


