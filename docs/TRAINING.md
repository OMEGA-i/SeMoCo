# Training

Train the tokenizer, export motion representations, then train the semantic prior
and six-part Flow. Data formats are described in [Data Preparation](DATA.md).

## 1. Tokenizer

```bash
semoco-train-tokenizer --config configs/humanml3d_tokenizer.yaml \
  --windows data/train_windows.npy --teacher data/train_tmr.npy \
  --feature-stats data/feature_stats.npz --output runs/tokenizer
```

The tokenizer uses one semantic VQ and seven motion RVQ layers. A frozen TMR
motion encoder supplies semantic targets for each 64-frame training window.

For Ω-MotionVerse, use `configs/omega_tokenizer.yaml` and supply
`--template data/soma77_template.npz` for the geometry loss.

| Setting | HumanML3D | Ω-MotionVerse |
|---|---:|---:|
| Motion features | 263 | 499 |
| Temporal stride | 4 | 4 |
| Downsampling stages | 2 | 1 |
| Batch size | 256 | 1024 |
| Learning rate | 2e-4 | 4e-4 |
| Semantic loss weight | 0.15 | 0.15 |

## 2. Export training representations

Freeze the trained tokenizer and export the aligned motion and text features:

```bash
semoco-export --dataset humanml3d --codec runs/tokenizer/last.pt \
  --feature-stats data/feature_stats.npz --manifest data/train_manifest.json \
  --output data/store --split train
```

The store contains semantic codes, continuous motion latents, text features,
body-part targets, and training-set normalization statistics. See the
[export manifest](DATA.md#export-manifest) for input preparation.

## 3. Semantic prior

```bash
semoco-train-ar --config configs/humanml3d_ar.yaml \
  --store data/store --output runs/ar
```

For HumanML3D, average the 4k, 6k, 8k, and 10k checkpoints:

```bash
python -m semoco.cli.average_ar \
  --inputs runs/ar/model_004000.pt runs/ar/model_006000.pt \
           runs/ar/model_008000.pt runs/ar/model_010000.pt \
  --output runs/ar/averaged.pt
semoco-export-checkpoint ar --dataset humanml3d \
  --input runs/ar/averaged.pt --output checkpoints/humanml3d/semantic_ar.pt
```

For Ω-MotionVerse, use `configs/omega_ar.yaml` and the 20k checkpoint.
The prior predicts semantic codes with text and motion-length conditioning.
Both datasets use batch size 256, learning rate 3e-4, and 4k warmup.

## 4. Six-part Flow

Train the HumanML3D Flow for 40k updates:

```bash
semoco-train-flow --recipe hml-base --store data/store --targets data/store \
  --output runs/flow40k --seed 3407
```

Continue for 2k updates with the frozen semantic prior:

```bash
semoco-train-flow --recipe hml-continue --store data/store --targets data/store \
  --init runs/flow40k/ema_040000.pt --ar checkpoints/humanml3d/semantic_ar.pt \
  --output runs/flow --seed 3432
```

The base stage uses batch size 256, learning rate 2e-4, and EMA decay 0.999.
Continuation uses batch size 128, learning rate 3e-5, and EMA decay 0.998.

For Ω-MotionVerse, initialize from the pretrained temporal backbone and train the
six-part stage for 20k updates:

```bash
semoco-train-flow --recipe omega-warm --store data/omega_store --targets data/omega_store \
  --ordinary-parent checkpoints/omega/temporal_parent40k.pt \
  --output runs/omega_flow --steps 20000 --seed 3407
```

This stage uses batch size 256, learning rate 5e-5, and EMA decay 0.999.

## Resume and export

Add `--resume` to resume from `last.pt` in the output directory. Resume checkpoints
include optimizer and random states. Export inference models with
`semoco-export-checkpoint`, as described in [Model Files](CHECKPOINTS.md).
