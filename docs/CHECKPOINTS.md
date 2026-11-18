# Model files

Download the [pretrained models](../README.md#pretrained-models) into `checkpoints/`:

```text
checkpoints/
  humanml3d/
    tokenizer.pt
    semantic_ar.pt
    flow.pt
    kin_stats.npz
    feature_stats.npz
  omega/
    tokenizer.pt
    semantic_ar.pt
    flow.pt
    kin_stats.npz
    feature_stats.npz
```

Each dataset has its own tokenizer, generator, and normalization statistics.
The inference presets in `configs/` use this layout.

| File | Contents |
|---|---|
| `tokenizer.pt` | Tokenizer configuration, weights, and optional feature statistics |
| `semantic_ar.pt` | Semantic autoregressive model configuration and weights |
| `flow.pt` | Six-part Flow configuration and EMA weights |
| `kin_stats.npz` | Motion-latent mean and standard deviation |
| `feature_stats.npz` | Motion-feature mean and standard deviation |

Statistics are computed on the training set. HumanML3D uses 263 motion features;
Ω-MotionVerse uses 499. Both use 512-dimensional motion latents.

Training also uses a frozen TMR motion teacher and its normalization statistics.
Text conditioning uses Flan-T5-XL and the matching TMR text encoder. The Ω Flow
training preset starts from `checkpoints/omega/temporal_parent40k.pt`.

## Export trained models

Export model-only files from training checkpoints:

```bash
semoco-export-checkpoint tokenizer --dataset humanml3d \
  --input runs/tokenizer/last.pt --output checkpoints/humanml3d/tokenizer.pt
semoco-export-checkpoint ar --dataset humanml3d \
  --input runs/ar/averaged.pt --output checkpoints/humanml3d/semantic_ar.pt
semoco-export-checkpoint flow --dataset humanml3d \
  --input runs/flow/last.pt --output checkpoints/humanml3d/flow.pt
```

The exported files contain model configuration, weights, and normalization statistics.
