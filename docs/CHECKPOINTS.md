# Model files

Download the [pretrained models](../README.md#pretrained-models) into `checkpoints/`:

```text
checkpoints/
  humanml3d/
    tokenizer.pt
    semantic_ar.pt
    flow.pt
    text_encoder.pt
    kin_stats.npz
    feature_stats.npz
  omega/
    tokenizer.pt
    semantic_ar.pt
    flow.pt
    text_encoder.pt
    kin_stats.npz
    feature_stats.npz
    temporal_parent40k.pt
    evaluator_v2.pt
    norm_stats.json
```

Each dataset has its own tokenizer, generator, text encoder, and normalization statistics.
The inference presets in `configs/` use this layout.

| File | Contents |
|---|---|
| `tokenizer.pt` | Tokenizer configuration, weights, and feature statistics |
| `semantic_ar.pt` | Semantic autoregressive model configuration and weights |
| `flow.pt` | Six-part Flow configuration and EMA weights |
| `text_encoder.pt` | Frozen TMR sentence encoder for `semoco-encode-text` |
| `kin_stats.npz` | Motion-latent mean and standard deviation |
| `feature_stats.npz` | Motion-feature mean and standard deviation |
| `temporal_parent40k.pt` | Ω only. Pretrained temporal Flow backbone for six-part training |
| `evaluator_v2.pt`, `norm_stats.json` | Ω only. Independent V2 motion/text evaluator and its 30 Hz statistics |

Statistics are computed on the training set. HumanML3D uses 263 motion features;
Ω-MotionVerse uses 499. Both use 512-dimensional motion latents.

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
