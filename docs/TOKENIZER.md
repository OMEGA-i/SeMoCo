# Use the tokenizer

Train a tokenizer using the command in [Training](TRAINING.md), or load a
matched pretrained tokenizer and its feature statistics.

## Encode motion

```bash
semoco-tokenize encode --dataset humanml3d \
  --codec checkpoints/humanml3d/tokenizer.pt \
  --feature-stats checkpoints/humanml3d/feature_stats.npz \
  --input data/motion.npy --output runs/codes.npz
```

Input is one physical feature sequence, or a fixed-length batch. Frame count must
be divisible by four. Use `--normalized` only when the input is already normalized
with the matching tokenizer statistics. The output contains all eight codebooks.

## Decode or reconstruct

```bash
semoco-tokenize decode --dataset humanml3d \
  --codec checkpoints/humanml3d/tokenizer.pt \
  --feature-stats checkpoints/humanml3d/feature_stats.npz \
  --input runs/codes.npz --output runs/reconstructed.npz
```

`reconstruct` accepts motion features directly and performs both operations.
For Ω, use `--dataset omega` and its own models/statistics.

Decoded output includes normalized features and, when statistics are available,
physical motion features. Joint recovery is dataset-specific, as described in
[Generation](INFERENCE.md).
