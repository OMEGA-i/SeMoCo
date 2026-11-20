# Generate motion

Install the text extra and place the matched models and statistics according to
[Model Files](CHECKPOINTS.md). The dataset presets in `configs/` select the
matching model files and sampling settings. Paths in these presets are relative
to the repository root.

## Generate from text prompts

Prepare `data/prompts.json` with captions and requested frame counts:

```json
[{"caption": "A person walks forward.", "frames": 196}]
```

Encode the prompts and generate motion:

```bash
semoco-encode-text --dataset humanml3d --prompts data/prompts.json \
  --output data/prompts.npz
semoco-generate --config configs/humanml3d_inference.yaml \
  --inputs data/prompts.npz --output runs/motion.npz
```

For Ω-MotionVerse, use `--dataset omega` and `configs/omega_inference.yaml`.
`semoco-encode-text` automatically pairs Flan-T5-XL token states with the
dataset's frozen sentence encoder (`checkpoints/<dataset>/text_encoder.pt`,
using DistilBERT on HumanML3D and Flan-T5-XL on Ω-MotionVerse). Precomputed
normalized `[N, 256]` sentence embeddings may also be supplied directly via
`--retrieval data/tmr_text.npy`.

Explicit command-line arguments override preset values. For example, `--seed 8`
changes the sampling seed, `--steps 32` sets the Flow integration steps, and
`--batch-size 32` controls batching.

## Output

`motion.npz` contains motion features, clip offsets, semantic codes, and motion
latents. When feature statistics are supplied, it also contains `physical_features`
in physical units. `offsets` separates individual clips in the concatenated arrays.

HumanML3D uses the standard 263-dimensional motion representation at 20 Hz.
Use `semoco.data.humanml3d.recover_joints` on each clip's physical features to
recover its 22 joint positions. Ω-MotionVerse uses UMR499 at 50 Hz; recover its
77 joints with `semoco.data.omega.recover_joints` (using the packaged SOMA77
template and neutral canonical anchor by default).

To encode or reconstruct existing motion independently of the generator, see
[Use the tokenizer](TOKENIZER.md).
