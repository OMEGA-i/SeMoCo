# Generate motion

Install the text extra and place the matched models and statistics according to
[Models and Data](CHECKPOINTS.md). The dataset presets in `configs/` select the
matching model files and sampling settings. Paths in these presets are relative
to the repository root.

## Generate from text features

Prepare `data/prompts.json` with captions and requested frame counts:

```json
[{"caption": "A person walks forward.", "frames": 196}]
```

Encode these captions with the dataset's frozen TMR text encoder and save the
normalized 256-dimensional sentence embeddings in `data/tmr_text.npy`, in the
same order. HumanML3D uses the text encoder from
[TMR's HumanML3D Guo-feature model](https://github.com/Mathux/TMR#pretrained-models-dvd);
its `encode_text.py` entry point supplies the latent mean. Normalize each mean to
unit length. Ω uses the matched Flan-TMR text encoder from the model bundle.
The TMR text encoder and tokenizer's motion teacher serve different purposes.

Then prepare the Flan tokens and generate motion:

```bash
semoco-encode-text --prompts data/prompts.json --retrieval data/tmr_text.npy \
  --output data/prompts.npz
```

```bash
semoco-generate --config configs/humanml3d_inference.yaml \
  --inputs data/prompts.npz --output runs/motion.npz
```

For Ω-MotionVerse, replace the preset with `configs/omega_inference.yaml`.
Explicit command-line arguments override preset values. For example, `--seed 8`
changes the sampling seed and `--steps 32` sets the Flow integration steps.

The input NPZ contains encoded text, a text mask, and the requested frame lengths.
The [data guide](DATA.md) gives the precise format. `TextConditioner` supplies the
frozen Flan encoding and combines it with the dataset's frozen TMR text features;
use the same text encoder as the training checkpoint.

## Output

`motion.npz` contains motion features, clip offsets, semantic codes, and motion
latents. When feature statistics are supplied, it also contains the motion
features in physical units. Offsets separate the clips in the concatenated arrays.

HumanML3D uses the standard 263-dimensional motion representation at 20 Hz.
The released `feature_stats.npz` is the Guo evaluator normalization. Use
`semoco.data.humanml3d.recover_joints` on each clip's physical features to
recover its 22 joint positions. Ω-MotionVerse uses UMR499 at 50 Hz. Recover its
77 joints with `semoco.data.omega.recover_joints`. The packaged SOMA77
template is the default, and the default anchor is the neutral identity pose.

To encode or reconstruct existing motion independently of the generator, see
[Use the tokenizer](TOKENIZER.md).

## Models

Use the tokenizer, semantic prior, Flow EMA, and statistics for the same dataset.
HumanML3D uses 40k Flow training followed by 2k continuation. Ω-MotionVerse uses
20k six-part Flow training after temporal-backbone pretraining.
