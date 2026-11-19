# SeMoCo

<div align="center">

[![Project Page](https://img.shields.io/badge/Project-Page-007ec6?style=plastic)][project]
[![arXiv](https://img.shields.io/badge/arXiv-2608.24334-b31b1b?style=plastic)][paper]
[![Hugging Face Models](https://img.shields.io/badge/Hugging%20Face-Models-d4b414?style=plastic)][models]
[![License](https://img.shields.io/badge/License-Apache%202.0-77aa1c?style=plastic)](LICENSE)

</div>

![SeMoCo teaser](assets/teaser.png)

SeMoCo generates human motion from text through semantic motion tokens and body-structured flow matching. A semantic-first tokenizer separates action semantics from kinematic detail, while an autoregressive semantic prior guides a six-part flow model to synthesize full-body motion. The generated motions support virtual character animation and transfer to humanoid robots.

## Installation

Requires Python 3.11+ and PyTorch with CUDA.

```bash
git clone https://github.com/OMEGA-i/SeMoCo.git
cd SeMoCo
pip install -e '.[text]'
```

## Pretrained Models

Download the [pretrained models][models] and extract them into `checkpoints/`.
See [Model Files](docs/CHECKPOINTS.md) for the directory structure.

## Inference

```bash
semoco-generate --config configs/humanml3d_inference.yaml \
  --inputs data/prompts.npz --output runs/motion.npz
```

Use `configs/omega_inference.yaml` for Ω-MotionVerse.
See [Generation](docs/INFERENCE.md) and [Tokenization](docs/TOKENIZER.md) for details.

## Training

Follow [Data Preparation](docs/DATA.md), then train the tokenizer and generator:

```bash
# Tokenizer
semoco-train-tokenizer --config configs/humanml3d_tokenizer.yaml \
  --windows data/train_windows.npy --teacher data/train_tmr.npy \
  --feature-stats data/feature_stats.npz --output runs/tokenizer

# Generator
semoco-train-ar --config configs/humanml3d_ar.yaml \
  --store data/store --output runs/ar
semoco-train-flow --recipe hml-base --store data/store --targets data/store \
  --output runs/flow
```

See [Training](docs/TRAINING.md) for continuation training and Ω-MotionVerse.

## Evaluation

```bash
semoco-score --dataset humanml3d --embeddings runs/test/seed*.npz
```

See [Evaluation](docs/EVALUATION.md) for the evaluators and test protocol.

## Acknowledgements

We thank [TMR](https://github.com/Mathux/TMR),
[TMR-SOMA-RP](https://huggingface.co/nvidia/TMR-SOMA-RP-v1),
[HumanML3D](https://github.com/EricGuo5513/HumanML3D), and
[SOMA-X](https://github.com/huster-wgm/SOMA-X) for their open-source resources.

## License

[Apache-2.0](LICENSE).

[models]: https://huggingface.co/poisonousID/SeMoCo
[project]: https://omega-i.github.io/SeMoCo-Page/
[paper]: https://arxiv.org/abs/2608.24334
