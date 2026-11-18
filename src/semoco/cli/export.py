"""Export aligned motion latents, semantic codes, text, and part targets."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from ..generator.parts import pack_part_targets
from ..data.humanml3d import recover_joints
from ..tokenizer.codec import load_codec, encode_flow_targets, decode_continuous


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="JSON list with motion/text paths and Omega reconstructed_joints caches")
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--feature-stats", type=Path)
    parser.add_argument("--dataset", choices=("humanml3d", "omega"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    codec, norms = load_codec(args.codec, dataset=args.dataset, device=args.device)
    if args.feature_stats:
        with np.load(args.feature_stats, allow_pickle=False) as stats:
            norms = tuple(torch.tensor(stats[k], device=args.device) for k in ("mean", "std"))
    if norms is None:
        parser.error("provide feature mean/std for this tokenizer")
    mean, std = norms
    if not (torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all()):
        parser.error("invalid feature normalization")
    entries = json.loads(args.manifest.read_text())
    base = args.manifest.parent
    codes, latents, targets, words, index, text_index = [], [], [], [], [], []
    code_offset = text_offset = 0
    for item in entries:
        # The manifest defines the aligned crop. Export never chooses a new crop.
        motion = np.load(base / item["motion"], allow_pickle=False)
        with np.load(base / item["text"], allow_pickle=False) as text:
            embedding = text["text"]
        if motion.ndim != 2 or len(motion) == 0 or len(motion) % 4:
            raise ValueError("export motions must be nonempty, aligned [4T,D] crops")
        if embedding.ndim != 2 or embedding.shape[1] != 2048 or not len(embedding):
            raise ValueError("each text cache must contain [L,2048] Flan/TMR tokens")
        with torch.inference_mode():
            x = (torch.tensor(motion, dtype=torch.float32, device=args.device)[None] - mean) / std
            q0, y = encode_flow_targets(codec, x)
            # Auxiliary targets use the same semantic+continuous decode as
            # generation, followed by the dataset's physical joint recovery.
            reconstructed = decode_continuous(codec, q0, y, torch.tensor([len(q0[0])]))[0]
            if args.dataset == "humanml3d":
                joints = recover_joints(reconstructed * std + mean).cpu()
            else:
                with np.load(base / item["reconstructed_joints"], allow_pickle=False) as joint:
                    if "normalized_features" not in joint:
                        raise ValueError("joint recovery cache must retain normalized_features")
                    np.testing.assert_allclose(joint["normalized_features"], reconstructed.cpu().numpy(),
                                               atol=1e-5, rtol=1e-4)
                    joints = torch.from_numpy(joint["joints"])
            target = pack_part_targets(joints, args.dataset).numpy()
        if len(target) != len(q0[0]):
            raise ValueError("recovered joints and latent crops differ")
        codes.append(q0[0, :, None].cpu().numpy().astype(np.int16))
        latents.append(y[0].cpu().numpy())
        targets.append(target)
        words.append(embedding)
        index.append(dict(code_start=code_offset, code_len=len(q0[0])))
        text_index.append(dict(text_start=text_offset, text_len=len(embedding)))
        code_offset += len(q0[0]); text_offset += len(embedding)
    if not index:
        parser.error("manifest is empty")
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    prefix = args.split
    # No source paths or recording identifiers are copied into exported indices.
    for suffix, values in (("codes", codes), ("kin_latent", latents), ("text_emb.flantmr", words)):
        np.save(out / f"{prefix}.{suffix}.npy", np.concatenate(values))
    (out / f"{prefix}.index.json").write_text(json.dumps(index) + "\n")
    (out / f"{prefix}.text_index.flantmr.json").write_text(json.dumps(text_index) + "\n")
    np.save(out / f"{prefix}.targets.npy", np.concatenate(targets))
    if args.split == "train":
        for name, values in (("kin_stats", latents), ("targets_norm", targets)):
            array = np.concatenate(values).astype(np.float64)
            np.savez(out / f"{name}.npz", mean=array.mean(0).astype(np.float32),
                     std=np.maximum(array.std(0), 1e-5).astype(np.float32))
        # FlowStore uses one train-only target array for training.
        np.save(out / "targets.npy", np.concatenate(targets))
    print(f"Exported {len(index)} aligned {prefix} rows")


if __name__ == "__main__":
    main()
