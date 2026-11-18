"""Encode, decode, or reconstruct motion with the main 1+7 tokenizer."""
import argparse
from pathlib import Path
import numpy as np
import torch
from ..tokenizer.codec import load_codec
from ..data.flow_store import load_statistics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("encode", "decode", "reconstruct"))
    for key in ("codec", "input", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--dataset", choices=("humanml3d", "omega"), required=True)
    parser.add_argument("--feature-stats", type=Path)
    parser.add_argument("--normalized", action="store_true", help="Input features are already codec-normalized.")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.output.suffix != ".npz" or args.output.resolve() == args.input.resolve():
        parser.error("write a new .npz output")
    codec, norm = load_codec(args.codec, dataset=args.dataset, device=args.device)
    if args.feature_stats:
        norm = load_statistics(args.feature_stats, codec.cfg.input_dim, torch.device(args.device))
    source = np.load(args.input, allow_pickle=False)
    if isinstance(source, np.lib.npyio.NpzFile):
        with source:
            values = source["codes" if args.mode == "decode" else "features"]
    else:
        values = source
    single = values.ndim == 2
    if single:
        values = values[None]
    if values.ndim != 3:
        parser.error("input must contain one sequence [T,D] or a fixed-length batch [B,T,D]")
    output = {}
    with torch.inference_mode():
        if args.mode == "decode":
            if values.shape[-1] != 8 or not np.issubdtype(values.dtype, np.integer):
                parser.error("decode expects integer codes with eight codebooks")
            codes = torch.tensor(values, dtype=torch.long, device=args.device)
            if not ((codes >= 0).all() and (codes[..., 0] < codec.cfg.semantic_codebook_size).all()
                    and (codes[..., 1:] < codec.cfg.codebook_size).all()):
                parser.error("code index outside the tokenizer vocabulary")
        else:
            if values.shape[-1] != codec.cfg.input_dim or not values.shape[1] or values.shape[1] % 4:
                parser.error("motion features must have complete four-frame groups and match the dataset width")
            features = torch.tensor(values, dtype=torch.float32, device=args.device)
            if not features.isfinite().all():
                parser.error("motion features must be finite")
            if not args.normalized:
                if norm is None:
                    parser.error("provide --feature-stats or already normalized features with --normalized")
                features = (features - norm[0]) / norm[1]
            codes = codec.encode(features)
        output["codes"] = codes.cpu().numpy()
        if args.mode != "encode":
            rec = codec.decode(codes).float()
            output["features"] = rec.cpu().numpy()
            if norm is not None:
                output["physical_features"] = (rec * norm[1] + norm[0]).cpu().numpy()
    if single:
        output = {key: value[0] for key, value in output.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **output)


if __name__ == "__main__":
    main()
