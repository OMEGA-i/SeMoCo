"""Export model-only checkpoints without training metadata or machine paths."""
import argparse
from dataclasses import asdict
from pathlib import Path
import torch
from ..tokenizer.codec import read_codec
from ..generator.ar.checkpoint import read_semantic_ar
from ..generator.flow import PartFlow


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("tokenizer", "ar", "flow"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=("humanml3d", "omega"), required=True)
    args = parser.parse_args(argv)
    if args.input.resolve() == args.output.resolve():
        parser.error("write to a new checkpoint")
    record = torch.load(args.input, map_location="cpu", mmap=True, weights_only=False)
    if args.kind == "tokenizer":
        model, norm = read_codec(record)
        expected = 263 if args.dataset == "humanml3d" else 499
        if model.cfg.input_dim != expected:
            parser.error("dataset does not match tokenizer")
        exported = dict(format="semoco.tokenizer.v1", config=asdict(model.cfg), model=model.state_dict())
        if norm is not None:
            exported["norm"] = dict(zip(("mean", "std"), norm))
    elif args.kind == "ar":
        model = read_semantic_ar(record)
        exported = dict(format="semoco.semantic_ar.v1", model_config=asdict(model.cfg), model=model.state_dict())
    else:
        model = PartFlow.load_ema(args.input, dataset=args.dataset, device="cpu")
        cfg = dict(dataset=args.dataset, width=model.y_in.out_features, depth=len(model.blocks),
                   heads=model.blocks[0].sa.num_heads, ydim=model.ydim,
                   text_dim=model.txt.in_features, q0_vocab=model.q0.num_embeddings,
                   max_t=model.pos.shape[1], dropout=0.)
        exported = dict(format="semoco.flow.v1", config={"model": cfg}, ema=model.state_dict())
    # Whitelist-only reconstruction above drops optimizer states, source names,
    # private data indices, selection history, and arbitrary checkpoint metadata.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pending = args.output.with_suffix(".pending")
    torch.save(exported, pending)
    pending.replace(args.output)
    print(f"Exported {args.kind}: {sum(t.numel() for t in model.parameters()):,} active parameters")


if __name__ == "__main__":
    main()
