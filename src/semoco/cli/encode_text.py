"""Prepare generation inputs from captions and matching frozen TMR text features."""
import argparse
import json
from pathlib import Path

import numpy as np

from ..generator.text import TextConditioner


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True, help="JSON list of caption/frames objects")
    parser.add_argument("--retrieval", type=Path, required=True, help="Matching normalized TMR text embeddings [N,256] NPY")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="google/flan-t5-xl")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    prompts = json.loads(args.prompts.read_text())
    if not isinstance(prompts, list) or not prompts or args.batch_size < 1:
        parser.error("provide a nonempty prompt list and a positive batch size")
    if any(not isinstance(p, dict) or not isinstance(p.get("caption"), str) or not p["caption"].strip()
           or type(p.get("frames")) is not int or p["frames"] < 4 for p in prompts):
        parser.error("each prompt requires a nonempty caption and an integer frames >= 4")
    retrieval = np.load(args.retrieval, allow_pickle=False)
    if retrieval.shape != (len(prompts), 256) or not np.isfinite(retrieval).all():
        parser.error("retrieval must be finite [number of prompts,256]")
    if not np.allclose(np.linalg.norm(retrieval, axis=-1), 1., atol=1e-3):
        parser.error("retrieval must contain normalized TMR text embeddings")
    if args.output.suffix != ".npz":
        parser.error("--output must end in .npz")
    encoder = TextConditioner(args.model, device=args.device, local_files_only=args.local_files_only)
    rows = []
    for begin in range(0, len(prompts), args.batch_size):
        text, valid = encoder.encode([p["caption"] for p in prompts[begin:begin + args.batch_size]],
                                     retrieval[begin:begin + args.batch_size])
        rows.extend(x[m].cpu().numpy() for x, m in zip(text, valid, strict=True))
    width = max(map(len, rows))
    text = np.zeros((len(rows), width, 2048), dtype=np.float32)
    valid = np.zeros((len(rows), width), dtype=bool)
    for i, row in enumerate(rows):
        text[i, -len(row):] = row
        valid[i, -len(row):] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, text=text, text_valid=valid,
             frames=np.array([p["frames"] for p in prompts], dtype=np.int64))


if __name__ == "__main__":
    main()
