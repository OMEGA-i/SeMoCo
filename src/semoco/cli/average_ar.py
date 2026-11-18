"""Average compatible semantic-prior checkpoints with equal weights."""
import argparse
from dataclasses import asdict
from pathlib import Path
import torch
from ..generator.ar.checkpoint import read_semantic_ar
from ..training.common import save


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    average = config = None
    for path in args.inputs:
        if path.resolve() == args.output.resolve():
            parser.error("output must differ from inputs")
        model = read_semantic_ar(torch.load(path, map_location="cpu", mmap=True, weights_only=False))
        current = asdict(model.cfg)
        if config is not None and current != config:
            raise ValueError("cannot average different semantic model configurations")
        config = current
        state = model.state_dict()
        if average is None:
            average = {k: v.float().clone() / len(args.inputs) for k, v in state.items()}
        else:
            for key, tensor in state.items():
                average[key].add_(tensor.float(), alpha=1 / len(args.inputs))
        del model
    save(args.output, dict(format="semoco.semantic_ar.v1", model_config=config, model=average))


if __name__ == "__main__":
    main()
