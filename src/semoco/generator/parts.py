"""Training targets aligned to the four physical frames of a codec token."""

from __future__ import annotations

import torch


PART_NAMES = ("root", "torso", "left_arm", "right_arm", "left_leg", "right_leg")
HUMANML_GROUPS = (
    (1, 2), (3, 6, 9, 12, 15), (13, 16, 18, 20),
    (14, 17, 19, 21), (4, 7, 10), (5, 8, 11),
)
# SOMA77 joint order, after dropping the rig's separate Root node.
SOMA_GROUPS = (
    (67, 72), tuple(range(1, 11)), tuple(range(11, 39)),
    tuple(range(39, 67)), (68, 69, 70, 71), (73, 74, 75, 76),
)
PART_GROUPS = {"humanml3d": HUMANML_GROUPS, "omega": SOMA_GROUPS}
PART_DIMS = {
    name: tuple(4 * 3 * (len(group) + (i == 0)) for i, group in enumerate(groups))
    for name, groups in PART_GROUPS.items()
}


def pack_part_targets(joints: torch.Tensor, dataset: str) -> torch.Tensor:
    """Pack one reconstructed joint sequence into [tokens, sum(PART_DIMS)].

    HumanML3D receives [4T, 22, 3]. Omega receives [1+4T, 77, 3];
    the first pose is the UMR499 recovery anchor, not an extra codec frame.
    """
    groups = PART_GROUPS[dataset]
    expected_joints = 22 if dataset == "humanml3d" else 77
    if joints.ndim != 3 or joints.shape[1:] != (expected_joints, 3):
        raise ValueError(f"{dataset} requires joint trajectories in [frames, {expected_joints}, 3]")
    frames = joints if dataset == "humanml3d" else joints[1:]
    if len(frames) == 0 or len(frames) % 4:
        raise ValueError("the target must contain four aligned frames per motion token")
    root = frames[:, :1]
    relative = frames - root
    packed = []
    for i, group in enumerate(groups):
        part = relative[:, group]
        if i == 0:
            part = torch.cat((part, root - joints[:1, :1]), dim=1)
        packed.append(part.reshape(len(frames) // 4, -1))
    return torch.cat(packed, dim=-1)


def part_loss(
    predictions: list[torch.Tensor],
    target: torch.Tensor,
    valid: torch.Tensor,
    time: torch.Tensor,
    denominator: torch.Tensor,
    dims: tuple[int, ...],
) -> torch.Tensor:
    """Equal weight for each group, with t² attenuation and masked token counts."""
    weight = valid * time[:, None].square()
    losses = []
    for i, (pred, gt) in enumerate(zip(predictions, target.split(dims, dim=-1), strict=True)):
        squared = (pred.float() - gt.float()).square()
        if i == 0:
            # Four frames, two root-relative hip joints and one root trajectory.
            frames = squared.unflatten(-1, (4, 3, 3))
            error = .5 * (frames[..., :2, :].mean((-3, -2, -1)) + frames[..., 2, :].mean((-2, -1)))
        else:
            error = squared.mean(-1)
        losses.append((error * weight).sum() / denominator.clamp_min(1))
    return torch.stack(losses).mean()
