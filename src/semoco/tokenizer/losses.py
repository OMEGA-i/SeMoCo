"""Reconstruction, commitment, and frozen-teacher semantic objectives."""
import torch
from torch import nn
from torch.nn import functional as F


def semantic_loss(prediction, teacher):
    return (1 - (F.normalize(prediction.float(), dim=-1) *
                 F.normalize(teacher.detach().float(), dim=-1)).sum(-1)).mean()


def humanml_loss(reconstruction, target, commitment):
    # Position channels are the root-relative joint block of Guo263.
    return (F.smooth_l1_loss(reconstruction.float(), target.float())
            + .5 * F.smooth_l1_loss(reconstruction[..., 4:67].float(), target[..., 4:67].float())
            + .02 * commitment)


class OmegaReconstructionLoss(nn.Module):
    """UMR499 smooth reconstruction with root and forward-kinematics geometry losses."""
    def __init__(self, template, mean, std):
        super().__init__()
        from .geometry import SomaFK
        self.fk = SomaFK(template)
        self.register_buffer("mean", torch.as_tensor(mean).float())
        self.register_buffer("std", torch.as_tensor(std).float())

    def joints(self, x):
        physical = x.float() * self.std + self.mean
        rotations = physical[..., 9:471].reshape(*x.shape[:2], 77, 6)
        return self.fk(rotations)

    def forward(self, rec, target, commitment):
        rec, target = rec.float(), target.float()
        predicted, actual = self.joints(rec), self.joints(target)
        pv, av = predicted[:, 1:] - predicted[:, :-1], actual[:, 1:] - actual[:, :-1]
        feet = [69, 70, 74, 75]
        # Contact weights come from
        # the normalized target channels, as supplied to the native objective.
        contact = target[:, 1:, -4:, None]
        sliding = (pv[:, :, feet].abs() * contact).sum() / (contact.sum() * 3 + 1e-6)
        return (F.smooth_l1_loss(rec, target) + 10 * F.smooth_l1_loss(rec[..., :15], target[..., :15])
                + .02 * commitment + F.l1_loss(predicted, actual) + .5 * F.l1_loss(pv, av)
                + .25 * F.l1_loss(pv[:, 1:] - pv[:, :-1], av[:, 1:] - av[:, :-1]) + .5 * sliding)
