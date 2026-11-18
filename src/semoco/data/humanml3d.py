"""Recover HumanML3D joint positions from physical Guo263 features."""
import torch


def _rotate_inverse_yaw(angle, vectors):
    cosine, sine = torch.cos(2 * angle), torch.sin(2 * angle)
    x, y, z = vectors.unbind(-1)
    return torch.stack((cosine * x - sine * z, y, sine * x + cosine * z), -1)


def recover_joints(features: torch.Tensor) -> torch.Tensor:
    """Convert [..., frames, 263] physical features to [..., frames, 22, 3].

    The first root pose has zero horizontal translation and yaw. Root velocity
    at frame t moves the root to t+1; the feature's rotation velocity stores
    the half-angle used by HumanML3D's quaternion convention.
    """
    if features.ndim < 2 or features.shape[-1] != 263 or not features.shape[-2]:
        raise ValueError("expected nonempty physical Guo263 features")
    angle = torch.cat((torch.zeros_like(features[..., :1, 0]), features[..., :-1, 0]), -1).cumsum(-1)
    delta = torch.zeros_like(features[..., :3])
    delta[..., 1:, 0] = features[..., :-1, 1]
    delta[..., 1:, 2] = features[..., :-1, 2]
    root = _rotate_inverse_yaw(angle, delta).cumsum(-2)
    root[..., 1] = features[..., 3]
    local = features[..., 4:67].reshape(*features.shape[:-1], 21, 3)
    joints = _rotate_inverse_yaw(angle[..., None], local)
    horizontal = root.clone()
    horizontal[..., 1] = 0
    return torch.cat((root[..., None, :], joints + horizontal[..., None, :]), -2)
