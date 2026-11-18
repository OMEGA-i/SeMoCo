"""Frechet distance in a fixed motion-embedding space."""
import numpy as np

def fid(gen_emb: np.ndarray, gt_emb: np.ndarray) -> float:
    """Frechet distance between generated and GT motion embedding distributions."""
    from scipy import linalg
    mu_g, cov_g = gen_emb.mean(0), np.cov(gen_emb, rowvar=False)
    mu_r, cov_r = gt_emb.mean(0), np.cov(gt_emb, rowvar=False)
    diff = mu_g - mu_r
    covmean = linalg.sqrtm(cov_g @ cov_r)
    if isinstance(covmean, tuple):
        covmean = covmean[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(cov_g + cov_r - 2.0 * covmean))
