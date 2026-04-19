import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights


# --------------- Metrics: PSNR / SSIM / FID ---------------

# FID is disabled by default (only PSNR / SSIM are computed); set to True to re-enable
FID_ENABLED = False


def compute_psnr_batch(x, y):
    """
    x, y: (B, C, H, W), values are assumed to be in [0, 1]
    Returns the mean PSNR (dB)
    """
    mse = F.mse_loss(x, y, reduction="none")
    mse = mse.view(mse.size(0), -1).mean(dim=1)  # per-image
    psnr = -10.0 * torch.log10(mse + 1e-8)
    return psnr.mean().item()


def _gaussian_kernel(window_size=11, sigma=1.5, channels=3, device="cpu"):
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g = g / g.sum()
    kernel_1d = g.view(1, 1, -1)
    kernel_2d = kernel_1d.transpose(1, 2) @ kernel_1d  # (1,1,H,W)
    kernel_2d = kernel_2d.to(device)
    kernel = kernel_2d.expand(channels, 1, window_size, window_size).contiguous()
    return kernel


def compute_ssim_batch(x, y, window_size=11, sigma=1.5):
    """
    Simplified SSIM following the classical formulation.
    x, y: (B, C, H, W), values are assumed to be in [0, 1]
    Returns the mean SSIM
    """
    device = x.device
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    channels = x.size(1)
    window = _gaussian_kernel(window_size, sigma, channels, device)

    mu_x = F.conv2d(x, window, padding=window_size // 2, groups=channels)
    mu_y = F.conv2d(y, window, padding=window_size // 2, groups=channels)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, window, padding=window_size // 2, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=window_size // 2, groups=channels) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=window_size // 2, groups=channels) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
        (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    )
    return ssim_map.mean().item()


_inception_models = {}


def _get_inception(device):
    """
    Return the Inception v3 model instance on the given device.
    To avoid device-mismatch issues in multi-GPU / multi-process settings,
    we maintain a separate model copy per device.
    """
    global _inception_models
    dev = torch.device(device)
    dev_key = str(dev)
    if dev_key not in _inception_models:
        # Use torchvision's official Inception v3 pretrained weights directly
        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        model.aux_logits = False  # ensure forward returns a single tensor
        model.fc = nn.Identity()  # take the penultimate-layer feature (2048)
        model.eval()
        model.to(dev)
        _inception_models[dev_key] = model
    return _inception_models[dev_key]


def _get_inception_features(x, device):
    """
    x: (N, 3, H, W), [0,1]
    return: (N, 2048) features
    """
    model = _get_inception(device)

    # Use the standard preprocessing provided by torchvision that matches the pretrained weights
    # (includes resize / center crop / normalize, depending on the weights)
    weights = Inception_V3_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()

    # Make sure it is a float tensor; the [0,1] range is guaranteed upstream — here we only apply geometric + normalization transforms.
    # Note: preprocess internally performs the proper resize / normalize.
    x = preprocess(x)

    with torch.no_grad():
        feats = model(x.to(device))
    return feats.view(feats.size(0), -1)


def compute_fid(real, fake, device):
    """
    Compute a simple FID based on the mean and covariance of Inception features.
    real, fake: (N, 3, H, W), [0,1]
    Note: in multi-GPU / multi-process environments, the given device must match the device of real/fake.
    When FID_ENABLED=False the FID is not computed and nan is returned (saves time and avoids Inception-related errors).
    """
    if not FID_ENABLED:
        return float("nan")
    # Extract Inception features on the given device
    real_f = _get_inception_features(real, device)
    fake_f = _get_inception_features(fake, device)

    return compute_fid_from_features(real_f, fake_f, device)


def compute_fid_from_features(real_f, fake_f, device):
    """
    Compute FID directly from Inception features.
    real_f, fake_f: (N, 2048)
    """
    real_f = real_f.to(device)
    fake_f = fake_f.to(device)

    mu_r = real_f.mean(dim=0)
    mu_f = fake_f.mean(dim=0)

    def cov(x):
        x = x - x.mean(dim=0, keepdim=True)
        return x.T @ x / (x.size(0) - 1)

    sigma_r = cov(real_f)
    sigma_f = cov(fake_f)

    # Add a small diagonal noise to the covariance matrices to mitigate singular / ill-conditioned issues for small batches
    eps = 1e-6
    eye = torch.eye(sigma_r.size(0), device=device, dtype=sigma_r.dtype)
    sigma_r = sigma_r + eps * eye
    sigma_f = sigma_f + eps * eye

    diff = mu_r - mu_f
    diff_sq = diff.dot(diff)

    # sqrtm(sigma_r * sigma_f) via eigen decomposition
    cov_prod = sigma_r @ sigma_f
    # Numerical safety: enforce symmetry
    cov_prod = (cov_prod + cov_prod.T) * 0.5

    eigvals = eigvecs = None
    # Gradually increase the diagonal noise and retry the decomposition; if it still fails, return NaN to avoid crashing the whole experiment
    for jitter in (0.0, 1e-4, 1e-2):
        cov_jit = cov_prod + jitter * eye
        cov_jit = (cov_jit + cov_jit.T) * 0.5
        try:
            eigvals, eigvecs = torch.linalg.eigh(cov_jit)
            break
        except Exception:
            eigvals = eigvecs = None

    if eigvals is None or eigvecs is None:
        # In extreme cases where convergence still fails, return NaN as a placeholder FID without interrupting the program
        print("[FID] Warning: torch.linalg.eigh failed to converge; returning NaN FID.")
        return float("nan")
    eigvals = torch.clamp(eigvals, min=0)
    sqrt_cov = eigvecs @ torch.diag(torch.sqrt(eigvals)) @ eigvecs.T

    trace_term = torch.trace(sigma_r + sigma_f - 2.0 * sqrt_cov)
    fid = diff_sq + trace_term
    return fid.item()


class FIDAccumulator:
    """
    Accumulates Inception features across multiple batches during evaluation, then computes FID once at the end.
    Usage:
        acc = FIDAccumulator(device)
        for real_batch, fake_batch in loader:
            acc.update(real_batch, fake_batch)
        fid = acc.compute()
    """

    def __init__(self, device):
        self.device = device
        self.real_feats = []
        self.fake_feats = []

    @torch.no_grad()
    def update(self, real_batch, fake_batch):
        """
        real_batch, fake_batch: (B, 3, H, W), values in [0,1]
        """
        rf = _get_inception_features(real_batch, self.device).detach().cpu()
        ff = _get_inception_features(fake_batch, self.device).detach().cpu()
        self.real_feats.append(rf)
        self.fake_feats.append(ff)

    def compute(self):
        if not self.real_feats or not self.fake_feats:
            return float("nan")
        real_f = torch.cat(self.real_feats, dim=0).to(self.device)
        fake_f = torch.cat(self.fake_feats, dim=0).to(self.device)
        return compute_fid_from_features(real_f, fake_f, self.device)

    def reset(self):
        self.real_feats.clear()
        self.fake_feats.clear()


