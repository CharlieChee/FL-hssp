#!/usr/bin/env python3
"""
GIA baseline: same batch (10 distinct labels), with/without pooling.
Fixed seed, identical images for both experiments.
"""
import os, random, torch, numpy as np
import torch.nn as nn
import matplotlib as mpl
mpl.use("Agg")

SEED = 2026
BATCH_SIZE = 10
GIA_STEPS = 4000
N_RESTARTS = 5
DATASET = "cifar10"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
OUTDIR = "expdata/gia_baseline"
os.makedirs(OUTDIR, exist_ok=True)

from _common import save_batch_img
from cnn import gia_reconstruct_batch, BasicCNN, get_loaders
import cnn_data
from cnn_metrics import compute_psnr_batch, compute_ssim_batch

ds_cfg = cnn_data.DATASET_CONFIG[DATASET]
mean_t = torch.tensor(ds_cfg["mean"], dtype=torch.float32, device=DEVICE).view(1, -1, 1, 1)
std_t = torch.tensor(ds_cfg["std"], dtype=torch.float32, device=DEVICE).view(1, -1, 1, 1)

def unnorm(x):
    return (x * std_t + mean_t).clamp(0.0, 1.0)

def collect_distinct_labels(device, use_pool, dataset_name="cifar10", seed=2026):
    """Collect one sample per class (10 distinct labels for CIFAR-10)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_loader, _, cfg = get_loaders(dataset_name, batch_size=500, num_workers=0, download=False)
    num_classes = cfg["num_classes"]

    # Collect one per class
    collected = {}
    for data_batch, target_batch in train_loader:
        for i in range(data_batch.shape[0]):
            lbl = target_batch[i].item()
            if lbl not in collected and lbl < 10:
                collected[lbl] = data_batch[i]
            if len(collected) == 10:
                break
        if len(collected) == 10:
            break

    # Stack in label order 0-9
    data = torch.stack([collected[i] for i in range(10)]).to(device)
    target = torch.arange(10, device=device)
    print(f"  Labels: {target.cpu().tolist()}")

    # Fixed model weights via seed
    torch.manual_seed(seed + 100)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + 100)

    model = BasicCNN(
        in_channels=cfg["in_channels"],
        img_size=cfg["img_size"],
        num_classes=cfg["num_classes"],
        use_bn=False,
        use_pool=use_pool,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    model.train()
    for p in model.parameters():
        p.grad = None

    out = model(data)
    loss = criterion(out, target)
    loss.backward()

    true_grads = [p.grad.detach().clone() for p in model.parameters() if p.requires_grad]
    real_feat = model.last_feature.detach().clone()
    return model, data.detach(), target.detach(), true_grads, real_feat

results = {}

for use_pool in [False, True]:
    tag = "pool" if use_pool else "nopool"
    print(f"\n{'='*60}")
    print(f"  {tag.upper()} (bs={BATCH_SIZE}, steps={GIA_STEPS}, restarts={N_RESTARTS}, distinct labels)")
    print(f"{'='*60}")

    net, true_images, true_labels, true_grads, fc_input = \
        collect_distinct_labels(DEVICE, use_pool=use_pool, seed=SEED)

    real_unnorm = unnorm(true_images)

    # Best of N restarts for naive GIA
    best_naive = None
    best_naive_psnr = -float("inf")
    for r in range(N_RESTARTS):
        dummy = gia_reconstruct_batch(
            model=net, target=true_labels, true_grads=true_grads,
            real_feat=None, feat_lambda=0.0,
            steps=GIA_STEPS, lr=0.1, device=DEVICE)
        recon = unnorm(dummy)
        p = compute_psnr_batch(recon, real_unnorm)
        print(f"  [naive restart {r+1}] PSNR={p:.2f} dB")
        if p > best_naive_psnr:
            best_naive_psnr = p
            best_naive = recon.clone()

    # Best of N restarts for HLCP+GIA
    best_feat = None
    best_feat_psnr = -float("inf")
    for r in range(N_RESTARTS):
        dummy = gia_reconstruct_batch(
            model=net, target=true_labels, true_grads=true_grads,
            real_feat=fc_input.to(DEVICE), feat_lambda=1.0, known_rate=1.0,
            steps=GIA_STEPS, lr=0.1, device=DEVICE)
        recon = unnorm(dummy)
        p = compute_psnr_batch(recon, real_unnorm)
        print(f"  [feat restart {r+1}] PSNR={p:.2f} dB")
        if p > best_feat_psnr:
            best_feat_psnr = p
            best_feat = recon.clone()

    naive_psnr = compute_psnr_batch(best_naive, real_unnorm)
    naive_ssim = compute_ssim_batch(best_naive, real_unnorm)
    feat_psnr = compute_psnr_batch(best_feat, real_unnorm)
    feat_ssim = compute_ssim_batch(best_feat, real_unnorm)

    print(f"\n  Best naive: PSNR={naive_psnr:.2f} dB, SSIM={naive_ssim:.4f}")
    print(f"  Best feat:  PSNR={feat_psnr:.2f} dB, SSIM={feat_ssim:.4f}")

    results[tag] = {
        "naive_psnr": naive_psnr, "naive_ssim": naive_ssim,
        "feat_psnr": feat_psnr, "feat_ssim": feat_ssim,
    }

    save_batch_img(real_unnorm, f"{OUTDIR}/real_{tag}.png")
    save_batch_img(best_naive, f"{OUTDIR}/naive_{tag}.png")
    save_batch_img(best_feat, f"{OUTDIR}/feat_{tag}.png")

print(f"\n{'='*60}")
print("Summary (10 distinct labels, known labels):")
for tag in ["nopool", "pool"]:
    r = results[tag]
    print(f"  {tag:>6s}: naive PSNR={r['naive_psnr']:.2f}, SSIM={r['naive_ssim']:.4f} | "
          f"HLCP+GIA PSNR={r['feat_psnr']:.2f}, SSIM={r['feat_ssim']:.4f}")
