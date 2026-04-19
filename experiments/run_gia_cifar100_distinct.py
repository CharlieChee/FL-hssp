#!/usr/bin/env python3
"""
CIFAR-100 GIA: 10 distinct labels, known label, no pool, 5 restarts.
"""
import os, random, torch, numpy as np
import torch.nn as nn
import matplotlib as mpl
mpl.use("Agg")

SEED = 2026
BATCH_SIZE = 10
GIA_STEPS = 4000
N_RESTARTS = 5
DATASET = "cifar100"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
OUTDIR = "expdata/gia_cifar100_distinct"
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

# Collect 10 distinct labels
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

train_loader, _, cfg = get_loaders(DATASET, batch_size=500, num_workers=0, download=False)

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

data = torch.stack([collected[i] for i in range(10)]).to(DEVICE)
target = torch.arange(10, device=DEVICE)
print(f"Labels: {target.cpu().tolist()}")

# Fixed model
torch.manual_seed(SEED + 100)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED + 100)

model = BasicCNN(
    in_channels=cfg["in_channels"],
    img_size=cfg["img_size"],
    num_classes=cfg["num_classes"],
    use_bn=False,
    use_pool=False,
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
model.train()
for p in model.parameters():
    p.grad = None

out = model(data)
loss = criterion(out, target)
loss.backward()

true_grads = [p.grad.detach().clone() for p in model.parameters() if p.requires_grad]
fc_input = model.last_feature.detach().clone()

real_unnorm = unnorm(data)

# Best of N restarts for naive GIA
print(f"\nNaive GIA ({N_RESTARTS} restarts):")
best_naive = None
best_naive_psnr = -float("inf")
for r in range(N_RESTARTS):
    dummy = gia_reconstruct_batch(
        model=model, target=target, true_grads=true_grads,
        real_feat=None, feat_lambda=0.0,
        steps=GIA_STEPS, lr=0.1, device=DEVICE)
    recon = unnorm(dummy)
    p = compute_psnr_batch(recon, real_unnorm)
    print(f"  [restart {r+1}] PSNR={p:.2f} dB")
    if p > best_naive_psnr:
        best_naive_psnr = p
        best_naive = recon.clone()

# Best of N restarts for HLCP+GIA
print(f"\nHLCP+GIA ({N_RESTARTS} restarts):")
best_feat = None
best_feat_psnr = -float("inf")
for r in range(N_RESTARTS):
    dummy = gia_reconstruct_batch(
        model=model, target=target, true_grads=true_grads,
        real_feat=fc_input.to(DEVICE), feat_lambda=1.0, known_rate=1.0,
        steps=GIA_STEPS, lr=0.1, device=DEVICE)
    recon = unnorm(dummy)
    p = compute_psnr_batch(recon, real_unnorm)
    print(f"  [restart {r+1}] PSNR={p:.2f} dB")
    if p > best_feat_psnr:
        best_feat_psnr = p
        best_feat = recon.clone()

naive_psnr = compute_psnr_batch(best_naive, real_unnorm)
naive_ssim = compute_ssim_batch(best_naive, real_unnorm)
feat_psnr = compute_psnr_batch(best_feat, real_unnorm)
feat_ssim = compute_ssim_batch(best_feat, real_unnorm)

print(f"\nResults (10 distinct labels, CIFAR-100, no pool):")
print(f"  Naive:    PSNR={naive_psnr:.2f} dB, SSIM={naive_ssim:.4f}")
print(f"  HLCP+GIA: PSNR={feat_psnr:.2f} dB, SSIM={feat_ssim:.4f}")

save_batch_img(real_unnorm, f"{OUTDIR}/real.png")
save_batch_img(best_naive, f"{OUTDIR}/naive.png")
save_batch_img(best_feat, f"{OUTDIR}/hlcp.png")
