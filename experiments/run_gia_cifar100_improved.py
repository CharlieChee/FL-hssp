#!/usr/bin/env python3
"""
Improved GIA on CIFAR-100: cosine similarity + TV regularization (Geiping et al. 2020 style).
10 distinct labels, known label, no pool, 8 restarts.
"""
import os, random, torch, numpy as np
import torch.nn as nn
import matplotlib as mpl
mpl.use("Agg")

SEED = 2026
BATCH_SIZE = 10
GIA_STEPS = 8000
N_RESTARTS = 8
DATASET = "cifar100"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
OUTDIR = "expdata/gia_cifar100_improved"
os.makedirs(OUTDIR, exist_ok=True)

from _common import cosine_grad_loss, save_batch_img, total_variation
from cnn import BasicCNN, get_loaders
import cnn_data
from cnn_metrics import compute_psnr_batch, compute_ssim_batch


def improved_gia(model, target, true_grads, device, steps=8000, lr=0.1,
                 tv_weight=1e-4, n_restarts=8, real_for_eval=None):
    """
    Inverting Gradients style: cosine similarity + TV + learning rate scheduling.
    """
    model.eval()
    attack_params = [p for p in model.parameters() if p.requires_grad]
    in_channels = model.conv1.in_channels
    img_size = model.img_size
    bs = target.shape[0]

    best_dummy = None
    best_psnr = -float("inf")

    for restart in range(n_restarts):
        dummy_data = torch.randn(
            (bs, in_channels, img_size, img_size),
            device=device, requires_grad=True
        )

        optimizer = torch.optim.Adam([dummy_data], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-4)
        criterion = nn.CrossEntropyLoss()

        best_loss = float("inf")
        no_improve = 0
        patience = 500

        for it in range(steps):
            optimizer.zero_grad()
            out = model(dummy_data)
            loss = criterion(out, target)
            dummy_grads = torch.autograd.grad(
                loss, attack_params, create_graph=True, retain_graph=True, only_inputs=True
            )

            # Cosine similarity loss
            grad_loss = cosine_grad_loss(dummy_grads, true_grads)

            # TV regularization
            tv_loss = total_variation(dummy_data)

            total_loss = grad_loss + tv_weight * tv_loss

            cur_val = total_loss.item()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            if cur_val < best_loss - 1e-8:
                best_loss = cur_val
                no_improve = 0
            else:
                no_improve += 1

            if (it + 1) % 500 == 0:
                print(f"    step {it+1}/{steps}, loss={cur_val:.6e}, "
                      f"grad={grad_loss.item():.4e}, tv={tv_loss.item():.4e}")

            if no_improve >= patience:
                print(f"    early stop at step {it+1}")
                break

        # Evaluate this restart
        with torch.no_grad():
            if real_for_eval is not None:
                ds_cfg = cnn_data.DATASET_CONFIG[DATASET]
                mean_t = torch.tensor(ds_cfg["mean"], device=device).view(1, -1, 1, 1)
                std_t = torch.tensor(ds_cfg["std"], device=device).view(1, -1, 1, 1)
                recon = (dummy_data * std_t + mean_t).clamp(0, 1)
                p = compute_psnr_batch(recon, real_for_eval)
            else:
                p = -best_loss
            print(f"  [restart {restart+1}] PSNR={p:.2f} dB, best_loss={best_loss:.6e}")
            if p > best_psnr:
                best_psnr = p
                best_dummy = dummy_data.detach().clone()

    return best_dummy


# ========== Setup ==========
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

# ========== Run improved GIA ==========
print(f"\n{'='*60}")
print(f"Improved GIA (cosine + TV, {N_RESTARTS} restarts, {GIA_STEPS} steps)")
print(f"{'='*60}")

best_dummy = improved_gia(
    model, target, true_grads, DEVICE,
    steps=GIA_STEPS, lr=0.1, tv_weight=1e-4,
    n_restarts=N_RESTARTS, real_for_eval=real_unnorm
)
best_recon = unnorm(best_dummy)

# Also run HLCP+GIA for comparison (use the simple DLG one, it's already perfect)
from cnn import gia_reconstruct_batch
print(f"\n{'='*60}")
print(f"HLCP+GIA (for comparison, 3 restarts)")
print(f"{'='*60}")
best_feat = None
best_feat_psnr = -float("inf")
for r in range(3):
    dummy = gia_reconstruct_batch(
        model=model, target=target, true_grads=true_grads,
        real_feat=fc_input.to(DEVICE), feat_lambda=1.0, known_rate=1.0,
        steps=4000, lr=0.1, device=DEVICE)
    recon = unnorm(dummy)
    p = compute_psnr_batch(recon, real_unnorm)
    print(f"  [restart {r+1}] PSNR={p:.2f} dB")
    if p > best_feat_psnr:
        best_feat_psnr = p
        best_feat = recon.clone()

# Metrics
improved_psnr = compute_psnr_batch(best_recon, real_unnorm)
improved_ssim = compute_ssim_batch(best_recon, real_unnorm)
feat_psnr = compute_psnr_batch(best_feat, real_unnorm)
feat_ssim = compute_ssim_batch(best_feat, real_unnorm)

print(f"\n{'='*60}")
print(f"Results (10 distinct labels, CIFAR-100, no pool):")
print(f"  Improved GIA (cosine+TV): PSNR={improved_psnr:.2f} dB, SSIM={improved_ssim:.4f}")
print(f"  HLCP+GIA:                 PSNR={feat_psnr:.2f} dB, SSIM={feat_ssim:.4f}")
print(f"{'='*60}")

save_batch_img(real_unnorm, f"{OUTDIR}/real.png")
save_batch_img(best_recon, f"{OUTDIR}/improved_gia.png")
save_batch_img(best_feat, f"{OUTDIR}/hlcp.png")
