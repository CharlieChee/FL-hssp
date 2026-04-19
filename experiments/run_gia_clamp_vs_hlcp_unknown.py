#!/usr/bin/env python3
"""
Adam+Clamp vs HLCP+GIA on CIFAR-100, bs=10/20/40, UNKNOWN labels, distinct labels, no pool.
"""
import os, random, torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib as mpl
mpl.use("Agg")

SEED = 2026
DATASET = "cifar100"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
OUTDIR = "expdata/gia_clamp_vs_hlcp_unknown"
os.makedirs(OUTDIR, exist_ok=True)

from _common import cosine_grad_loss, save_batch_img, total_variation
from cnn import BasicCNN, get_loaders, gia_reconstruct_batch_unknown_label
import cnn_data
from cnn_metrics import compute_psnr_batch, compute_ssim_batch


def adam_clamp_gia_unknown(model, true_grads, device, mean_t, std_t,
                           real_for_eval, batch_size, num_classes,
                           steps=6000, lr=0.1, tv_weight=1e-4, n_restarts=5):
    """Adam+Clamp GIA with unknown labels: optimize both data and label logits."""
    model.eval()
    attack_params = [p for p in model.parameters() if p.requires_grad]
    in_channels = model.conv1.in_channels
    img_size = model.img_size

    best_dummy = None
    best_psnr = -float("inf")

    for restart in range(n_restarts):
        torch.manual_seed(SEED + restart * 1000 + 7777)
        dummy_data = torch.randn(
            (batch_size, in_channels, img_size, img_size),
            device=device, requires_grad=True
        )
        dummy_label_logits = torch.randn(
            (batch_size, num_classes), device=device, requires_grad=True
        )

        optimizer = torch.optim.Adam([dummy_data, dummy_label_logits], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=steps, eta_min=1e-5)

        best_loss = float("inf")
        no_improve = 0
        patience = 500

        for it in range(steps):
            optimizer.zero_grad()
            out = model(dummy_data)
            # Soft cross-entropy with optimizable labels
            log_probs = F.log_softmax(out, dim=1)
            pseudo_y = F.softmax(dummy_label_logits, dim=1)
            loss = -(pseudo_y * log_probs).sum(dim=1).mean()

            dg = torch.autograd.grad(loss, attack_params,
                                      create_graph=True, retain_graph=True,
                                      only_inputs=True)
            gl = cosine_grad_loss(dg, true_grads)
            total = gl + tv_weight * total_variation(dummy_data)
            cur_val = total.item()
            total.backward()
            optimizer.step()
            scheduler.step()

            # Pixel clamp
            with torch.no_grad():
                pixel = dummy_data * std_t + mean_t
                pixel.clamp_(0.0, 1.0)
                dummy_data.data.copy_((pixel - mean_t) / std_t)

            if cur_val < best_loss - 1e-8:
                best_loss = cur_val
                no_improve = 0
            else:
                no_improve += 1
            if (it + 1) % 1000 == 0:
                print(f"    step {it+1}/{steps}, loss={cur_val:.6e}")
            if no_improve >= patience:
                print(f"    early stop at step {it+1}")
                break

        with torch.no_grad():
            recon = (dummy_data * std_t + mean_t).clamp(0, 1)
            p = compute_psnr_batch(recon, real_for_eval)
            s = compute_ssim_batch(recon, real_for_eval)
            print(f"  [restart {restart+1}] PSNR={p:.2f} dB, SSIM={s:.4f}")
            if p > best_psnr:
                best_psnr = p
                best_dummy = dummy_data.detach().clone()

    return best_dummy


ds_cfg = cnn_data.DATASET_CONFIG[DATASET]
mean_t = torch.tensor(ds_cfg["mean"], dtype=torch.float32, device=DEVICE).view(1, -1, 1, 1)
std_t = torch.tensor(ds_cfg["std"], dtype=torch.float32, device=DEVICE).view(1, -1, 1, 1)

def unnorm(x):
    return (x * std_t + mean_t).clamp(0.0, 1.0)

# Load data - collect 40 distinct labels
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
        if lbl not in collected and lbl < 40:
            collected[lbl] = data_batch[i]
        if len(collected) == 40:
            break
    if len(collected) == 40:
        break

print(f"Collected {len(collected)} distinct labels")

for BS in [10, 20, 40]:
    print(f"\n{'#'*70}")
    print(f"# Batch Size = {BS}, UNKNOWN labels")
    print(f"{'#'*70}")

    data = torch.stack([collected[i] for i in range(BS)]).to(DEVICE)
    target = torch.arange(BS, device=DEVICE)
    print(f"True labels (hidden from attacker): {target.cpu().tolist()}")

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

    # Adam+Clamp (unknown labels)
    print(f"\n--- Adam+Clamp unknown (bs={BS}, 5 restarts) ---")
    dummy_clamp = adam_clamp_gia_unknown(
        model, true_grads, DEVICE, mean_t, std_t,
        real_for_eval=real_unnorm, batch_size=BS,
        num_classes=cfg["num_classes"],
        steps=6000, lr=0.1, tv_weight=1e-4, n_restarts=5)
    recon_clamp = unnorm(dummy_clamp)

    # HLCP+GIA (unknown labels)
    print(f"\n--- HLCP+GIA unknown (bs={BS}, 3 restarts) ---")
    best_feat = None
    best_feat_psnr = -float("inf")
    for r in range(3):
        dummy = gia_reconstruct_batch_unknown_label(
            model=model, true_grads=true_grads,
            real_feat=fc_input.to(DEVICE), feat_lambda=1.0, known_rate=1.0,
            steps=4000, lr=0.1, device=DEVICE,
            batch_size=BS, num_classes=cfg["num_classes"])
        recon = unnorm(dummy)
        p = compute_psnr_batch(recon, real_unnorm)
        print(f"  [restart {r+1}] PSNR={p:.2f} dB")
        if p > best_feat_psnr:
            best_feat_psnr = p
            best_feat = recon.clone()

    clamp_psnr = compute_psnr_batch(recon_clamp, real_unnorm)
    clamp_ssim = compute_ssim_batch(recon_clamp, real_unnorm)
    hlcp_psnr = compute_psnr_batch(best_feat, real_unnorm)
    hlcp_ssim = compute_ssim_batch(best_feat, real_unnorm)

    print(f"\n{'='*60}")
    print(f"Results (bs={BS}, CIFAR-100, UNKNOWN labels, no pool):")
    print(f"  Adam+Clamp: PSNR={clamp_psnr:.2f} dB, SSIM={clamp_ssim:.4f}")
    print(f"  HLCP+GIA:   PSNR={hlcp_psnr:.2f} dB, SSIM={hlcp_ssim:.4f}")
    print(f"{'='*60}")

    save_batch_img(real_unnorm, f"{OUTDIR}/real_bs{BS}.png")
    save_batch_img(recon_clamp, f"{OUTDIR}/clamp_bs{BS}.png")
    save_batch_img(best_feat, f"{OUTDIR}/hlcp_bs{BS}.png")
