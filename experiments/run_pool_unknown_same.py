#!/usr/bin/env python3
"""Pool=True, UNKNOWN + SAME label (all samples share one label), bs=10/20/40"""
import os, random, torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib as mpl; mpl.use("Agg")

SEED = 2026
DATASET = "cifar100"
SAME_LABEL = 3  # all samples from class 3
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
OUTDIR = "expdata/pool_unknown_same"
os.makedirs(OUTDIR, exist_ok=True)

from _common import cosine_grad_loss, save_batch_img, total_variation
from cnn import BasicCNN, get_loaders, gia_reconstruct_batch_unknown_label
import cnn_data
from cnn_metrics import compute_psnr_batch, compute_ssim_batch

def adam_clamp_gia_unknown(model, true_grads, device, mean_t, std_t,
                           real_for_eval, batch_size, num_classes,
                           steps=6000, lr=0.1, tv_weight=1e-4, n_restarts=5):
    model.eval()
    ap = [p for p in model.parameters() if p.requires_grad]
    ic, isz = model.conv1.in_channels, model.img_size
    best_dummy, best_psnr = None, -float("inf")
    for r in range(n_restarts):
        torch.manual_seed(SEED + r*1000 + 7777)
        dd = torch.randn((batch_size, ic, isz, isz), device=device, requires_grad=True)
        dl = torch.randn((batch_size, num_classes), device=device, requires_grad=True)
        opt = torch.optim.Adam([dd, dl], lr=lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-5)
        bl, ni, pat = float("inf"), 0, 500
        for it in range(steps):
            opt.zero_grad()
            out = model(dd)
            log_p = F.log_softmax(out, dim=1); py = F.softmax(dl, dim=1)
            loss = -(py * log_p).sum(dim=1).mean()
            dg = torch.autograd.grad(loss, ap, create_graph=True, retain_graph=True, only_inputs=True)
            total = cosine_grad_loss(dg, true_grads) + tv_weight * total_variation(dd)
            cv = total.item(); total.backward(); opt.step(); sch.step()
            with torch.no_grad():
                px = dd * std_t + mean_t; px.clamp_(0,1); dd.data.copy_((px - mean_t) / std_t)
            if cv < bl - 1e-8: bl, ni = cv, 0
            else: ni += 1
            if (it+1) % 1000 == 0: print(f"    step {it+1}/{steps}, loss={cv:.6e}")
            if ni >= pat: print(f"    early stop at step {it+1}"); break
        with torch.no_grad():
            rec = (dd * std_t + mean_t).clamp(0,1)
            p = compute_psnr_batch(rec, real_for_eval); s = compute_ssim_batch(rec, real_for_eval)
            print(f"  [restart {r+1}] PSNR={p:.2f} dB, SSIM={s:.4f}")
            if p > best_psnr: best_psnr, best_dummy = p, dd.detach().clone()
    return best_dummy

ds_cfg = cnn_data.DATASET_CONFIG[DATASET]
mean_t = torch.tensor(ds_cfg["mean"], dtype=torch.float32, device=DEVICE).view(1,-1,1,1)
std_t = torch.tensor(ds_cfg["std"], dtype=torch.float32, device=DEVICE).view(1,-1,1,1)
def unnorm(x): return (x * std_t + mean_t).clamp(0,1)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
train_loader, _, cfg = get_loaders(DATASET, batch_size=500, num_workers=0, download=False)
# Collect all samples from SAME_LABEL
all_data, all_labels = [], []
for db, tb in train_loader: all_data.append(db); all_labels.append(tb)
all_data = torch.cat(all_data); all_labels = torch.cat(all_labels)

same_mask = (all_labels == SAME_LABEL)
same_data = all_data[same_mask]
print(f"Class {SAME_LABEL} has {same_data.shape[0]} samples")

for BS in [10, 20, 40]:
    print(f"\n{'#'*70}\n# BS={BS}, UNKNOWN+SAME_LABEL({SAME_LABEL}), Pool=True\n{'#'*70}")
    data = same_data[:BS].to(DEVICE)
    target = torch.full((BS,), SAME_LABEL, dtype=torch.long, device=DEVICE)
    print(f"Labels (hidden): {target.cpu().tolist()}")

    torch.manual_seed(SEED + 100)
    model = BasicCNN(in_channels=cfg["in_channels"], img_size=cfg["img_size"],
                     num_classes=cfg["num_classes"], use_bn=False, use_pool=True).to(DEVICE)
    model.train()
    for p in model.parameters(): p.grad = None
    out = model(data); loss = nn.CrossEntropyLoss()(out, target); loss.backward()
    true_grads = [p.grad.detach().clone() for p in model.parameters() if p.requires_grad]
    fc_input = model.last_feature.detach().clone()
    real_unnorm = unnorm(data)

    print(f"\n--- Adam+Clamp unknown (bs={BS}) ---")
    dc = adam_clamp_gia_unknown(model, true_grads, DEVICE, mean_t, std_t, real_unnorm,
                                batch_size=BS, num_classes=cfg["num_classes"])
    rc = unnorm(dc)

    print(f"\n--- HLCP+GIA unknown (bs={BS}) ---")
    bf, bfp = None, -float("inf")
    for r in range(3):
        d = gia_reconstruct_batch_unknown_label(model=model, true_grads=true_grads,
            real_feat=fc_input.to(DEVICE), feat_lambda=1.0, known_rate=1.0,
            steps=4000, lr=0.1, device=DEVICE, batch_size=BS, num_classes=cfg["num_classes"])
        rec = unnorm(d); p = compute_psnr_batch(rec, real_unnorm)
        print(f"  [restart {r+1}] PSNR={p:.2f} dB")
        if p > bfp: bfp, bf = p, rec.clone()

    cp = compute_psnr_batch(rc, real_unnorm); cs = compute_ssim_batch(rc, real_unnorm)
    hp = compute_psnr_batch(bf, real_unnorm); hs = compute_ssim_batch(bf, real_unnorm)
    print(f"\n{'='*60}\nResults (bs={BS}, Pool=True, UNKNOWN+SAME_LABEL):")
    print(f"  Adam+Clamp: PSNR={cp:.2f} dB, SSIM={cs:.4f}")
    print(f"  HLCP+GIA:   PSNR={hp:.2f} dB, SSIM={hs:.4f}\n{'='*60}")
    save_batch_img(real_unnorm, f"{OUTDIR}/real_bs{BS}.png")
    save_batch_img(rc, f"{OUTDIR}/clamp_bs{BS}.png")
    save_batch_img(bf, f"{OUTDIR}/hlcp_bs{BS}.png")
