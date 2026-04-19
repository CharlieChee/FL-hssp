#!/usr/bin/env python3
"""
Compare multiple GIA methods on CIFAR-100: bs=10, distinct labels, no pool.
Methods:
  1) DLG (MSE + Adam)          — baseline
  2) Cosine + TV (Adam)        — Geiping et al. 2020
  3) L-BFGS (MSE)              — original DLG optimizer
  4) Cosine + MSE + TV (Adam)  — combined
  5) Cosine + strong TV (Adam) — heavier regularization
  6) Adam + pixel clamp        — project to valid range each step
"""
import os, random, torch, numpy as np
import torch.nn as nn
import matplotlib as mpl
mpl.use("Agg")

SEED = 2026
BATCH_SIZE = 10
DATASET = "cifar100"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
OUTDIR = "expdata/gia_cifar100_methods"
os.makedirs(OUTDIR, exist_ok=True)

from _common import cosine_grad_loss, mse_grad_loss, save_batch_img, total_variation
from cnn import BasicCNN, get_loaders
import cnn_data
from cnn_metrics import compute_psnr_batch, compute_ssim_batch


def run_method(name, model, target, true_grads, device, real_for_eval,
               loss_fn="mse", optimizer_type="adam", lr=0.1, steps=4000,
               tv_weight=0.0, n_restarts=5, do_clamp=False,
               mean_t=None, std_t=None):
    """Generic GIA runner with different loss/optimizer combos."""
    print(f"\n{'='*60}")
    print(f"Method: {name} ({n_restarts} restarts, {steps} steps)")
    print(f"  loss={loss_fn}, opt={optimizer_type}, lr={lr}, tv={tv_weight}, clamp={do_clamp}")
    print(f"{'='*60}")

    model.eval()
    attack_params = [p for p in model.parameters() if p.requires_grad]
    in_channels = model.conv1.in_channels
    img_size = model.img_size
    bs = target.shape[0]
    criterion = nn.CrossEntropyLoss()

    best_dummy = None
    best_psnr = -float("inf")

    for restart in range(n_restarts):
        torch.manual_seed(SEED + restart * 1000 + hash(name) % 10000)
        dummy_data = torch.randn(
            (bs, in_channels, img_size, img_size),
            device=device, requires_grad=True
        )

        if optimizer_type == "adam":
            optimizer = torch.optim.Adam([dummy_data], lr=lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=steps, eta_min=1e-5)
        elif optimizer_type == "lbfgs":
            optimizer = torch.optim.LBFGS([dummy_data], lr=lr,
                                           max_iter=10, history_size=100)
            scheduler = None
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

        best_loss = float("inf")
        no_improve = 0
        patience = 500

        for it in range(steps):
            if optimizer_type == "lbfgs":
                def closure():
                    optimizer.zero_grad()
                    out = model(dummy_data)
                    loss = criterion(out, target)
                    dg = torch.autograd.grad(loss, attack_params,
                                              create_graph=True,
                                              retain_graph=True,
                                              only_inputs=True)
                    if loss_fn == "mse":
                        gl = mse_grad_loss(dg, true_grads)
                    elif loss_fn == "cosine":
                        gl = cosine_grad_loss(dg, true_grads)
                    elif loss_fn == "cosine+mse":
                        gl = cosine_grad_loss(dg, true_grads) + mse_grad_loss(dg, true_grads)
                    else:
                        gl = mse_grad_loss(dg, true_grads)
                    total = gl
                    if tv_weight > 0:
                        total = total + tv_weight * total_variation(dummy_data)
                    total.backward()
                    return total
                cur_val = optimizer.step(closure).item()
            else:
                optimizer.zero_grad()
                out = model(dummy_data)
                loss = criterion(out, target)
                dg = torch.autograd.grad(loss, attack_params,
                                          create_graph=True,
                                          retain_graph=True,
                                          only_inputs=True)
                if loss_fn == "mse":
                    gl = mse_grad_loss(dg, true_grads)
                elif loss_fn == "cosine":
                    gl = cosine_grad_loss(dg, true_grads)
                elif loss_fn == "cosine+mse":
                    gl = cosine_grad_loss(dg, true_grads) + mse_grad_loss(dg, true_grads)
                else:
                    gl = mse_grad_loss(dg, true_grads)

                total = gl
                if tv_weight > 0:
                    total = total + tv_weight * total_variation(dummy_data)
                cur_val = total.item()
                total.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            # Pixel clamping: project dummy_data to plausible normalized range
            if do_clamp and mean_t is not None:
                with torch.no_grad():
                    # clamp in pixel space [0,1], then convert back to normalized
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
            ds_cfg = cnn_data.DATASET_CONFIG[DATASET]
            mt = torch.tensor(ds_cfg["mean"], device=device).view(1, -1, 1, 1)
            st = torch.tensor(ds_cfg["std"], device=device).view(1, -1, 1, 1)
            recon = (dummy_data * st + mt).clamp(0, 1)
            p = compute_psnr_batch(recon, real_for_eval)
            s = compute_ssim_batch(recon, real_for_eval)
            print(f"  [restart {restart+1}] PSNR={p:.2f} dB, SSIM={s:.4f}")
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

# ========== Run all methods ==========
methods = {
    "DLG (MSE+Adam)": dict(
        loss_fn="mse", optimizer_type="adam", lr=0.1, steps=4000,
        tv_weight=0.0, n_restarts=5, do_clamp=False),
    "InvGrad (Cos+TV)": dict(
        loss_fn="cosine", optimizer_type="adam", lr=0.1, steps=6000,
        tv_weight=1e-4, n_restarts=5, do_clamp=False),
    "L-BFGS (MSE)": dict(
        loss_fn="mse", optimizer_type="lbfgs", lr=1.0, steps=2000,
        tv_weight=0.0, n_restarts=5, do_clamp=False),
    "Cos+MSE+TV": dict(
        loss_fn="cosine+mse", optimizer_type="adam", lr=0.1, steps=6000,
        tv_weight=1e-4, n_restarts=5, do_clamp=False),
    "Cos+StrongTV": dict(
        loss_fn="cosine", optimizer_type="adam", lr=0.1, steps=6000,
        tv_weight=1e-2, n_restarts=5, do_clamp=False),
    "Adam+Clamp": dict(
        loss_fn="cosine", optimizer_type="adam", lr=0.1, steps=6000,
        tv_weight=1e-4, n_restarts=5, do_clamp=True),
}

results = {}
for mname, kw in methods.items():
    dummy = run_method(mname, model, target, true_grads, DEVICE,
                       real_for_eval=real_unnorm, mean_t=mean_t, std_t=std_t, **kw)
    recon = unnorm(dummy)
    psnr = compute_psnr_batch(recon, real_unnorm)
    ssim = compute_ssim_batch(recon, real_unnorm)
    results[mname] = (psnr, ssim)
    save_batch_img(recon, f"{OUTDIR}/{mname.replace(' ', '_').replace('+', '_').replace('(', '').replace(')', '')}.png")

# Also run HLCP+GIA as reference
from cnn import gia_reconstruct_batch
print(f"\n{'='*60}")
print(f"HLCP+GIA (reference, 3 restarts)")
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

feat_psnr = compute_psnr_batch(best_feat, real_unnorm)
feat_ssim = compute_ssim_batch(best_feat, real_unnorm)
results["HLCP+GIA"] = (feat_psnr, feat_ssim)

# Summary
print(f"\n{'='*60}")
print(f"SUMMARY: CIFAR-100, bs=10, distinct labels, no pool")
print(f"{'='*60}")
print(f"{'Method':<25s} {'PSNR (dB)':>10s} {'SSIM':>8s}")
print(f"{'-'*45}")
for mname, (p, s) in results.items():
    print(f"{mname:<25s} {p:>10.2f} {s:>8.4f}")

save_batch_img(real_unnorm, f"{OUTDIR}/real.png")
save_batch_img(best_feat, f"{OUTDIR}/hlcp_gia.png")

# Save results to text
with open(f"{OUTDIR}/results.txt", "w") as f:
    f.write("CIFAR-100, bs=10, 10 distinct labels, no pool\n")
    f.write(f"{'Method':<25s} {'PSNR (dB)':>10s} {'SSIM':>8s}\n")
    f.write(f"{'-'*45}\n")
    for mname, (p, s) in results.items():
        f.write(f"{mname:<25s} {p:>10.2f} {s:>8.4f}\n")

print(f"\nResults saved to {OUTDIR}/results.txt")
