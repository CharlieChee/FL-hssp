#!/usr/bin/env python3
"""
Unified reconstruction comparison: Adam+Clamp GIA vs HLCP+GIA.
Usage: python -u run_all_recon_compare.py --dataset cifar10 --pool 0 --label_dist same --label_known 1
  runs bs=10,20,40 for given (dataset, pool, label_dist, label_known).
Total: 2 datasets * 2 pool * 2 label_dist * 2 label_known = 16 jobs, each does 3 batch sizes.
"""
import os, argparse, random, torch, numpy as np, json
import torch.nn as nn
import torch.nn.functional as F
import matplotlib as mpl; mpl.use("Agg")

SEED = 2026
SAME_LABEL_CIFAR10 = 3
SAME_LABEL_CIFAR100 = 3

from _common import cosine_grad_loss, save_batch_img, total_variation
from cnn import BasicCNN, get_loaders, gia_reconstruct_batch, gia_reconstruct_batch_unknown_label
import cnn_data
from cnn_metrics import compute_psnr_batch, compute_ssim_batch


def adam_clamp_known(model, target, true_grads, device, mean_t, std_t,
                     real_for_eval, steps=6000, lr=0.1, tv_weight=1e-4, n_restarts=5):
    model.eval()
    ap = [p for p in model.parameters() if p.requires_grad]
    ic, isz, bs = model.conv1.in_channels, model.img_size, target.shape[0]
    crit = nn.CrossEntropyLoss()
    best_dummy, best_psnr = None, -float("inf")
    for r in range(n_restarts):
        torch.manual_seed(SEED + r*1000 + 7777)
        dd = torch.randn((bs, ic, isz, isz), device=device, requires_grad=True)
        opt = torch.optim.Adam([dd], lr=lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-5)
        bl, ni, pat = float("inf"), 0, 500
        for it in range(steps):
            opt.zero_grad()
            out = model(dd); loss = crit(out, target)
            dg = torch.autograd.grad(loss, ap, create_graph=True, retain_graph=True, only_inputs=True)
            total = cosine_grad_loss(dg, true_grads) + tv_weight * total_variation(dd)
            cv = total.item(); total.backward(); opt.step(); sch.step()
            with torch.no_grad():
                px = dd * std_t + mean_t; px.clamp_(0,1); dd.data.copy_((px - mean_t) / std_t)
            if cv < bl - 1e-8: bl, ni = cv, 0
            else: ni += 1
            if (it+1) % 2000 == 0: print(f"    step {it+1}/{steps}, loss={cv:.6e}")
            if ni >= pat: print(f"    early stop at step {it+1}"); break
        with torch.no_grad():
            rec = (dd * std_t + mean_t).clamp(0,1)
            p = compute_psnr_batch(rec, real_for_eval); s = compute_ssim_batch(rec, real_for_eval)
            print(f"  [restart {r+1}] PSNR={p:.2f}, SSIM={s:.4f}")
            if p > best_psnr: best_psnr, best_dummy = p, dd.detach().clone()
    return best_dummy


def adam_clamp_unknown(model, true_grads, device, mean_t, std_t,
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
            if (it+1) % 2000 == 0: print(f"    step {it+1}/{steps}, loss={cv:.6e}")
            if ni >= pat: print(f"    early stop at step {it+1}"); break
        with torch.no_grad():
            rec = (dd * std_t + mean_t).clamp(0,1)
            p = compute_psnr_batch(rec, real_for_eval); s = compute_ssim_batch(rec, real_for_eval)
            print(f"  [restart {r+1}] PSNR={p:.2f}, SSIM={s:.4f}")
            if p > best_psnr: best_psnr, best_dummy = p, dd.detach().clone()
    return best_dummy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["cifar10", "cifar100"])
    parser.add_argument("--pool", type=int, required=True, choices=[0, 1])
    parser.add_argument("--label_dist", required=True, choices=["same", "mixed"])
    parser.add_argument("--label_known", type=int, required=True, choices=[0, 1])
    args = parser.parse_args()

    DATASET = args.dataset
    USE_POOL = bool(args.pool)
    LABEL_DIST = args.label_dist
    LABEL_KNOWN = bool(args.label_known)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pool_str = "pool" if USE_POOL else "nopool"
    known_str = "known" if LABEL_KNOWN else "unknown"
    tag = f"{DATASET}_{pool_str}_{LABEL_DIST}_{known_str}"
    OUTDIR = f"expdata/recon_compare/{tag}"
    os.makedirs(OUTDIR, exist_ok=True)

    print(f"{'#'*70}")
    print(f"# {tag}")
    print(f"{'#'*70}")

    ds_cfg = cnn_data.DATASET_CONFIG[DATASET]
    mean_t = torch.tensor(ds_cfg["mean"], dtype=torch.float32, device=DEVICE).view(1,-1,1,1)
    std_t = torch.tensor(ds_cfg["std"], dtype=torch.float32, device=DEVICE).view(1,-1,1,1)
    def unnorm(x): return (x * std_t + mean_t).clamp(0,1)

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    train_loader, _, cfg = get_loaders(DATASET, batch_size=500, num_workers=0, download=False)
    all_data, all_labels = [], []
    for db, tb in train_loader: all_data.append(db); all_labels.append(tb)
    all_data = torch.cat(all_data); all_labels = torch.cat(all_labels)

    same_label = SAME_LABEL_CIFAR10 if DATASET == "cifar10" else SAME_LABEL_CIFAR100
    num_classes = cfg["num_classes"]

    results_all = {}

    for BS in [10, 20, 40]:
        print(f"\n{'='*60}")
        print(f"BS={BS}, {tag}")
        print(f"{'='*60}")

        # Select data
        if LABEL_DIST == "same":
            mask = (all_labels == same_label)
            pool_data = all_data[mask]
            data = pool_data[:BS].to(DEVICE)
            target = torch.full((BS,), same_label, dtype=torch.long, device=DEVICE)
        else:  # mixed
            torch.manual_seed(SEED + BS)
            idx = torch.randperm(len(all_data))[:BS]
            data = all_data[idx].to(DEVICE)
            target = all_labels[idx].to(DEVICE)

        print(f"Labels: {target.cpu().tolist()}")
        print(f"Unique: {len(target.unique())}/{BS}")

        # Model
        torch.manual_seed(SEED + 100)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED + 100)
        model = BasicCNN(in_channels=cfg["in_channels"], img_size=cfg["img_size"],
                         num_classes=num_classes, use_bn=False, use_pool=USE_POOL).to(DEVICE)
        model.train()
        for p in model.parameters(): p.grad = None
        out = model(data); loss = nn.CrossEntropyLoss()(out, target); loss.backward()
        true_grads = [p.grad.detach().clone() for p in model.parameters() if p.requires_grad]
        fc_input = model.last_feature.detach().clone()
        real_unnorm = unnorm(data)

        # === Adam+Clamp ===
        print(f"\n--- Adam+Clamp ({known_str}) ---")
        if LABEL_KNOWN:
            dc = adam_clamp_known(model, target, true_grads, DEVICE, mean_t, std_t, real_unnorm)
        else:
            dc = adam_clamp_unknown(model, true_grads, DEVICE, mean_t, std_t, real_unnorm,
                                    batch_size=BS, num_classes=num_classes)
        rc = unnorm(dc)

        # === HLCP+GIA ===
        print(f"\n--- HLCP+GIA ({known_str}) ---")
        bf, bfp = None, -float("inf")
        for r in range(3):
            if LABEL_KNOWN:
                d = gia_reconstruct_batch(model=model, target=target, true_grads=true_grads,
                    real_feat=fc_input.to(DEVICE), feat_lambda=1.0, known_rate=1.0,
                    steps=4000, lr=0.1, device=DEVICE)
            else:
                d = gia_reconstruct_batch_unknown_label(model=model, true_grads=true_grads,
                    real_feat=fc_input.to(DEVICE), feat_lambda=1.0, known_rate=1.0,
                    steps=4000, lr=0.1, device=DEVICE, batch_size=BS, num_classes=num_classes)
            rec = unnorm(d); p = compute_psnr_batch(rec, real_unnorm)
            print(f"  [restart {r+1}] PSNR={p:.2f}")
            if p > bfp: bfp, bf = p, rec.clone()

        # Metrics
        cp = compute_psnr_batch(rc, real_unnorm); cs = compute_ssim_batch(rc, real_unnorm)
        hp = compute_psnr_batch(bf, real_unnorm); hs = compute_ssim_batch(bf, real_unnorm)

        print(f"\nResults: {tag}, bs={BS}")
        print(f"  Adam+Clamp: PSNR={cp:.2f}, SSIM={cs:.4f}")
        print(f"  HLCP+GIA:   PSNR={hp:.2f}, SSIM={hs:.4f}")

        results_all[f"bs{BS}"] = {
            "clamp_psnr": round(float(cp), 2), "clamp_ssim": round(float(cs), 4),
            "hlcp_psnr": round(float(hp), 2), "hlcp_ssim": round(float(hs), 4),
        }

        # Save images
        save_batch_img(real_unnorm, f"{OUTDIR}/real_bs{BS}.png")
        save_batch_img(rc, f"{OUTDIR}/clamp_bs{BS}.png")
        save_batch_img(bf, f"{OUTDIR}/hlcp_bs{BS}.png")
        print(f"  Saved to {OUTDIR}/")

    # Save summary JSON
    with open(f"{OUTDIR}/results.json", "w") as f:
        json.dump({"tag": tag, "results": results_all}, f, indent=2)
    print(f"\nDone: {tag}")
    print(json.dumps(results_all, indent=2))


if __name__ == "__main__":
    main()
