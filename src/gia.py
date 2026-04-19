"""GIA (gradient inversion attack) reconstruction pipeline.

Two main reconstruction routines, each driven by ``cnn.py``:

  - ``gia_reconstruct_batch``                   — known-label DLG-style attack.
  - ``gia_reconstruct_batch_unknown_label``     — joint optimisation of images and
                                                  one-hot label distributions.

Both compare three objectives in one call:

  - ``naive``                : pure gradient matching (``L_grad``).
  - ``feat`` / ``hssp+gia``  : ``L_grad + lambda * MSE(real_feat, dummy_feat)``,
                                where ``real_feat`` is FC1's input that the
                                lattice attack would recover.
  - probe / norm variants    : reuse the same code path with different scaling.

The ``run_gia_demo*`` functions wrap an entire demo (data sampling, model init,
both reconstructions, metric reporting, optional figure dumping) for one
``(batch_size, run_idx)`` configuration. They are dispatched by ``cnn.py``'s
multiprocessing pool worker ``_gia_single_run_worker``.
"""
import gc
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image

import cnn_data
from cnn_data import get_loaders
from cnn_metrics import compute_psnr_batch, compute_ssim_batch, compute_fid, FID_ENABLED
from cnn_model import BasicCNN

# GIA mode names. ``ALL_GIA_MODES`` includes the unknown-label variant.
KNOWN_GIA_MODES = ("gia", "gia_loss_probe", "gia_norm")
ALL_GIA_MODES = KNOWN_GIA_MODES + ("gia_unknown",)


def _partial_feat_mse_loss(
    dummy_feat,
    real_feat,
    known_rate,
    feat_row_idx_holder,
    known_residual=False,
):
    """
    FC feature matching: n_known = round(known_rate * B) sample rows are treated as exactly known, and per-row MSE is applied.
    If known_residual=True and 0 < n_known < B: for the remaining |U| unknown samples, also add an MSE term between
    mean(dummy_feat[U]) and mean(real_feat[U]) (the unknown part as a whole).
    If known_residual=True and n_known=0: use only the MSE between mean(dummy_feat) and mean(real_feat) over the whole batch.
    When known_residual=False, keep the original behavior: per-row MSE on known rows only, no supervision on unknown rows.
    """
    if real_feat is None:
        return torch.tensor(0.0, device=dummy_feat.device)
    B = int(real_feat.shape[0])
    if B <= 0:
        return torch.tensor(0.0, device=dummy_feat.device)
    kr = float(max(0.0, min(1.0, known_rate)))
    n_known = int(round(kr * B))
    n_known = max(0, min(B, n_known))

    if n_known <= 0:
        if known_residual:
            return F.mse_loss(dummy_feat.mean(dim=0), real_feat.mean(dim=0))
        return torch.tensor(0.0, device=dummy_feat.device)

    if n_known >= B:
        return F.mse_loss(dummy_feat, real_feat)

    if feat_row_idx_holder.get("idx") is None:
        perm = torch.randperm(B, device=dummy_feat.device)
        feat_row_idx_holder["idx"] = perm[:n_known]
    idx_k = feat_row_idx_holder["idx"]
    row_loss = F.mse_loss(dummy_feat[idx_k], real_feat[idx_k])
    if not known_residual:
        return row_loss

    mask = torch.zeros(B, dtype=torch.bool, device=dummy_feat.device)
    mask[idx_k] = True
    unk_idx = ~mask
    if unk_idx.sum().item() == 0:
        return row_loss
    du = dummy_feat[unk_idx].mean(dim=0)
    ru = real_feat[unk_idx].mean(dim=0)
    agg_loss = F.mse_loss(du, ru)
    return row_loss + agg_loss


def collect_true_batch_and_grads(device, use_bn=False, batch_size=10, use_pool=False, dataset_name="cifar10"):
    """
    Take one batch (default batch_size=10), run a forward + backward pass, and return:
    - model: model after one training step
    - data: real images (B, 3, 32, 32)
    - target: real labels (B,)
    - true_grads: list of parameter gradients produced by this batch on the model

    Assumed setting (strongest attacker):
    - the attacker knows the model architecture and current parameters
    - the attacker knows the real labels of this batch
    """
    # GIA itself already runs in multiple processes; to avoid DataLoader spawning more child
    # processes, num_workers is fixed to 0 here so data is loaded in a single process.
    train_loader, _, cfg = get_loaders(
        dataset_name,
        batch_size=batch_size,
        num_workers=0,
        download=False,
    )

    data, target = next(iter(train_loader))
    data, target = data.to(device), target.to(device)

    model = BasicCNN(
        in_channels=cfg["in_channels"],
        img_size=cfg["img_size"],
        num_classes=cfg["num_classes"],
        use_bn=use_bn,
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
    # Record the FC layer input (the flattened feature from the last conv output).
    real_feat = model.last_feature.detach().clone()
    return model, data.detach(), target.detach(), true_grads, real_feat


def collect_true_batch_and_grads_same_label(
    device,
    use_bn=False,
    batch_size=10,
    label=1,
    use_pool=False,
    dataset_name="cifar10",
):
    """
    Similar to collect_true_batch_and_grads, but forces every sample in the batch to come from the same class
    of the specified dataset. Default label=1 (e.g., automobile in CIFAR-10).
    """
    # Likewise disable DataLoader multi-process loading inside the GIA multi-process setting.
    train_loader, _, cfg = get_loaders(
        dataset_name,
        batch_size=batch_size,
        num_workers=0,
        download=False,
    )
    dataset = train_loader.dataset

    # Handle the field name for targets across different torchvision versions.
    if hasattr(dataset, "targets"):
        targets = np.array(dataset.targets)
    else:
        targets = np.array(dataset.train_labels)

    candidate_indices = np.where(targets == label)[0]
    if len(candidate_indices) < batch_size:
        raise ValueError(
            f"Label {label} has only {len(candidate_indices)} samples, "
            f"but batch_size={batch_size}."
        )

    chosen_indices = np.random.choice(candidate_indices, size=batch_size, replace=False)
    imgs = []
    labels = []
    for idx in chosen_indices:
        img, lab = dataset[idx]
        imgs.append(img)
        labels.append(lab)

    data = torch.stack(imgs, dim=0).to(device)
    target = torch.tensor(labels, dtype=torch.long, device=device)

    model = BasicCNN(
        in_channels=cfg["in_channels"],
        img_size=cfg["img_size"],
        num_classes=cfg["num_classes"],
        use_bn=use_bn,
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


def gia_reconstruct_batch(
    model,
    target,
    true_grads,
    real_feat=None,
    feat_lambda=1.0,
    known_rate=1.0,
    known_residual=False,
    feat_known_idx=None,
    steps=4000,
    lr=0.1,
    device="cpu",
    save_every=None,
    save_dir=None,
    mean=None,
    std=None,
    tag="known_grad",
    snapshots=None,
    loss_probe=None,
    normalize_terms=False,
    norm_eps=1e-8,
):
    """
    Simple DLG-style GIA:
    - keep model and target fixed
    - optimize dummy_data so its gradients approach true_grads as closely as possible

    Required assumptions:
    - the attacker possesses the model architecture and parameters (model)
    - the attacker possesses the real labels of the batch (target)
    - only the gradients true_grads from a single round of local update are used, with no noise or aggregation
    - known_rate: per-row MSE applied on round(known_rate*B) FC rows; 1=full batch; with known_residual,
      add a mean-vector MSE term on unknown rows, and when known_rate=0 only that full-batch mean term is used.
    """
    model = model.to(device)
    model.eval()  # do not update parameters, only compute gradients
    feat_row_idx_holder = {"idx": None}
    if feat_known_idx is not None:
        feat_row_idx_holder["idx"] = feat_known_idx.detach().to(device)

    # Randomly initialize dummy input (channel count matches the first conv to support MNIST/CIFAR).
    in_channels = model.conv1.in_channels
    dummy_data = torch.randn(
        (target.shape[0], in_channels, model.img_size, model.img_size),
        device=device,
        requires_grad=True,
    )
    dummy_labels = target.clone()  # labels are assumed known here; if unknown they can be optimized jointly

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([dummy_data], lr=lr)

    # Parameters whose gradients are to be matched (all trainable parameters).
    attack_params = [p for p in model.parameters() if p.requires_grad]

    best_loss_val = float("inf")
    no_improve = 0
    patience = 300  # early-stop if loss does not decrease for `patience` consecutive steps

    grad_init_raw = None
    feat_init_raw = None
    grad_final_raw = None
    feat_final_raw = None
    used_total_init = None
    used_total_final = None
    last_it = 0

    for it in range(steps):
        last_it = it + 1
        optimizer.zero_grad()

        # Forward pass with dummy input.
        out = model(dummy_data)
        loss = criterion(out, dummy_labels)

        # Compute gradients w.r.t. parameters (do not update parameters; only used for matching).
        dummy_grads = torch.autograd.grad(
            loss,
            attack_params,
            create_graph=True,   # need a graph through gradients to backprop into dummy_data
            retain_graph=True,   # total_loss will be backpropagated again afterward
            only_inputs=True,
        )

        # Gradient matching loss.
        grad_loss = 0.0
        for g_fake, g_true in zip(dummy_grads, true_grads):
            grad_loss = grad_loss + F.mse_loss(g_fake, g_true)

        # If the FC input (last conv feature) is known, add a feature-matching term (optionally on a known_rate subset).
        if real_feat is not None:
            dummy_feat = model.last_feature
            feat_loss = _partial_feat_mse_loss(
                dummy_feat,
                real_feat,
                known_rate,
                feat_row_idx_holder,
                known_residual=known_residual,
            )
        else:
            feat_loss = torch.tensor(0.0, device=device)

        if grad_init_raw is None:
            grad_init_raw = grad_loss.detach()
            feat_init_raw = feat_loss.detach()

        # Experiment 2: normalize each term by its initial value to reduce scale drift between terms.
        if normalize_terms:
            grad_term = grad_loss / (grad_init_raw + norm_eps)
            if real_feat is not None:
                feat_term = feat_loss / (feat_init_raw + norm_eps)
                total_loss = grad_term + feat_lambda * feat_term
            else:
                total_loss = grad_term
        else:
            total_loss = grad_loss + feat_lambda * feat_loss

        cur_loss_val = total_loss.item()
        if used_total_init is None:
            used_total_init = cur_loss_val

        total_loss.backward()
        optimizer.step()

        # Early-stopping logic: stop if total_loss has not decreased for a long time.
        if cur_loss_val < best_loss_val - 1e-8:
            best_loss_val = cur_loss_val
            no_improve = 0
        else:
            no_improve += 1

        if (it + 1) % 50 == 0:
            print(
                f"GIA step {it+1}/{steps}, "
                f"total_loss={cur_loss_val:.6e}, "
                f"grad_loss={grad_loss.item():.6e}, feat_loss={feat_loss.item():.6e}"
            )

        if no_improve >= patience:
            print(
                f"GIA early stop at step {it+1}, best_loss={best_loss_val:.6e}"
            )
            break

        grad_final_raw = grad_loss.detach()
        feat_final_raw = feat_loss.detach()
        used_total_final = cur_loss_val

    # Save only the last iteration (including the early-stop case).
    if save_every is not None and save_dir is not None and last_it > 0:
        with torch.no_grad():
            snap = dummy_data.detach()
            if mean is not None and std is not None:
                snap_vis = snap * std + mean
                snap_vis = snap_vis.clamp(0.0, 1.0)
            else:
                snap_vis = snap
            filename = f"{tag}_step_{last_it}.png"
            save_image(
                snap_vis.cpu(),
                os.path.join(save_dir, filename),
                nrow=snap.shape[0],
            )
            if snapshots is not None:
                snapshots.clear()
                snapshots.append(snap_vis.detach().cpu())

    if grad_final_raw is None:
        grad_final_raw = grad_init_raw if grad_init_raw is not None else torch.tensor(0.0, device=device)
    if feat_final_raw is None:
        feat_final_raw = feat_init_raw if feat_init_raw is not None else torch.tensor(0.0, device=device)
    if used_total_final is None:
        used_total_final = used_total_init if used_total_init is not None else float("nan")

    if loss_probe is not None:
        loss_probe.clear()
        loss_probe.update({
            "grad_init": float(grad_init_raw.item()) if grad_init_raw is not None else float("nan"),
            "grad_final": float(grad_final_raw.item()),
            "lambda_feat_init": float(feat_lambda * feat_init_raw.item()) if feat_init_raw is not None else 0.0,
            "lambda_feat_final": float(feat_lambda * feat_final_raw.item()),
            "used_total_init": float(used_total_init) if used_total_init is not None else float("nan"),
            "used_total_final": float(used_total_final),
            "normalized_objective": int(bool(normalize_terms)),
        })

    return dummy_data.detach()


def gia_reconstruct_batch_unknown_label(
    model,
    true_grads,
    real_feat=None,
    feat_lambda=1.0,
    known_rate=1.0,
    known_residual=False,
    known_peel=False,
    peel_known_idx=None,
    feat_known_idx=None,
    steps=4000,
    lr=0.1,
    device="cpu",
    batch_size=10,
    num_classes=10,
    save_every=None,
    save_dir=None,
    mean=None,
    std=None,
    tag="[unknown]_grad",
    snapshots=None,
):
    """
    DLG-style GIA (unknown-label version):
    - do not use the real labels; labels are also optimization variables (dummy_label_logits)
    - the rest of the pipeline mirrors gia_reconstruct_batch
    - known_rate / known_residual: same semantics as in _partial_feat_mse_loss inside gia_reconstruct_batch
    - known_peel: if True, gradient matching is computed only on "unknown rows" (known rows are peeled out of the gradient mixture)
      peel_known_idx: specifies which rows count as "known rows"; the rest are "unknown rows" (used for masking the gradient-only term)
    """
    model = model.to(device)
    model.eval()
    feat_row_idx_holder = {"idx": None}

    # Randomly initialize dummy input and dummy label logits.
    in_channels = model.conv1.in_channels
    dummy_data = torch.randn(
        (batch_size, in_channels, model.img_size, model.img_size),
        device=device,
        requires_grad=True,
    )
    dummy_label_logits = torch.randn(
        (batch_size, num_classes), device=device, requires_grad=True
    )

    optimizer = optim.Adam([dummy_data, dummy_label_logits], lr=lr)

    # Peel only changes the gradient-matching sample subset; it keeps the forward pass and features intact, preserving the differentiable path through dummy_data.
    attack_params = [p for p in model.parameters() if p.requires_grad]
    if known_peel and peel_known_idx is not None:
        feat_row_idx_holder["idx"] = peel_known_idx.detach().to(device)
    # Fix the "known row" selection used by feature matching (for a fair comparison between partial and residual).
    if feat_known_idx is not None:
        feat_row_idx_holder["idx"] = feat_known_idx.detach().to(device)

    best_loss_val = float("inf")
    no_improve = 0
    patience = 300
    last_it = 0

    for it in range(steps):
        last_it = it + 1
        optimizer.zero_grad()

        out = model(dummy_data)  # (B, num_classes)
        log_probs = F.log_softmax(out, dim=1)
        pseudo_y = F.softmax(dummy_label_logits, dim=1)
        ce_vec = -(pseudo_y * log_probs).sum(dim=1)  # (B,)

        grad_loss = torch.zeros((), device=device)
        if known_peel:
            # unknown_idx = complement(peel_known_idx); used to shrink the gradient mixture to the unknown block.
            known_idx = feat_row_idx_holder.get("idx", None)
            if known_idx is None:
                # If the caller did not pass peel_known_idx, fall back to the original (full-batch) gradient matching.
                ce_loss = ce_vec.mean()
                dummy_grads = torch.autograd.grad(
                    ce_loss,
                    attack_params,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                )
                for g_fake, g_true in zip(dummy_grads, true_grads):
                    grad_loss = grad_loss + F.mse_loss(g_fake, g_true)
            else:
                B = int(ce_vec.shape[0])
                known_idx = known_idx.to(device).view(-1).long()
                mask_unknown = torch.ones(B, dtype=torch.bool, device=device)
                mask_unknown[known_idx] = False
                unknown_idx = torch.nonzero(mask_unknown, as_tuple=False).view(-1)
                if unknown_idx.numel() > 0 and len(true_grads) > 0:
                    ce_loss_u = ce_vec[unknown_idx].mean()
                    dummy_grads = torch.autograd.grad(
                        ce_loss_u,
                        attack_params,
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True,
                    )
                    for g_fake, g_true in zip(dummy_grads, true_grads):
                        grad_loss = grad_loss + F.mse_loss(g_fake, g_true)
                else:
                    grad_loss = torch.zeros((), device=device)
        else:
            ce_loss = ce_vec.mean()
            dummy_grads = torch.autograd.grad(
                ce_loss,
                attack_params,
                create_graph=True,
                retain_graph=True,   # total_loss will be backpropagated again afterward
                only_inputs=True,
            )
            for g_fake, g_true in zip(dummy_grads, true_grads):
                grad_loss = grad_loss + F.mse_loss(g_fake, g_true)

        total_loss = grad_loss

        if real_feat is not None:
            dummy_feat = model.last_feature
            feat_loss = _partial_feat_mse_loss(
                dummy_feat,
                real_feat,
                known_rate,
                feat_row_idx_holder,
                known_residual=known_residual,
            )
            total_loss = total_loss + feat_lambda * feat_loss
        else:
            feat_loss = torch.tensor(0.0, device=device)

        cur_loss_val = total_loss.item()

        total_loss.backward()
        optimizer.step()

        if cur_loss_val < best_loss_val - 1e-8:
            best_loss_val = cur_loss_val
            no_improve = 0
        else:
            no_improve += 1

        if (it + 1) % 50 == 0:
            print(
                f"[unknown label] GIA step {it+1}/{steps}, "
                f"total_loss={cur_loss_val:.6e}, "
                f"grad_loss={grad_loss.item():.6e}, feat_loss={feat_loss.item():.6e}"
            )

        if no_improve >= patience:
            print(
                f"[unknown label] GIA early stop at step {it+1}, best_loss={best_loss_val:.6e}"
            )
            break

    # Save only the last iteration (including the early-stop case).
    if save_every is not None and save_dir is not None and last_it > 0:
        with torch.no_grad():
            snap = dummy_data.detach()
            if mean is not None and std is not None:
                snap_vis = snap * std + mean
                snap_vis = snap_vis.clamp(0.0, 1.0)
            else:
                snap_vis = snap
            filename = f"{tag}_step_{last_it}.png"
            save_image(
                snap_vis.cpu(),
                os.path.join(save_dir, filename),
                nrow=snap.shape[0],
            )
            if snapshots is not None:
                snapshots.clear()
                snapshots.append(snap_vis.detach().cpu())

    return dummy_data.detach()


def run_gia_demo(
    device,
    same_label=False,
    label=1,
    use_pool=False,
    steps=4000,
    batch_size=4,
    dataset_name="cifar10",
    save_fig=False,
    collect_loss_probe=False,
    normalize_objective=False,
    norm_eps=1e-8,
    known_rate=1.0,
    known_residual=False,
):
    """
    Demo flow:
    1. take a real batch (default batch_size=10), compute gradients, record true_grads and FC feature real_feat
    2. run two GIA variants:
       - naive: gradient matching only
       - feat:  gradient matching + FC input feature matching
    3. if save_fig=True, save the real images and both reconstructions to ./expdata/gia/{dataset_name}/
    """
    save_dir = None
    if save_fig:
        base_dir = os.path.join("./expdata/gia", dataset_name)
        os.makedirs(base_dir, exist_ok=True)
        bs = batch_size
        exp_tag = "known"
        exp_tag += "_sameLabel" if same_label else "_mixed"
        exp_tag += "_pool" if use_pool else "_nopool"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(base_dir, f"{timestamp}_{exp_tag}_bs{bs}_steps{steps}")
        os.makedirs(save_dir, exist_ok=True)
    bs = batch_size

    # 1. Collect real data and gradients.
    if same_label:
        # Use a "same-class batch" (all samples drawn from the same class of the same dataset).
        model, real_data, real_target, true_grads, real_feat = collect_true_batch_and_grads_same_label(
            device=device, use_bn=False, batch_size=bs, label=label, use_pool=use_pool, dataset_name=dataset_name
        )
    else:
        # Standard random batch.
        model, real_data, real_target, true_grads, real_feat = collect_true_batch_and_grads(
            device=device, use_bn=False, batch_size=bs, use_pool=use_pool, dataset_name=dataset_name
        )

    # Look up normalization parameters dynamically per dataset to support MNIST/CIFAR-10/CIFAR-100.
    ds_cfg = cnn_data.DATASET_CONFIG[dataset_name]
    mean = torch.tensor(ds_cfg["mean"], dtype=torch.float32, device=device).view(1, -1, 1, 1)
    std = torch.tensor(ds_cfg["std"], dtype=torch.float32, device=device).view(1, -1, 1, 1)

    # Save the real image (after de-normalization); only written when save_fig is set.
    real_unnorm = real_data * std + mean
    real_unnorm = real_unnorm.clamp(0.0, 1.0)
    if save_fig and save_dir:
        save_image(
            real_unnorm.cpu(),
            os.path.join(save_dir, "real_batch.png"),
            nrow=bs,
        )

    # Extra optimization strategy for small batches: longer iterations, smaller learning rate, multiple restarts.
    # Goal: reduce the optimization instability at bs<10 and the "lower scores at low batch sizes" phenomenon.
    attack_steps = steps
    attack_lr = 0.1
    n_restarts = 1
    if bs < 10:
        attack_steps = max(steps, 10000)
        attack_lr = 0.05
        n_restarts = 3
        print(
            f"[GIA][SmallBS] bs={bs} < 10, use steps={attack_steps}, "
            f"lr={attack_lr:g}, restarts={n_restarts}"
        )

    # 2. Run naive GIA (gradient matching only).
    print("Running naive GIA (gradient-only) to reconstruct the batch...")
    best_naive_psnr = -float("inf")
    dummy_data = None
    snaps_grad_only = []
    best_naive_probe = None
    real_eval_for_select = real_unnorm.clamp(0.0, 1.0).to(device)
    for r in range(n_restarts):
        cur_snaps = []
        cur_probe = {}
        cur_dummy = gia_reconstruct_batch(
            model=model,
            target=real_target,
            true_grads=true_grads,
            real_feat=None,          # do not use feature information
            feat_lambda=0.0,
            steps=attack_steps,
            lr=attack_lr,
            device=device,
            save_every=500 if save_fig else None,
            save_dir=save_dir,
            mean=mean,
            std=std,
            tag=f"known_grad_only_r{r+1}",
            snapshots=cur_snaps,
            loss_probe=cur_probe,
            normalize_terms=False,
            norm_eps=norm_eps,
        )
        cur_unnorm = (cur_dummy * std + mean).clamp(0.0, 1.0)
        cur_psnr = compute_psnr_batch(cur_unnorm.to(device), real_eval_for_select)
        if cur_psnr > best_naive_psnr:
            best_naive_psnr = cur_psnr
            dummy_data = cur_dummy
            snaps_grad_only = cur_snaps
            best_naive_probe = cur_probe
    print(f"[GIA][SmallBS] best naive restart PSNR={best_naive_psnr:.4f} dB")

    # 3. Save reconstruction results (after de-normalization).
    dummy_unnorm = dummy_data * std + mean
    dummy_unnorm = dummy_unnorm.clamp(0.0, 1.0)
    if save_fig and save_dir:
        save_image(
            dummy_unnorm.cpu(),
            os.path.join(save_dir, "reconstructed_batch_grad_only_final.png"),
            nrow=bs,
        )

    # 3. Run GIA with the FC feature prior.
    print("Running GIA with feature matching (gradient + FC input) ...")
    # Fix the "known row" selection shared by partial and residual so both use the same row subset for per-row supervision.
    feat_known_idx_partial = None
    if known_rate > 0.0 and known_rate < 1.0:
        nk_partial = max(0, min(bs, int(round(float(known_rate) * bs))))
        feat_perm = torch.randperm(bs, device=device)
        feat_known_idx_partial = feat_perm[:nk_partial]

    def _run_feat_known(use_residual, known_rate_run, feat_known_idx_run, save_basename, tag_prefix):
        best_psnr = -float("inf")
        best_dummy = None
        best_snaps = []
        best_probe = None
        k_run = float(max(0.0, min(1.0, known_rate_run)))
        for r in range(n_restarts):
            cur_snaps = []
            cur_probe = {}
            cur_dummy_feat = gia_reconstruct_batch(
                model=model,
                target=real_target,
                true_grads=true_grads,
                real_feat=real_feat.to(device),
                feat_lambda=1.0,
                known_rate=k_run,
                known_residual=use_residual,
                feat_known_idx=feat_known_idx_run,
                steps=attack_steps,
                lr=attack_lr,
                device=device,
                save_every=500 if save_fig else None,
                save_dir=save_dir,
                mean=mean,
                std=std,
                tag=f"{tag_prefix}_r{r+1}",
                snapshots=cur_snaps,
                loss_probe=cur_probe,
                normalize_terms=normalize_objective,
                norm_eps=norm_eps,
            )
            cur_feat_unnorm = (cur_dummy_feat * std + mean).clamp(0.0, 1.0)
            cur_psnr = compute_psnr_batch(cur_feat_unnorm.to(device), real_eval_for_select)
            if cur_psnr > best_psnr:
                best_psnr = cur_psnr
                best_dummy = cur_dummy_feat
                best_snaps = cur_snaps
                best_probe = cur_probe
        print(f"[GIA][SmallBS] best feat restart PSNR={best_psnr:.4f} dB [{tag_prefix}]")
        best_unnorm = (best_dummy * std + mean).clamp(0.0, 1.0)
        if save_fig and save_dir:
            save_image(
                best_unnorm.cpu(),
                os.path.join(save_dir, save_basename),
                nrow=bs,
            )
        return best_dummy, best_unnorm, best_snaps, best_probe

    # partial-only (per-row known rows).
    dummy_data_feat, dummy_feat_unnorm, snaps_grad_feat, best_feat_probe = _run_feat_known(
        use_residual=False,
        known_rate_run=known_rate,
        feat_known_idx_run=feat_known_idx_partial,
        save_basename="reconstructed_batch_grad_plus_feat_partial_final.png",
        tag_prefix="known_grad_plus_feat_partial",
    )
    # oracle (known_rate=1, per-row MSE on all rows, upper-bound reference).
    dummy_data_feat_k1, dummy_feat_k1_unnorm, snaps_grad_feat_k1, best_feat_probe_k1 = None, None, [], None
    need_oracle = float(known_rate) < 1.0 - 1e-15
    if need_oracle:
        dummy_data_feat_k1, dummy_feat_k1_unnorm, snaps_grad_feat_k1, best_feat_probe_k1 = _run_feat_known(
            use_residual=False,
            known_rate_run=1.0,
            feat_known_idx_run=None,
            save_basename="reconstructed_batch_grad_plus_feat_k1_final.png",
            tag_prefix="known_grad_plus_feat_k1",
        )
    # residual compare (per-row + unknown-mean term), meaningful only when 0<known_rate<1.
    dummy_data_feat_res, dummy_feat_res_unnorm, snaps_grad_feat_res, best_feat_probe_res = None, None, [], None
    need_residual_compare = float(known_rate) > 0.0 and float(known_rate) < 1.0
    if need_residual_compare:
        dummy_data_feat_res, dummy_feat_res_unnorm, snaps_grad_feat_res, best_feat_probe_res = _run_feat_known(
            use_residual=True,
            known_rate_run=known_rate,
            feat_known_idx_run=feat_known_idx_partial,
            save_basename="reconstructed_batch_grad_plus_feat_residual_final.png",
            tag_prefix="known_grad_plus_feat_residual",
        )

    # 4. Generate summary images (only when save_fig).
    if save_fig and save_dir:
        # overall: real + grad-only + partial (+ residual) (+ oracle k=1, as ideal upper bound)
        overall_rows = [real_unnorm.cpu(), dummy_unnorm.cpu(), dummy_feat_unnorm.cpu()]
        if need_residual_compare and dummy_feat_res_unnorm is not None:
            overall_rows.append(dummy_feat_res_unnorm.cpu())
        if need_oracle and dummy_feat_k1_unnorm is not None:
            overall_rows.append(dummy_feat_k1_unnorm.cpu())
        summary_overall = torch.cat(overall_rows, dim=0)
        save_image(
            summary_overall,
            os.path.join(save_dir, "summary_known_overall.png"),
            nrow=bs,
        )
        if snaps_grad_only:
            grid_tensors = [real_unnorm.cpu()] + snaps_grad_only
            summary = torch.cat(grid_tensors, dim=0)
            save_image(
                summary,
                os.path.join(save_dir, "summary_known_grad_only.png"),
                nrow=bs,
            )
        if snaps_grad_feat:
            grid_tensors = [real_unnorm.cpu()] + snaps_grad_feat
            summary = torch.cat(grid_tensors, dim=0)
            save_image(
                summary,
                os.path.join(save_dir, "summary_known_grad_plus_feat.png"),
                nrow=bs,
            )
        if need_oracle and snaps_grad_feat_k1:
            grid_tensors = [real_unnorm.cpu()] + snaps_grad_feat_k1
            summary = torch.cat(grid_tensors, dim=0)
            save_image(
                summary,
                os.path.join(save_dir, "summary_known_grad_plus_feat_k1.png"),
                nrow=bs,
            )
        if need_residual_compare and snaps_grad_feat_res:
            grid_tensors = [real_unnorm.cpu()] + snaps_grad_feat_res
            summary = torch.cat(grid_tensors, dim=0)
            save_image(
                summary,
                os.path.join(save_dir, "summary_known_grad_plus_feat_residual.png"),
                nrow=bs,
            )
        print(
            "Saved real_batch.png, reconstructed_batch_grad_only.png and "
            "reconstructed_batch_grad_plus_feat*.png to",
            save_dir,
        )

    # 5. Compute and print only the "final" metrics: PSNR / SSIM / FID.
    device_eval = device
    real_eval = real_unnorm.clamp(0.0, 1.0).to(device_eval)

    def eval_and_print(name, recon_tensor):
        recon_eval = recon_tensor.clamp(0.0, 1.0).to(device_eval)
        psnr = compute_psnr_batch(recon_eval, real_eval)
        ssim = compute_ssim_batch(recon_eval, real_eval)
        fid = compute_fid(real_eval, recon_eval, device_eval)
        if FID_ENABLED:
            print(
                f"[Metrics][{name}] PSNR={psnr:.4f} dB, "
                f"SSIM={ssim:.4f}, FID={fid:.4f}"
            )
        else:
            print(
                f"[Metrics][{name}] PSNR={psnr:.4f} dB, SSIM={ssim:.4f}"
            )
        return psnr, ssim, fid

    psnr_grad_only, ssim_grad_only, fid_grad_only = eval_and_print(
        "grad_only_final", dummy_unnorm
    )
    psnr_feat, ssim_feat, fid_feat = eval_and_print(
        "grad_plus_feat_final", dummy_feat_unnorm
    )
    psnr_feat_k1, ssim_feat_k1, fid_feat_k1 = None, None, None
    if need_oracle and dummy_feat_k1_unnorm is not None:
        psnr_feat_k1, ssim_feat_k1, fid_feat_k1 = eval_and_print(
            "grad_plus_feat_oracle | known_rate=1", dummy_feat_k1_unnorm
        )
    psnr_feat_res, ssim_feat_res, fid_feat_res = None, None, None
    if need_residual_compare and dummy_feat_res_unnorm is not None:
        psnr_feat_res, ssim_feat_res, fid_feat_res = eval_and_print(
            "grad_plus_feat_partial+known_residual_final", dummy_feat_res_unnorm
        )

    # Return final metrics so multi-run statistics are easy to aggregate.
    ret = {
        "grad_only_final": {
            "psnr": psnr_grad_only,
            "ssim": ssim_grad_only,
            "fid": fid_grad_only,
        },
        "grad_plus_feat_final": {
            "psnr": psnr_feat,
            "ssim": ssim_feat,
            "fid": fid_feat,
        },
    }
    if need_oracle and psnr_feat_k1 is not None:
        ret["grad_plus_feat_oracle_final"] = {
            "psnr": psnr_feat_k1,
            "ssim": ssim_feat_k1,
            "fid": fid_feat_k1,
        }
    if need_residual_compare and psnr_feat_res is not None:
        ret["grad_plus_feat_residual_final"] = {
            "psnr": psnr_feat_res,
            "ssim": ssim_feat_res,
            "fid": fid_feat_res,
        }
    if collect_loss_probe:
        ret["loss_probe"] = {
            "naive": best_naive_probe if best_naive_probe is not None else {},
            "feat_partial": best_feat_probe if best_feat_probe is not None else {},
            "feat_oracle": best_feat_probe_k1 if best_feat_probe_k1 is not None else {},
            "feat_residual": best_feat_probe_res if best_feat_probe_res is not None else {},
        }
    return ret


def run_gia_demo_unknown_label(
    device,
    same_label=False,
    label=1,
    use_pool=False,
    steps=4000,
    batch_size=4,
    dataset_name="cifar10",
    save_fig=False,
    noise_on_fc=0.0,
    known_rate=1.0,
    known_peel=False,
):
    """
    GIA demo for the unknown-label scenario:
    1. still use a real batch to generate true_grads and real_feat (the attacker does not know the label, but the server does)
    2. on the attacker side, do not use the real labels and rely only on gradients:
       - naive: gradient matching only
       - partial: gradient + FC features (per-row MSE on the known_rate row subset only)
       - oracle: known_rate=1, per-row MSE on all rows (additionally run when known_rate<1)
       - +residual: same known_rate as partial, but with a mean-residual term on unknown rows (additionally run when known_rate<1)
    3. if save_fig=True, save the real images and reconstructions to ./expdata/gia_unknown/{dataset_name}/
    """
    save_dir = None
    if save_fig:
        base_dir = os.path.join("./expdata/gia_unknown", dataset_name)
        os.makedirs(base_dir, exist_ok=True)
        bs = batch_size
        exp_tag = "unknown"
        exp_tag += "_sameLabel" if same_label else "_mixed"
        exp_tag += "_pool" if use_pool else "_nopool"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(base_dir, f"{timestamp}_{exp_tag}_bs{bs}_steps{steps}")
        os.makedirs(save_dir, exist_ok=True)
    bs = batch_size
    kr = float(max(0.0, min(1.0, known_rate)))

    if same_label:
        # Use a "same-class batch" (all samples drawn from the same class of the same dataset).
        model, real_data, real_target, true_grads, real_feat = collect_true_batch_and_grads_same_label(
            device=device, use_bn=False, batch_size=bs, label=label, use_pool=use_pool, dataset_name=dataset_name
        )
    else:
        # Standard random batch.
        model, real_data, real_target, true_grads, real_feat = collect_true_batch_and_grads(
            device=device, use_bn=False, batch_size=bs, use_pool=use_pool, dataset_name=dataset_name
        )

    ds_cfg = cnn_data.DATASET_CONFIG[dataset_name]
    mean = torch.tensor(ds_cfg["mean"], dtype=torch.float32, device=device).view(1, -1, 1, 1)
    std = torch.tensor(ds_cfg["std"], dtype=torch.float32, device=device).view(1, -1, 1, 1)
    num_classes = int(ds_cfg["num_classes"])

    # known_peel: peel the known rows out of the gradient mixture (match gradients only on unknown rows).
    peel_perm = None
    peel_known_idx_naive = None
    peel_known_idx_partial = None
    peel_known_idx_oracle = None
    true_grads_unknown_partial = None
    true_grads_unknown_oracle = None
    if known_peel:
        peel_perm = torch.randperm(bs, device=device)

        def _compute_true_grads_for_subset(sub_idx):
            if sub_idx.numel() == 0:
                return []
            for p in model.parameters():
                p.grad = None
            model.train()
            out_sub = model(real_data[sub_idx])
            ce_sub = nn.CrossEntropyLoss()(out_sub, real_target[sub_idx])
            ce_sub.backward()
            grads_sub = [p.grad.detach().clone() for p in model.parameters() if p.requires_grad]
            model.eval()
            return grads_sub

        peel_known_idx_naive = peel_perm[:0]  # known_rate=0 -> all rows are unknown
        n_known_partial = max(0, min(bs, int(round(kr * bs))))
        peel_known_idx_partial = peel_perm[:n_known_partial]
        peel_unknown_idx_partial = peel_perm[n_known_partial:]
        true_grads_unknown_partial = _compute_true_grads_for_subset(peel_unknown_idx_partial)

        peel_known_idx_oracle = peel_perm  # known_rate=1 -> unknown set is empty
        true_grads_unknown_oracle = []

    real_unnorm = real_data * std + mean
    real_unnorm = real_unnorm.clamp(0.0, 1.0)
    if save_fig and save_dir:
        save_image(
            real_unnorm.cpu(),
            os.path.join(save_dir, "real_batch_unknown_label.png"),
            nrow=bs,
        )

    # Extra optimization strategy for small batches: longer iterations, smaller learning rate, multiple restarts.
    attack_steps = steps
    attack_lr = 0.1
    n_restarts = 1
    if bs < 10:
        attack_steps = max(steps, 10000)
        attack_lr = 0.05
        n_restarts = 3
        print(
            f"[GIA][SmallBS] bs={bs} < 10, use steps={attack_steps}, "
            f"lr={attack_lr:g}, restarts={n_restarts}"
        )

    print("Running naive GIA (gradient-only, unknown label) ...")
    best_naive_psnr = -float("inf")
    dummy_data = None
    snaps_unknown_grad = []
    real_eval_for_select = real_unnorm.clamp(0.0, 1.0).to(device)
    for r in range(n_restarts):
        cur_snaps = []
        cur_dummy = gia_reconstruct_batch_unknown_label(
            model=model,
            true_grads=true_grads,
            real_feat=None,
            feat_lambda=0.0,
            known_rate=0.0,
            steps=attack_steps,
            lr=attack_lr,
            device=device,
            batch_size=bs,
            num_classes=num_classes,
            save_every=500 if save_fig else None,
            save_dir=save_dir,
            mean=mean,
            std=std,
            tag=f"unknown_grad_only_r{r+1}",
            snapshots=cur_snaps,
            known_peel=known_peel,
            peel_known_idx=peel_known_idx_naive,
        )
        cur_unnorm = (cur_dummy * std + mean).clamp(0.0, 1.0)
        cur_psnr = compute_psnr_batch(cur_unnorm.to(device), real_eval_for_select)
        if cur_psnr > best_naive_psnr:
            best_naive_psnr = cur_psnr
            dummy_data = cur_dummy
            snaps_unknown_grad = cur_snaps
    print(f"[GIA][SmallBS] best unknown naive restart PSNR={best_naive_psnr:.4f} dB")

    dummy_unnorm = dummy_data * std + mean
    dummy_unnorm = dummy_unnorm.clamp(0.0, 1.0)
    if save_fig and save_dir:
        save_image(
            dummy_unnorm.cpu(),
            os.path.join(save_dir, "reconstructed_batch_unknown_grad_only_final.png"),
            nrow=bs,
        )

    # Add noise to the FC input feature (to simulate HSSP estimation error); sigma=0 reduces to the original HSSP+GIA.
    if noise_on_fc > 0.0:
        noisy_real_feat = real_feat.to(device) + noise_on_fc * torch.randn_like(real_feat.to(device))
    else:
        noisy_real_feat = real_feat.to(device)

    def _run_feat_unknown(
        known_rate_run,
        save_image_basename,
        step_tag_prefix,
        use_residual,
        peel_known_idx_for_grad=None,
        true_grads_for_grad=None,
        feat_known_idx_for_feat=None,
    ):
        best_psnr = -float("inf")
        best_dummy = None
        best_snaps = []
        k_run = float(max(0.0, min(1.0, known_rate_run)))
        nk = max(0, min(bs, int(round(k_run * bs))))
        res_tag = "+mean_residual_for_unknown_rows" if use_residual else ""
        print(
            f"Running GIA with feature matching (unknown label, +FC input, sigma={noise_on_fc:g}, "
            f"known_rate={k_run:g} -> {nk}/{bs} known rows MSE{res_tag}) [{step_tag_prefix}] ..."
        )
        for r in range(n_restarts):
            cur_snaps = []
            cur_dummy_feat = gia_reconstruct_batch_unknown_label(
                model=model,
                true_grads=true_grads_for_grad if known_peel else true_grads,
                real_feat=noisy_real_feat,
                feat_lambda=1.0,
                known_rate=k_run,
                known_residual=use_residual,
                known_peel=known_peel,
                peel_known_idx=peel_known_idx_for_grad,
                feat_known_idx=feat_known_idx_for_feat,
                steps=attack_steps,
                lr=attack_lr,
                device=device,
                batch_size=bs,
                num_classes=num_classes,
                save_every=500 if save_fig else None,
                save_dir=save_dir,
                mean=mean,
                std=std,
                tag=f"{step_tag_prefix}_r{r+1}",
                snapshots=cur_snaps,
            )
            cur_feat_unnorm = (cur_dummy_feat * std + mean).clamp(0.0, 1.0)
            cur_psnr = compute_psnr_batch(cur_feat_unnorm.to(device), real_eval_for_select)
            if cur_psnr > best_psnr:
                best_psnr = cur_psnr
                best_dummy = cur_dummy_feat
                best_snaps = cur_snaps
        print(f"[GIA][SmallBS] best unknown feat restart PSNR={best_psnr:.4f} dB [{step_tag_prefix}]")
        unnorm = (best_dummy * std + mean).clamp(0.0, 1.0)
        if save_fig and save_dir:
            save_image(
                unnorm.cpu(),
                os.path.join(save_dir, save_image_basename),
                nrow=bs,
            )
        return best_dummy, unnorm, best_snaps

    need_oracle = kr < 1.0 - 1e-15
    need_residual_compare = kr < 1.0 - 1e-15
    # Fix the "known row" selection shared by partial and residual so both use the same row subset for per-row supervision.
    feat_known_idx_partial = None
    if kr > 0.0 and kr < 1.0:
        nk_partial = max(0, min(bs, int(round(float(kr) * bs))))
        feat_perm = torch.randperm(bs, device=device)
        feat_known_idx_partial = feat_perm[:nk_partial]

    # Partial-only (no unknown-row mean term): matches the previous "per-row known only" variant; when kr=0 this has the same objective as naive.
    if kr <= 0.0:
        dummy_data_feat = dummy_data
        dummy_feat_unnorm = dummy_unnorm
        snaps_unknown_feat = list(snaps_unknown_grad)
        print(
            "[GIA] known_rate=0: partial feat (no row feat, no residual) reuses naive reconstruction."
        )
    else:
        dummy_data_feat, dummy_feat_unnorm, snaps_unknown_feat = _run_feat_unknown(
            kr,
            "reconstructed_batch_unknown_grad_plus_feat_final.png",
            "unknown_grad_plus_feat_partial",
            use_residual=False,
            peel_known_idx_for_grad=peel_known_idx_partial if known_peel else None,
            true_grads_for_grad=true_grads_unknown_partial if known_peel else None,
            feat_known_idx_for_feat=feat_known_idx_partial,
        )

    if need_oracle:
        dummy_data_feat_k1, dummy_feat_k1_unnorm, snaps_unknown_feat_k1 = _run_feat_unknown(
            1.0,
            "reconstructed_batch_unknown_grad_plus_feat_k1_final.png",
            "unknown_grad_plus_feat_k1",
            use_residual=False,
            peel_known_idx_for_grad=peel_known_idx_oracle if known_peel else None,
            true_grads_for_grad=true_grads_unknown_oracle if known_peel else None,
            feat_known_idx_for_feat=None,
        )
    else:
        dummy_data_feat_k1 = dummy_data_feat
        dummy_feat_k1_unnorm = dummy_feat_unnorm
        snaps_unknown_feat_k1 = list(snaps_unknown_feat)

    dummy_data_feat_res, dummy_feat_res_unnorm, snaps_unknown_feat_res = None, None, []
    if need_residual_compare:
        dummy_data_feat_res, dummy_feat_res_unnorm, snaps_unknown_feat_res = _run_feat_unknown(
            kr,
            "reconstructed_batch_unknown_grad_plus_feat_residual_final.png",
            "unknown_grad_plus_feat_residual",
            use_residual=True,
            peel_known_idx_for_grad=peel_known_idx_partial if known_peel else None,
            true_grads_for_grad=true_grads_unknown_partial if known_peel else None,
            feat_known_idx_for_feat=feat_known_idx_partial,
        )

    # 4. Generate summary images (only when save_fig).
    if save_fig and save_dir:
        overall_rows = [real_unnorm.cpu()]
        if snaps_unknown_grad:
            overall_rows.append(snaps_unknown_grad[-1])
        if snaps_unknown_feat:
            overall_rows.append(snaps_unknown_feat[-1])
        if need_residual_compare and snaps_unknown_feat_res:
            overall_rows.append(snaps_unknown_feat_res[-1])
        if need_oracle and snaps_unknown_feat_k1:
            overall_rows.append(snaps_unknown_feat_k1[-1])
        if len(overall_rows) > 1:
            summary_overall = torch.cat(overall_rows, dim=0)
            save_image(
                summary_overall,
                os.path.join(save_dir, "summary_unknown_overall.png"),
                nrow=bs,
            )
        if snaps_unknown_grad:
            grid_tensors = [real_unnorm.cpu(), snaps_unknown_grad[-1]]
            summary = torch.cat(grid_tensors, dim=0)
            save_image(
                summary,
                os.path.join(save_dir, "summary_unknown_grad_only.png"),
                nrow=bs,
            )
        if snaps_unknown_feat:
            grid_tensors = [real_unnorm.cpu(), snaps_unknown_feat[-1]]
            summary = torch.cat(grid_tensors, dim=0)
            save_image(
                summary,
                os.path.join(save_dir, "summary_unknown_grad_plus_feat.png"),
                nrow=bs,
            )

    # 5. Compute and print metrics only on the "final" reconstruction: PSNR / SSIM / FID.
    device_eval = device
    real_eval = real_unnorm.clamp(0.0, 1.0).to(device_eval)

    def eval_and_print(name, recon_tensor):
        recon_eval = recon_tensor.clamp(0.0, 1.0).to(device_eval)
        psnr = compute_psnr_batch(recon_eval, real_eval)
        ssim = compute_ssim_batch(recon_eval, real_eval)
        fid = compute_fid(real_eval, recon_eval, device_eval)
        if FID_ENABLED:
            print(
                f"[Metrics][{name}] PSNR={psnr:.4f} dB, "
                f"SSIM={ssim:.4f}, FID={fid:.4f}"
            )
        else:
            print(
                f"[Metrics][{name}] PSNR={psnr:.4f} dB, SSIM={ssim:.4f}"
            )
        return psnr, ssim, fid

    cmp_parts = ["naive(grad only)", f"partial(rows only)|known_rate={kr:g}"]
    if need_oracle:
        cmp_parts.append("oracle|known_rate=1")
    if need_residual_compare:
        cmp_parts.append(f"partial+known_residual|known_rate={kr:g}")
    if known_peel:
        cmp_parts = [p + "|peel_grad_unknownRows" for p in cmp_parts]
    print("[Metrics][compare] " + " vs ".join(cmp_parts))
    psnr_grad_only, ssim_grad_only, fid_grad_only = eval_and_print(
        "unknown_grad_only_final", dummy_unnorm
    )
    psnr_feat, ssim_feat, fid_feat = eval_and_print(
        f"unknown_grad_plus_feat_partial | known_rate={kr:g}", dummy_feat_unnorm
    )
    psnr_feat_k1, ssim_feat_k1, fid_feat_k1 = None, None, None
    if need_oracle:
        psnr_feat_k1, ssim_feat_k1, fid_feat_k1 = eval_and_print(
            "unknown_grad_plus_feat_oracle | known_rate=1", dummy_feat_k1_unnorm
        )
    psnr_feat_res, ssim_feat_res, fid_feat_res = None, None, None
    if need_residual_compare:
        psnr_feat_res, ssim_feat_res, fid_feat_res = eval_and_print(
            f"unknown_grad_plus_feat_partial+known_residual | known_rate={kr:g}",
            dummy_feat_res_unnorm,
        )

    if save_fig and save_dir:
        print(
            "Saved real/reconstructed batches, snapshots, summary images and metrics to",
            save_dir,
        )

    # Return final metrics so multi-run statistics are easy to aggregate.
    ret_metrics = {
        "unknown_grad_only_final": {
            "psnr": psnr_grad_only,
            "ssim": ssim_grad_only,
            "fid": fid_grad_only,
        },
        "unknown_grad_plus_feat_final": {
            "psnr": psnr_feat,
            "ssim": ssim_feat,
            "fid": fid_feat,
        },
    }
    if need_oracle and psnr_feat_k1 is not None:
        ret_metrics["unknown_grad_plus_feat_oracle_final"] = {
            "psnr": psnr_feat_k1,
            "ssim": ssim_feat_k1,
            "fid": fid_feat_k1,
        }
    if need_residual_compare and psnr_feat_res is not None:
        ret_metrics["unknown_grad_plus_feat_residual_final"] = {
            "psnr": psnr_feat_res,
            "ssim": ssim_feat_res,
            "fid": fid_feat_res,
        }
    return ret_metrics


def _gia_single_run_worker(args_tuple):
    """
    Multiprocessing worker for a single GIA run.
    args_tuple: (mode, same_label, label_idx, use_pool, gia_steps, batch_size, run_idx, task_idx, gpu_ids, dataset_name, save_fig, noise_on_fc, known_rate, known_residual, known_peel)
    Returns a dict:
      {
        "naive": {"psnr": ..., "ssim": ..., "fid": ...},
        "feat":  {"psnr": ..., "ssim": ..., "fid": ...},
      }
    where:
      - known-label modes: naive=grad_only_final, feat=grad_plus_feat_final
      - unknown-label mode: naive=unknown_grad_only_final, feat=unknown_grad_plus_feat_final
    """
    metrics = None
    m_naive = None
    m_feat = None

    (
        mode,
        same_label,
        label_idx,
        use_pool,
        gia_steps,
        batch_size,
        run_idx,
        task_idx,
        gpu_ids,
        dataset_name,
        save_fig,
        noise_on_fc,
        known_rate,
        known_residual,
        known_peel,
    ) = args_tuple

    # Set a different random seed per process so batches and random numbers differ across workers.
    base_seed = int(datetime.now().timestamp() * 1e6) & 0x7FFFFFFF
    seed = base_seed + task_idx * 9973
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Pick a GPU for each worker (if available); otherwise fall back to CPU.
    if torch.cuda.is_available() and gpu_ids:
        gpu_id = gpu_ids[task_idx % len(gpu_ids)]
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        device_tag = f"cuda:{gpu_id}"
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_tag = str(device)

    # Real-time progress line for sweep monitoring.
    print(
        f"[GIA][TaskStart] mode={mode} dataset={dataset_name} bs={batch_size} "
        f"run={run_idx + 1} same_label={int(bool(same_label))} pool={int(bool(use_pool))} "
        f"sigma={noise_on_fc:g} known_rate={known_rate:g} "
        f"known_residual={int(bool(known_residual))} known_peel={int(bool(known_peel))} device={device_tag}"
    )

    # If same_label=True and label_idx is not provided, randomly select a class index for this run.
    # Look up the number of classes dynamically per dataset to avoid wrong assumptions on MNIST/CIFAR100.
    effective_label_idx = label_idx
    if same_label and label_idx is None:
        max_classes = int(cnn_data.DATASET_CONFIG[dataset_name]["num_classes"])
        effective_label_idx = random.randint(0, max_classes - 1)
        print(
            f"[Run {run_idx + 1}] same_label mode with no label_idx provided; "
            f"this run randomly selected label_idx={effective_label_idx}"
        )

    try:
        if mode in KNOWN_GIA_MODES:
            metrics = run_gia_demo(
                device=device,
                same_label=same_label,
                label=effective_label_idx,
                use_pool=use_pool,
                steps=gia_steps,
                batch_size=batch_size,
                dataset_name=dataset_name,
                save_fig=save_fig,
                collect_loss_probe=(mode in ("gia_loss_probe", "gia_norm")),
                normalize_objective=(mode == "gia_norm"),
                known_rate=known_rate,
                known_residual=known_residual,
            )
            naive_key = "grad_only_final"
            feat_key = "grad_plus_feat_final"
        elif mode == "gia_unknown":
            metrics = run_gia_demo_unknown_label(
                device=device,
                same_label=same_label,
                label=effective_label_idx,
                use_pool=use_pool,
                steps=gia_steps,
                batch_size=batch_size,
                dataset_name=dataset_name,
                save_fig=save_fig,
                noise_on_fc=noise_on_fc,
                known_rate=known_rate,
                known_peel=known_peel,
            )
            naive_key = "unknown_grad_only_final"
            feat_key = "unknown_grad_plus_feat_final"
        else:
            raise ValueError(f"Unsupported mode for GIA worker: {mode}")

        m_naive = metrics[naive_key]
        m_feat = metrics[feat_key]
        m_oracle = metrics.get("unknown_grad_plus_feat_oracle_final") if mode == "gia_unknown" else None
        m_res = metrics.get("unknown_grad_plus_feat_residual_final") if mode == "gia_unknown" else None
        if FID_ENABLED:
            print(
                f"[Run {run_idx + 1}][bs={batch_size}][naive] PSNR={m_naive['psnr']:.4f} dB, "
                f"SSIM={m_naive['ssim']:.4f}, FID={m_naive['fid']:.4f}"
            )
            print(
                f"[Run {run_idx + 1}][bs={batch_size}][hssp+gia|sigma={noise_on_fc:g}|known_rate={known_rate:g}] PSNR={m_feat['psnr']:.4f} dB, "
                f"SSIM={m_feat['ssim']:.4f}, FID={m_feat['fid']:.4f}"
            )
            if m_oracle is not None:
                print(
                    f"[Run {run_idx + 1}][bs={batch_size}][hssp+gia|sigma={noise_on_fc:g}|known_rate=1|oracle] PSNR={m_oracle['psnr']:.4f} dB, "
                    f"SSIM={m_oracle['ssim']:.4f}, FID={m_oracle['fid']:.4f}"
                )
            if m_res is not None:
                print(
                    f"[Run {run_idx + 1}][bs={batch_size}][hssp+gia|sigma={noise_on_fc:g}|known_rate={known_rate:g}|known_residual] PSNR={m_res['psnr']:.4f} dB, "
                    f"SSIM={m_res['ssim']:.4f}, FID={m_res['fid']:.4f}"
                )
        else:
            print(
                f"[Run {run_idx + 1}][bs={batch_size}][naive] "
                f"PSNR={m_naive['psnr']:.4f} dB, SSIM={m_naive['ssim']:.4f}"
            )
            print(
                f"[Run {run_idx + 1}][bs={batch_size}][hssp+gia|sigma={noise_on_fc:g}|known_rate={known_rate:g}] "
                f"PSNR={m_feat['psnr']:.4f} dB, SSIM={m_feat['ssim']:.4f}"
            )
            if m_oracle is not None:
                print(
                    f"[Run {run_idx + 1}][bs={batch_size}][hssp+gia|sigma={noise_on_fc:g}|known_rate=1|oracle] "
                    f"PSNR={m_oracle['psnr']:.4f} dB, SSIM={m_oracle['ssim']:.4f}"
                )
            if m_res is not None:
                print(
                    f"[Run {run_idx + 1}][bs={batch_size}][hssp+gia|sigma={noise_on_fc:g}|known_rate={known_rate:g}|known_residual] "
                    f"PSNR={m_res['psnr']:.4f} dB, SSIM={m_res['ssim']:.4f}"
                )
        ret = {"naive": m_naive, "feat": m_feat, "noise_on_fc": noise_on_fc}
        if m_oracle is not None:
            ret["feat_oracle"] = m_oracle
        if m_res is not None:
            ret["feat_residual"] = m_res
        if "loss_probe" in metrics:
            ret["loss_probe"] = metrics["loss_probe"]
        return ret
    finally:
        # Conservative cleanup to reduce per-process GPU memory retention.
        del metrics
        del m_naive
        del m_feat
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


