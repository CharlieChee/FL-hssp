import os
from datetime import datetime
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image

from cnn_data import get_loaders
from cnn_model import BasicCNN
from cnn_metrics import compute_psnr_batch, compute_ssim_batch, compute_fid, FID_ENABLED


# --------------- GIA: DLG-style gradient inversion ---------------


def collect_true_batch_and_grads(device, use_bn=False, batch_size=10, use_pool=False, dataset_name="cifar10"):
    """
    Take one batch (default batch_size=10), run a single forward + backward, and return:
    - model: the model after one training step
    - data: real images (B, 3, 32, 32)
    - target: real labels (B,)
    - true_grads: the list of parameter gradients produced by this batch on the model

    Assumptions (strongest-attacker viewpoint):
    - The attacker knows the model architecture and current parameters.
    - The attacker knows the real labels of this batch.
    """
    # GIA itself already runs in multiple processes; to avoid the DataLoader
    # spawning extra subprocesses, we fix num_workers to 0 to use single-process data loading.
    train_loader, _, cfg = get_loaders(dataset_name, batch_size=batch_size, num_workers=0)

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
    # Record the FC layer input (flattened features from the last conv layer's output)
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
    Similar to collect_true_batch_and_grads, but forces all samples in this batch
    to come from the same class of the specified dataset.
    The default label=1 (e.g., automobile in CIFAR-10).
    """
    # Likewise disable DataLoader multi-process loading under the GIA multi-process scenario
    train_loader, _, cfg = get_loaders(dataset_name, batch_size=batch_size, num_workers=0)
    dataset = train_loader.dataset

    # Be compatible with the different `targets` attribute names across torchvision versions
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
    steps=4000,
    lr=0.1,
    device="cpu",
    save_every=None,
    save_dir=None,
    mean=None,
    std=None,
    tag="known_grad",
    snapshots=None,
):
    """
    Simple DLG-style GIA:
    - Fix model and target.
    - Optimize dummy_data so that its gradients match true_grads as closely as possible.

    Required assumptions:
    - The attacker has the model architecture and parameters (model).
    - The attacker has the real labels of this batch (target).
    - Only the gradients true_grads from a single local update are used, without noise / aggregation masking.
    """
    model = model.to(device)
    model.eval()  # do not update parameters; only compute gradients

    # Randomly initialize the dummy input
    dummy_data = torch.randn(
        (target.shape[0], 3, model.img_size, model.img_size),
        device=device,
        requires_grad=True,
    )
    dummy_labels = target.clone()  # assume labels are known here; otherwise they can be optimized jointly

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([dummy_data], lr=lr)

    # The list of parameters for gradient matching (all trainable parameters)
    attack_params = [p for p in model.parameters() if p.requires_grad]

    best_loss_val = float("inf")
    no_improve = 0
    patience = 300  # early stop if no decrease for `patience` consecutive steps
    last_it = 0

    for it in range(steps):
        last_it = it + 1
        optimizer.zero_grad()

        # Forward with the dummy input
        out = model(dummy_data)
        loss = criterion(out, dummy_labels)

        # Take gradients w.r.t. parameters (do not update parameters; only used for matching)
        dummy_grads = torch.autograd.grad(
            loss,
            attack_params,
            create_graph=True,   # need to build the gradient graph to backprop into dummy_data
            retain_graph=True,   # we still need to backward through total_loss afterwards
            only_inputs=True,
        )

        # Gradient-matching loss
        grad_loss = 0.0
        for g_fake, g_true in zip(dummy_grads, true_grads):
            grad_loss = grad_loss + F.mse_loss(g_fake, g_true)

        total_loss = grad_loss

        # If the FC layer input (last conv-layer feature) is known, add a feature-matching term
        if real_feat is not None:
            dummy_feat = model.last_feature
            feat_loss = F.mse_loss(dummy_feat, real_feat)
            total_loss = total_loss + feat_lambda * feat_loss
        else:
            feat_loss = torch.tensor(0.0, device=device)

        cur_loss_val = total_loss.item()

        total_loss.backward()
        optimizer.step()

        # Early-stop logic: stop if total_loss has not decreased for a long time
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

    # Only save the last iteration (including early-stopped runs)
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


def run_fc_hssp_analysis(device, dataset_name="cifar10", batch_size=10, use_pool=False, run_id=None):
    """
    Gradient-structure analysis utility:
    - Take a single mini-batch and forward through the CNN.
    - Save:
      1) The FC layer input batch matrix X_fc (each row is the flattened feature of one sample).
      2) The ReLU mask matrix R_fc built from the FC output (a 0-1 matrix of shape num_classes x batch_size).
    - Do NOT perform GIA, training, or any other operation.
    - When run_id is not None, save to bs{batch_size}/run_{run_id:04d}/ for multi-run experiments.
    """
    save_dir = os.path.join("./expdata/fc_hssp", dataset_name, f"bs{batch_size}")
    if run_id is not None:
        save_dir = os.path.join(save_dir, "run_{:04d}".format(int(run_id)))
    os.makedirs(save_dir, exist_ok=True)

    # Only datasets already present in DATASET_CONFIG are supported
    # FC HSSP analysis runs in the main process, so we keep DataLoader multi-process loading for speed
    train_loader, _, cfg = get_loaders(dataset_name, batch_size=batch_size, num_workers=2)
    data, target = next(iter(train_loader))
    data, target = data.to(device), target.to(device)

    model = BasicCNN(
        in_channels=cfg["in_channels"],
        img_size=cfg["img_size"],
        num_classes=cfg["num_classes"],
        use_bn=False,
        use_pool=use_pool,
    ).to(device)

    # Forward + backward (via autograd.grad) to obtain:
    # - The gradient of the first FC layer weights dL/dW_fc1 (the gradient transmitted to the server)
    model.train()
    for p in model.parameters():
        p.grad = None

    logits = model(data)  # triggers the computation of last_feature and fc1_relu_mask
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, target)

    # Compute the gradient w.r.t. the FC1 weights
    grad_fc1_weight = torch.autograd.grad(
        loss,
        model.fc1.weight,
        retain_graph=False,
        create_graph=False,
    )[0]

    # 1) The input batch matrix of the first FC layer (flatten -> 1000): X_fc, shape = (B, D)
    fc_input = model.last_feature.detach().cpu().numpy()

    # 2) The ReLU mask matrix of the first FC layer: fc1_relu_mask, shape = (B, 1000), transposed to (1000, B)
    relu_mask_bc = model.fc1_relu_mask.detach().cpu().numpy()  # (batch_size, 1000)
    relu_mask = relu_mask_bc.T  # (fc_hidden_dim=1000, batch_size)

    # 2b) FC1 linear output (pre-ReLU), shape (1000, B)
    fc1_pre_bc = model.fc1_pre_relu.detach().cpu().numpy()  # (B, 1000)
    fc1_pre_relu = fc1_pre_bc.T.astype(np.float64)          # (1000, B)

    # 2c) FC1 post-ReLU output, shape (1000, B)
    fc1_act_bc = model.fc1_act.detach().cpu().numpy()        # (B, 1000)
    fc1_act = fc1_act_bc.T.astype(np.float64)                # (1000, B)

    # 3) The first FC layer weight-gradient matrix: dL/dW_fc1, shape = (fc_hidden_dim, D)
    fc1_weight_grad = grad_fc1_weight.detach().cpu().numpy()

    # 4) True gradient factors: delta and backprop_masked
    #    FC2 gradient: G_W2 = (1/B) * delta^T @ fc1_act_bc
    #    FC1 gradient: G_W1 = (1/B) * backprop_masked^T @ Z
    B = data.shape[0]
    with torch.no_grad():
        softmax_out = torch.softmax(logits, dim=1)                # (B, K)
        onehot = torch.zeros_like(softmax_out)
        onehot.scatter_(1, target.unsqueeze(1), 1.0)
        delta_bc = (softmax_out - onehot).cpu().numpy()           # (B, K)
        delta = delta_bc.T.astype(np.float64)                     # (K, B)

        W2 = model.fc2.weight.detach().cpu().numpy()              # (K, 1000)
        bp = delta_bc @ W2                                        # (B, 1000)
        bp_masked_bc = bp * relu_mask_bc                          # (B, 1000)
        bp_masked = bp_masked_bc.T.astype(np.float64)             # (1000, B)

    np.save(os.path.join(save_dir, "fc_input_batch.npy"), fc_input)
    np.save(os.path.join(save_dir, "relu_mask_fc.npy"), relu_mask)
    np.save(os.path.join(save_dir, "fc1_pre_relu_fc.npy"), fc1_pre_relu)
    np.save(os.path.join(save_dir, "fc1_act_fc.npy"), fc1_act)
    np.save(os.path.join(save_dir, "fc1_weight_grad.npy"), fc1_weight_grad)
    np.save(os.path.join(save_dir, "delta_fc.npy"), delta)
    np.save(os.path.join(save_dir, "backprop_masked_fc.npy"), bp_masked)

    print(f"[FC HSSP] Saved files to {save_dir}:")
    print(f"  fc_input_batch     (Z, FC1 input):        {fc_input.shape}  range=[{fc_input.min():.6f}, {fc_input.max():.6f}]")
    print(f"  relu_mask_fc       (ReLU mask):            {relu_mask.shape}")
    print(f"  fc1_pre_relu_fc    (FC1 pre-ReLU):         {fc1_pre_relu.shape}  neg={int((fc1_pre_relu<0).sum())}/{fc1_pre_relu.size}")
    print(f"  fc1_act_fc         (FC1 post-ReLU):        {fc1_act.shape}  neg={int((fc1_act<0).sum())}/{fc1_act.size}")
    print(f"  delta_fc           (softmax-onehot):       {delta.shape}  neg={int((delta<0).sum())}/{delta.size}")
    print(f"  backprop_masked_fc (bp through FC2*mask):  {bp_masked.shape}  neg={int((bp_masked<0).sum())}/{bp_masked.size}")
    print(f"  fc1_weight_grad    (dL/dW_fc1):            {fc1_weight_grad.shape}")
    print(f"  Verify: G_W1 == (1/B) * bp_masked @ Z ?  max_diff={np.abs(bp_masked @ fc_input - fc1_weight_grad * B).max():.2e}")


def run_gia_demo(device, same_label=False, label=1, use_pool=False, steps=4000, batch_size=4, dataset_name="cifar10", save_fig=False):
    """
    Demo procedure:
    1. Take a real batch (default size 10), compute gradients once, record true_grads and FC input feature real_feat.
    2. Run two GIA variants:
       - naive: only match gradients.
       - feat:  match gradients + FC input feature.
    3. If save_fig=True, save the original images and both reconstructions to ./expdata/gia/{dataset_name}/.
    """
    save_dir = None
    if save_fig:
        base_dir = os.path.join("./expdata/gia", dataset_name)
        os.makedirs(base_dir, exist_ok=True)
        bs = batch_size  # the batch size used by GIA
        exp_tag = "known"
        exp_tag += "_sameLabel" if same_label else "_mixed"
        exp_tag += "_pool" if use_pool else "_nopool"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(base_dir, f"{timestamp}_{exp_tag}_bs{bs}_steps{steps}")
        os.makedirs(save_dir, exist_ok=True)
    bs = batch_size

    # 1. Collect real data and gradients
    if same_label:
        # Use a "same-class batch" (all samples come from the same class of the same dataset)
        model, real_data, real_target, true_grads, real_feat = collect_true_batch_and_grads_same_label(
            device=device, use_bn=False, batch_size=bs, label=label, use_pool=use_pool, dataset_name=dataset_name
        )
    else:
        # A normal random batch
        model, real_data, real_target, true_grads, real_feat = collect_true_batch_and_grads(
            device=device, use_bn=False, batch_size=bs, use_pool=use_pool, dataset_name=dataset_name
        )

    # Normalization parameters (only CIFAR-10 is supported)
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1).to(device)

    # Save the real images (after de-normalization); only written when save_fig is true
    real_unnorm = real_data * std + mean
    real_unnorm = real_unnorm.clamp(0.0, 1.0)
    if save_fig and save_dir:
        save_image(
            real_unnorm.cpu(),
            os.path.join(save_dir, "real_batch.png"),
            nrow=bs,
        )

    # 2. Run naive GIA (gradient-only)
    snaps_grad_only = []
    print("Running naive GIA (gradient-only) to reconstruct the batch...")
    dummy_data = gia_reconstruct_batch(
        model=model,
        target=real_target,
        true_grads=true_grads,
        real_feat=None,          # do not use feature information
        feat_lambda=0.0,
        steps=steps,
        lr=0.1,
        device=device,
        save_every=500 if save_fig else None,
        save_dir=save_dir,
        mean=mean,
        std=std,
        tag="known_grad_only",
        snapshots=snaps_grad_only,
    )

    # 3. Save the reconstruction result (after de-normalization)
    dummy_unnorm = dummy_data * std + mean
    dummy_unnorm = dummy_unnorm.clamp(0.0, 1.0)
    if save_fig and save_dir:
        save_image(
            dummy_unnorm.cpu(),
            os.path.join(save_dir, "reconstructed_batch_grad_only_final.png"),
            nrow=bs,
        )

    # 3. Run GIA with the FC feature prior
    snaps_grad_feat = []
    print("Running GIA with feature matching (gradient + FC input) ...")
    dummy_data_feat = gia_reconstruct_batch(
        model=model,
        target=real_target,
        true_grads=true_grads,
        real_feat=real_feat.to(device),
        feat_lambda=1.0,
        steps=steps,
        lr=0.1,
        device=device,
        save_every=500 if save_fig else None,
        save_dir=save_dir,
        mean=mean,
        std=std,
        tag="known_grad_plus_feat",
        snapshots=snaps_grad_feat,
    )

    dummy_feat_unnorm = dummy_data_feat * std + mean
    dummy_feat_unnorm = dummy_feat_unnorm.clamp(0.0, 1.0)
    if save_fig and save_dir:
        save_image(
            dummy_feat_unnorm.cpu(),
            os.path.join(save_dir, "reconstructed_batch_grad_plus_feat_final.png"),
            nrow=bs,
        )

    # 4. Generate summary figure: original images + reconstructions from several steps (only when save_fig)
    if save_fig and save_dir:
        if snaps_grad_only:
            grid_tensors = [real_unnorm.cpu(), snaps_grad_only[-1]]
            summary = torch.cat(grid_tensors, dim=0)
            save_image(
                summary,
                os.path.join(save_dir, "summary_known_grad_only.png"),
                nrow=bs,
            )
        if snaps_grad_feat:
            grid_tensors = [real_unnorm.cpu(), snaps_grad_feat[-1]]
            summary = torch.cat(grid_tensors, dim=0)
            save_image(
                summary,
                os.path.join(save_dir, "summary_known_grad_plus_feat.png"),
                nrow=bs,
            )
        print(
            "Saved real_batch.png, reconstructed_batch_grad_only.png and "
            "reconstructed_batch_grad_plus_feat.png to",
            save_dir,
        )

    # 5. Compute and print only the "final" metrics: PSNR / SSIM / FID
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

    # Return the final metrics to facilitate multi-run statistics
    return {
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


def gia_reconstruct_batch_unknown_label(
    model,
    true_grads,
    real_feat=None,
    feat_lambda=1.0,
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
    - Do not use the real labels; labels are also treated as variables to optimize (dummy_label_logits).
    - The rest of the procedure is similar to gia_reconstruct_batch.
    """
    model = model.to(device)
    model.eval()

    # Randomly initialize the dummy input and dummy label logits
    dummy_data = torch.randn(
        (batch_size, 3, model.img_size, model.img_size),
        device=device,
        requires_grad=True,
    )
    dummy_label_logits = torch.randn(
        (batch_size, num_classes), device=device, requires_grad=True
    )

    optimizer = optim.Adam([dummy_data, dummy_label_logits], lr=lr)

    attack_params = [p for p in model.parameters() if p.requires_grad]

    best_loss_val = float("inf")
    no_improve = 0
    patience = 300

    for it in range(steps):
        optimizer.zero_grad()

        out = model(dummy_data)  # (B, num_classes)
        log_probs = F.log_softmax(out, dim=1)
        pseudo_y = F.softmax(dummy_label_logits, dim=1)
        ce_loss = -(pseudo_y * log_probs).sum(dim=1).mean()

        dummy_grads = torch.autograd.grad(
            ce_loss,
            attack_params,
            create_graph=True,
            retain_graph=True,   # we still need to backward through total_loss afterwards
            only_inputs=True,
        )

        grad_loss = 0.0
        for g_fake, g_true in zip(dummy_grads, true_grads):
            grad_loss = grad_loss + F.mse_loss(g_fake, g_true)

        total_loss = grad_loss

        if real_feat is not None:
            dummy_feat = model.last_feature
            feat_loss = F.mse_loss(dummy_feat, real_feat)
            total_loss = total_loss + feat_lambda * feat_loss
        else:
            feat_loss = torch.tensor(0.0, device=device)

        cur_loss_val = total_loss.item()

        total_loss.backward()
        optimizer.step()

        # If required, periodically save intermediate reconstruction results
        if (
            save_every is not None
            and save_dir is not None
            and ((it + 1) % save_every == 0 or (it + 1) == steps)
        ):
            with torch.no_grad():
                snap = dummy_data.detach()
                if mean is not None and std is not None:
                    snap_vis = snap * std + mean
                    snap_vis = snap_vis.clamp(0.0, 1.0)
                else:
                    snap_vis = snap
                filename = f"{tag}_step_{it+1}.png"
                save_image(
                    snap_vis.cpu(),
                    os.path.join(save_dir, filename),
                    nrow=snap.shape[0],
                )
                if snapshots is not None:
                    snapshots.append(snap_vis.detach().cpu())

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

    return dummy_data.detach()


def run_gia_demo_unknown_label(
    device, same_label=False, label=1, use_pool=False, steps=4000, batch_size=4, dataset_name="cifar10", save_fig=False
):
    """
    GIA demo for the unknown-label scenario:
    1. Still use a real batch to generate true_grads and real_feat (attacker does not know the labels; server does).
    2. The attacker side does not use the real labels and only relies on gradients:
       - naive: only match gradients.
       - feat:  match gradients + FC input feature.
    3. If save_fig=True, save the original images and both reconstructions to ./expdata/gia_unknown/{dataset_name}/.
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

    if same_label:
        # Use a "same-class batch" (all samples come from the same class of the same dataset)
        model, real_data, real_target, true_grads, real_feat = collect_true_batch_and_grads_same_label(
            device=device, use_bn=False, batch_size=bs, label=label, use_pool=use_pool, dataset_name=dataset_name
        )
    else:
        # A normal random batch
        model, real_data, real_target, true_grads, real_feat = collect_true_batch_and_grads(
            device=device, use_bn=False, batch_size=bs, use_pool=use_pool, dataset_name=dataset_name
        )

    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1).to(device)

    real_unnorm = real_data * std + mean
    real_unnorm = real_unnorm.clamp(0.0, 1.0)
    if save_fig and save_dir:
        save_image(
            real_unnorm.cpu(),
            os.path.join(save_dir, "real_batch_unknown_label.png"),
            nrow=bs,
        )

    print("Running naive GIA (gradient-only, unknown label) ...")
    snaps_unknown_grad = []
    dummy_data = gia_reconstruct_batch_unknown_label(
        model=model,
        true_grads=true_grads,
        real_feat=None,
        feat_lambda=0.0,
        steps=steps,
        lr=0.1,
        device=device,
        batch_size=bs,
        num_classes=10,
        save_every=500 if save_fig else None,
        save_dir=save_dir,
        mean=mean,
        std=std,
        tag="unknown_grad_only",
        snapshots=snaps_unknown_grad,
    )

    dummy_unnorm = dummy_data * std + mean
    dummy_unnorm = dummy_unnorm.clamp(0.0, 1.0)
    if save_fig and save_dir:
        save_image(
            dummy_unnorm.cpu(),
            os.path.join(save_dir, "reconstructed_batch_unknown_grad_only_final.png"),
            nrow=bs,
        )

    print("Running GIA with feature matching (unknown label, +FC input) ...")
    snaps_unknown_feat = []
    dummy_data_feat = gia_reconstruct_batch_unknown_label(
        model=model,
        true_grads=true_grads,
        real_feat=real_feat.to(device),
        feat_lambda=1.0,
        steps=steps,
        lr=0.1,
        device=device,
        batch_size=bs,
        num_classes=10,
        save_every=500 if save_fig else None,
        save_dir=save_dir,
        mean=mean,
        std=std,
        tag="unknown_grad_plus_feat",
        snapshots=snaps_unknown_feat,
    )

    dummy_feat_unnorm = dummy_data_feat * std + mean
    dummy_feat_unnorm = dummy_feat_unnorm.clamp(0.0, 1.0)
    if save_fig and save_dir:
        save_image(
            dummy_feat_unnorm.cpu(),
            os.path.join(
                save_dir, "reconstructed_batch_unknown_grad_plus_feat_final.png"
            ),
            nrow=bs,
        )

    # 4. Generate summary figure (only when save_fig)
    if save_fig and save_dir:
        overall_rows = [real_unnorm.cpu()]
        if snaps_unknown_grad:
            overall_rows.append(snaps_unknown_grad[-1])
        if snaps_unknown_feat:
            overall_rows.append(snaps_unknown_feat[-1])
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

    # 5. Compute and print metrics only on the "final" reconstruction: PSNR / SSIM / FID
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
        "unknown_grad_only_final", dummy_unnorm
    )
    psnr_feat, ssim_feat, fid_feat = eval_and_print(
        "unknown_grad_plus_feat_final", dummy_feat_unnorm
    )

    if save_fig and save_dir:
        print(
            "Saved real/reconstructed batches, snapshots, summary images and metrics to",
            save_dir,
        )

    # Return the final metrics to facilitate multi-run statistics
    return {
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


def _gia_single_run_worker(args_tuple):
    """
    Multi-process worker for a single GIA run.
    args_tuple: (mode, same_label, label_idx, use_pool, gia_steps, batch_size, run_idx, task_idx, gpu_ids, dataset_name, save_fig)
    Returns a dict:
      {
        "naive": {"psnr": ..., "ssim": ..., "fid": ...},
        "feat":  {"psnr": ..., "ssim": ..., "fid": ...},
      }
    where:
      - Known-label mode: naive=grad_only_final, feat=grad_plus_feat_final
      - Unknown-label mode: naive=unknown_grad_only_final, feat=unknown_grad_plus_feat_final
    """
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
    ) = args_tuple

    # Use a different random seed for each process to ensure different batches and random numbers
    base_seed = int(datetime.now().timestamp() * 1e6) & 0x7FFFFFFF
    seed = base_seed + task_idx * 9973
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Assign a GPU to each worker (if available); otherwise fall back to CPU
    if torch.cuda.is_available() and gpu_ids:
        gpu_id = gpu_ids[task_idx % len(gpu_ids)]
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # If same_label=True and label_idx is not specified, randomly choose a class index for the current run.
    # Here we simply assume 10 classes (the current GIA demo is based on CIFAR-10).
    effective_label_idx = label_idx
    if same_label and label_idx is None:
        max_classes = 10
        effective_label_idx = random.randint(0, max_classes - 1)
        print(
            f"[Run {run_idx + 1}] same_label mode without an explicit label_idx; "
            f"this run randomly selects label_idx={effective_label_idx}"
        )

    if mode == "gia":
        metrics = run_gia_demo(
            device=device,
            same_label=same_label,
            label=effective_label_idx,
            use_pool=use_pool,
            steps=gia_steps,
            batch_size=batch_size,
            dataset_name=dataset_name,
            save_fig=save_fig,
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
        )
        naive_key = "unknown_grad_only_final"
        feat_key = "unknown_grad_plus_feat_final"
    else:
        raise ValueError(f"Unsupported mode for GIA worker: {mode}")

    m_naive = metrics[naive_key]
    m_feat = metrics[feat_key]
    if FID_ENABLED:
        print(
            f"[Run {run_idx + 1}][naive] PSNR={m_naive['psnr']:.4f} dB, "
            f"SSIM={m_naive['ssim']:.4f}, FID={m_naive['fid']:.4f}"
        )
        print(
            f"[Run {run_idx + 1}][hssp+gia] PSNR={m_feat['psnr']:.4f} dB, "
            f"SSIM={m_feat['ssim']:.4f}, FID={m_feat['fid']:.4f}"
        )
    else:
        print(
            f"[Run {run_idx + 1}][naive] PSNR={m_naive['psnr']:.4f} dB, SSIM={m_naive['ssim']:.4f}"
        )
        print(
            f"[Run {run_idx + 1}][hssp+gia] PSNR={m_feat['psnr']:.4f} dB, SSIM={m_feat['ssim']:.4f}"
        )
    return {"naive": m_naive, "feat": m_feat}

