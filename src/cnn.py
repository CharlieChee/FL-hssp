"""
Basic CNN for image classification and gradient inversion demos.
- Supports MNIST, CIFAR-10, CIFAR-100 (adaptive).
- All conv: kernel_size=3, stride=1, no pooling.
"""

import argparse
import os
from datetime import datetime
import multiprocessing as mp
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from cnn_metrics import compute_psnr_batch, compute_ssim_batch, compute_fid, FID_ENABLED
import cnn_data
import cnn_model
import cnn_gia
import cnn_train

# Re-export the canonical implementations from the focused submodules, so
# legacy ``from cnn import BasicCNN, get_loaders, train_epoch, gia_reconstruct_batch, ...``
# imports keep working across the rest of the repo (experiments/, cnn_gia, scripts).
from cnn_model import BasicCNN, print_model_structure
from cnn_data import DATASET_CONFIG, get_loaders
from cnn_train import train_epoch, evaluate
from gia import (
    KNOWN_GIA_MODES,
    ALL_GIA_MODES,
    gia_reconstruct_batch,
    gia_reconstruct_batch_unknown_label,
    run_gia_demo,
    run_gia_demo_unknown_label,
    _gia_single_run_worker,
)
# ``run_fc_hssp_analysis`` lives in cnn_gia.py (with the ``run_id`` argument main() uses).
from cnn_gia import run_fc_hssp_analysis

# Public re-exports so legacy ``from cnn import ...`` imports keep working in
# experiments/, cnn_gia.py, and other parts of the repo. Listed in __all__ so
# ruff F401 does not flag the re-export-only imports above as unused.
__all__ = [
    "BasicCNN",
    "print_model_structure",
    "DATASET_CONFIG",
    "get_loaders",
    "train_epoch",
    "evaluate",
    "compute_psnr_batch",
    "compute_ssim_batch",
    "compute_fid",
    "FID_ENABLED",
    "KNOWN_GIA_MODES",
    "ALL_GIA_MODES",
    "gia_reconstruct_batch",
    "gia_reconstruct_batch_unknown_label",
    "run_fc_hssp_analysis",
    "run_gia_demo",
    "run_gia_demo_unknown_label",
    "main",
]

# --------------- Main ---------------
# This file is now thin: it owns only the argparse / dispatch logic. The actual
# implementations live in:
#   * cnn_model.py    : BasicCNN, print_model_structure
#   * cnn_data.py     : DATASET_CONFIG, get_loaders (with MNIST concurrency lock)
#   * cnn_train.py    : train_epoch, evaluate
#   * cnn_metrics.py  : compute_psnr_batch, compute_ssim_batch, compute_fid
#   * gia.py          : GIA reconstruction (known + unknown labels), demo runners,
#                        multiprocessing worker
#   * cnn_gia.py      : run_fc_hssp_analysis (the FC-tensor exporter)


def main():
    parser = argparse.ArgumentParser(description="Basic CNN: MNIST / CIFAR-10 / CIFAR-100")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["mnist", "cifar10", "cifar100"],
        help="Dataset: mnist, cifar10, cifar100",
    )
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--log_interval",
        type=int,
        default=50,
        help="Print every N batches (0 = only epoch summary)",
    )
    parser.add_argument(
        "--bn",
        action="store_true",
        help="Use BatchNorm after each conv (default: no BN)",
    )
    parser.add_argument(
        "--pool",
        action="store_true",
        help="Insert 2x2 MaxPool between conv layers to reduce spatial size (default: off).",
    )
    parser.add_argument(
        "--gia_steps",
        type=int,
        default=4000,
        help="Number of iterations for GIA optimization (used only when mode=gia/gia_unknown).",
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=1,
        help="Number of repeated GIA runs used to report mean/variance (>1 triggers automatic multiprocessing).",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=None,
        help=(
            "Maximum number of GIA processes to run in parallel. "
            "For example, n_runs=10, n_jobs=5 caps parallelism at 5 processes; "
            "if unspecified, n_jobs defaults to n_runs."
        ),
    )
    parser.add_argument(
        "--batch_size_sweep",
        type=str,
        default=None,
        help=(
            "Sweep over multiple batch_size values in GIA mode, "
            "comma-separated, e.g. '1,2,3,4,5,6,7,8,9,10'. "
            "Range syntax '1,...,128' (or descending '128,...,1') is also supported."
        ),
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated list of GPU ids, e.g. '0,1,2'. Defaults to all visible GPUs.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "gia", "gia_loss_probe", "gia_norm", "gia_unknown", "fc_hssp"],
        help=(
            "Run mode: "
            "train=standard training, "
            "gia=known-label GIA, "
            "gia_loss_probe=experiment 1: record and plot the initial/final values of L_grad and lambda*L_feat vs batch, "
            "gia_norm=experiment 2: use the normalized objective L_grad/L_grad(0)+lambda*L_feat/L_feat(0), "
            "gia_unknown=unknown-label GIA, "
            "fc_hssp=only run the FC-layer gradient structure / HSSP analysis and save the related matrices"
        ),
    )
    parser.add_argument(
        "--same_label",
        action="store_true",
        help="Use a same-class batch of CIFAR-10 in the GIA demo (default: off, use a mixed-class batch).",
    )
    parser.add_argument(
        "--label_idx",
        type=int,
        default=None,
        help="Class index (0-9) of CIFAR-10 used when --same_label is enabled. If unspecified, the program randomly picks a class internally.",
    )
    parser.add_argument(
        "--save_fig",
        action="store_true",
        help="Save the real images and the naive GIA / HSSP+GIA reconstructions to an expdata subdirectory in GIA mode; otherwise no images are saved (metrics only).",
    )
    parser.add_argument(
        "--noise_on_fc",
        type=str,
        default="0",
        help=(
            "Active only when mode=gia_unknown. "
            "Gaussian-noise sigma added to the FC input feature in HSSP+GIA. "
            "Accepts a single value or a comma-separated list (e.g. '0,1e-4,1e-3,1e-2,1e-1'). "
            "Defaults to 0 if unspecified."
        ),
    )
    parser.add_argument(
        "--known_rate",
        type=float,
        default=1.0,
        help=(
            "For the FC feature term in GIA, the fraction of batch samples whose true features are exactly known, in [0,1]. "
            "gia_unknown: when 0<known_rate<1, partial (per-row only), oracle (k=1), and partial+known_residual are run in sequence; "
            "when known_rate=0, partial coincides with naive, but oracle and the mean-only residual are still run. "
            "Known-label gia variants: see --known_residual; with known_rate=0 and that switch off, feat has the same objective as naive."
        ),
    )
    parser.add_argument(
        "--known_residual",
        action="store_true",
        help=(
            "Active only in the feat branch of mode=gia / gia_loss_probe / gia_norm (known-label). "
            "Under the unknown-label mode=gia_unknown, partial-only and partial+known_residual are always both run with log output (this switch is not needed there)."
        ),
    )
    parser.add_argument(
        "--known_peel",
        action="store_true",
        help=(
            "Active only when mode=gia_unknown. When enabled, gradient matching is peeled: known rows are removed from the gradient mixture "
            "and gradients are computed/matched only on unknown rows (the feature term still follows the known_rate/known_residual rules)."
        ),
    )
    args = parser.parse_args()
    args.known_rate = float(max(0.0, min(1.0, args.known_rate)))

    noise_on_fc_list = []
    for item in str(args.noise_on_fc).split(","):
        s = item.strip()
        if not s:
            continue
        noise_on_fc_list.append(float(s))
    if not noise_on_fc_list:
        noise_on_fc_list = [0.0]
    if args.mode != "gia_unknown" and any(abs(v) > 0.0 for v in noise_on_fc_list):
        print("[GIA] --noise_on_fc is only active under mode=gia_unknown; the current mode ignores this argument.")
    if args.batch_size_sweep is None and len(noise_on_fc_list) > 1:
        print(
            "[GIA] Without --batch_size_sweep, only the first value of --noise_on_fc is used: "
            f"{noise_on_fc_list[0]:g}"
        )

    # If n_jobs is not specified, default to n_runs.
    if args.n_jobs is None:
        args.n_jobs = args.n_runs

    # When same_label is enabled without an explicit label_idx, each GIA run picks its own random class;
    # here we only print a notice and do not globally fix label_idx.
    if args.same_label and args.label_idx is None:
        print(
            "[GIA] --same_label is enabled and --label_idx is not specified; "
            "each GIA run will randomly choose its own class index."
        )

    # Parse the list of usable GPUs.
    if torch.cuda.is_available():
        if args.gpus is not None:
            gpu_ids = [
                int(x) for x in args.gpus.split(",") if x.strip() != ""
            ]
        else:
            gpu_ids = list(range(torch.cuda.device_count()))
    else:
        gpu_ids = []

    # Default primary device: the first GPU; fall back to CPU if none are available.
    if torch.cuda.is_available() and gpu_ids:
        primary_gpu = gpu_ids[0]
        torch.cuda.set_device(primary_gpu)
        device = torch.device(f"cuda:{primary_gpu}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode in ALL_GIA_MODES:
        # Pre-download in the main process first to avoid MNIST download races across GIA worker processes.
        if args.dataset == "mnist":
            print("[MNIST] Preflight: preparing dataset in main process (download/check)...")
            get_loaders("mnist", batch_size=1, num_workers=0, download=True, mnist_retries=2)
            print("[MNIST] Preflight done. Worker processes will read local files only.")

        # GIA mode: supports multiple runs with parallel metric aggregation.
        # When batch_size_sweep is set, run n_runs iterations for each batch_size separately.
        if args.batch_size_sweep is not None:
            sweep_str = args.batch_size_sweep.strip()
            # Extended syntax:
            # - 1,...,50 means all integers from 1 to 50 (ascending)
            # - 50,...,1 means all integers from 50 to 1 (descending)
            # - 1,50 still means just {1, 50}
            if "..." in sweep_str:
                left, right = sweep_str.split("...")
                start = int(left.split(",")[0].strip())
                end = int(right.split(",")[-1].strip())
                if start <= 0 or end <= 0:
                    raise ValueError(f"Invalid batch_size_sweep range: {sweep_str}")
                if end >= start:
                    bs_list = list(range(start, end + 1))
                else:
                    bs_list = list(range(start, end - 1, -1))
            else:
                bs_list = [
                    int(x) for x in sweep_str.split(",") if x.strip() != ""
                ]
        else:
            bs_list = None
        if bs_list is not None:
            # Always run sweep from large batch_size to small batch_size,
            # so heavy configurations fail fast and are diagnosed earlier.
            bs_list = sorted(bs_list, reverse=True)
        # Case 1: no sweep is configured; run GIA on a single batch_size.
        if bs_list is None:
            batch_size_single = args.batch_size
            if args.n_runs <= 1:
                if args.mode in KNOWN_GIA_MODES:
                    run_gia_demo(
                        device=device,
                        same_label=args.same_label,
                        label=args.label_idx,
                        use_pool=args.pool,
                        steps=args.gia_steps,
                        batch_size=batch_size_single,
                        dataset_name=args.dataset,
                        save_fig=args.save_fig,
                        collect_loss_probe=(args.mode in ("gia_loss_probe", "gia_norm")),
                        normalize_objective=(args.mode == "gia_norm"),
                        known_rate=args.known_rate,
                        known_residual=args.known_residual,
                    )
                else:
                    run_gia_demo_unknown_label(
                        device=device,
                        same_label=args.same_label,
                        label=args.label_idx,
                        use_pool=args.pool,
                        steps=args.gia_steps,
                        batch_size=batch_size_single,
                        dataset_name=args.dataset,
                        save_fig=args.save_fig,
                        noise_on_fc=noise_on_fc_list[0],
                        known_rate=args.known_rate,
                        known_peel=args.known_peel,
                    )
                return

            # Run n_runs iterations in parallel via multiprocessing; each worker is bound to one GPU (round-robin).
            n_runs = args.n_runs
            print(
                f"[GIA] Running {args.mode} for {n_runs} runs "
                f"(multiprocessing, max jobs={args.n_jobs})..."
            )
            ctx = mp.get_context("spawn")
            worker_args = []
            for i in range(n_runs):
                # With a single batch_size, task_idx equals run_idx.
                worker_args.append(
                    (
                        args.mode,
                        args.same_label,
                        args.label_idx,
                        args.pool,
                        args.gia_steps,
                        batch_size_single,
                        i,          # run_idx
                        i,          # task_idx
                        gpu_ids,
                        args.dataset,
                        args.save_fig,
                        noise_on_fc_list[0],
                        args.known_rate,
                        args.known_residual,
                        args.known_peel,
                    )
                )
            with ctx.Pool(
                processes=min(args.n_jobs, n_runs, mp.cpu_count()),
                maxtasksperchild=1,
            ) as pool:
                # Use chunksize=1 to avoid chunk-aligned GPU hotspotting.
                # With default chunksize, each worker receives a contiguous task block;
                # when task_idx drives GPU choice (task_idx % num_gpus), many workers can
                # start on the same GPU at once (e.g., GPU0), causing avoidable OOM.
                results = pool.map(_gia_single_run_worker, worker_args, chunksize=1)

            # Aggregate metrics for naive GIA and HSSP+GIA separately.
            psnrs_naive = np.array(
                [m["naive"]["psnr"] for m in results], dtype=np.float64
            )
            ssims_naive = np.array(
                [m["naive"]["ssim"] for m in results], dtype=np.float64
            )
            fids_naive = np.array(
                [m["naive"]["fid"] for m in results], dtype=np.float64
            )

            psnrs_feat = np.array(
                [m["feat"]["psnr"] for m in results], dtype=np.float64
            )
            ssims_feat = np.array(
                [m["feat"]["ssim"] for m in results], dtype=np.float64
            )
            fids_feat = np.array(
                [m["feat"]["fid"] for m in results], dtype=np.float64
            )

            def _summary(prefix, name, arr):
                mean = float(arr.mean())
                std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
                print(f"[GIA][{prefix}][{name}] mean={mean:.4f}, std={std:.4f}")

            print(f"[GIA] Summary over {n_runs} runs (naive vs hssp+gia separated):")
            _summary("naive", "PSNR (dB)", psnrs_naive)
            _summary("naive", "SSIM", ssims_naive)
            _summary("naive", "FID", fids_naive)
            _summary("hssp+gia", "PSNR (dB)", psnrs_feat)
            _summary("hssp+gia", "SSIM", ssims_feat)
            _summary("hssp+gia", "FID", fids_feat)
            return

        # Case 2: batch_size_sweep is set; run n_runs GIA iterations for each batch_size.
        # Create the overall experiment directory (by dataset + mode + timestamp).
        sweep_base_dir = os.path.join(
            "./expdata/gia_sweep" if args.mode in KNOWN_GIA_MODES else "./expdata/gia_unknown_sweep",
            args.dataset,
        )
        os.makedirs(sweep_base_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sweep_dir = os.path.join(
            sweep_base_dir,
            f"{timestamp}_{args.mode}_sameLabel{int(args.same_label)}_pool{int(args.pool)}",
        )
        os.makedirs(sweep_dir, exist_ok=True)

        print(
            f"[GIA][Sweep] batch_size list = {bs_list}, n_runs = {args.n_runs}, "
            f"results will be saved to {sweep_dir}"
        )

        if args.mode == "gia_unknown":
            sweep_noise_list = noise_on_fc_list
        else:
            sweep_noise_list = [0.0]
        print(f"[GIA][Sweep] noise_on_fc list = {sweep_noise_list}")

        all_bs = []
        psnr_naive_means, psnr_naive_stds = [], []
        ssim_naive_means, ssim_naive_stds = [], []
        fid_naive_means, fid_naive_stds = [], []
        psnr_feat_means_by_sigma = {sigma: [] for sigma in sweep_noise_list}
        psnr_feat_stds_by_sigma = {sigma: [] for sigma in sweep_noise_list}
        ssim_feat_means_by_sigma = {sigma: [] for sigma in sweep_noise_list}
        ssim_feat_stds_by_sigma = {sigma: [] for sigma in sweep_noise_list}
        fid_feat_means_by_sigma = {sigma: [] for sigma in sweep_noise_list}
        fid_feat_stds_by_sigma = {sigma: [] for sigma in sweep_noise_list}
        probe_grad_init_means, probe_grad_final_means = [], []
        probe_lfeat_init_means, probe_lfeat_final_means = [], []

        # For the sweep, build the full list of (batch_size, run_idx) tasks and dispatch them uniformly,
        # so different batch_sizes can also run in parallel with one another.
        ctx = mp.get_context("spawn")
        all_tasks = []
        task_idx = 0
        for bs in bs_list:
            for sigma in sweep_noise_list:
                for i in range(args.n_runs):
                    all_tasks.append(
                        (
                            args.mode,
                            args.same_label,
                            args.label_idx,
                            args.pool,
                            args.gia_steps,
                            bs,
                            i,          # run_idx
                            task_idx,   # task_idx: globally unique; used for GPU assignment and seeding
                            gpu_ids,
                            args.dataset,
                            args.save_fig,
                            sigma,
                            args.known_rate,
                            args.known_residual,
                            args.known_peel,
                        )
                    )
                    task_idx += 1

        print(
            f"[GIA][Sweep] Submitting {len(all_tasks)} total runs "
            f"(batch_sizes={bs_list}, noise_on_fc={sweep_noise_list}, "
            f"known_rate={args.known_rate:g}, known_residual={int(args.known_residual)}, "
            f"n_runs={args.n_runs}, max jobs={args.n_jobs})..."
        )

        with ctx.Pool(
            processes=min(args.n_jobs, len(all_tasks), mp.cpu_count()),
            maxtasksperchild=1,
        ) as pool:
            # Keep fine-grained task distribution so GPU round-robin by task_idx works.
            all_results = pool.map(_gia_single_run_worker, all_tasks, chunksize=1)

        # Aggregate results per batch_size.
        results_by_bs_naive = {bs: [] for bs in bs_list}
        results_by_bs_sigma = {
            bs: {sigma: [] for sigma in sweep_noise_list}
            for bs in bs_list
        }
        naive_ref_sigma = sweep_noise_list[0]
        for task, res in zip(all_tasks, all_results):
            bs = task[5]
            sigma = task[11]
            results_by_bs_sigma[bs][sigma].append(res)
            if abs(float(sigma) - float(naive_ref_sigma)) <= 1e-15:
                results_by_bs_naive[bs].append(res)

        def _mean_std(arr):
            """
            Compute the mean and standard deviation, ignoring NaNs.
            If the array is all NaN, return (nan, nan); downstream code fills these via interpolation/nearest-neighbor.
            """
            valid = arr[~np.isnan(arr)]
            if valid.size == 0:
                return float("nan"), float("nan")
            mean = float(valid.mean())
            std = float(valid.std(ddof=1)) if valid.size > 1 else 0.0
            return mean, std

        for bs in bs_list:
            results = results_by_bs_naive[bs]
            if not results:
                continue

            psnrs_naive = np.array(
                [m["naive"]["psnr"] for m in results], dtype=np.float64
            )
            ssims_naive = np.array(
                [m["naive"]["ssim"] for m in results], dtype=np.float64
            )
            fids_naive = np.array(
                [m["naive"]["fid"] for m in results], dtype=np.float64
            )

            m_psnr_naive, s_psnr_naive = _mean_std(psnrs_naive)
            m_ssim_naive, s_ssim_naive = _mean_std(ssims_naive)
            m_fid_naive, s_fid_naive = _mean_std(fids_naive)

            print(
                f"[GIA][Sweep][bs={bs}] "
                f"naive: PSNR={m_psnr_naive:.4f}±{s_psnr_naive:.4f}, "
                f"SSIM={m_ssim_naive:.4f}±{s_ssim_naive:.4f}, "
                f"FID={m_fid_naive:.4f}±{s_fid_naive:.4f}"
            )
            for sigma in sweep_noise_list:
                sigma_results = results_by_bs_sigma[bs][sigma]
                psnrs_feat_sigma = np.array(
                    [m["feat"]["psnr"] for m in sigma_results], dtype=np.float64
                )
                ssims_feat_sigma = np.array(
                    [m["feat"]["ssim"] for m in sigma_results], dtype=np.float64
                )
                fids_feat_sigma = np.array(
                    [m["feat"]["fid"] for m in sigma_results], dtype=np.float64
                )
                m_psnr_feat, s_psnr_feat = _mean_std(psnrs_feat_sigma)
                m_ssim_feat, s_ssim_feat = _mean_std(ssims_feat_sigma)
                m_fid_feat, s_fid_feat = _mean_std(fids_feat_sigma)
                print(
                    f"[GIA][Sweep][bs={bs}] "
                    f"hssp+gia(sigma={sigma:g}): PSNR={m_psnr_feat:.4f}±{s_psnr_feat:.4f}, "
                    f"SSIM={m_ssim_feat:.4f}±{s_ssim_feat:.4f}, "
                    f"FID={m_fid_feat:.4f}±{s_fid_feat:.4f}"
                )
                psnr_feat_means_by_sigma[sigma].append(m_psnr_feat)
                psnr_feat_stds_by_sigma[sigma].append(s_psnr_feat)
                ssim_feat_means_by_sigma[sigma].append(m_ssim_feat)
                ssim_feat_stds_by_sigma[sigma].append(s_ssim_feat)
                fid_feat_means_by_sigma[sigma].append(m_fid_feat)
                fid_feat_stds_by_sigma[sigma].append(s_fid_feat)

            all_bs.append(bs)
            psnr_naive_means.append(m_psnr_naive)
            psnr_naive_stds.append(s_psnr_naive)

            ssim_naive_means.append(m_ssim_naive)
            ssim_naive_stds.append(s_ssim_naive)

            fid_naive_means.append(m_fid_naive)
            fid_naive_stds.append(s_fid_naive)

            if args.mode in ("gia_loss_probe", "gia_norm"):
                feat_probes = [
                    m.get("loss_probe", {}).get("feat", {})
                    for m in results
                    if m.get("loss_probe", {}).get("feat", None) is not None
                ]
                if feat_probes:
                    gi = np.array([p.get("grad_init", np.nan) for p in feat_probes], dtype=np.float64)
                    gf = np.array([p.get("grad_final", np.nan) for p in feat_probes], dtype=np.float64)
                    li = np.array([p.get("lambda_feat_init", np.nan) for p in feat_probes], dtype=np.float64)
                    lf = np.array([p.get("lambda_feat_final", np.nan) for p in feat_probes], dtype=np.float64)
                    probe_grad_init_means.append(float(np.nanmean(gi)))
                    probe_grad_final_means.append(float(np.nanmean(gf)))
                    probe_lfeat_init_means.append(float(np.nanmean(li)))
                    probe_lfeat_final_means.append(float(np.nanmean(lf)))
                else:
                    probe_grad_init_means.append(float("nan"))
                    probe_grad_final_means.append(float("nan"))
                    probe_lfeat_init_means.append(float("nan"))
                    probe_lfeat_final_means.append(float("nan"))

        def _fill_nan_with_neighbors(values):
            """
            Simple NaN fill using left/right neighbors:
            - if NaN has both left and right neighbors: linear interpolation
            - if only one side has a neighbor: copy the nearest neighbor
            - if the input is all NaN: return unchanged
            """
            arr = np.array(values, dtype=np.float64)
            if not np.any(np.isnan(arr)):
                return arr
            idx_valid = np.where(~np.isnan(arr))[0]
            if idx_valid.size == 0:
                return arr
            for i in range(len(arr)):
                if not np.isnan(arr[i]):
                    continue
                left = idx_valid[idx_valid < i]
                right = idx_valid[idx_valid > i]
                if left.size and right.size:
                    li, ri = left[-1], right[0]
                    t = (i - li) / (ri - li)
                    arr[i] = (1.0 - t) * arr[li] + t * arr[ri]
                elif left.size:
                    arr[i] = arr[left[-1]]
                elif right.size:
                    arr[i] = arr[right[0]]
            return arr

        # Fill NaNs across all mean/std arrays to avoid gaps when certain batch_sizes (e.g. bs=1) have no FID at all.
        psnr_naive_means = _fill_nan_with_neighbors(psnr_naive_means)
        psnr_naive_stds = _fill_nan_with_neighbors(psnr_naive_stds)

        ssim_naive_means = _fill_nan_with_neighbors(ssim_naive_means)
        ssim_naive_stds = _fill_nan_with_neighbors(ssim_naive_stds)

        fid_naive_means = _fill_nan_with_neighbors(fid_naive_means)
        fid_naive_stds = _fill_nan_with_neighbors(fid_naive_stds)
        for sigma in sweep_noise_list:
            psnr_feat_means_by_sigma[sigma] = _fill_nan_with_neighbors(psnr_feat_means_by_sigma[sigma])
            psnr_feat_stds_by_sigma[sigma] = _fill_nan_with_neighbors(psnr_feat_stds_by_sigma[sigma])
            ssim_feat_means_by_sigma[sigma] = _fill_nan_with_neighbors(ssim_feat_means_by_sigma[sigma])
            ssim_feat_stds_by_sigma[sigma] = _fill_nan_with_neighbors(ssim_feat_stds_by_sigma[sigma])
            fid_feat_means_by_sigma[sigma] = _fill_nan_with_neighbors(fid_feat_means_by_sigma[sigma])
            fid_feat_stds_by_sigma[sigma] = _fill_nan_with_neighbors(fid_feat_stds_by_sigma[sigma])
        if args.mode in ("gia_loss_probe", "gia_norm"):
            probe_grad_init_means = _fill_nan_with_neighbors(probe_grad_init_means)
            probe_grad_final_means = _fill_nan_with_neighbors(probe_grad_final_means)
            probe_lfeat_init_means = _fill_nan_with_neighbors(probe_lfeat_init_means)
            probe_lfeat_final_means = _fill_nan_with_neighbors(probe_lfeat_final_means)

        # Save the sweep results to an npz file.
        np.savez(
            os.path.join(sweep_dir, "metrics_sweep_summary.npz"),
            batch_sizes=np.array(all_bs, dtype=np.int32),
            noise_on_fc=np.array(sweep_noise_list, dtype=np.float64),
            psnr_naive_mean=np.array(psnr_naive_means, dtype=np.float64),
            psnr_naive_std=np.array(psnr_naive_stds, dtype=np.float64),
            psnr_feat_mean=np.array(
                [np.array(psnr_feat_means_by_sigma[s], dtype=np.float64) for s in sweep_noise_list],
                dtype=np.float64,
            ),
            psnr_feat_std=np.array(
                [np.array(psnr_feat_stds_by_sigma[s], dtype=np.float64) for s in sweep_noise_list],
                dtype=np.float64,
            ),
            ssim_naive_mean=np.array(ssim_naive_means, dtype=np.float64),
            ssim_naive_std=np.array(ssim_naive_stds, dtype=np.float64),
            ssim_feat_mean=np.array(
                [np.array(ssim_feat_means_by_sigma[s], dtype=np.float64) for s in sweep_noise_list],
                dtype=np.float64,
            ),
            ssim_feat_std=np.array(
                [np.array(ssim_feat_stds_by_sigma[s], dtype=np.float64) for s in sweep_noise_list],
                dtype=np.float64,
            ),
            fid_naive_mean=np.array(fid_naive_means, dtype=np.float64),
            fid_naive_std=np.array(fid_naive_stds, dtype=np.float64),
            fid_feat_mean=np.array(
                [np.array(fid_feat_means_by_sigma[s], dtype=np.float64) for s in sweep_noise_list],
                dtype=np.float64,
            ),
            fid_feat_std=np.array(
                [np.array(fid_feat_stds_by_sigma[s], dtype=np.float64) for s in sweep_noise_list],
                dtype=np.float64,
            ),
        )
        if args.mode in ("gia_loss_probe", "gia_norm"):
            np.savez(
                os.path.join(sweep_dir, "loss_scale_probe.npz"),
                batch_sizes=np.array(all_bs, dtype=np.int32),
                grad_init_mean=np.array(probe_grad_init_means, dtype=np.float64),
                grad_final_mean=np.array(probe_grad_final_means, dtype=np.float64),
                lambda_feat_init_mean=np.array(probe_lfeat_init_means, dtype=np.float64),
                lambda_feat_final_mean=np.array(probe_lfeat_final_means, dtype=np.float64),
            )

        # Also save the experiment configuration parameters to a text file.
        with open(os.path.join(sweep_dir, "config.txt"), "w") as f:
            f.write(f"timestamp: {timestamp}\n")
            f.write(f"mode: {args.mode}\n")
            f.write(f"dataset: {args.dataset}\n")
            f.write(f"same_label: {args.same_label}\n")
            f.write(f"label_idx: {args.label_idx}\n")
            f.write(f"use_pool: {args.pool}\n")
            f.write(f"gia_steps: {args.gia_steps}\n")
            f.write(f"n_runs: {args.n_runs}\n")
            f.write(f"batch_size_sweep: {bs_list}\n")
            f.write(f"noise_on_fc: {sweep_noise_list}\n")
            f.write(f"known_rate: {args.known_rate}\n")
            f.write(f"known_residual: {args.known_residual}\n")
            f.write(f"gpus: {gpu_ids}\n")

        # Plotting: x-axis is batch_size, y-axis is PSNR / SSIM in two subplots;
        # each subplot contains two lines (naive vs hssp+gia), with a light shaded +/-1 std band.
        all_bs_arr = np.array(all_bs, dtype=np.int32)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # PSNR
        ax0 = axes[0]
        # naive
        ax0.plot(all_bs_arr, psnr_naive_means, "o-", label="naive PSNR", color="C0")
        ax0.fill_between(
            all_bs_arr,
            np.array(psnr_naive_means) - np.array(psnr_naive_stds),
            np.array(psnr_naive_means) + np.array(psnr_naive_stds),
            color="C0",
            alpha=0.2,
        )
        for i_sigma, sigma in enumerate(sweep_noise_list):
            color = f"C{i_sigma + 1}"
            y = np.array(psnr_feat_means_by_sigma[sigma], dtype=np.float64)
            e = np.array(psnr_feat_stds_by_sigma[sigma], dtype=np.float64)
            ax0.plot(all_bs_arr, y, "o-", label=f"hssp+gia sigma={sigma:g} PSNR", color=color)
            ax0.fill_between(
                all_bs_arr,
                y - e,
                y + e,
                color=color,
                alpha=0.15,
            )
        ax0.set_xlabel("batch size")
        ax0.set_ylabel("PSNR (dB)")
        ax0.set_title("PSNR vs batch size")
        ax0.grid(True)
        ax0.legend()

        # SSIM
        ax1 = axes[1]
        ax1.plot(all_bs_arr, ssim_naive_means, "o-", label="naive SSIM", color="C0")
        ax1.fill_between(
            all_bs_arr,
            np.array(ssim_naive_means) - np.array(ssim_naive_stds),
            np.array(ssim_naive_means) + np.array(ssim_naive_stds),
            color="C0",
            alpha=0.2,
        )
        for i_sigma, sigma in enumerate(sweep_noise_list):
            color = f"C{i_sigma + 1}"
            y = np.array(ssim_feat_means_by_sigma[sigma], dtype=np.float64)
            e = np.array(ssim_feat_stds_by_sigma[sigma], dtype=np.float64)
            ax1.plot(all_bs_arr, y, "o-", label=f"hssp+gia sigma={sigma:g} SSIM", color=color)
            ax1.fill_between(
                all_bs_arr,
                y - e,
                y + e,
                color=color,
                alpha=0.15,
            )
        ax1.set_xlabel("batch size")
        ax1.set_ylabel("SSIM")
        ax1.set_title("SSIM vs batch size")
        ax1.grid(True)
        ax1.legend()

        fig.suptitle(
            f"{args.mode} metrics vs batch size (n_runs={args.n_runs}, same_label={args.same_label}, pool={args.pool})"
        )
        plt.tight_layout()
        fig.savefig(os.path.join(sweep_dir, "metrics_vs_batch_size.png"))
        plt.close(fig)

        # Also save two standalone plots: PSNR and SSIM (each shows naive + hssp+gia for every sigma).
        fig_psnr, ax_psnr = plt.subplots(1, 1, figsize=(7, 5))
        ax_psnr.plot(all_bs_arr, psnr_naive_means, "o-", label="naive PSNR", color="C0")
        ax_psnr.fill_between(
            all_bs_arr,
            np.array(psnr_naive_means) - np.array(psnr_naive_stds),
            np.array(psnr_naive_means) + np.array(psnr_naive_stds),
            color="C0",
            alpha=0.2,
        )
        for i_sigma, sigma in enumerate(sweep_noise_list):
            color = f"C{i_sigma + 1}"
            y = np.array(psnr_feat_means_by_sigma[sigma], dtype=np.float64)
            e = np.array(psnr_feat_stds_by_sigma[sigma], dtype=np.float64)
            ax_psnr.plot(all_bs_arr, y, "o-", label=f"hssp+gia sigma={sigma:g} PSNR", color=color)
            ax_psnr.fill_between(all_bs_arr, y - e, y + e, color=color, alpha=0.15)
        ax_psnr.set_xlabel("batch size")
        ax_psnr.set_ylabel("PSNR (dB)")
        ax_psnr.set_title("PSNR vs batch size")
        ax_psnr.grid(True)
        ax_psnr.legend()
        fig_psnr.tight_layout()
        fig_psnr.savefig(os.path.join(sweep_dir, "psnr_vs_batch_size.png"))
        plt.close(fig_psnr)

        fig_ssim, ax_ssim = plt.subplots(1, 1, figsize=(7, 5))
        ax_ssim.plot(all_bs_arr, ssim_naive_means, "o-", label="naive SSIM", color="C0")
        ax_ssim.fill_between(
            all_bs_arr,
            np.array(ssim_naive_means) - np.array(ssim_naive_stds),
            np.array(ssim_naive_means) + np.array(ssim_naive_stds),
            color="C0",
            alpha=0.2,
        )
        for i_sigma, sigma in enumerate(sweep_noise_list):
            color = f"C{i_sigma + 1}"
            y = np.array(ssim_feat_means_by_sigma[sigma], dtype=np.float64)
            e = np.array(ssim_feat_stds_by_sigma[sigma], dtype=np.float64)
            ax_ssim.plot(all_bs_arr, y, "o-", label=f"hssp+gia sigma={sigma:g} SSIM", color=color)
            ax_ssim.fill_between(all_bs_arr, y - e, y + e, color=color, alpha=0.15)
        ax_ssim.set_xlabel("batch size")
        ax_ssim.set_ylabel("SSIM")
        ax_ssim.set_title("SSIM vs batch size")
        ax_ssim.grid(True)
        ax_ssim.legend()
        fig_ssim.tight_layout()
        fig_ssim.savefig(os.path.join(sweep_dir, "ssim_vs_batch_size.png"))
        plt.close(fig_ssim)

        if args.mode in ("gia_loss_probe", "gia_norm"):
            fig_probe, axes_probe = plt.subplots(1, 2, figsize=(12, 5))
            axp0, axp1 = axes_probe
            axp0.plot(all_bs_arr, probe_grad_init_means, "o-", label="L_grad init", color="C0")
            axp0.plot(all_bs_arr, probe_grad_final_means, "o-", label="L_grad final", color="C1")
            axp0.set_xlabel("batch size")
            axp0.set_ylabel("L_grad (raw)")
            axp0.set_title("Gradient matching term")
            axp0.grid(True)
            axp0.legend()

            axp1.plot(all_bs_arr, probe_lfeat_init_means, "o-", label="lambda*L_feat init", color="C2")
            axp1.plot(all_bs_arr, probe_lfeat_final_means, "o-", label="lambda*L_feat final", color="C3")
            axp1.set_xlabel("batch size")
            axp1.set_ylabel("lambda*L_feat (raw)")
            axp1.set_title("Feature constraint term")
            axp1.grid(True)
            axp1.legend()

            fig_probe.suptitle(
                f"{args.mode} loss-scale probe (n_runs={args.n_runs}, same_label={args.same_label}, pool={args.pool})"
            )
            fig_probe.tight_layout()
            fig_probe.savefig(os.path.join(sweep_dir, "loss_scale_probe.png"))
            plt.close(fig_probe)

        print(f"[GIA][Sweep] Saved metrics summary and plots to {sweep_dir}")
        return
    if args.mode == "fc_hssp":
        # Only save the gradient / HSSP-related data for the first FC layer (i.e., the first linear layer):
        # - do not run GIA, training, or other steps
        # - only save the FC input batch matrix and the 0/1 coefficient matrix from the ReLU mask
        # - for multi-run experiments, the environment variable HSSP_RUN_ID provides run_id, and the output is saved to bs{}/run_{:04d}/
        run_id = os.environ.get("HSSP_RUN_ID")
        run_id = int(run_id) if run_id is not None and str(run_id).strip().isdigit() else None
        cnn_gia.run_fc_hssp_analysis(
            device=device,
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            use_pool=args.pool,
            run_id=run_id,
        )
        return

    print(f"Loading {args.dataset.upper()}...")
    # Training mode runs in the main process, so DataLoader multi-process loading can be used safely.
    train_loader, test_loader, cfg = cnn_data.get_loaders(args.dataset, args.batch_size, num_workers=2)

    model = cnn_model.BasicCNN(
        in_channels=cfg["in_channels"],
        img_size=cfg["img_size"],
        num_classes=cfg["num_classes"],
        use_bn=args.bn,
        use_pool=args.pool,
    ).to(device)

    # In training mode, use DataParallel when multiple GPUs are specified.
    if torch.cuda.is_available() and len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    cnn_model.print_model_structure(
        model, cfg["in_channels"], cfg["img_size"], cfg["num_classes"], use_bn=args.bn
    )
    print(
        f"Dataset: {args.dataset} | in_channels={cfg['in_channels']} | "
        f"img_size={cfg['img_size']} | num_classes={cfg['num_classes']}"
    )
    print(
        f"Device: {device} | Epochs: {args.epochs} | LR: {args.lr} | "
        f"Batch: {args.batch_size} | log_interval: {args.log_interval} | BN: {args.bn}"
    )
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        print(f"\n>>> Epoch {epoch}/{args.epochs}")
        train_loss, train_acc = cnn_train.train_epoch(
            model,
            device,
            train_loader,
            optimizer,
            criterion,
            epoch=epoch,
            log_interval=args.log_interval,
        )
        print(
            f"  [Epoch {epoch} Summary] Train Loss: {train_loss:.4f}  "
            f"Train Acc: {train_acc:.2f}%"
        )

    print("-" * 60)
    print("Training finished.")


if __name__ == "__main__":
    main()
