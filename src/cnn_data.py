"""Dataset configuration, mirror handling, and DataLoader construction.

This module owns:

- ``DATASET_CONFIG``            — per-dataset metadata (shape, normalization stats, class).
- ``_apply_dataset_mirror`` / ``_restore_dataset_mirror``
                                — optional redirect of torchvision download URLs to a
                                  regional mirror (``TORCH_DATASETS_MIRROR`` env var).
- MNIST download helpers        — a cross-process file lock plus an EOFError-aware retry
                                  wrapper, to keep many parallel workers from corrupting
                                  ``data/MNIST/raw``.
- ``get_loaders``               — build ``(train_loader, test_loader, cfg)`` for a
                                  dataset. Called from everywhere (training, GIA,
                                  FC-HSSP export).
"""
import os
import time

import torch
from torchvision import datasets, transforms


# --------------- Dataset mirrors ---------------

_DATASET_MIRROR_USER = os.environ.get("TORCH_DATASETS_MIRROR", "").strip()
# Mirror candidates in order; empty string means the official source as fallback.
_DATASET_MIRROR_CANDIDATES = [
    "https://mirrors.aliyun.com/pytorch-wheels/datasets",
    "https://mirrors.tuna.tsinghua.edu.cn/datasets",
    "https://mirrors.bfsu.edu.cn/datasets",
    "https://mirrors.cloud.tencent.com/pytorch/datasets",
    "https://mirror.sjtu.edu.cn/pytorch-wheels/datasets",
    "",
]


def _get_mirror_list():
    """Return the ordered list of mirrors to try (official source included)."""
    if not _DATASET_MIRROR_USER:
        return _DATASET_MIRROR_CANDIDATES
    user_list = [u.strip().rstrip("/") for u in _DATASET_MIRROR_USER.split(",") if u.strip()]
    seen, out = set(), []
    for base in user_list + _DATASET_MIRROR_CANDIDATES:
        if base not in seen:
            seen.add(base)
            out.append(base)
    return out


def _apply_dataset_mirror(dataset_name, mirror_base):
    """Redirect torchvision's download URL to ``mirror_base``; return a token for restore."""
    if not mirror_base:
        return None, None
    base = mirror_base.rstrip("/")
    saved = {}
    try:
        if dataset_name == "cifar10":
            saved["url"] = datasets.CIFAR10.url
            datasets.CIFAR10.url = f"{base}/cifar-10-python.tar.gz"
        elif dataset_name == "cifar100":
            saved["url"] = datasets.CIFAR100.url
            datasets.CIFAR100.url = f"{base}/cifar-100-python.tar.gz"
        elif dataset_name == "mnist":
            saved["resources"] = list(datasets.MNIST.resources)
            new_res = []
            for info in datasets.MNIST.resources:
                fname = info[0].split("/")[-1]
                new_url = f"{base}/mnist/{fname}"
                new_res.append((new_url,) + tuple(info[1:]))
            datasets.MNIST.resources = new_res
        return saved, dataset_name
    except Exception:
        return None, None


def _restore_dataset_mirror(dataset_name, saved):
    """Restore the original download URL saved by ``_apply_dataset_mirror``."""
    if not saved or not dataset_name:
        return
    try:
        if dataset_name == "cifar10" and "url" in saved:
            datasets.CIFAR10.url = saved["url"]
        elif dataset_name == "cifar100" and "url" in saved:
            datasets.CIFAR100.url = saved["url"]
        elif dataset_name == "mnist" and "resources" in saved:
            datasets.MNIST.resources = saved["resources"]
    except Exception:
        pass


# --------------- MNIST download helpers (concurrency-safe) ---------------

def _acquire_file_lock(lock_path, timeout_sec=180, poll_sec=0.2):
    """Cross-process mutex via atomic file creation. Returns the open fd."""
    start = time.time()
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"pid={os.getpid()} time={time.time():.6f}\n".encode("utf-8"))
            return fd
        except FileExistsError:
            if time.time() - start > timeout_sec:
                raise TimeoutError(f"Timeout waiting dataset lock: {lock_path}")
            time.sleep(poll_sec)


def _release_file_lock(fd, lock_path):
    """Release a lock created by ``_acquire_file_lock``."""
    try:
        os.close(fd)
    except Exception:
        pass
    try:
        os.unlink(lock_path)
    except (FileNotFoundError, Exception):
        pass


def _cleanup_mnist_raw_files(root):
    """Remove potentially corrupted MNIST raw archives / idx files. Returns the list removed."""
    raw_dir = os.path.join(root, "MNIST", "raw")
    removed = []
    if not os.path.isdir(raw_dir):
        return removed

    names = []
    for info in datasets.MNIST.resources:
        gz_name = info[0].split("/")[-1]
        names.append(gz_name)
        if gz_name.endswith(".gz"):
            names.append(gz_name[:-3])

    for name in sorted(set(names)):
        p = os.path.join(raw_dir, name)
        if os.path.isfile(p):
            try:
                os.remove(p)
                removed.append(p)
            except Exception:
                pass
    return removed


def _download_mnist_with_retries(DatasetCls, root, transform, max_retries=2):
    """Download MNIST in a critical section; on ``EOFError`` clean up raw files and retry."""
    last_exc = None
    for attempt in range(1, max_retries + 2):
        try:
            train_set = DatasetCls(root=root, train=True, download=True, transform=transform)
            test_set = DatasetCls(root=root, train=False, download=True, transform=transform)
            return train_set, test_set
        except EOFError as e:
            last_exc = e
            removed = _cleanup_mnist_raw_files(root)
            print(
                f"[MNIST][DownloadRetry] attempt={attempt}/{max_retries + 1} "
                f"hit EOFError, removed {len(removed)} possibly-corrupted files."
            )
            for p in removed:
                print(f"  - {p}")
            if attempt > max_retries:
                break
        except Exception:
            # Do not swallow non-EOF errors (network, permission, full disk, etc.).
            raise
    raise last_exc if last_exc is not None else RuntimeError("MNIST download failed.")


# --------------- Dataset config ---------------

DATASET_CONFIG = {
    "mnist": {
        "in_channels": 1,
        "img_size": 28,
        "num_classes": 10,
        "root": "./data",
        "dataset_cls": datasets.MNIST,
        "mean": (0.1307,),
        "std": (0.3081,),
    },
    "cifar10": {
        "in_channels": 3,
        "img_size": 32,
        "num_classes": 10,
        "root": "./data",
        "dataset_cls": datasets.CIFAR10,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
    },
    "cifar100": {
        "in_channels": 3,
        "img_size": 32,
        "num_classes": 100,
        "root": "./data",
        "dataset_cls": datasets.CIFAR100,
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
    },
}


# --------------- DataLoader construction ---------------

def get_loaders(dataset_name, batch_size=128, num_workers=2, download=True, mnist_retries=2):
    """Build ``(train_loader, test_loader, cfg)`` for a supported dataset.

    For MNIST, downloads are serialised via a cross-process file lock and retried
    on ``EOFError`` (which typically means a corrupted partial archive).
    """
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Choose from {list(DATASET_CONFIG.keys())}"
        )
    cfg = DATASET_CONFIG[dataset_name]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg["mean"], cfg["std"]),
    ])
    DatasetCls = cfg["dataset_cls"]
    root = cfg["root"]

    if dataset_name == "mnist" and download:
        os.makedirs(root, exist_ok=True)
        lock_path = os.path.join(root, ".mnist_download.lock")
        lock_fd = _acquire_file_lock(lock_path)
        try:
            train_set, test_set = _download_mnist_with_retries(
                DatasetCls=DatasetCls, root=root, transform=transform, max_retries=mnist_retries,
            )
        finally:
            _release_file_lock(lock_fd, lock_path)
    else:
        train_set = DatasetCls(root=root, train=True, download=download, transform=transform)
        test_set = DatasetCls(root=root, train=False, download=download, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    return train_loader, test_loader, cfg
