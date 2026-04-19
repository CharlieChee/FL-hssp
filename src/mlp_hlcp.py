"""
MLP-HLCP: attack the first layer of an MLP with mHLCP to directly recover image pixels.

MLP structure:  input(3072) -> fc1(1000) -> ReLU -> fc2(100) -> ReLU -> fc3(10)

At the first layer: output = fc1(input) = input @ W^T + bias
  - input: (batch_size, 3072) = A^T  <- the image pixels themselves!
  - W: (1000, 3072) = fc1.weight
  - output_pre_relu: (batch_size, 1000)

Build the mHLCP: H = X . A mod x_0
  - X: (m, n) = a subsample of fc1_pre_relu with m rows and n=batch_size columns
  - A: (n, l) = the image pixel matrix with n=batch_size rows and l=3072 columns
  - Recovering A is recovering the images!

Usage:
  python mlp_hlcp.py --batch_size 10
  python mlp_hlcp.py --batch_size 20
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import os
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -- MLP model (consistent with fl.py) --
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        self.input_flat = x.detach().clone()  # save raw input = image pixels
        x = self.fc1(x)
        self.fc1_pre_relu = x.detach().clone()  # save fc1 output (pre-ReLU)
        x = F.relu(x)
        self.fc1_relu_mask = (self.fc1_pre_relu > 0).float().detach()  # ReLU mask
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


def extract_hlcp_data(batch_size, seed=0):
    """Extract one batch of MLP first-layer data and build the X and A required by HLCP."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = MLP()
    model.eval()

    # Take the first batch
    images, labels = next(iter(loader))

    with torch.no_grad():
        output = model(images)

    # A = input images (batch_size, 3072); after normalization values are in [-1, 1]
    A_float = model.input_flat.numpy()  # (batch_size, 3072)

    # X = fc1_pre_relu (batch_size, 1000) -> transpose to (1000, batch_size)
    X_float = model.fc1_pre_relu.numpy().T  # (1000, batch_size)

    # ReLU mask
    relu_mask = model.fc1_relu_mask.numpy().T  # (1000, batch_size)

    # Raw images (not normalized, in 0-1 range)
    images_raw = images.numpy()  # (batch_size, 3, 32, 32)

    return A_float, X_float, relu_mask, images_raw, labels.numpy()


def save_images(images_raw, labels, path, title=""):
    """Save a set of images as a grid."""
    n = len(images_raw)
    cols = min(n, 10)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2.2 * rows))
    if rows == 1:
        axes = [axes] if cols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    class_names = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]

    for i in range(len(axes)):
        ax = axes[i]
        ax.axis("off")
        if i < n:
            img = images_raw[i].transpose(1, 2, 0)  # (3,32,32) -> (32,32,3)
            img = img * 0.5 + 0.5  # de-normalize to [0,1]
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            if labels is not None:
                ax.set_title(class_names[labels[i]], fontsize=9)

    if title:
        fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print("saved %s" % path)


def save_recovered_images(A_recovered, A_true, nfound_indices, n, path, title=""):
    """Save recovered images: those in NFound show the recovered image; unrecovered ones show gray/blur placeholders."""
    cols = min(n, 10)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2.2 * rows))
    if rows == 1:
        axes = [axes] if cols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    for i in range(len(axes)):
        ax = axes[i]
        ax.axis("off")
        if i >= n:
            continue

        if i in nfound_indices:
            # Exact recovery: display the recovered image
            pixel_vec = A_recovered[i]  # (3072,)
            img = pixel_vec.reshape(3, 32, 32).transpose(1, 2, 0)
            img = img * 0.5 + 0.5  # de-normalize
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.set_title("recovered", fontsize=8, color="green")
        else:
            # Not recovered: show a gray placeholder
            ax.imshow(np.ones((32, 32, 3)) * 0.7)
            ax.set_title("failed", fontsize=8, color="red")

    if title:
        fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print("saved %s" % path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--int_scale", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="expdata/mlp_hlcp")
    args = parser.parse_args()

    n = args.batch_size
    int_scale = args.int_scale
    out_dir = os.path.join(args.output_dir, "bs%d" % n)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("MLP-HLCP: batch_size=%d, int_scale=%d" % (n, int_scale))
    print("=" * 60)

    # 1. Extract data
    A_float, X_float, relu_mask, images_raw, labels = extract_hlcp_data(n, seed=args.seed)
    print("A (image pixels): shape=%s  range=[%.2f, %.2f]" %
          (A_float.shape, A_float.min(), A_float.max()))
    print("X (fc1 pre-relu): shape=%s" % (X_float.shape,))

    # 2. Save the original images
    save_images(images_raw, labels, os.path.join(out_dir, "original.png"),
                title="Original Images (batch_size=%d)" % n)

    # 3. Quantize
    if int_scale == -1:
        # Binarize: nonzero -> 1, zero -> 0
        A_int = (A_float != 0).astype(np.int64)
        X_int = (X_float != 0).astype(np.int64)
        print("Quantization: binarize (nonzero -> 1)")
    else:
        # A: image pixels in [-1, 1] (after normalization), quantize to integers.
        # To make A non-negative (required by HLCP), first shift to [0, 2] before quantization.
        A_shifted = A_float + 1.0  # [0, 2]
        A_int = np.floor(A_shifted * int_scale).astype(np.int64)  # [0, 2*int_scale]
        # X: fc1_pre_relu, quantize
        X_int = np.floor(X_float * int_scale).astype(np.int64)
        print("Quantization: floor(. * %d)" % int_scale)

    print("A_int: shape=%s  range=[%d, %d]" % (A_int.shape, A_int.min(), A_int.max()))
    print("X_int: shape=%s  range=[%d, %d]" % (X_int.shape, X_int.min(), X_int.max()))

    # 4. Save npy files for use by sage
    np.save(os.path.join(out_dir, "A_int.npy"), A_int)
    np.save(os.path.join(out_dir, "X_int.npy"), X_int)
    np.save(os.path.join(out_dir, "relu_mask.npy"), relu_mask)
    np.save(os.path.join(out_dir, "images_raw.npy"), images_raw)
    np.save(os.path.join(out_dir, "labels.npy"), labels)
    np.save(os.path.join(out_dir, "A_float.npy"), A_float)

    print("\nData saved to %s/" % out_dir)
    print("  A_int.npy: (%d, %d) image pixels (quantized)" % A_int.shape)
    print("  X_int.npy: (%d, %d) fc1 pre-relu (quantized)" % X_int.shape)
    print("  images_raw.npy: raw images")
    print("\nNext step: run the mHLCP attack in sage")
    print("  sage mlp_hlcp_attack.sage --batch_size %d" % n)


if __name__ == "__main__":
    main()
