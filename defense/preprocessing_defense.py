#!/usr/bin/env python3
"""Defense 1: input preprocessing (JPEG compression + Gaussian smoothing).

JPEG's quantization of high-frequency DCT coefficients destroys most of the
small, high-frequency PGD perturbation; the Gaussian blur mops up what's
left, at the cost of some clean-image detail.

Standalone timing/sanity check:
    python defense/preprocessing_defense.py --checkpoint checkpoints/best_model.pth \
        --data-dir data/i-raven --jpeg-quality 75 --gaussian-sigma 1.0
"""
import argparse
import io
import os
import sys
import time

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def jpeg_compress_batch(images, quality=75):
    """JPEG-encodes and decodes every panel independently.

    images: (B, N, H, W) float tensor in [0, 1] (N = number of panels, e.g. 8 or 16)
    returns: same shape/dtype/device, round-tripped through JPEG.
    """
    device = images.device
    arr = (images.detach().clamp(0, 1) * 255).round().byte().cpu().numpy()
    B, N, H, W = arr.shape
    out = np.empty_like(arr, dtype=np.float32)

    for b in range(B):
        for n in range(N):
            img = Image.fromarray(arr[b, n], mode="L")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            decoded = np.array(Image.open(buf), dtype=np.float32)
            out[b, n] = decoded

    out = torch.from_numpy(out / 255.0).to(device=device, dtype=images.dtype)
    return out


def preprocess_defense(images, jpeg_quality=75, gaussian_sigma=1.0):
    """Applies JPEG compression then Gaussian smoothing.

    images: (B, N, H, W) float tensor in [0, 1]
    returns: cleaned images, same shape, in [0, 1]
    """
    cleaned = jpeg_compress_batch(images, quality=jpeg_quality)
    if gaussian_sigma > 0:
        k = int(2 * round(3 * gaussian_sigma) + 1)
        k = max(3, k)
        cleaned = TF.gaussian_blur(cleaned, kernel_size=[k, k], sigma=[gaussian_sigma, gaussian_sigma])
    return cleaned.clamp(0, 1)


class PreprocessingDefense:
    def __init__(self, jpeg_quality=75, gaussian_sigma=1.0):
        self.jpeg_quality = jpeg_quality
        self.gaussian_sigma = gaussian_sigma

    def __call__(self, images):
        return preprocess_defense(images, self.jpeg_quality, self.gaussian_sigma)


def _cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--data-dir", default="data/i-raven")
    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument("--gaussian-sigma", type=float, default=1.0)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    from common.dataset import IRavenDataset
    from common.model import load_model
    from common.utils import get_device
    from torch.utils.data import DataLoader

    device = get_device(args.device)
    model, _ = load_model(args.checkpoint, device=device)
    dataset = IRavenDataset(args.data_dir, "test")
    loader = DataLoader(dataset, batch_size=args.num_samples, shuffle=True)
    context, candidates, target, _config = next(iter(loader))
    context, candidates, target = context.to(device), candidates.to(device), target.to(device)

    defense = PreprocessingDefense(args.jpeg_quality, args.gaussian_sigma)

    with torch.no_grad():
        clean_pred = model(context, candidates).argmax(dim=1)

    t0 = time.time()
    clean_context = defense(context)
    clean_candidates = defense(candidates)
    elapsed = time.time() - t0
    ms_per_sample = 1000 * elapsed / context.size(0)

    with torch.no_grad():
        defended_pred = model(clean_context, clean_candidates).argmax(dim=1)

    print(f"clean_acc (no defense) = {(clean_pred == target).float().mean().item() * 100:.1f}%")
    print(f"clean_acc (after defense preprocessing) = {(defended_pred == target).float().mean().item() * 100:.1f}%")
    print(f"preprocessing overhead: {ms_per_sample:.2f} ms/sample "
          f"(jpeg_quality={args.jpeg_quality}, gaussian_sigma={args.gaussian_sigma})")


if __name__ == "__main__":
    _cli()
