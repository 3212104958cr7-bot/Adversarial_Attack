"""PredRNet model construction, checkpoint I/O, and a thin wrapper that
operates directly on [0, 1]-scaled images (the convention used by the
attack/defense code), rather than PredRNet's native [-1, 1] input.
"""
import argparse

import torch
import torch.nn as nn

from common.paths import PREDRNET_DIR  # noqa: F401
from networks import create_net

DEFAULT_ARCH_KWARGS = dict(
    arch="predrnet_raven",
    num_extra_stages=3,
    block_drop=0.1,
    classifier_drop=0.1,
    classifier_hidreduce=4,
    num_filters=32,
    in_channels=1,
    enable_rc=False,
)


def build_predrnet(**overrides):
    """Builds the raw PredRNet nn.Module (16 grayscale panels in -> 8 logits out)."""
    kwargs = {**DEFAULT_ARCH_KWARGS, **overrides}
    args = argparse.Namespace(**kwargs)
    return create_net(args)


class RPMModel(nn.Module):
    """Wraps PredRNet so callers work purely in [0, 1] pixel space.

    forward(context, candidates) -> logits
        context:    (B, 8, H, W) float in [0, 1]
        candidates: (B, 8, H, W) float in [0, 1]
        logits:     (B, 8) - one score per candidate answer
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, context, candidates):
        images = torch.cat([context, candidates], dim=1)  # (B, 16, H, W)
        images = images * 2.0 - 1.0  # equivalent to predrnet.utils.normalize_image on 0-255 scale
        return self.net(images)

    def extract_pe_features(self, context, candidates):
        """Returns the pooled prediction-error relation features (pre-classifier),
        one (featr_dims,) vector per candidate, shape (B, 8, featr_dims).
        Used by attack/analysis.py for the t-SNE feature-space analysis.
        """
        images = torch.cat([context, candidates], dim=1)
        images = images * 2.0 - 1.0
        net = self.net
        b = images.size(0)
        img_featrs = net.extract_features(images)
        relations = net.extract_relations(img_featrs)
        relations = relations.reshape(b, net.ou_channels, -1)
        relations = torch.nn.functional.adaptive_avg_pool1d(relations, net.featr_dims)
        return relations  # (B, 8, featr_dims)


def strip_module_prefix(state_dict):
    """Undoes nn.DataParallel's 'module.' prefix, if present."""
    if not any(k.startswith("module.") for k in state_dict):
        return state_dict
    return {k[len("module."):] if k.startswith("module.") else k: v
            for k, v in state_dict.items()}


def save_checkpoint(path, net, epoch, best_acc, arch_kwargs, optimizer=None):
    state = {
        "state_dict": net.state_dict(),
        "epoch": epoch,
        "best_acc": best_acc,
        "arch_kwargs": arch_kwargs,
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    torch.save(state, path)


def load_model(checkpoint_path, device="cpu", **arch_overrides):
    """Builds a PredRNet + RPMModel and loads weights from a checkpoint saved
    by save_checkpoint() above (also tolerates the original PredRNet repo's
    checkpoint.pth.tar format, which wraps state_dict with a 'module.' prefix).
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    arch_kwargs = ckpt.get("arch_kwargs", {})
    arch_kwargs.update(arch_overrides)

    net = build_predrnet(**arch_kwargs)
    state_dict = strip_module_prefix(ckpt["state_dict"])
    net.load_state_dict(state_dict)
    net.to(device)

    model = RPMModel(net).to(device)
    return model, ckpt
