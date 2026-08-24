#!/usr/bin/env python3
"""C&W-style targeted attack (Carlini & Wagner, 2017 formulation, adapted to
an L-inf budget) that forces PredRNet to pick the *worst* distractor: the
candidate answer image that is furthest (pixel L2) from the true correct
answer. This demonstrates intentional misdirection rather than plain
confusion, as a secondary experiment to the untargeted PGD attack.

Standalone smoke-test:
    python attack/cw_attack.py --checkpoint checkpoints/best_model.pth \
        --data-dir data/i-raven --num-samples 8
"""
import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.utils import set_seed, get_device


def select_worst_distractor(answer_images, true_label):
    """For each problem, returns the candidate index (!= true_label) whose
    image is furthest in pixel L2 distance from the true answer image.

    answer_images: (B, 8, H, W) in [0, 1]
    true_label: (B,)
    returns: (B,) target label for the targeted attack
    """
    B, C, H, W = answer_images.shape
    correct = answer_images[torch.arange(B), true_label]  # (B, H, W)
    diffs = (answer_images - correct.unsqueeze(1)).flatten(2).pow(2).sum(-1)  # (B, C)
    diffs[torch.arange(B), true_label] = -1.0  # exclude the true label itself
    return diffs.argmax(dim=1)


class CWAttack:
    """Targeted C&W-style attack on the 8 context images.

    Optimizes, in tanh space (so the [0, 1] box constraint is automatic):
        cost = ||x_adv - x||_2^2 + c * f(x_adv)
        f(x_adv) = max(max_{i != target} logit_i - logit_target, -kappa)
    which pushes the target class's logit above all others by margin kappa.
    An optional L-inf epsilon further constrains the search, matching the
    attack budget used for the untargeted PGD experiments.
    """

    def __init__(self, c=1.0, kappa=0.0, num_steps=100, lr=0.01, epsilon=None):
        self.c = c
        self.kappa = kappa
        self.num_steps = num_steps
        self.lr = lr
        self.epsilon = epsilon  # None = unconstrained (pure C&W); else L-inf ball

    @staticmethod
    def _to_w(x):
        x = torch.clamp(x, 1e-6, 1 - 1e-6)
        return torch.atanh(2 * x - 1)

    @staticmethod
    def _to_x(w):
        return 0.5 * (torch.tanh(w) + 1)

    def perturb(self, model, context_images, answer_images, true_label, target_label=None):
        """Returns (adversarial context images, target labels used)."""
        was_training = model.training
        model.eval()

        x = context_images.detach()
        candidates = answer_images.detach()

        if target_label is None:
            target_label = select_worst_distractor(candidates, true_label)
        target_label = target_label.detach()

        w = self._to_w(x).clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([w], lr=self.lr)

        best_adv = x.clone()
        best_l2 = torch.full((x.size(0),), float("inf"), device=x.device)
        one_hot = F.one_hot(target_label, candidates.size(1)).float()

        for _ in range(self.num_steps):
            x_adv = self._to_x(w)
            if self.epsilon is not None:
                delta = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
                x_adv = torch.clamp(x + delta, 0.0, 1.0)

            logits = model(x_adv, candidates)
            target_logit = (logits * one_hot).sum(dim=1)
            other_logit = (logits - one_hot * 1e4).max(dim=1)[0]
            f_loss = torch.clamp(other_logit - target_logit + self.kappa, min=0.0)

            l2 = (x_adv - x).flatten(1).pow(2).sum(dim=1)
            cost = (l2 + self.c * f_loss).sum()

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            with torch.no_grad():
                success = f_loss == 0.0
                improved = success & (l2 < best_l2)
                if improved.any():
                    best_l2 = torch.where(improved, l2, best_l2)
                    best_adv = torch.where(
                        improved.view(-1, 1, 1, 1), x_adv.detach(), best_adv
                    )

        # fall back to the last iterate for samples that never fully succeeded
        never_succeeded = torch.isinf(best_l2)
        if never_succeeded.any():
            with torch.no_grad():
                x_adv = self._to_x(w)
                if self.epsilon is not None:
                    delta = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
                    x_adv = torch.clamp(x + delta, 0.0, 1.0)
                best_adv = torch.where(
                    never_succeeded.view(-1, 1, 1, 1), x_adv, best_adv
                )

        if was_training:
            model.train()
        return best_adv.detach(), target_label


def _smoke_test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--data-dir", default="data/i-raven")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=None, help="in 1/255 units, optional")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device("auto")

    from common.dataset import IRavenDataset
    from common.model import load_model
    from torch.utils.data import DataLoader

    model, _ = load_model(args.checkpoint, device=device)
    dataset = IRavenDataset(args.data_dir, "test")
    loader = DataLoader(dataset, batch_size=args.num_samples, shuffle=True)
    context, candidates, target, config = next(iter(loader))
    context, candidates, target = context.to(device), candidates.to(device), target.to(device)

    eps = args.epsilon / 255 if args.epsilon is not None else None
    attack = CWAttack(c=args.c, num_steps=args.steps, epsilon=eps)

    with torch.no_grad():
        clean_pred = model(context, candidates).argmax(dim=1)

    adv_context, worst_target = attack.perturb(model, context, candidates, target)

    with torch.no_grad():
        adv_pred = model(adv_context, candidates).argmax(dim=1)

    hijack_rate = (adv_pred == worst_target).float().mean().item()
    print(f"clean_acc={(clean_pred == target).float().mean().item() * 100:.1f}%  "
          f"hijack_to_worst_distractor_rate={hijack_rate * 100:.1f}%")


if __name__ == "__main__":
    _smoke_test()
