#!/usr/bin/env python3
"""PGD (Madry et al., 2018) white-box attack on the 8 context images of an
RPM problem. The 8 answer candidates are left untouched, matching the
realistic threat model described in coding_agent_prompt.md.

Standalone smoke-test:
    python attack/pgd_attack.py --checkpoint checkpoints/best_model.pth \
        --data-dir data/i-raven --epsilon 8 --steps 20 --num-samples 8
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.utils import set_seed, get_device


class PGDAttack:
    """L-infinity PGD attack.

    Args:
        epsilon: max perturbation per pixel, in [0, 1] units (default 8/255).
        alpha: step size, in [0, 1] units (default 2/255).
        num_steps: number of gradient steps.
        random_start: whether to start from a random point in the epsilon ball.
    """

    def __init__(self, epsilon=8 / 255, alpha=2 / 255, num_steps=20, random_start=True):
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.random_start = random_start

    def perturb(self, model, context_images, answer_images, target_label):
        """Returns adversarial context images (context_images.shape), pixels
        clamped to [0, 1] and constrained to an L-inf epsilon ball around the
        original context images.

        model: callable model(context, candidates) -> logits (B, num_candidates)
        context_images: (B, 8, H, W) in [0, 1] -- the images being attacked
        answer_images: (B, 8, H, W) in [0, 1] -- candidates, held fixed
        target_label: (B,) ground-truth candidate index (attack maximizes the
            loss against this label, i.e. an untargeted attack)
        """
        was_training = model.training
        model.eval()

        x = context_images.detach()
        candidates = answer_images.detach()

        if self.random_start:
            delta = torch.empty_like(x).uniform_(-self.epsilon, self.epsilon)
            x_adv = torch.clamp(x + delta, 0.0, 1.0).detach()
        else:
            x_adv = x.clone().detach()

        for _ in range(self.num_steps):
            x_adv.requires_grad_(True)
            logits = model(x_adv, candidates)
            loss = F.cross_entropy(logits, target_label)
            grad = torch.autograd.grad(loss, x_adv)[0]

            x_adv = x_adv.detach() + self.alpha * grad.sign()
            delta = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
            x_adv = torch.clamp(x + delta, 0.0, 1.0).detach()

        if was_training:
            model.train()
        return x_adv


def _smoke_test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--data-dir", default="data/i-raven")
    parser.add_argument("--epsilon", type=float, default=8, help="in 1/255 units")
    parser.add_argument("--alpha", type=float, default=2, help="in 1/255 units")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--num-samples", type=int, default=8)
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

    attack = PGDAttack(epsilon=args.epsilon / 255, alpha=args.alpha / 255, num_steps=args.steps)

    with torch.no_grad():
        clean_pred = model(context, candidates).argmax(dim=1)

    adv_context = attack.perturb(model, context, candidates, target)

    with torch.no_grad():
        adv_pred = model(adv_context, candidates).argmax(dim=1)

    clean_correct = (clean_pred == target)
    flipped = clean_correct & (adv_pred != target)
    asr = flipped.sum().item() / max(clean_correct.sum().item(), 1)

    print(f"clean_acc={clean_correct.float().mean().item() * 100:.1f}%  "
          f"adv_acc={(adv_pred == target).float().mean().item() * 100:.1f}%  "
          f"ASR={asr * 100:.1f}% (on {clean_correct.sum().item()} clean-correct samples)")
    print(f"max perturbation: {(adv_context - context).abs().max().item():.4f} "
          f"(epsilon={args.epsilon / 255:.4f})")


if __name__ == "__main__":
    _smoke_test()
