#!/usr/bin/env python3
"""Defense 2: lightweight adversarial fine-tuning.

Fine-tunes the pretrained PredRNet checkpoint for a few epochs on a 50/50 mix
of clean and PGD-perturbed context images, using a weaker attack (fewer
steps) than the one used for evaluation, per coding_agent_prompt.md.

Usage:
    python defense/adversarial_training.py --checkpoint checkpoints/best_model.pth \
        --data-dir data/i-raven --epochs 10 --epsilon 8 --steps 7
"""
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.dataset import IRavenDataset
from common.model import load_model, save_checkpoint
from common.utils import set_seed, get_device, setup_logger, append_json_log
from attack.pgd_attack import PGDAttack


def bce_matrix_loss(logits, target):
    labels = torch.zeros_like(logits)
    labels.scatter_(1, target.view(-1, 1), 1.0)
    return F.binary_cross_entropy_with_logits(logits, labels)


@torch.no_grad()
def evaluate_clean(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for context, candidates, target, _config in loader:
        context, candidates, target = context.to(device), candidates.to(device), target.to(device)
        pred = model(context, candidates).argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)
    return 100.0 * correct / max(total, 1)


def adv_train_one_epoch(model, loader, optimizer, attack, device, epoch, logger, print_freq=50):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    t0 = time.time()

    for i, (context, candidates, target, _config) in enumerate(loader):
        context = context.to(device)
        candidates = candidates.to(device)
        target = target.to(device)

        # generate PGD examples for the whole batch, then keep only half of them
        adv_context = attack.perturb(model, context, candidates, target)
        half = context.size(0) // 2
        mix_mask = torch.zeros(context.size(0), dtype=torch.bool, device=device)
        mix_mask[torch.randperm(context.size(0), device=device)[:half]] = True
        mixed_context = torch.where(mix_mask.view(-1, 1, 1, 1), adv_context, context)

        model.train()
        optimizer.zero_grad()
        logits = model(mixed_context, candidates)
        loss = bce_matrix_loss(logits, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * target.size(0)
        correct += (logits.argmax(dim=1) == target).sum().item()
        total += target.size(0)

        if i % print_freq == 0 or i == len(loader) - 1:
            logger.info(
                f"[adv-train] epoch {epoch} [{i + 1}/{len(loader)}] "
                f"loss {running_loss / total:.4f} acc {100.0 * correct / total:.2f}"
            )

    return running_loss / max(total, 1), 100.0 * correct / max(total, 1), time.time() - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--data-dir", default="data/i-raven")
    parser.add_argument("--image-size", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--epsilon", type=float, default=8, help="PGD epsilon for adv examples, in 1/255 units")
    parser.add_argument("--alpha", type=float, default=2, help="PGD step size, in 1/255 units")
    parser.add_argument("--steps", type=int, default=7, help="PGD steps for adv-training examples (weaker than eval attack)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--out-checkpoint", default="checkpoints/adv_trained_model.pth")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--print-freq", type=int, default=50)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_checkpoint), exist_ok=True)

    logger = setup_logger("adv_train", os.path.join(args.log_dir, "adv_train.log"))
    logger.info(f"device={device}")

    model, ckpt = load_model(args.checkpoint, device=device)
    arch_kwargs = ckpt.get("arch_kwargs", {})
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_set = IRavenDataset(args.data_dir, "train", image_size=args.image_size)
    val_set = IRavenDataset(args.data_dir, "val", image_size=args.image_size)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=device.type == "cuda")

    attack = PGDAttack(epsilon=args.epsilon / 255, alpha=args.alpha / 255, num_steps=args.steps)

    log_path = os.path.join(args.log_dir, "defense_log.json")
    best_val = evaluate_clean(model, val_loader, device)
    logger.info(f"starting val_acc (clean) = {best_val:.2f}")

    for epoch in range(args.epochs):
        train_loss, train_acc, epoch_time = adv_train_one_epoch(
            model, train_loader, optimizer, attack, device, epoch, logger, args.print_freq
        )
        val_acc = evaluate_clean(model, val_loader, device)
        logger.info(f"[adv-train] epoch {epoch} done in {epoch_time:.1f}s "
                    f"loss={train_loss:.4f} train_acc={train_acc:.2f} val_acc={val_acc:.2f}")

        append_json_log(log_path, {
            "phase": "adversarial_training", "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_acc_clean": val_acc, "epoch_time_sec": epoch_time,
        })

        save_checkpoint(args.out_checkpoint, model.net, epoch, val_acc, arch_kwargs, optimizer)

    logger.info(f"saved adversarially fine-tuned model to {args.out_checkpoint}")


if __name__ == "__main__":
    main()
