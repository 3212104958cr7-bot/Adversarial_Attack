#!/usr/bin/env python3
"""Stage 1: train PredRNet on I-RAVEN.

    python train.py --data-dir data/i-raven --epochs 100 --batch-size 128

Logs per-epoch train loss / val accuracy to logs/training_log.json, evaluates
the best checkpoint on the test set (overall + per figure-configuration), and
saves it to checkpoints/best_model.pth.
"""
import argparse
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common.dataset import IRavenDataset, CONFIG_NAMES
from common.model import build_predrnet, RPMModel, save_checkpoint, DEFAULT_ARCH_KWARGS
from common.utils import set_seed, get_device, setup_logger, append_json_log, save_json


def bce_matrix_loss(logits, target):
    """Same loss PredRNet was trained with: binary cross-entropy of each of
    the 8 candidate logits against a one-hot label (loss.py:BinaryCrossEntropy
    in the original repo), rather than plain softmax cross-entropy.
    """
    labels = torch.zeros_like(logits)
    labels.scatter_(1, target.view(-1, 1), 1.0)
    return F.binary_cross_entropy_with_logits(logits, labels)


@torch.no_grad()
def evaluate(model, loader, device, per_config=False):
    model.eval()
    correct, total = 0, 0
    config_correct = {c: 0 for c in CONFIG_NAMES}
    config_total = {c: 0 for c in CONFIG_NAMES}

    for context, candidates, target, config in loader:
        context = context.to(device)
        candidates = candidates.to(device)
        target = target.to(device)

        logits = model(context, candidates)
        pred = logits.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)

        if per_config:
            for c, ok in zip(config, (pred == target).tolist()):
                config_total[c] += 1
                config_correct[c] += int(ok)

    acc = 100.0 * correct / max(total, 1)
    if not per_config:
        return acc

    per_cfg_acc = {
        c: (100.0 * config_correct[c] / config_total[c] if config_total[c] else None)
        for c in CONFIG_NAMES
    }
    return acc, per_cfg_acc


def train_one_epoch(model, loader, optimizer, device, epoch, logger, print_freq=50):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    t0 = time.time()

    for i, (context, candidates, target, _config) in enumerate(loader):
        context = context.to(device)
        candidates = candidates.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        logits = model(context, candidates)
        loss = bce_matrix_loss(logits, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * target.size(0)
        correct += (logits.argmax(dim=1) == target).sum().item()
        total += target.size(0)

        if i % print_freq == 0 or i == len(loader) - 1:
            logger.info(
                f"epoch {epoch} [{i + 1}/{len(loader)}] "
                f"loss {running_loss / total:.4f} acc {100.0 * correct / total:.2f}"
            )

    return running_loss / max(total, 1), 100.0 * correct / max(total, 1), time.time() - t0


def main():
    parser = argparse.ArgumentParser(description="Train PredRNet on I-RAVEN")
    parser.add_argument("--data-dir", default="data/i-raven")
    parser.add_argument("--image-size", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-extra-stages", type=int, default=3)
    parser.add_argument("--block-drop", type=float, default=0.1)
    parser.add_argument("--classifier-drop", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--ckpt-dir", default="checkpoints")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--print-freq", type=int, default=50)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    logger = setup_logger("train", os.path.join(args.log_dir, "train.log"))
    logger.info(f"device={device}")

    train_set = IRavenDataset(args.data_dir, "train", image_size=args.image_size)
    val_set = IRavenDataset(args.data_dir, "val", image_size=args.image_size)
    test_set = IRavenDataset(args.data_dir, "test", image_size=args.image_size)
    logger.info(f"train={len(train_set)} val={len(val_set)} test={len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=device.type == "cuda")

    arch_kwargs = {
        **DEFAULT_ARCH_KWARGS,
        "num_extra_stages": args.num_extra_stages,
        "block_drop": args.block_drop,
        "classifier_drop": args.classifier_drop,
    }
    net = build_predrnet(**arch_kwargs)
    model = RPMModel(net).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    training_log_path = os.path.join(args.log_dir, "training_log.json")
    best_acc = 0.0
    best_path = os.path.join(args.ckpt_dir, "best_model.pth")

    for epoch in range(args.epochs):
        train_loss, train_acc, epoch_time = train_one_epoch(
            model, train_loader, optimizer, device, epoch, logger, args.print_freq
        )
        val_acc = evaluate(model, val_loader, device)
        logger.info(f"epoch {epoch} done in {epoch_time:.1f}s train_loss={train_loss:.4f} "
                    f"train_acc={train_acc:.2f} val_acc={val_acc:.2f}")

        append_json_log(training_log_path, {
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_acc": val_acc, "epoch_time_sec": epoch_time,
        })

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(best_path, net, epoch, best_acc, arch_kwargs, optimizer)
            logger.info(f"new best val_acc={best_acc:.2f}, saved to {best_path}")

    # final test evaluation with the best checkpoint
    ckpt = torch.load(best_path, map_location=device)
    from common.model import strip_module_prefix
    net.load_state_dict(strip_module_prefix(ckpt["state_dict"]))
    test_acc, per_cfg_acc = evaluate(model, test_loader, device, per_config=True)
    logger.info(f"FINAL test_acc={test_acc:.2f} per_config={per_cfg_acc}")

    save_json(os.path.join(args.log_dir, "final_test_results.json"), {
        "overall_test_acc": test_acc,
        "per_config_test_acc": per_cfg_acc,
        "best_val_acc": best_acc,
        "best_epoch": ckpt["epoch"],
    })


if __name__ == "__main__":
    main()
