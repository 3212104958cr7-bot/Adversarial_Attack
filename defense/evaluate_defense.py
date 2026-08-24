#!/usr/bin/env python3
"""Stage 4 evaluation: compares Clean / After-PGD-Attack / After-PGD-Attack+Defense
accuracy (and ASR) for both defenses, and measures the preprocessing defense's
inference-time overhead.

Threat model:
  - Preprocessing defense: the attacker does not adapt to the defense (a
    standard simplifying assumption for non-differentiable input
    transforms); the PGD attack is generated once against the undefended
    base model, then the defense is applied at inference time.
  - Adversarial-training defense: the retrained weights *are* the deployed
    model, so the white-box PGD attack is regenerated directly against it.

Usage:
    python defense/evaluate_defense.py --defense preprocessing \
        --checkpoint checkpoints/best_model.pth --data-dir data/i-raven
    python defense/evaluate_defense.py --defense adversarial_training \
        --checkpoint checkpoints/best_model.pth \
        --adv-checkpoint checkpoints/adv_trained_model.pth --data-dir data/i-raven
    python defense/evaluate_defense.py --defense both ...
"""
import argparse
import csv
import os
import sys
import time

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.dataset import IRavenDataset
from common.model import load_model
from common.utils import set_seed, get_device, append_json_log
from attack.pgd_attack import PGDAttack
from attack.evaluate_attack import sample_subset_indices
from defense.preprocessing_defense import PreprocessingDefense


@torch.no_grad()
def _predict(model, context, candidates):
    return model(context, candidates).argmax(dim=1)


def evaluate_preprocessing_defense(model, loader, attack, defense, device):
    n = clean_correct_n = adv_correct_n = def_correct_n = flipped_by_attack = flipped_after_defense = 0
    overhead_ms_total = 0.0

    for context, candidates, target, _config in loader:
        context, candidates, target = context.to(device), candidates.to(device), target.to(device)

        clean_pred = _predict(model, context, candidates)
        clean_correct = clean_pred == target

        adv_context = attack.perturb(model, context, candidates, target)
        adv_pred = _predict(model, adv_context, candidates)

        t0 = time.time()
        def_context = defense(adv_context)
        def_candidates = defense(candidates)
        overhead_ms_total += 1000 * (time.time() - t0)
        def_pred = _predict(model, def_context, def_candidates)

        n += target.size(0)
        clean_correct_n += clean_correct.sum().item()
        adv_correct_n += (adv_pred == target).sum().item()
        def_correct_n += (def_pred == target).sum().item()
        flipped_by_attack += (clean_correct & (adv_pred != target)).sum().item()
        flipped_after_defense += (clean_correct & (def_pred != target)).sum().item()

    return {
        "clean_acc": 100.0 * clean_correct_n / n,
        "adv_acc": 100.0 * adv_correct_n / n,
        "defended_acc": 100.0 * def_correct_n / n,
        "asr_no_defense": 100.0 * flipped_by_attack / max(clean_correct_n, 1),
        "asr_with_defense": 100.0 * flipped_after_defense / max(clean_correct_n, 1),
        "overhead_ms_per_sample": overhead_ms_total / n,
        "n": n,
    }


def evaluate_adv_trained_defense(base_model, adv_model, loader, attack_on_base, attack_on_adv, device):
    n = clean_correct_n = base_adv_correct_n = adv_model_correct_n = 0
    flipped_base = flipped_adv_model = 0

    for context, candidates, target, _config in loader:
        context, candidates, target = context.to(device), candidates.to(device), target.to(device)

        clean_pred = _predict(base_model, context, candidates)
        clean_correct = clean_pred == target

        base_adv_context = attack_on_base.perturb(base_model, context, candidates, target)
        base_adv_pred = _predict(base_model, base_adv_context, candidates)

        adv_model_adv_context = attack_on_adv.perturb(adv_model, context, candidates, target)
        adv_model_pred = _predict(adv_model, adv_model_adv_context, candidates)

        n += target.size(0)
        clean_correct_n += clean_correct.sum().item()
        base_adv_correct_n += (base_adv_pred == target).sum().item()
        adv_model_correct_n += (adv_model_pred == target).sum().item()
        flipped_base += (clean_correct & (base_adv_pred != target)).sum().item()
        flipped_adv_model += (clean_correct & (adv_model_pred != target)).sum().item()

    return {
        "clean_acc": 100.0 * clean_correct_n / n,
        "adv_acc": 100.0 * base_adv_correct_n / n,
        "defended_acc": 100.0 * adv_model_correct_n / n,
        "asr_no_defense": 100.0 * flipped_base / max(clean_correct_n, 1),
        "asr_with_defense": 100.0 * flipped_adv_model / max(clean_correct_n, 1),
        "n": n,
    }


def write_rows(csv_writer, defense_name, stats):
    csv_writer.writerow({
        "defense": defense_name, "metric": "Overall Accuracy",
        "clean": f"{stats['clean_acc']:.2f}",
        "after_attack": f"{stats['adv_acc']:.2f}",
        "after_attack_plus_defense": f"{stats['defended_acc']:.2f}",
    })
    csv_writer.writerow({
        "defense": defense_name, "metric": "ASR",
        "clean": "-",
        "after_attack": f"{stats['asr_no_defense']:.2f}",
        "after_attack_plus_defense": f"{stats['asr_with_defense']:.2f}",
    })
    csv_writer.writerow({
        "defense": defense_name, "metric": "Accuracy drop vs. clean",
        "clean": "-",
        "after_attack": f"{stats['clean_acc'] - stats['adv_acc']:.2f}",
        "after_attack_plus_defense": f"{stats['clean_acc'] - stats['defended_acc']:.2f}",
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--defense", choices=["preprocessing", "adversarial_training", "both"], default="both")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--adv-checkpoint", default="checkpoints/adv_trained_model.pth")
    parser.add_argument("--data-dir", default="data/i-raven")
    parser.add_argument("--epsilon", type=float, default=8, help="in 1/255 units")
    parser.add_argument("--alpha", type=float, default=2)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument("--gaussian-sigma", type=float, default=1.0)
    parser.add_argument("--num-samples", type=int, default=200, help="per figure-configuration")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--output", default="results/defense_comparison_table.csv")
    parser.add_argument("--log-dir", default="logs")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    dataset = IRavenDataset(args.data_dir, "test")
    indices = sample_subset_indices(dataset, args.num_samples, args.seed)
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=False)

    attack = PGDAttack(epsilon=args.epsilon / 255, alpha=args.alpha / 255, num_steps=args.steps)

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["defense", "metric", "clean", "after_attack", "after_attack_plus_defense"])
        writer.writeheader()

        if args.defense in ("preprocessing", "both"):
            print("Evaluating preprocessing defense (JPEG + Gaussian)...")
            base_model, _ = load_model(args.checkpoint, device=device)
            defense = PreprocessingDefense(args.jpeg_quality, args.gaussian_sigma)
            stats = evaluate_preprocessing_defense(base_model, loader, attack, defense, device)
            write_rows(writer, "preprocessing", stats)
            append_json_log(os.path.join(args.log_dir, "defense_log.json"), {
                "phase": "evaluation", "defense": "preprocessing", **stats,
            })
            print(f"  clean={stats['clean_acc']:.1f}% adv={stats['adv_acc']:.1f}% "
                  f"defended={stats['defended_acc']:.1f}% "
                  f"overhead={stats['overhead_ms_per_sample']:.2f} ms/sample")

        if args.defense in ("adversarial_training", "both"):
            if not os.path.exists(args.adv_checkpoint):
                print(f"WARNING: {args.adv_checkpoint} not found; run defense/adversarial_training.py first. Skipping.")
            else:
                print("Evaluating adversarial-training defense...")
                base_model, _ = load_model(args.checkpoint, device=device)
                adv_model, _ = load_model(args.adv_checkpoint, device=device)
                attack_on_adv = PGDAttack(epsilon=args.epsilon / 255, alpha=args.alpha / 255, num_steps=args.steps)
                stats = evaluate_adv_trained_defense(base_model, adv_model, loader, attack, attack_on_adv, device)
                write_rows(writer, "adversarial_training", stats)
                append_json_log(os.path.join(args.log_dir, "defense_log.json"), {
                    "phase": "evaluation", "defense": "adversarial_training", **stats,
                })
                print(f"  clean={stats['clean_acc']:.1f}% adv(base)={stats['adv_acc']:.1f}% "
                      f"adv(adv-trained)={stats['defended_acc']:.1f}%")

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
