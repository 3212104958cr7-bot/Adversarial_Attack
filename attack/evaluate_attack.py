#!/usr/bin/env python3
"""Stage 2 evaluation: run the PGD attack (and optionally the C&W targeted
variant) over I-RAVEN test problems, compute Attack Success Rate (ASR), run
an epsilon/num_steps ablation, and save visualized adversarial examples.

Usage:
    python attack/evaluate_attack.py --checkpoint checkpoints/best_model.pth \
        --data-dir data/i-raven --epsilon 8 --steps 20 --output results/
"""
import argparse
import os
import random
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.dataset import IRavenDataset, CONFIG_NAMES
from common.model import load_model
from common.utils import set_seed, get_device, save_json, setup_logger
from attack.pgd_attack import PGDAttack
from attack.cw_attack import CWAttack, select_worst_distractor


def sample_subset_indices(dataset, num_per_config, seed):
    """Picks up to num_per_config indices per figure-configuration without
    materializing any images, using the underlying file list.
    """
    by_config = defaultdict(list)
    for i, fname in enumerate(dataset.base.file_names):
        folder = fname.split(os.sep)[0]
        by_config[folder].append(i)

    rng = random.Random(seed)
    indices = []
    for folder, idxs in by_config.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        indices.extend(idxs[:num_per_config])
    return indices


def summarize(records):
    """Overall + per-config clean accuracy, adversarial accuracy, and ASR.
    ASR = fraction of clean-correct samples that flip to incorrect under attack.
    """
    def _summ(recs):
        n = len(recs)
        if n == 0:
            return {"n": 0, "clean_acc": None, "adv_acc": None, "asr": None}
        clean_correct = [r["clean_pred"] == r["true_label"] for r in recs]
        adv_correct = [r["adv_pred"] == r["true_label"] for r in recs]
        n_clean_correct = sum(clean_correct)
        n_flipped = sum(
            1 for cc, r in zip(clean_correct, recs)
            if cc and r["adv_pred"] != r["true_label"]
        )
        return {
            "n": n,
            "clean_acc": 100.0 * n_clean_correct / n,
            "adv_acc": 100.0 * sum(adv_correct) / n,
            "asr": 100.0 * n_flipped / n_clean_correct if n_clean_correct else None,
        }

    result = {"overall": _summ(records)}
    by_config = defaultdict(list)
    for r in records:
        by_config[r["config"]].append(r)
    for cfg, recs in by_config.items():
        result[cfg] = _summ(recs)
    return result


def run_attack_pass(model, loader, device, attack):
    records = []
    for context, candidates, target, config in loader:
        context = context.to(device)
        candidates = candidates.to(device)
        target = target.to(device)

        with torch.no_grad():
            clean_pred = model(context, candidates).argmax(dim=1)

        adv_context = attack.perturb(model, context, candidates, target)

        with torch.no_grad():
            adv_pred = model(adv_context, candidates).argmax(dim=1)

        for i in range(target.size(0)):
            records.append({
                "config": config[i],
                "true_label": int(target[i]),
                "clean_pred": int(clean_pred[i]),
                "adv_pred": int(adv_pred[i]),
            })
    return records


def save_example_figure(path, context, adv_context, clean_pred, adv_pred, true_label, config):
    delta = (adv_context - context)
    delta_vis = torch.clamp(delta * 10 + 0.5, 0, 1)  # amplify x10, recenter for display

    fig, axes = plt.subplots(3, 8, figsize=(16, 6))
    for j in range(8):
        axes[0, j].imshow(context[j].cpu(), cmap="gray", vmin=0, vmax=1)
        axes[0, j].axis("off")
        axes[1, j].imshow(adv_context[j].cpu(), cmap="gray", vmin=0, vmax=1)
        axes[1, j].axis("off")
        axes[2, j].imshow(delta_vis[j].cpu(), cmap="seismic", vmin=0, vmax=1)
        axes[2, j].axis("off")
    axes[0, 0].set_ylabel("original", fontsize=10)
    axes[1, 0].set_ylabel("adversarial", fontsize=10)
    axes[2, 0].set_ylabel("perturbation x10", fontsize=10)

    fig.suptitle(
        f"[{config}] true={true_label}  clean_pred={clean_pred}  adv_pred={adv_pred}"
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def run_primary_attack_with_examples(model, dataset, indices, device, attack, examples_per_config, out_dir):
    """Runs the primary attack sample-by-sample (batch size 1) so we can both
    collect per-sample records and save N visualized examples per config.
    """
    loader = DataLoader(Subset(dataset, indices), batch_size=32, shuffle=False)
    saved_per_config = defaultdict(int)
    records = []

    for context, candidates, target, config in loader:
        context_d = context.to(device)
        candidates_d = candidates.to(device)
        target_d = target.to(device)

        with torch.no_grad():
            clean_pred = model(context_d, candidates_d).argmax(dim=1)
        adv_context = attack.perturb(model, context_d, candidates_d, target_d)
        with torch.no_grad():
            adv_pred = model(adv_context, candidates_d).argmax(dim=1)

        for i in range(target_d.size(0)):
            cfg = config[i]
            records.append({
                "config": cfg,
                "true_label": int(target_d[i]),
                "clean_pred": int(clean_pred[i]),
                "adv_pred": int(adv_pred[i]),
            })
            if saved_per_config[cfg] < examples_per_config:
                fig_path = os.path.join(
                    out_dir, "adversarial_examples", cfg,
                    f"example_{saved_per_config[cfg]}.png"
                )
                save_example_figure(
                    fig_path, context[i], adv_context[i].cpu(),
                    int(clean_pred[i]), int(adv_pred[i]), int(target_d[i]), cfg,
                )
                saved_per_config[cfg] += 1

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--data-dir", default="data/i-raven")
    parser.add_argument("--image-size", type=int, default=80)
    parser.add_argument("--output", default="results")
    parser.add_argument("--num-samples", type=int, default=1000,
                         help="max samples per figure-configuration for the primary attack run")
    parser.add_argument("--ablation-num-samples", type=int, default=200,
                         help="max samples per configuration for each ablation grid point (kept smaller for speed)")
    parser.add_argument("--epsilon", type=float, default=8, help="primary epsilon, in 1/255 units")
    parser.add_argument("--alpha", type=float, default=2, help="PGD step size, in 1/255 units")
    parser.add_argument("--steps", type=int, default=20, help="primary num_steps")
    parser.add_argument("--epsilons", default="2,4,8,16", help="comma list, in 1/255 units, for the ablation table")
    parser.add_argument("--steps-list", default="5,10,20,40", help="comma list of num_steps for the ablation table")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--include-cw", action="store_true", help="also run the C&W targeted variant")
    parser.add_argument("--cw-steps", type=int, default=100)
    parser.add_argument("--cw-c", type=float, default=1.0)
    parser.add_argument("--examples-per-config", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    os.makedirs(args.output, exist_ok=True)
    logger = setup_logger("evaluate_attack", os.path.join("logs", "attack_log.json").replace(".json", ".log"))

    model, _ = load_model(args.checkpoint, device=device)
    dataset = IRavenDataset(args.data_dir, "test", image_size=args.image_size)

    # ---- primary attack: full per-sample records + visualized examples ----
    primary_indices = sample_subset_indices(dataset, args.num_samples, args.seed)
    logger.info(f"primary attack: {len(primary_indices)} samples, epsilon={args.epsilon}/255, steps={args.steps}")

    primary_attack = PGDAttack(epsilon=args.epsilon / 255, alpha=args.alpha / 255, num_steps=args.steps)
    records = run_primary_attack_with_examples(
        model, dataset, primary_indices, device, primary_attack,
        args.examples_per_config, args.output,
    )
    save_json(os.path.join(args.output, "attack_results.json"), records)
    primary_summary = summarize(records)
    primary_asr = primary_summary["overall"]["asr"]
    primary_asr_str = f"{primary_asr:.2f}%" if primary_asr is not None else "N/A (no clean-correct samples)"
    logger.info(f"primary ASR (overall) = {primary_asr_str}")

    # ---- ablation over epsilon x num_steps ----
    summary_rows = []
    if not args.skip_ablation:
        epsilons = [float(e) for e in args.epsilons.split(",")]
        steps_list = [int(s) for s in args.steps_list.split(",")]
        ablation_indices = sample_subset_indices(dataset, args.ablation_num_samples, args.seed)
        ablation_loader = DataLoader(Subset(dataset, ablation_indices), batch_size=args.batch_size, shuffle=False)

        for eps in epsilons:
            for steps in steps_list:
                attack = PGDAttack(epsilon=eps / 255, alpha=args.alpha / 255, num_steps=steps)
                recs = run_attack_pass(model, ablation_loader, device, attack)
                summ = summarize(recs)
                asr = summ["overall"]["asr"]
                asr_str = f"{asr:.2f}%" if asr is not None else "N/A (no clean-correct samples)"
                logger.info(f"epsilon={eps}/255 steps={steps} -> ASR={asr_str}")
                for cfg, s in summ.items():
                    summary_rows.append({
                        "config": cfg, "epsilon_255": eps, "num_steps": steps,
                        "n": s["n"], "clean_acc": s["clean_acc"],
                        "adv_acc": s["adv_acc"], "asr": s["asr"],
                    })
    else:
        for cfg, s in primary_summary.items():
            summary_rows.append({
                "config": cfg, "epsilon_255": args.epsilon, "num_steps": args.steps,
                "n": s["n"], "clean_acc": s["clean_acc"],
                "adv_acc": s["adv_acc"], "asr": s["asr"],
            })

    import csv
    csv_path = os.path.join(args.output, "attack_summary_table.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "epsilon_255", "num_steps", "n", "clean_acc", "adv_acc", "asr"])
        writer.writeheader()
        writer.writerows(summary_rows)
    logger.info(f"wrote {csv_path}")

    # ---- optional C&W targeted variant ----
    if args.include_cw:
        cw_indices = sample_subset_indices(dataset, min(args.num_samples, 200), args.seed)
        cw_loader = DataLoader(Subset(dataset, cw_indices), batch_size=args.batch_size, shuffle=False)
        cw_attack = CWAttack(c=args.cw_c, num_steps=args.cw_steps, epsilon=args.epsilon / 255)

        cw_records = []
        for context, candidates, target, config in cw_loader:
            context, candidates, target = context.to(device), candidates.to(device), target.to(device)
            with torch.no_grad():
                clean_pred = model(context, candidates).argmax(dim=1)
            adv_context, worst_target = cw_attack.perturb(model, context, candidates, target)
            with torch.no_grad():
                adv_pred = model(adv_context, candidates).argmax(dim=1)
            for i in range(target.size(0)):
                cw_records.append({
                    "config": config[i],
                    "true_label": int(target[i]),
                    "clean_pred": int(clean_pred[i]),
                    "adv_pred": int(adv_pred[i]),
                    "target_worst_distractor": int(worst_target[i]),
                    "hijacked_to_target": bool(adv_pred[i] == worst_target[i]),
                })
        save_json(os.path.join(args.output, "cw_attack_results.json"), cw_records)
        hijack_rate = 100.0 * sum(r["hijacked_to_target"] for r in cw_records) / max(len(cw_records), 1)
        logger.info(f"C&W hijack-to-worst-distractor rate = {hijack_rate:.2f}%")

    print("Done.")
    print(f"  {os.path.join(args.output, 'attack_results.json')}")
    print(f"  {csv_path}")
    print(f"  {os.path.join(args.output, 'adversarial_examples/')}")


if __name__ == "__main__":
    main()
