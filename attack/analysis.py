#!/usr/bin/env python3
"""Stage 3: analyze *why* the PGD attack succeeds and *what* it breaks.

Automates:
  1. Per-configuration accuracy-drop breakdown (from attack_results.json)
  2. Gradient-magnitude saliency maps (a lightweight Grad-CAM stand-in) over
     the context images, to see whether perturbation-sensitive pixels align
     with the geometric shapes
  3. t-SNE of PredRNet's prediction-error (PE) embeddings, clean vs adversarial

Step 4 (rule disruption: Number/Position/Type/Size/Color) is left as a
qualitative section in the generated report, as specified.

Usage:
    python attack/analysis.py --checkpoint checkpoints/best_model.pth \
        --data-dir data/i-raven --attack-results results/attack_results.json \
        --output results/analysis
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.dataset import IRavenDataset
from common.model import load_model
from common.utils import set_seed, get_device, save_json
from attack.pgd_attack import PGDAttack
from attack.evaluate_attack import sample_subset_indices, summarize


def per_config_breakdown(attack_results_path, out_csv):
    with open(attack_results_path) as f:
        records = json.load(f)
    summary = summarize(records)

    import csv
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "n", "clean_acc", "adv_acc", "asr", "acc_drop"])
        writer.writeheader()
        for cfg, s in summary.items():
            row = dict(s)
            row["config"] = cfg
            row["acc_drop"] = (s["clean_acc"] - s["adv_acc"]) if s["clean_acc"] is not None else None
            writer.writerow(row)
    return summary


def saliency_map(model, context, candidates, target):
    """Gradient-magnitude map of the loss w.r.t. each context pixel -- a
    cheap, architecture-agnostic stand-in for Grad-CAM that works with
    PredRNet's non-convolutional-classifier-head architecture.
    """
    context = context.clone().detach().requires_grad_(True)
    logits = model(context, candidates)
    loss = F.cross_entropy(logits, target)
    grad = torch.autograd.grad(loss, context)[0]
    sal = grad.abs()
    sal = sal / (sal.amax(dim=(-2, -1), keepdim=True) + 1e-8)
    return sal.detach()


def save_saliency_figure(path, context, sal, config, idx):
    fig, axes = plt.subplots(2, 8, figsize=(16, 4.5))
    for j in range(8):
        axes[0, j].imshow(context[j].cpu(), cmap="gray", vmin=0, vmax=1)
        axes[0, j].axis("off")
        axes[1, j].imshow(context[j].cpu(), cmap="gray", vmin=0, vmax=1)
        axes[1, j].imshow(sal[j].cpu(), cmap="jet", alpha=0.5)
        axes[1, j].axis("off")
    fig.suptitle(f"[{config}] sample {idx} -- top: context, bottom: gradient-magnitude overlay")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def run_saliency_analysis(model, dataset, indices, device, out_dir, per_config=3):
    loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False)
    saved = defaultdict(int)
    for context, candidates, target, config in loader:
        cfg = config[0]
        if saved[cfg] >= per_config:
            continue
        context_d, candidates_d, target_d = context.to(device), candidates.to(device), target.to(device)
        sal = saliency_map(model, context_d, candidates_d, target_d)
        save_saliency_figure(
            os.path.join(out_dir, "saliency", f"{cfg}_{saved[cfg]}.png"),
            context[0], sal[0], cfg, saved[cfg],
        )
        saved[cfg] += 1


def collect_pe_embeddings(model, loader, device, attack=None):
    feats, correct_flags, sources = [], [], []
    for context, candidates, target, config in loader:
        context = context.to(device)
        candidates = candidates.to(device)
        target = target.to(device)

        if attack is not None:
            images = attack.perturb(model, context, candidates, target)
        else:
            images = context

        with torch.no_grad():
            emb = model.extract_pe_features(images, candidates)  # (B, 8, D)
            true_emb = emb[torch.arange(emb.size(0)), target]
            pred = model(images, candidates).argmax(dim=1)

        feats.append(true_emb.cpu())
        correct_flags.extend((pred == target).cpu().tolist())
        sources.extend(["adv" if attack is not None else "clean"] * target.size(0))

    return torch.cat(feats).numpy(), correct_flags, sources


def run_tsne_analysis(model, dataset, indices, device, attack, out_path, max_samples=400):
    from sklearn.manifold import TSNE

    indices = indices[:max_samples]
    loader = DataLoader(Subset(dataset, indices), batch_size=32, shuffle=False)

    clean_feats, clean_correct, clean_src = collect_pe_embeddings(model, loader, device, attack=None)
    adv_feats, adv_correct, adv_src = collect_pe_embeddings(model, loader, device, attack=attack)

    all_feats = np.concatenate([clean_feats, adv_feats], axis=0)
    all_correct = clean_correct + adv_correct
    all_src = clean_src + adv_src

    tsne = TSNE(n_components=2, init="pca", random_state=42, perplexity=min(30, len(all_feats) // 4 or 1))
    proj = tsne.fit_transform(all_feats)

    fig, ax = plt.subplots(figsize=(7, 6))
    groups = {
        ("clean", True): ("tab:blue", "clean, correct"),
        ("clean", False): ("tab:cyan", "clean, wrong"),
        ("adv", True): ("tab:orange", "adv, correct"),
        ("adv", False): ("tab:red", "adv, wrong"),
    }
    for (src, correct), (color, label) in groups.items():
        mask = [(s == src and c == correct) for s, c in zip(all_src, all_correct)]
        mask = np.array(mask)
        if mask.any():
            ax.scatter(proj[mask, 0], proj[mask, 1], c=color, label=label, s=12, alpha=0.7)
    ax.legend()
    ax.set_title("t-SNE of PE embeddings for the correct-answer slot\n(clean vs. adversarial)")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


REPORT_TEMPLATE = """# Adversarial Attack Effect Analysis

## 1. Per-configuration breakdown

| Config | N | Clean Acc | Adv Acc | ASR | Acc Drop |
|---|---|---|---|---|---|
{table_rows}

{config_discussion}

## 2. Perturbation localization (gradient-magnitude saliency)

See `saliency/*.png`. Each figure overlays |dLoss/dPixel| on the context
images for a few examples per configuration. Bright regions indicate pixels
the attack (and the model's own decision) are most sensitive to; check
whether they concentrate on the geometric shapes (task-relevant) versus the
uniform background (task-irrelevant).

## 3. Feature-space analysis (t-SNE of PE embeddings)

See `tsne_pe_features.png`. Each point is PredRNet's pooled prediction-error
embedding for the correct-answer slot, projected to 2D. Adversarial points
drifting away from the "clean, correct" cluster indicate the attack disrupts
the internal relational representation, not just the final logits.

## 4. Rule disruption analysis (qualitative)

For a sample of successful attacks, manually inspect which of the five RAVEN
rule attributes (Number, Position, Type, Size, Color) the perturbation most
visibly targets or which the model's confusion correlates with. This project
does not auto-label attribute-level distortion; use the saved adversarial
examples in `results/adversarial_examples/` and the per-config ASR above
(e.g., whether attribute-dense configs like 3x3Grid degrade faster than
single-object configs like Center) to guide this qualitative pass, and record
findings here.
"""


def render_report(summary, out_path):
    rows = []
    ordered = ["overall"] + [k for k in summary if k != "overall"]
    for cfg in ordered:
        s = summary[cfg]
        if s["n"] == 0 or s["asr"] is None:
            continue
        rows.append(
            f"| {cfg} | {s['n']} | {s['clean_acc']:.1f} | {s['adv_acc']:.1f} | "
            f"{s['asr']:.1f} | {(s['clean_acc'] - s['adv_acc']):.1f} |"
        )

    non_overall = {k: v for k, v in summary.items() if k != "overall" and v["n"] > 0 and v["asr"] is not None}
    if non_overall:
        hardest = max(non_overall.items(), key=lambda kv: kv[1]["asr"])
        easiest = min(non_overall.items(), key=lambda kv: kv[1]["asr"])
        discussion = (
            f"The attack is most effective on **{hardest[0]}** (ASR={hardest[1]['asr']:.1f}%) "
            f"and least effective on **{easiest[0]}** (ASR={easiest[1]['asr']:.1f}%). "
            "Compare panel complexity (single-object configs like Center vs. "
            "multi-object grids like 3x3Grid) against this ranking to see whether "
            "attack success correlates with reasoning complexity."
        )
    else:
        discussion = "(not enough per-config data to compare configurations)"

    text = REPORT_TEMPLATE.format(table_rows="\n".join(rows), config_discussion=discussion)
    with open(out_path, "w") as f:
        f.write(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--data-dir", default="data/i-raven")
    parser.add_argument("--attack-results", default="results/attack_results.json")
    parser.add_argument("--output", default="results/analysis")
    parser.add_argument("--epsilon", type=float, default=8, help="in 1/255 units, used to regenerate adv samples for saliency/tsne")
    parser.add_argument("--alpha", type=float, default=2)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--saliency-per-config", type=int, default=3)
    parser.add_argument("--tsne-samples-per-config", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    os.makedirs(args.output, exist_ok=True)

    print("1/3 per-configuration breakdown...")
    summary = per_config_breakdown(args.attack_results, os.path.join(args.output, "per_config_breakdown.csv"))

    model, _ = load_model(args.checkpoint, device=device)
    dataset = IRavenDataset(args.data_dir, "test")
    attack = PGDAttack(epsilon=args.epsilon / 255, alpha=args.alpha / 255, num_steps=args.steps)
    indices = sample_subset_indices(dataset, args.tsne_samples_per_config, args.seed)

    print("2/3 gradient-magnitude saliency maps...")
    run_saliency_analysis(model, dataset, indices, device, args.output, per_config=args.saliency_per_config)

    print("3/3 t-SNE of PE embeddings...")
    run_tsne_analysis(
        model, dataset, indices, device, attack,
        os.path.join(args.output, "tsne_pe_features.png"),
        max_samples=len(indices),
    )

    render_report(summary, os.path.join(args.output, "analysis_report.md"))
    print(f"Done. See {args.output}/analysis_report.md")


if __name__ == "__main__":
    main()
