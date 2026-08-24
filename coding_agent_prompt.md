# Coding Agent Prompt: Adversarial Attack on Abstract Visual Reasoning Models

## Project Overview

You are implementing a complete adversarial robustness evaluation pipeline targeting **abstract visual reasoning models** trained on the RAVEN/I-RAVEN dataset. The project has four stages:

1. Select and train a visual reasoning model on I-RAVEN
2. Implement an adversarial attack against the trained model
3. Evaluate the attack's effect on model performance
4. Implement a lightweight defense and compare before/after

The final goal is a clean, reproducible codebase with experiment logs, accuracy tables, and visualizations suitable for a research paper targeting TIFS/TDSC.

---

## Stage 1: Model Selection and Training

### Target Model: PredRNet

Use **PredRNet** (Predictive Reasoning Network) as the victim model. It is state-of-the-art on RAVEN/I-RAVEN and has open-source code.

- Paper: "Neural Prediction Errors enable Analogical Visual Reasoning in Human Standard Intelligence Tests" (ICML 2023)
- Official repo: https://github.com/ZjjConan/AVR-PredRNet

**Task:**

1. Clone the PredRNet repository and set up the environment (Python 3.8+, PyTorch ≥ 1.10).
2. Download the **I-RAVEN** dataset (preferred over original RAVEN due to its unbiased answer set).
   - I-RAVEN can be generated using the script from: https://github.com/husheng12345/SRAN
   - Dataset has 7 figure configurations: Center, 2x2Grid, 3x3Grid, Left-Right, Up-Down, Out-InCenter, Out-InGrid
   - Each RPM problem: one 3×3 matrix (8 context images + 1 correct answer image) and 7 distractor images
3. Train PredRNet on I-RAVEN with default hyperparameters. Log:
   - Training loss per epoch
   - Validation accuracy per epoch
   - Final test accuracy (overall + per figure configuration)
4. Save the best checkpoint (`best_model.pth`).

**Expected clean accuracy:** ~60–75% overall on I-RAVEN.

**Directory structure to create:**
```
project/
├── data/
│   └── i-raven/          # dataset root
├── models/
│   └── predrnet/         # cloned repo
├── checkpoints/
│   └── best_model.pth
├── attack/
├── defense/
├── results/
└── logs/
```

---

## Stage 2: Adversarial Attack Implementation

### Attack Choice: PGD (Projected Gradient Descent) — White-box Attack

Implement a **PGD-based perturbation attack** applied to the **context images** of the RPM matrix. The goal is to add an imperceptible perturbation ε to the 8 context images such that PredRNet selects the wrong answer.

**Why PGD:**
- Standard, well-understood white-box attack (Madry et al., 2018)
- Directly applicable to classification-style RPM solvers
- Provides a solid baseline before exploring transfer-based or patch attacks

**Attack formulation:**

Given an RPM problem with context images X = {x₁, ..., x₈} and correct answer index y*:

```
x_adv = x + δ
where δ = PGD(loss, x, y*, ε, α, num_steps)
```

- Maximize cross-entropy loss: L(f(X_adv, candidates), y*)
- Constraint: ||δ||_∞ ≤ ε
- Attack only the 8 context images (not the answer candidates), as this is the more realistic threat model

**Implementation steps:**

1. Write `attack/pgd_attack.py`:
   - Class `PGDAttack` with parameters: `epsilon` (default 8/255), `alpha` (step size, default 2/255), `num_steps` (default 20), `random_start=True`
   - Method `perturb(model, context_images, answer_images, target_label)` → returns adversarial context images
   - Ensure pixel values stay in [0, 1] after perturbation

2. Write `attack/evaluate_attack.py`:
   - Load the saved checkpoint
   - Run PGD attack on the full test set (or a representative subset of 1000 samples per config)
   - Record per-sample: clean prediction, adversarial prediction, correct label
   - Compute: Attack Success Rate (ASR) = fraction of correctly classified samples that become misclassified after attack

3. Ablation: vary epsilon ∈ {2/255, 4/255, 8/255, 16/255} and num_steps ∈ {5, 10, 20, 40}. Record ASR for each combination in a table.

4. Also implement a **C&W-style targeted attack** variant as a secondary experiment:
   - Target: force model to select the *worst* distractor (the one most different from the correct answer)
   - This demonstrates intentional misdirection rather than just confusion

**Outputs:**
- `results/attack_results.json`: per-sample results
- `results/attack_summary_table.csv`: ASR by config and epsilon
- `results/adversarial_examples/`: save 5 visualized examples per config showing:
  - Original context images
  - Adversarial context images
  - Perturbation (amplified ×10 for visibility)
  - Clean prediction vs. adversarial prediction vs. ground truth

---

## Stage 3: Attack Effect Analysis

Beyond raw ASR numbers, analyze *why* the attack succeeds and *what* it breaks.

**Analysis tasks:**

1. **Per-configuration breakdown**: Does the attack work better on simpler configs (Center) or complex ones (3x3Grid)? Report accuracy drop per config.

2. **Perturbation localization**: Use Grad-CAM or simple gradient magnitude maps to show which pixels are most perturbed and whether they align with task-relevant regions (the geometric shapes).

3. **Feature space analysis**: Extract PredRNet's internal prediction error features (the PE embeddings) for clean vs. adversarial inputs. Use t-SNE to visualize whether adversarial samples drift away from the correct class cluster.

4. **Rule disruption analysis**: For a sample of adversarial examples, manually or programmatically check which visual attributes are most distorted (Number, Position, Type, Size, Color — the five RAVEN rule attributes). This connects the attack to the abstract reasoning failure.

Write `attack/analysis.py` to automate steps 1–3. Step 4 can be a short qualitative section.

**Output:** `results/analysis/` with figures and a `analysis_report.md`.

---

## Stage 4: Defense Implementation

Implement **two lightweight defense strategies** and compare their effectiveness.

### Defense 1: Input Preprocessing — JPEG Compression + Gaussian Smoothing

Apply preprocessing to the input images before feeding them to the model:

```python
def preprocess_defense(images, jpeg_quality=75, gaussian_sigma=1.0):
    # Apply JPEG compression (destroys high-frequency adversarial noise)
    # Apply Gaussian blur
    return cleaned_images
```

Write `defense/preprocessing_defense.py`.

### Defense 2: Adversarial Training (Light version)

Fine-tune the pretrained PredRNet for 10 additional epochs using a mix of:
- 50% clean samples
- 50% PGD-perturbed samples (ε=8/255, 7 steps — weaker than the evaluation attack)

Write `defense/adversarial_training.py`:
- Load `best_model.pth`
- Run adversarial fine-tuning loop
- Save `checkpoints/adv_trained_model.pth`

### Evaluation of Defenses

For both defenses, measure:

| Metric | Clean | After PGD Attack | After PGD Attack + Defense |
|--------|-------|-----------------|---------------------------|
| Overall Accuracy | | | |
| ASR | — | | |
| Accuracy drop vs. clean | — | | |

Also measure inference-time overhead (ms per sample) for the preprocessing defense.

Write `defense/evaluate_defense.py` to produce the comparison table automatically.

---

## Code Quality Requirements

- All scripts should be runnable from the command line with argparse:
  ```bash
  python attack/pgd_attack.py --checkpoint checkpoints/best_model.pth --epsilon 8 --steps 20 --output results/
  python defense/evaluate_defense.py --defense preprocessing --checkpoint checkpoints/best_model.pth
  ```
- Use `torch.no_grad()` appropriately; ensure no gradient leakage during evaluation
- Seed all random operations (`torch.manual_seed(42)`, `numpy.random.seed(42)`) for reproducibility
- Log all experiment configs to `logs/` using Python's `logging` module or `json`
- Write a `README.md` with setup instructions and commands to reproduce all results

---

## Key Files to Produce

```
attack/
├── pgd_attack.py          # PGD attack class
├── cw_attack.py           # C&W targeted variant
├── evaluate_attack.py     # run attack on test set, compute ASR
└── analysis.py            # Grad-CAM, t-SNE, per-config breakdown

defense/
├── preprocessing_defense.py   # JPEG + Gaussian defense
├── adversarial_training.py    # adversarial fine-tuning
└── evaluate_defense.py        # compare clean/attacked/defended

results/
├── attack_summary_table.csv
├── attack_results.json
├── defense_comparison_table.csv
├── adversarial_examples/      # visualizations
└── analysis/                  # figures

logs/
├── training_log.json
├── attack_log.json
└── defense_log.json

README.md
requirements.txt
```

---

## References (for context)

- **PredRNet**: Yang et al., "Neural Prediction Errors enable Analogical Visual Reasoning", ICML 2023. Code: https://github.com/ZjjConan/AVR-PredRNet
- **I-RAVEN**: Hu et al., "Stratified Rule-Aware Network for Abstract Visual Reasoning", AAAI 2021. Code: https://github.com/husheng12345/SRAN
- **RAVEN Dataset**: Zhang et al., "RAVEN: A Dataset for Relational and Analogical Visual Reasoning", CVPR 2019
- **PGD Attack**: Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks", ICLR 2018
- **RPAttack** (patch attack reference): Huang et al., ICME 2021
- **T-SEA** (transfer attack reference): Huang et al., CVPR 2023
- **Radar adversarial reference**: Ozbulak et al., "Investigating the significance of adversarial attacks and their relation to interpretability for radar-based human activity recognition systems", CVIU 2021

---

## Notes for the Coding Agent

- The RPM task is a **10-way classification** problem (choose 1 correct answer from 8 candidates; some implementations use all 16 images)
- PredRNet processes all 16 images (8 context + 8 candidates) jointly; the attack should only perturb the 8 context images to remain realistic
- I-RAVEN images are **160×160 grayscale PNGs**, normalized to [0, 1]
- If GPU memory is limited, process attacks in batches of 32 problems at a time
- The I-RAVEN dataset is ~3.5 GB; make sure disk space is available before downloading
- If PredRNet training is too slow, ResNet-18 with the DRT module (from the original RAVEN paper) is an acceptable alternative victim model — it achieves ~59% accuracy and is simpler to train
