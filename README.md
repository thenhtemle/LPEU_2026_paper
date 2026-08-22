# LPEU: Local Prototype Erasure Unlearning

Official implementation of **"Local Prototype Erasure Unlearning: Representation-Level Forgetting for Deep Face Recognition"** (FAIR 2026, under review).

Most identity-unlearning methods for face recognition work at the **output level**: they optimize the classifier to mispredict the forgotten identity. This does not guarantee the identity's *representation* has been destroyed — an adversary with access to embeddings can still re-identify a nominally forgotten subject by similarity search.

LPEU targets the representation directly. It builds per-identity prototypes from an ArcFace-trained backbone, locates the forgotten identity's nearest-neighbour identities, and optimizes a two-branch objective that **disperses the forgotten cluster** while **anchoring its geometric neighbours**. The forget branch follows its own two-state **Migrate → Shatter** schedule — Migrate relocates the cluster as a whole, Shatter is what actually breaks it apart — decoupled from the retain anchor's own schedule. Because the two branches conflict on shared parameters, their gradients are reconciled by projection.

The forget unit here is an **individual person**: 100 of 809 eligible individuals on MUFAC (~12%, keyed by `family_id::person_id` since a bare person id repeats across families), and 100 of 1,500 identities on VGGFace2. The featured configuration, **Hue C**, adds a class-matched divergence term that pulls the forget-set output distribution toward the frozen original model's distribution on genuinely unseen data — targeting the paper-defined forgetting score (FS) directly rather than through a proxy such as entropy maximization. This extends our earlier household-level formulation of LPEU to a finer-grained forget unit and a substantially larger relative scale.

<p align="center">
  <img src="figures/cluster.png" alt="Local cluster erasure" width="88%">
</p>

<p align="center"><em>Before unlearning the forget identity's embeddings form a tight cluster; afterwards they are scattered and the original region is vacated, while neighbouring identities keep their position.</em></p>

## Method at a glance

<p align="center">
  <img src="figures/architecture.png" alt="LPEU architecture, updated with Hue C" width="80%">
</p>

<p align="center"><em>The trainable student (top) and frozen teacher (bottom) share the backbone, neck, and ArcFace head. The forget branch follows its own Migrate&nbsp;&rarr;&nbsp;Shatter schedule (Migrate: cluster relocates as one block; Shatter: the cluster actually breaks apart), decoupled from the retain anchor's own three-stage schedule. The Hue C divergence term (<code>L_div</code>) is staged on the same Migrate/Shatter switch.</em></p>

| Component | Role |
|---|---|
| Anti-prototype push (`L_erase`) | Moves forget embeddings away from their original cluster centre |
| Entropy correction / KL-to-uniform (`L_kl-uni`) | Drives the posterior to uniform, preventing *confident* misprediction |
| Negative-margin pairwise repulsion (`L_repel`) | Breaks the cluster's internal cohesion (margin τ = −0.3) |
| Class-matched divergence (`L_div`, Hue C — featured configuration) | Pulls the forget-set output distribution toward the frozen original model's distribution on genuinely unseen, class-matched data — targets the forgetting score (FS) directly |
| Global distillation + CE | Preserves overall retain-set behaviour |
| K-NN local anchor | Protects the identities geometrically closest to the forget cluster |
| Retain-prototype anchor | Holds retain samples at their own frozen prototypes |
| EWC | Data-free retain guard, active when the anchor loss is switched off |
| PCGrad + hard orthogonal projection | Removes forget/retain gradient conflict on shared parameters |
| Migrate → Shatter | Two-state forget-branch schedule, decoupled from the retain anchor's own three-stage schedule |

## Repository structure

```
figures/                        # figures used in this README
src/
├── mufac/
│   └── lpeu-v7-mufac.ipynb       # MUFAC experiments (person-level forget unit, Hue C)
└── vggface2/
    └── lpeu-v7-vggface2.ipynb    # VGGFace2 experiments (1,500 identities, Hue C)
.gitattributes
README.md
```

## Setup

```bash
pip install torch torchvision numpy pandas scikit-learn Pillow jupyter
```

Install a CUDA-enabled `torch` matching your GPU driver — see <https://pytorch.org/get-started/locally/>.

Verify:

```bash
python -c "import torch, torchvision, numpy, pandas, sklearn, PIL; print('OK')"
```

Tested with Python 3.12. A CUDA GPU with at least 8 GB VRAM is recommended; 16 GB RAM and 5 GB free disk (more for the 1,500-identity VGGFace2 run). Full details in [`HuongDanCaiDat.txt`](HuongDanCaiDat.txt).

## Datasets

| Dataset | Source |
|---|---|
| MUFAC (Korean Family, 128×128) | [Kaggle mirror](https://www.kaggle.com/datasets/thenhtemle/custom-korean-family-faces) · [original](https://postechackr-my.sharepoint.com/:u:/g/personal/dongbinna_postech_ac_kr/EbMhBPnmIb5MutZvGicPKggBWKm5hLs0iwKfGW7_TwQIKg?download=1) |
| VGGFace2 (1,500-identity subset) | [Kaggle](https://www.kaggle.com/datasets/dimarodionov/vggface2/train) |

On MUFAC the ArcFace head classifies **age** (the benchmark's original task) while the unlearning unit is an **individual person**, a separate field. Because a person never corresponds to any of the eight age classes, the codebase's `identity_equals_class` check freezes and restores the entire ArcFace head around every forget step — good unlearning should therefore reduce cluster recognizability *without* necessarily reducing retain accuracy. On VGGFace2, classification and forget unit coincide (identity *is* the class), so only the retain identities' ArcFace rows are frozen during the forget step, and this decoupling does not apply in the same way.

## Running

The notebooks were developed on Kaggle (GPU T4). Easiest path: upload the `.ipynb` to Kaggle, add the relevant dataset via **Add Input**, enable GPU, then run all cells top to bottom.

To run locally, edit `dataset_root` in the `Config` class (cell **B**) to point at your local data.

Cells are labelled `CELL B` … `CELL N`. **`CELL L` is the main entry point** and runs the full pipeline end to end; `CELL M` runs the optional ablation study. The VGGFace2 notebook has three extra cells (0–2) that restructure the dataset — run them **before** `CELL B`.

Two settings worth checking if you run outside Kaggle: `num_workers` in `Config` (Kaggle's own environment can require `0`), and the dataset cache path used to download/unzip VGGFace2 (pin it to a persistent directory so re-runs don't re-download).

Approximate runtime on a Kaggle T4: 1–2 h for original training, 10–20 min for the unlearning loop, 15–25 min for repair, and roughly 7× a single unlearning run for the full ablation. The VGGFace2 notebook's larger identity pool increases these times proportionally. Step-by-step instructions in [`HuongDanSuDung.txt`](HuongDanSuDung.txt).

## Results

Forget unit is an individual person; the entropy-maximizing and divergence terms are staged on the Migrate → Shatter schedule described above. FS is the paper-defined forgetting score, `|0.5 − M|` with `M` the MIA-attacker accuracy — lower is better (more private).

**MUFAC** (forget unit: person, n_forget = 100 of 809, w_div = 20, best-observed of four independent runs)

| Method | Retain Acc. | Forget Acc. | Sim Drop | FS (paper) |
|---|---|---|---|---|
| Original | 0.7286 | 0.9308 | 0.0000 | 0.0383 ± 0.021 |
| Fine-tune | 0.7247 | 0.9308 | 0.0000 | 0.0383 ± 0.020 |
| NegGrad | 0.7269 | 0.9308 | 0.0000 | 0.0367 ± 0.020 |
| **LPEU (Hue C)** | **0.7286** | 0.9308 | **0.0151** | **0.0200 ± 0.022** |

LPEU's FS was lower than its own run's Original in **all four** independent runs (Δ = −0.0017, −0.0067, −0.0050, −0.0183), a 48% relative reduction in the best run at unchanged retain accuracy — but **no individual run reaches bootstrap-based statistical significance** against any baseline (largest gap 0.0183 against a significance threshold of 0.043). Retain accuracy and cluster similarity drop, not subject to the same MIA sampling noise, are where LPEU's advantage is most solidly measured. Forget-set task accuracy never measurably moves at any stage, for any method — a structural consequence of the ArcFace head being fully frozen and restored around every forget step when the forget unit doesn't correspond to a classification class, not a tuning failure (see the paper's Discussion section).

**VGGFace2** (forget unit: person, N=1,500, n_forget = 100, Hue C mechanism, single run)

| Method | Retain Acc. | MIA-AUC (forget vs. retain) | FS_retain = &#124;0.5 − AUC&#124; |
|---|---|---|---|
| Original | 0.5496 | 0.9459 | 0.4459 |
| Fine-tune | 0.5487 | 0.9490 | 0.4490 |
| NegGrad | 0.5478 | 0.9460 | 0.4460 |
| **LPEU (Hue C)** | **0.5601** | **0.9143** | **0.4143** |

On VGGFace2 the standard unseen-identity FS protocol used above is structurally inapplicable (it returns a ceiling of `M = 1.000` for every method, including Original, since identity recognition cannot generalize to a never-seen person). This repository's VGGFace2 notebook instead reports a same-population forget-vs-retain comparison (`FS_retain`, not numerically comparable to the MUFAC FS above). LPEU is again the only method to move this privacy metric meaningfully (0.4459 → 0.4143, ~7% relative) while also leading on retain accuracy. One caveat, reported without reframing: cluster similarity drop was **not** favourable to LPEU on this single run (all methods below 0.01) — flagged for future replication rather than smoothed over.

## Known limitations

Stated plainly, as in the paper:

- FS improvements are directionally consistent across four runs but **do not individually reach statistical significance**; the best run is reported as the headline result while the full four-run spread is disclosed alongside it.
- In three of four runs, the same mechanism is associated with a **re-identification-retrieval regression** for a subset of forgotten individuals — not visible in the loss-based FS metric at all.
- Forget-set classification accuracy structurally cannot move on MUFAC (ArcFace head is frozen/restored whenever the forget unit isn't an output class), which means task-level accuracy cannot be used to validate forgetting strength here, by design.
- The VGGFace2 run is a **single run at one scale**, without the multi-run replication applied to MUFAC; its cluster-similarity-drop result was not favourable to LPEU this run.
- The standard unseen-identity FS protocol does not transfer to identity-classification tasks (confirmed empirically: it returns a ceiling for every method including Original on VGGFace2) — a structural, not a tuning, finding.
- MIA-AUC and MIA-accuracy-based FS can disagree in ranking (LPEU was worse on AUC but competitive-or-best on accuracy-based FS in the task-coupled staging round that led to Hue C) — a caution for any future work reading AUC alone.

## Citation

```bibtex
@inproceedings{le2026lpeu,
  title     = {Local Prototype Erasure Unlearning: Representation-Level Forgetting
               for Deep Face Recognition},
  author    = {Le, Thanh Tam and Le, Hoang Thai},
  booktitle = {Proceedings of the Conference on Fundamental and Applied
               IT Research (FAIR)},
  year      = {2026},
  note      = {Under review}
}
```

## Contact

- Le Thanh Tam — <lttam22@clc.fitus.edu.vn>
- Le Hoang Thai (corresponding author) — <lhthai@fit.hcmus.edu.vn>

Faculty of Information Technology, University of Science, VNU-HCM.