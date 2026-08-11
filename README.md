# LPEU: Local Prototype Erasure Unlearning

Official implementation of **"Local Prototype Erasure Unlearning: Representation-Level Forgetting for Deep Face Recognition"** (FAIR 2026, under review).

Most identity-unlearning methods for face recognition work at the **output level**: they optimize the classifier to mispredict the forgotten identity. This does not guarantee the identity's *representation* has been destroyed — an adversary with access to embeddings can still re-identify a nominally forgotten subject by similarity search.

LPEU targets the representation directly. It builds per-identity prototypes from an ArcFace-trained backbone, locates the forgotten identity's nearest-neighbour identities, and optimizes a two-branch objective that **disperses the forgotten cluster** while **anchoring its geometric neighbours**. Because the two branches conflict on shared parameters, their gradients are reconciled by projection.

## Method at a glance

| Component | Role |
|---|---|
| Anti-prototype push | Moves forget embeddings away from their original cluster centre |
| Entropy correction (KL-to-uniform) | Drives the posterior to uniform, preventing *confident* misprediction |
| Negative-margin pairwise repulsion | Breaks the cluster's internal cohesion (margin $\tau=-0.3$) |
| Global distillation + CE | Preserves overall retain-set behaviour |
| K-NN local anchor | Protects the 30 identities geometrically closest to the forget cluster |
| Retain-prototype anchor | Holds retain samples at their own frozen prototypes |
| EWC | Data-free retain guard, active when the anchor loss is switched off |
| PCGrad + hard orthogonal projection | Removes forget/retain gradient conflict on shared parameters |
| Three-stage schedule | ERASE → REFINE → STABLE, resolves a geometric deadlock |

## Repository structure

```
src/
├── mufac/
│   ├── lpeu-v7-mufac.ipynb          # main notebook — MUFAC experiments
│   ├── original_model.pt            # pretrained original model
│   ├── lpeu_mufac_model.pt          # after LPEU unlearning
│   ├── lpeu_mufac_repaired_model.pt # after post-unlearning repair
│   ├── results_mufac.json           # main metric results
│   └── ablation_result_mufac.json   # ablation study results
└── vggface2/
    ├── lpeu-v7-vgg-face2.ipynb      # main notebook — VGGFace2 experiments
    ├── original_model.pt
    ├── lpeu_v7_model.pt
    ├── lpeu_v7_repaired_model.pt
    ├── results_v7.json
    └── ablation_results_vggface2.json

HuongDanCaiDat.txt   # detailed installation guide (Vietnamese)
HuongDanSuDung.txt   # detailed usage guide (Vietnamese)
```

The `.pt` and `.json` files are **pre-computed results**, so every table in the paper can be reproduced without retraining.

## Setup

```bash
pip install torch torchvision numpy pandas scikit-learn Pillow jupyter
```

Install a CUDA-enabled `torch` matching your GPU driver — see <https://pytorch.org/get-started/locally/>.

Verify:

```bash
python -c "import torch, torchvision, numpy, pandas, sklearn, PIL; print('OK')"
```

Tested with Python 3.12. A CUDA GPU with at least 8 GB VRAM is recommended; 16 GB RAM and 5 GB free disk. Full details in [`HuongDanCaiDat.txt`](HuongDanCaiDat.txt).

## Datasets

| Dataset | Source |
|---|---|
| MUFAC (Korean Family, 128×128) | [Kaggle mirror](https://www.kaggle.com/datasets/thenhtemle/custom-korean-family-faces) · [original](https://postechackr-my.sharepoint.com/:u:/g/personal/dongbinna_postech_ac_kr/EbMhBPnmIb5MutZvGicPKggBWKm5hLs0iwKfGW7_TwQIKg?download=1) |
| VGGFace2 (300-identity subset) | [Kaggle](https://www.kaggle.com/datasets/hearfool/vggface2) |

On MUFAC the ArcFace head classifies **age** (the benchmark's original task) while the unlearning unit is the **household** — a separate label. Good household unlearning should therefore reduce family-cluster recognizability *without* necessarily reducing retain accuracy. On VGGFace2 classification and forget unit coincide.

## Running

The notebooks were developed on Kaggle (GPU T4). Easiest path: upload the `.ipynb` to Kaggle, add both datasets via **Add Input**, enable GPU, then run all cells top to bottom.

To run locally, edit `dataset_root` in the `Config` class (cell **B**) to point at your local data.

Cells are labelled `CELL B` … `CELL N`. **`CELL L` is the main entry point** and runs the full pipeline end to end; `CELL M` runs the optional ablation study. The VGGFace2 notebook has three extra cells (0–2) that restructure the dataset — run them **before** `CELL B`.

Approximate runtime on a Kaggle T4: 1–2 h for original training, 10–20 min for the unlearning loop, 15–25 min for repair, and roughly 7× a single unlearning run for the full ablation. Step-by-step instructions in [`HuongDanSuDung.txt`](HuongDanSuDung.txt).

## Results

Cluster-destruction metrics are *higher is better*; neighbour shift is *lower is better*; MIA AUC targets 0.5.

**MUFAC** (forget unit: household)

| Method | Retain Acc. | Forget Sim Drop | Compact. Incr. | Entropy Incr. | Neighbour Shift | MIA AUC |
|---|---|---|---|---|---|---|
| Original | 54.47% | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.561 |
| Fine-tune | 53.40% | −0.0027 | −0.0046 | 0.0005 | 0.0038 | 0.564 |
| NegGrad | 52.00% | −0.0126 | −0.0216 | −0.0162 | 0.0198 | 0.557 |
| **LPEU** | **54.84%** | **+0.0318** | **+0.0539** | **+0.0177** | 0.0410 | 0.554 |

**VGGFace2** (forget unit: identity, C=300)

| Method | Retain Acc. | Forget Sim Drop | Compact. Incr. | Entropy Incr. | Neighbour Shift | MIA AUC |
|---|---|---|---|---|---|---|
| Original | 28.92% | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.855 |
| Fine-tune | 30.46% | −0.0001 | −0.0003 | 0.0008 | 0.0080 | 0.765 |
| NegGrad | 29.91% | +0.0000 | −0.0002 | 0.0010 | 0.0189 | 0.750 |
| **LPEU** | **30.90%** | **+0.0022** | **+0.0043** | **+0.0048** | 0.0241 | 0.839 |

On MUFAC, LPEU is the only method with positive movement on every cluster-destruction metric while also leading on retain accuracy — above the original model. NegGrad in fact moves in the *wrong* direction on most cluster metrics.

### The forgetting–damage frontier

Plotting each ablation configuration with neighbour shift on the horizontal axis and forget similarity drop on the vertical reveals that all configurations lie close to a **single line** (slope 0.77, R² = 0.974). The protection mechanisms do **not** decouple forgetting from collateral damage — disabling one slides the operating point *along* the same frontier rather than off it.

Ranked by exchange rate, full LPEU reaches 0.776 while the best ablated variant reaches 0.779 — a difference a single-seed run cannot distinguish from noise. We therefore **do not claim** LPEU improves the exchange rate. What it demonstrably does is relocate the operating point: neighbour shift 0.041 against 0.384, a **9.4× reduction in absolute collateral damage**, while repair lifts retain accuracy above the original model.

## Known limitations

Stated plainly, as in the paper:

- Each configuration was run **once** at a fixed seed (42); statistical significance is unassessed. This particularly bounds the VGGFace2 conclusions, where effect sizes are ~0.002–0.004.
- On VGGFace2 the original embedding space is **near-collapsed** (intra-cluster similarity 0.978 versus 0.705 on MUFAC), leaving repulsion losses little geometric room to act. This is the leading explanation for the order-of-magnitude smaller effect there.
- **Active suppression is unresolved on VGGFace2**: forget accuracy reaches exactly 0% rather than converging to the 1/C random-guess floor.
- **MIA AUC on VGGFace2 is worse than the simplest baselines** (0.839 versus 0.750–0.765, target 0.5).
- No retrained-from-scratch reference model was computed, so metric targets are argued rather than measured.
- The metric suite does not yet include a direct re-identification (verification/retrieval) attack, so the retrieval threat motivating the work is argued but not measured.
- Results may vary slightly between runs despite the fixed seed, due to CUDA non-determinism.

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
