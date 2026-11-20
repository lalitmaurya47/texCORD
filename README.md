# TexCORD: Texture-Consistent Self-Supervised Learning and Regional Distillation for Lightweight Ultrasound Foundation Models

<p align="center">
  <b>Texture-aware self-supervised learning and ultrasound foundation-model distillation for compact, transferable ultrasound representations.</b>
</p>

<p align="center">
  <img src="assets/texcord_overview.png" alt="TexCORD overview" width="95%">
</p>

> **Note:** Replace the placeholder commands, paths, author information, and checkpoint links below with the final repository-specific values before release.

---

## Overview

**TexCORD** is a lightweight ultrasound representation-learning framework that combines:

- **Texture-consistency self-supervised learning (SSL)** to learn dense ultrasound-specific texture representations from unlabeled B-mode images.
- **Foreground-aware global knowledge distillation (KD)** to transfer image-level semantic information from a frozen ultrasound foundation model.
- **Texture-guided regional KD** to transfer teacher knowledge within soft regions defined by the student's learned texture assignments.
- **Lightweight student backbones**, including transformer- and CNN-based variants, for efficient downstream transfer.

TexCORD is designed to preserve local acoustic appearance while inheriting higher-level anatomical and semantic information from a large pretrained ultrasound teacher.

The framework is pretrained on approximately **106k public ultrasound images** aggregated from **34 datasets** spanning **14 anatomical groups**, and is evaluated on heterogeneous classification and segmentation benchmarks.

---

## Highlights

- **106k+** public ultrasound images for pretraining.
- **34 public datasets** spanning **14 anatomical groups**.
- Two lightweight student variants:
  - **TexCORD-PVT** — transformer-based student.
  - **TexCORD-ResNet** — convolutional student.
- Evaluated on **6 classification datasets** and **9 segmentation task-dataset pairs**.
- **88.31%** average classification accuracy with TexCORD-PVT.
- **83.54%** average segmentation Dice with TexCORD-PVT.
- Strong label efficiency: **80.44%** average classification accuracy using only **25%** of downstream labels, compared with **77.35%** for Scratch-PVT using 100% of the labels.
- Compact computational footprint relative to the large USFM teacher.

---

## Method

TexCORD combines local texture discovery with global and regional semantic distillation.

### 1. Multi-view Texture-Consistency SSL

For each B-mode image, TexCORD generates multiple spatial and intensity-perturbed views. The student predicts a dense soft assignment map over a small set of texture prototypes and reconstructs the ultrasound image from these assignments.

The self-supervised objective combines:

- reconstruction loss,
- contrastive representation learning,
- texture/logit consistency across augmented views,
- prototype-usage regularization to reduce collapse.

The goal is to learn texture representations that remain stable under realistic ultrasound appearance variations while retaining spatial structure.

### 2. Foreground-Aware Global Distillation

Ultrasound images often contain large black background regions. TexCORD uses a foreground mask so that teacher-student feature alignment focuses on meaningful image content.

Global KD aligns foreground-pooled student and teacher features, transferring image-level semantic and anatomical information from the frozen **USFM** teacher.

### 3. Texture-Guided Regional Distillation

TexCORD uses the student's learned soft texture assignments to define texture-consistent regions. Teacher and student features are pooled inside these regions and aligned separately.

This allows knowledge transfer to operate at the level of local acoustic structure instead of supervising the whole image uniformly.

### 4. Lightweight Students

| Model | Student type |
|---|---|
| **TexCORD-PVT** | Lightweight Pyramid Vision Transformer |
| **TexCORD-ResNet** | Compact convolutional network |

---

## Overall Objective

The complete objective is

$$
\mathcal{L}_{\mathrm{total}}
=
\mathcal{L}_{\mathrm{ssl}}
+
\lambda_g\mathcal{L}_{\mathrm{global}}
+
\lambda_t\mathcal{L}_{\mathrm{texture}}.
$$

The SSL component is

$$
\mathcal{L}_{\mathrm{ssl}}
=
\lambda_{\mathrm{rec}}\mathcal{L}_{\mathrm{rec}}
+
\lambda_{\mathrm{con}}\mathcal{L}_{\mathrm{con}}
+
\lambda_{\mathrm{logit}}\mathcal{L}_{\mathrm{logit}}
+
\lambda_{\mathrm{ent}}\mathcal{L}_{\mathrm{ent}}.
$$

---

## Pretraining Data

The pretraining corpus contains approximately **106,469 ultrasound images** collected from **34 publicly available datasets** covering **14 anatomical groups**.

Representative anatomical groups include:

- abdomen,
- breast,
- kidney,
- liver,
- cardiac,
- fetal abdomen,
- fetal head,
- lung,
- musculoskeletal,
- neck,
- thyroid,
- ovary,
- prostate,
- knee.

All images are converted to grayscale, resized to **224 x 224**, and normalized to **[-1, 1]**. The preprocessing pipeline also reduces non-anatomical overlays such as text, arrows, outlines, and measurement markers before pretraining.

> Dataset download links and licenses should be documented individually in the repository. Please follow the original terms of use for every public dataset.

---

## Downstream Evaluation

### Classification

The classification benchmarks cover:

- breast,
- skeletal muscle,
- uterine,
- liver,
- ovarian,
- fetal ultrasound.

### Segmentation

The segmentation benchmarks cover:

- breast tumour,
- gastrointestinal stromal tumour,
- multifidus muscle,
- ovarian tumour,
- thyroid nodule,
- brachial plexus,
- lung ultrasound structures/artifacts,
- pubic symphysis and fetal head,
- spinal cord structures/classes.

---

## Main Results

### Classification

| Method | Average Accuracy (%) | Average Rank |
|---|---:|---:|
| USFM | 78.76 | 8.42 |
| TinyUSFM | 84.86 | 5.17 |
| USF-MAE | 86.50 | 2.83 |
| TexCORD-ResNet | 87.54 | 2.33 |
| **TexCORD-PVT** | **88.31** | **1.67** |

TexCORD-PVT improves average classification accuracy by **1.81 percentage points** over USF-MAE, the strongest compared foundation-model baseline in the reported experiments.

### Segmentation

| Method | Average Dice (%) | Average Rank |
|---|---:|---:|
| USFM | 82.42 | 3.89 |
| TinyUSFM | 77.95 | 7.94 |
| USF-MAE | 78.68 | 7.89 |
| TexCORD-ResNet | 82.65 | 2.67 |
| **TexCORD-PVT** | **83.54** | **2.00** |

TexCORD-PVT improves average Dice by **1.12 percentage points** over USFM, the strongest compared foundation-model baseline in the reported segmentation experiments.

---

## Ablation Study

| Variant | Texture SSL | Global KD | Texture-guided Regional KD | Classification (%) | Segmentation Dice (%) |
|---|:---:|:---:|:---:|---:|---:|
| Supervised from scratch | ✗ | ✗ | ✗ | 77.35 | 74.27 |
| Texture SSL only | ✓ | ✗ | ✗ | 82.53 | 79.39 |
| Texture SSL + Global KD | ✓ | ✓ | ✗ | 84.74 | 81.28 |
| Texture SSL + Regional KD | ✓ | ✗ | ✓ | 87.62 | 82.12 |
| **Full TexCORD** | ✓ | ✓ | ✓ | **88.31** | **83.54** |

### Incremental contribution

- Texture SSL vs. scratch: **+5.18** classification points and **+5.12** Dice points.
- Global KD beyond Texture SSL: **+2.21** classification points and **+1.89** Dice points.
- Regional KD beyond Texture SSL: **+5.09** classification points and **+2.73** Dice points.
- Full TexCORD vs. Texture SSL: **+5.78** classification points and **+4.15** Dice points.
- Full TexCORD vs. scratch: **+10.96** classification points and **+9.27** Dice points.

The ablation indicates that texture-guided regional KD provides a larger individual gain beyond SSL than global KD, while combining both global and regional distillation gives the strongest overall result.

> If p-values are reported in the final repository, also specify the reference configuration, statistical test, number of independent runs, and whether multiple-comparison correction was used.

---

## Label Efficiency

| Label fraction | Scratch-PVT | TexCORD-PVT |
|---:|---:|---:|
| 1% | 58.12 | 61.09 |
| 5% | 64.48 | 72.35 |
| 10% | 66.85 | 73.78 |
| 25% | 70.95 | **80.44** |
| 50% | 75.76 | 83.73 |
| 100% | 77.35 | **88.31** |

With only **25%** of the labelled downstream training data, TexCORD-PVT reaches **80.44%** average classification accuracy, exceeding Scratch-PVT trained using **100%** of the labels (**77.35%**).

---

## Model Efficiency

### Classification

| Model | Avg. Accuracy (%) | Parameters | GFLOPs |
|---|---:|---:|---:|
| USF-MAE | 86.50 | 85.801M | 16.867 |
| TexCORD-ResNet | 87.54 | 13.879M | 1.740 |
| **TexCORD-PVT** | **88.31** | **15.156M** | **2.024** |

### Segmentation

| Model | Avg. Dice (%) | Parameters | GFLOPs |
|---|---:|---:|---:|
| USFM | 82.42 | 101.225M | 27.280 |
| TexCORD-ResNet | 82.65 | 14.593M | 2.412 |
| **TexCORD-PVT** | **83.54** | **15.878M** | **2.682** |

Relative to USFM in the reported segmentation comparison, TexCORD-PVT uses approximately **84.31% fewer parameters** and **90.17% fewer GFLOPs** while achieving a higher average Dice score.

---

## Quick Start

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```

### 2. Create the environment

```bash
conda create -n texcord python=<PYTHON_VERSION> -y
conda activate texcord
pip install -r requirements.txt
```

### 3. Prepare pretraining data

Organize the public ultrasound datasets according to the released data loader and provide a manifest/configuration containing the dataset roots.

```text
data/
├── dataset_01/
├── dataset_02/
├── ...
└── dataset_34/
```

### 4. Pretrain TexCORD

```bash
python <PRETRAIN_SCRIPT> \
    --config <PRETRAIN_CONFIG> \
    --teacher <USFM_CHECKPOINT> \
    --data-root <PRETRAIN_DATA_ROOT> \
    --output-dir <OUTPUT_DIR>
```

### 5. Fine-tune for classification

```bash
python <CLASSIFICATION_SCRIPT> \
    --config <CLASSIFICATION_CONFIG> \
    --checkpoint <TEXCORD_CHECKPOINT> \
    --data-root <DATASET_ROOT>
```

### 6. Fine-tune for segmentation

```bash
python <SEGMENTATION_SCRIPT> \
    --config <SEGMENTATION_CONFIG> \
    --checkpoint <TEXCORD_CHECKPOINT> \
    --data-root <DATASET_ROOT>
```

### 7. Evaluate

```bash
python <EVALUATION_SCRIPT> \
    --config <EVAL_CONFIG> \
    --checkpoint <CHECKPOINT>
```

---

## Suggested Repository Structure

```text
TexCORD/
├── README.md
├── requirements.txt
├── LICENSE
├── assets/
│   └── texcord_overview.png
├── configs/
│   ├── pretrain/
│   ├── classification/
│   └── segmentation/
├── datasets/
├── models/
├── losses/
├── scripts/
├── utils/
├── checkpoints/
└── results/
```

Rename these entries to match the actual codebase.

---

## Reproducibility Checklist

For the public release, document:

- exact dataset versions and download links,
- patient-level train/validation/test splits,
- overlap/near-duplicate audit between pretraining and downstream test data,
- optimizer and learning-rate schedule,
- pretraining batch size and number of epochs,
- all loss weights,
- NT-Xent temperature,
- augmentation probabilities and ranges,
- random seeds,
- teacher checkpoint and distilled teacher layer(s),
- hardware and training time,
- whether downstream encoders are frozen, partially tuned, or fully fine-tuned,
- definition of all reported metrics and uncertainty estimates.

---

## Checkpoints

| Model | Pretraining | Checkpoint |
|---|---|---|
| TexCORD-PVT | 106k+ public ultrasound images | Coming soon |
| TexCORD-ResNet | 106k+ public ultrasound images | Coming soon |

---

## Citation

If you use TexCORD in your research, please cite the paper.

```bibtex
@article{texcord,
  title   = {TexCORD: Texture-Consistent Self-Supervised Learning and Regional Distillation for Lightweight Ultrasound Foundation Models},
  author  = {TODO},
  journal = {TODO},
  year    = {TODO}
}
```

---

## License

Add the license for the **TexCORD code** here. Individual ultrasound datasets remain subject to their original licenses and terms of use.

---

## Acknowledgments

TexCORD builds on publicly available ultrasound datasets and a pretrained ultrasound foundation-model teacher. Please cite the corresponding dataset and model publications when using these resources.

---

## Contact

```text
Name:  <AUTHOR NAME>
Email: <EMAIL>
```
