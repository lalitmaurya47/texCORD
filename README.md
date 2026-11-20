# TexCORD: Texture-Consistent Self-Supervised Learning and Regional Distillation for Lightweight Ultrasound Foundation Models

<p align="center">
  <b>Texture-aware self-supervised learning and ultrasound foundation-model distillation for compact, transferable ultrasound representations.</b>
</p>

<p align="center">
  <img src="assets/texcord_overview.png" alt="TexCORD overview" width="95%">
</p>

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



## Acknowledgments

TexCORD builds on publicly available ultrasound datasets and a pretrained ultrasound foundation-model teacher. Please cite the corresponding dataset and model publications when using these resources.

---


