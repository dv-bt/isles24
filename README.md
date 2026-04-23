# Ischemic Stroke Lesion Segmentation with Swin Transformers

Capstone project for the [Diploma of Advanced Studies in Data Science](https://inf.ethz.ch/continuing-education/das-datascience.html) at ETH Zurich (2026).

## Overview

This repository contains the implementation of **Multi-Encoder Swin-UNETR (ME-Swin-UNETR)**, a modification of the [Swin-UNETR](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR) architecture for multimodal ischemic stroke lesion segmentation.

The model is evaluated on the [ISLES'24 challenge dataset](https://doi.org/10.5281/zenodo.16731717), which provides multimodal acute CT imaging of ischemic stroke cases with the goal of predicting final infarct lesion regions from pre-interventional data.

## Proposed Architecture

Standard Swin-UNETR fuses all input modalities at the input level (early channel fusion) before processing them with a single shared Swin Transformer encoder. ME-Swin-UNETR instead assigns a **dedicated pre-trained Swin Transformer encoder** to each input modality, enabling modality-specific feature extraction. At each encoder stage, the outputs of all branches are fused via a learned 1×1 convolution and passed as skip connections to the shared U-Net-like decoder.

## Dataset

The [ISLES'24 dataset](https://doi.org/10.5281/zenodo.16731717) consists of 149 real-world acute ischemic stroke cases from two centers (TUM Klinikum Rechts der Isar, Germany; Universitätsspital Zürich, Switzerland). Each case includes multimodal CT imaging (NCCT, CTA, and CTP-derived perfusion maps: CBV, CBF, MTT, Tmax), post-treatment MRI, tabular clinical data, and infarct lesion annotations.

This work uses CTA and CBF as input modalities, selected based on a preliminary single-modality evaluation. Only the final infarct lesion masks are used as training targets.

## Methods

- **Preprocessing**: skull stripping with SynthStrip (applied to NCCT, propagated to all modalities), per-modality intensity clipping, resampling to 1 mm isotropic spacing.
- **Training**: AdamW optimizer, cosine LR schedule with linear warmup, DiceCE loss, 80/20 stratified train/val split.
- **Inference**: sliding window, final checkpoint used.
- **Evaluation**: Dice score on the validation set. Dice score on the validation set; metrics computed using the [example ISLES'24 evaluation repository](https://github.com/ezequieldlrosa/isles24).

## Stack

- **Deep learning**: PyTorch, MONAI, PyTorch Lightning
- **Medical imaging I/O**: nibabel
- **Evaluation**: panoptica
- **Experiment tracking**: Weights & Biases

## Installation

```bash
git clone https://github.com/dv-bt/isles24.git
cd isles24
conda env create -f environment.yml
conda activate isles24
```

## References

- Riedel et al. (2025). *ISLES'24 – A Real-World Longitudinal Multimodal Stroke Dataset*. [doi:10.48550/arXiv.2408.11142](https://doi.org/10.48550/arXiv.2408.11142)
- de la Rosa et al. (2025). *ISLES'24: Final Infarct Prediction with Multimodal Imaging and Clinical Data. Where Do We Stand?* [doi:10.48550/arXiv.2408.10966](https://doi.org/10.48550/arXiv.2408.10966)
- Ren et al. (2025). *How We Won the ISLES'24 Challenge by Preprocessing*. [doi:10.48550/arXiv.2505.18424](https://doi:10.48550/arXiv.2505.18424)
- Tang et al. (2022). *Self-Supervised Pre-Training of Swin Transformers for 3D Medical Image Analysis*. [doi:10.48550/arXiv.2111.14791](https://doi.org/10.48550/arXiv.2111.14791)
- Hatamizadeh et al. (2022). *Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images*. [doi:10.48550/arXiv.2201.01266](https://doi.org/10.48550/arXiv.2201.01266)
- Hoopes et al. (2022). *SynthStrip: skull-stripping for any brain image*. NeuroImage. [doi:10.1016/j.neuroimage.2022.119474](https://doi.org/10.1016/j.neuroimage.2022.119474)