# Learning Multi-Scale Knowledge-Guided Features for Text-Guided Face Recognition

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

**CaptionFace-lite** lab twin: multi-scale image–text alignment for text-guided face recognition using CelebA attributes as caption proxies (GPTFace omitted; see [DEVIATIONS.md](./DEVIATIONS.md)).

**Repository:** https://github.com/biometric-community/learning-multi-scale-knowledge-guided-features-for-text-guided-face-recognition

**Paper:** [10.1109/TBIOM.2024.3466216](https://doi.org/10.1109/TBIOM.2024.3466216)

## Features

- ResNet-18 image encoder + attribute MLP text encoder
- MSFA-lite: global / local InfoNCE vs text embeddings
- Identity CE + attribute-aware BCE
- Eval on LFW / CALFW / CPLFW verification

## Setup

```bash
bash scripts/setup_env.sh
```

Point configs at local datasets under `projects/datasets/` (CelebA, LFW, CALFW, CPLFW).

## Train / Eval / Report

```bash
bash scripts/smoke.sh
bash scripts/train_full.sh
```

See [REPORT.md](./REPORT.md) for metrics and figures.

## Layout

```
tgfr/           # package (data, models, losses, train/eval/report)
configs/        # smoke.yaml, default.yaml
scripts/        # shell entrypoints
figures/        # tracked SVG embeds for REPORT.md
```

## License

[CC BY 4.0](./LICENSE)
