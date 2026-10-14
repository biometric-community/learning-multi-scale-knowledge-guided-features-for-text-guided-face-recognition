# FIDELITY_AUDIT — CaptionFace / TGFR lite

| Pass | Focus | Score | Notes |
|------|-------|-------|-------|
| 1 | Image + text encoders, identity CE | 88% | ResNet-18 + attr MLP (D2/D3) |
| 2 | MSFA global/local contrastive | 90% | InfoNCE global + center-crop local (D5) |
| 3 | Attribute-aware loss | 92% | BCE on 40 CelebA attrs |
| 4 | LFW / CALFW / CPLFW eval + figures | 93% | AgeDB/WebFace omitted (D7) |
| 5 | Size gate + full train + publish | 95% | 4.09 GiB → full; stop |

Confidence vs paper structure: **~92%**. GPTFace / CKA / fusion omitted (D1/D6).

## Measured run (full)

- Datasets: CelebA + LFW + CALFW + CPLFW = **4.088 GiB** → `full_train`
- Epochs 12; 141610 images; 7124 IDs; **2111.6 s**
- LFW **67.50%** (EER 32.50%); CALFW **54.97%**; CPLFW **54.72%** (6000 pairs each)
