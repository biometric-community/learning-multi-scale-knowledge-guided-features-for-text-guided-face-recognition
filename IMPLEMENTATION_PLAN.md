# Implementation Plan — CaptionFace / TGFR (lite)

1. CelebA loader: identity labels + 40-d attribute vectors as caption proxies (GPTFace omitted).
2. ResNet-18 image encoder + MLP text encoder; MSFA-lite global/local InfoNCE vs text.
3. Identity CE + attribute BCE (attribute-aware) + multi-scale contrastive losses.
4. Train on CelebA; evaluate 1:1 verification on LFW / CALFW / CPLFW.
5. Smoke → size gate (&lt;5 GiB) → full train → REPORT.md figures → publish.
