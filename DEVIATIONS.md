# DEVIATIONS — Text-Guided FR (CaptionFace-lite)

| ID | Item | Paper | Ours | Reason |
|----|------|-------|------|--------|
| D1 | Caption generator | GPTFace (FR encoder + GPT-2) | CelebA 40-d attributes as text proxies | GPTFace / face-caption corpora not local |
| D2 | Text encoder | ALIGN / BERT-scale | MLP on attributes | Attribute captions only |
| D3 | Image backbone | ArcFace / large FR | ResNet-18 | Lab twin compute |
| D4 | Train data | WebFace / MMCelebA-HQ captions | CelebA identities + attrs | Available locally |
| D5 | MSFA | Full multi-scale MSAM | Global emb + center-crop local + InfoNCE | Simplified MSFA-lite |
| D6 | CKA / fusion | Caption knowledge adaptation + fusion | Omitted | Scope |
| D7 | Eval | + AgeDB / WebFace260M | LFW + CALFW + CPLFW | Datasets available |
| D8 | Epochs | Paper schedule | 12 full / 1 smoke | Full-train budget |
