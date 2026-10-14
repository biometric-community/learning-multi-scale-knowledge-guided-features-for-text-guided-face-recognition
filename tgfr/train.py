"""Train TGFR on CelebA with attribute captions + multi-scale alignment."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tgfr.data.celeba import CelebACaptionDataset, resolve_celeba_root
from tgfr.losses.align import tgfr_loss
from tgfr.models.tgfr_net import TGFRNet


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _device(cfg: dict, override: Optional[str] = None) -> torch.device:
    name = override or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def train(cfg: dict, device_override: Optional[str] = None) -> Dict:
    device = _device(cfg, device_override)
    root = resolve_celeba_root(cfg)
    seed = int(cfg["train"].get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    max_n = cfg["data"].get("max_images")
    max_n = int(max_n) if max_n is not None else None
    image_size = int(cfg["model"].get("image_size", 112))
    train_ds = CelebACaptionDataset(
        root,
        split="train",
        image_size=image_size,
        train_ratio=float(cfg["data"].get("train_ratio", 0.7)),
        max_n=max_n,
        seed=seed,
        train=True,
    )
    loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"].get("batch_size", 64)),
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 4)),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    model = TGFRNet(
        n_classes=train_ds.num_classes,
        embed_dim=int(cfg["model"].get("embed_dim", 512)),
        attr_dim=int(cfg["model"].get("attr_dim", 40)),
        pretrained=bool(cfg["model"].get("pretrained_backbone", True)),
    ).to(device)

    opt = torch.optim.SGD(
        model.parameters(),
        lr=float(cfg["train"].get("lr", 0.05)),
        momentum=float(cfg["train"].get("momentum", 0.9)),
        weight_decay=float(cfg["train"].get("weight_decay", 5e-4)),
    )
    epochs = int(cfg["train"].get("epochs", 12))
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    log_dir = Path(cfg["paths"]["log_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    history = []
    t0 = time.time()
    model.train()
    for epoch in range(1, epochs + 1):
        parts = []
        pbar = tqdm(loader, desc=f"tgfr {epoch}/{epochs}", leave=False)
        for images, attrs, labels in pbar:
            images = images.to(device, non_blocking=True)
            attrs = attrs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            out = model(images, attrs)
            loss, d = tgfr_loss(
                out,
                labels,
                attrs,
                lambda_id=float(cfg["train"].get("lambda_id", 1.0)),
                lambda_global=float(cfg["train"].get("lambda_global", 0.5)),
                lambda_local=float(cfg["train"].get("lambda_local", 0.5)),
                lambda_attr=float(cfg["train"].get("lambda_attr", 0.2)),
                temperature=float(cfg["train"].get("temperature", 0.07)),
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            parts.append(d)
            pbar.set_postfix(loss=f"{d['total']:.3f}")
        mean = {k: float(np.mean([x[k] for x in parts])) for k in parts[0]}
        mean["epoch"] = epoch
        history.append(mean)
        print(
            f"[tgfr] epoch={epoch} total={mean['total']:.4f} id={mean['id']:.4f} "
            f"g={mean['global']:.4f} l={mean['local']:.4f} attr={mean['attr']:.4f}"
        )

    elapsed = time.time() - t0
    torch.save(
        {"model": model.state_dict(), "num_classes": train_ds.num_classes, "config": cfg},
        ckpt_dir / "last.pt",
    )
    summary = {
        "protocol": cfg["train"].get("protocol", "full"),
        "epochs": epochs,
        "n_train": len(train_ds),
        "num_classes": train_ds.num_classes,
        "elapsed_sec": elapsed,
        "history": history,
        "checkpoint": str(ckpt_dir / "last.pt"),
    }
    with open(log_dir / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"Wrote {log_dir / 'train_summary.json'} elapsed={elapsed:.1f}s")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    train(load_config(args.config), device_override=args.device)


if __name__ == "__main__":
    main()
