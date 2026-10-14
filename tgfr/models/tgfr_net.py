"""Image encoder + attribute/text MLP with global/local projections (MSFA-lite)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class TGFRNet(nn.Module):
    def __init__(
        self,
        n_classes: int,
        embed_dim: int = 512,
        attr_dim: int = 40,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        net = models.resnet18(weights=weights)
        self.features = nn.Sequential(*list(net.children())[:-1])
        self.fc = nn.Linear(512, embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim)
        self.classifier = nn.Linear(embed_dim, n_classes)
        # attribute / caption encoder (GPTFace substitute)
        self.text_mlp = nn.Sequential(
            nn.Linear(attr_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embed_dim),
        )
        # multi-scale heads
        self.proj_global = nn.Linear(embed_dim, embed_dim)
        self.proj_local = nn.Linear(embed_dim, embed_dim)
        self.proj_text = nn.Linear(embed_dim, embed_dim)
        # attribute prediction from image (attribute-aware loss)
        self.attr_head = nn.Linear(embed_dim, attr_dim)

    def embed_image(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.bn(self.fc(x))
        return F.normalize(x, p=2, dim=1)

    def embed_text(self, attrs: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.text_mlp(attrs), p=2, dim=1)

    def forward(self, images: torch.Tensor, attrs=None):
        emb = self.embed_image(images)
        # local: center crop embedding via spatial avg of center region
        # approximate by projecting same emb through local head (deviated MSFA)
        g = F.normalize(self.proj_global(emb), p=2, dim=1)
        # true local crop
        _, _, h, w = images.shape
        y1, y2 = h // 4, 3 * h // 4
        x1, x2 = w // 4, 3 * w // 4
        local_img = F.interpolate(images[:, :, y1:y2, x1:x2], size=(h, w), mode="bilinear", align_corners=False)
        emb_l = self.embed_image(local_img)
        l = F.normalize(self.proj_local(emb_l), p=2, dim=1)
        logits = self.classifier(emb)
        attr_logits = self.attr_head(emb)
        out = {
            "embed": emb,
            "global": g,
            "local": l,
            "logits": logits,
            "attr_logits": attr_logits,
        }
        if attrs is not None:
            t = self.embed_text(attrs)
            out["text"] = F.normalize(self.proj_text(t), p=2, dim=1)
            out["text_raw"] = t
        return out
