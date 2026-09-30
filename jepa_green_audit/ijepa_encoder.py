"""Optional adapter for Meta FAIR's archived I-JEPA research encoder.

This module does not download third-party code or weights. The caller must
explicitly provide a local checkout of the pinned upstream repository and a
local official checkpoint.

The upstream I-JEPA project is CC BY-NC 4.0. This integration is therefore
research/prototype-only unless a separate commercial license is obtained.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib
import sys
from typing import Any

import numpy as np

IJEPA_UPSTREAM_REPO = "https://github.com/facebookresearch/ijepa"
IJEPA_UPSTREAM_COMMIT = "52c1ae95d05f743e000e8f10a1f3a79b10cff048"
IJEPA_LICENSE = "CC BY-NC 4.0"
IJEPA_MODEL_NAME = "vit_huge"
IJEPA_PATCH_SIZE = 14
IJEPA_IMAGE_SIZE = 224
IJEPA_EMBED_DIM = 1280
IJEPA_CHECKPOINT_NAME = "IN1K-vit.h.14-300e.pth.tar"
IJEPA_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar"
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class IJEPAConfig:
    upstream_commit: str = IJEPA_UPSTREAM_COMMIT
    model_name: str = IJEPA_MODEL_NAME
    patch_size: int = IJEPA_PATCH_SIZE
    image_size: int = IJEPA_IMAGE_SIZE
    embed_dim: int = IJEPA_EMBED_DIM
    checkpoint_name: str = IJEPA_CHECKPOINT_NAME
    checkpoint_url: str = IJEPA_CHECKPOINT_URL
    license: str = IJEPA_LICENSE


def mean_pool_patch_tokens(tokens: np.ndarray) -> np.ndarray:
    """Mean-pool [batch, patches, dim] patch tokens into [batch, dim]."""
    arr = np.asarray(tokens)
    if arr.ndim != 3:
        raise ValueError("tokens must have shape [batch, patches, dim]")
    return arr.mean(axis=1)


def verify_upstream_checkout(upstream_dir: str | Path) -> None:
    """Fail fast if the expected upstream model file is not present."""
    root = Path(upstream_dir)
    required = root / "src" / "models" / "vision_transformer.py"
    if not required.exists():
        raise FileNotFoundError(
            f"{required} not found. Clone the pinned official I-JEPA repository first."
        )


class FrozenIJEPAEncoder:
    """Frozen feature extractor around the official I-JEPA ViT-H/14 encoder."""

    def __init__(
        self,
        upstream_dir: str | Path,
        checkpoint_path: str | Path,
        device: str = "cpu",
    ) -> None:
        verify_upstream_checkout(upstream_dir)
        self.upstream_dir = Path(upstream_dir).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(self.checkpoint_path)

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is optional. Install requirements-jepa.txt to use I-JEPA."
            ) from exc

        self.torch = torch
        self.device = torch.device(device)
        self.model = self._load_model()

    def _load_model(self):
        torch = self.torch
        upstream_str = str(self.upstream_dir)
        if upstream_str not in sys.path:
            sys.path.insert(0, upstream_str)

        vit = importlib.import_module("src.models.vision_transformer")
        model_factory = getattr(vit, IJEPA_MODEL_NAME)
        encoder = model_factory(
            img_size=[IJEPA_IMAGE_SIZE],
            patch_size=IJEPA_PATCH_SIZE,
        )

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=torch.device("cpu"),
            weights_only=False,
        )
        if "encoder" not in checkpoint:
            raise KeyError("Official I-JEPA checkpoint does not contain 'encoder'.")

        state = checkpoint["encoder"]
        msg = encoder.load_state_dict(state, strict=True)
        if msg.missing_keys or msg.unexpected_keys:
            raise RuntimeError(
                f"Checkpoint mismatch. missing={msg.missing_keys}, "
                f"unexpected={msg.unexpected_keys}"
            )

        encoder.eval()
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        encoder.to(self.device)
        return encoder

    def preprocess_rgb(self, rgb: np.ndarray):
        """Resize RGB deterministically to 224x224 and normalize."""
        torch = self.torch
        arr = np.asarray(rgb)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError("rgb must have shape [height, width, 3]")

        arr = arr.astype(np.float32)
        if arr.size and arr.max() > 1.0:
            arr /= 255.0
        arr = np.clip(arr, 0.0, 1.0)

        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=(IJEPA_IMAGE_SIZE, IJEPA_IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor.to(self.device)

    def encode_rgb(self, rgb: np.ndarray) -> np.ndarray:
        """Return one 1280-D mean-pooled embedding for an RGB image."""
        tensor = self.preprocess_rgb(rgb)
        with self.torch.inference_mode():
            patch_tokens = self.model(tensor)
            embedding = patch_tokens.mean(dim=1)
        return embedding.detach().cpu().numpy()[0].astype(np.float32)

    def metadata(self) -> dict[str, Any]:
        return {
            "family": "I-JEPA",
            "upstream_repo": IJEPA_UPSTREAM_REPO,
            "upstream_commit": IJEPA_UPSTREAM_COMMIT,
            "license": IJEPA_LICENSE,
            "model": "ViT-H/14",
            "image_size": IJEPA_IMAGE_SIZE,
            "embedding_dim": IJEPA_EMBED_DIM,
            "checkpoint": IJEPA_CHECKPOINT_NAME,
            "checkpoint_source": IJEPA_CHECKPOINT_URL,
            "frozen": True,
            "pooling": "mean over patch tokens",
            "normalization": "ImageNet mean/std",
        }
