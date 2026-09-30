# I-JEPA Research Encoder Integration

## Status

This repository supports an optional frozen I-JEPA image encoder for research/prototype experiments.

It is deliberately not installed by the default lightweight environment and is not used by the static GitHub Pages demo.

## Pinned upstream dependency

- Upstream: facebookresearch/ijepa
- Commit: 52c1ae95d05f743e000e8f10a1f3a79b10cff048
- Model: ViT-H/14
- Input: RGB 224 x 224
- Pretraining data: ImageNet-1K
- Published training: 300 epochs
- Checkpoint: IN1K-vit.h.14-300e.pth.tar
- Embedding dimension: 1280
- Adapter pooling: mean over output patch tokens
- Project license: CC BY-NC 4.0

Official checkpoint URL:
https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar

## Licensing boundary

The official archived I-JEPA repository is distributed under Creative Commons Attribution-NonCommercial 4.0.

Therefore:
- this adapter is suitable for non-commercial research/prototyping under the upstream terms;
- the checkpoint is not bundled in this repository;
- the upstream source is not vendored into this repository;
- a commercial AI4GOOD product should not assume it may use this dependency without separately resolving licensing.

This is a model-governance decision, not merely an installation detail.

## Explicit installation

Clone the exact pinned upstream source:

    mkdir -p vendor models
    git clone https://github.com/facebookresearch/ijepa.git vendor/ijepa
    git -C vendor/ijepa checkout 52c1ae95d05f743e000e8f10a1f3a79b10cff048

Download the published checkpoint explicitly:

    curl -L https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar \
      -o models/IN1K-vit.h.14-300e.pth.tar

Install the optional ML environment:

    python -m venv .venv-jepa
    source .venv-jepa/bin/activate
    pip install -r requirements-jepa.txt
    pip install -e .

## Deterministic preprocessing

The adapter:
1. accepts an RGB image;
2. converts values to [0, 1];
3. resizes deterministically to 224 x 224 with bilinear interpolation;
4. applies ImageNet channel normalization;
5. runs the frozen ViT-H/14 encoder;
6. mean-pools patch-token representations;
7. returns one 1280-dimensional float32 vector.

No random crop, color jitter, or augmentation is performed during inference.

## Frozen inference

Every parameter has requires_grad=False, the model is put into eval mode, and inference uses torch.inference_mode().

This is intentionally the first integration step. Domain adaptation is a separate experiment and must not be conflated with the frozen baseline.

## Pairwise change experiment

Given aligned RGB tiles:

    python scripts/ijepa_embed.py \
      --upstream vendor/ijepa \
      --checkpoint models/IN1K-vit.h.14-300e.pth.tar \
      --before data/interim/tile_2025.png \
      --after data/interim/tile_2026.png \
      --device cuda \
      --output outputs/ijepa_pair.json

The script outputs both embeddings and feeds them directly into cosine_embedding_change().

A larger distance means that the learned representation changed more. It does not identify the cause of the change and must not be interpreted by itself as vegetation loss.

## Hardware expectations

CPU:
Supported for functional testing and small experiments, but ViT-H inference is large and will be slow and memory intensive.

GPU:
Recommended for practical batch inference. Start with a modern CUDA-capable GPU with substantial VRAM and benchmark actual memory usage before scaling.

Pretraining:
Out of scope. The upstream project documents a much larger multi-GPU configuration for reproducing ViT-H/14 pretraining. The AI4GOOD MVP does not attempt this.

## Why we do not auto-download

The zero-cost web demo and standard CI must remain lightweight. The project will not unexpectedly:
- download a multi-gigabyte checkpoint;
- install GPU frameworks;
- accept a non-commercial license on the user's behalf;
- make browser availability depend on ML infrastructure.

## Completion validation

To close Issue #2, run one explicit local or GPU-backed inference with the official checkpoint and verify:
- a 1280-D embedding is produced;
- repeated inference on the same tile is deterministic within normal numerical tolerance;
- a before/after pair produces a finite cosine change value;
- provenance records the pinned encoder and checkpoint.
