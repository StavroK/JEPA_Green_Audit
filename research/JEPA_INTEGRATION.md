# JEPA Integration Strategy

## Architecture choice

For single satellite/aerial images, start with an **image JEPA (I-JEPA-style) encoder** or a comparable self-supervised image encoder. Do not use a video world model merely because it is newer.

For repeated observations through time, add a temporal experiment later. V-JEPA can become relevant when the input is represented as an ordered sequence and the research question specifically concerns temporal prediction rather than pairwise change detection.

## MVP strategy

Do **not** train a large JEPA from scratch.

1. establish the NDVI and supervised segmentation baselines;
2. use a pretrained self-supervised image encoder;
3. freeze the encoder and train a lightweight downstream head;
4. evaluate label efficiency;
5. domain-adapt using unlabeled urban imagery;
6. only then consider deeper fine-tuning or self-supervised pretraining.

## Encoder interface

The rest of this repository should consume embeddings without assuming a particular checkpoint implementation:

```python
embedding = encoder.encode(image_tile)
```

The downstream system then performs:
- segmentation / classification;
- embedding comparison across time;
- feature fusion with spectral indices.

## Current upstream references

Meta FAIR maintains the JEPA family of research implementations. The upstream code, model licenses, checkpoint requirements, and exact inference APIs should be reviewed before adding a pinned integration.

Relevant upstream projects:
- I-JEPA / image JEPA research implementation;
- V-JEPA;
- V-JEPA 2 / 2.1 for video representation learning and temporal prediction.

## Why the repository does not silently download model weights

A public prototype should not:
- download multi-gigabyte checkpoints unexpectedly;
- assume a GPU;
- hide third-party model-license obligations;
- make the GitHub Pages demo depend on server-side inference.

Actual checkpoint integration belongs in an optional ML environment and should include:
- explicit model/version;
- checksum or immutable revision;
- license note;
- expected VRAM/RAM;
- reproducible preprocessing.
