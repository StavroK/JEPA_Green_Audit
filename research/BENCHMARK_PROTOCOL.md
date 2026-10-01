# M3 Benchmark Protocol

## Goal

Measure how downstream vegetation-segmentation performance changes as the amount
of labeled data is reduced, while preventing obvious spatial leakage.

The benchmark is intentionally separated into an **engineering smoke test** and
a **scientific benchmark**. The smoke test can use weak labels to validate the
pipeline; claims about label efficiency require labels that are independent of
the evaluated feature being used.

## Resolution

Frozen I-JEPA ViT-H/14 produces a 16 × 16 grid of 1280-dimensional patch
representations for a 224 × 224 RGB input. M3 therefore starts with a coarse
16 × 16 segmentation benchmark before any higher-resolution decoder is added.

## Fixed geographic split

For the Fundidora pilot, the initial benchmark uses contiguous west-to-east
holdout bands:

- train: western 50%;
- validation: middle 25%;
- test: eastern 25%.

Training-label fractions are selected by **whole spatial blocks**, not random
pixels, so the 100%, 50%, 25%, 10%, 5%, and 1% conditions do not create
pixel-level label leakage.

The same geographic masks must be reused for every representation and every
label fraction.

## Representations

Initial linear-probe comparisons:

1. NDVI only;
2. frozen I-JEPA patch embedding;
3. frozen I-JEPA patch embedding + NDVI.

Required before M3 can close:

4. supervised image baseline;
5. non-JEPA self-supervised baseline;
6. repeatable uncertainty estimates across geographic blocks and seeds.

## Labels

### Smoke test

Sentinel-2 Scene Classification Layer (SCL) class 4 may be used as a
**weak vegetation proxy** to verify that the benchmark code, splits, probes,
and metrics work end to end.

This result must be labeled as weak supervision and must not be presented as
independent ground truth.

### Scientific benchmark

The final label-efficiency result must use vegetation labels that are
independent of NDVI and of the model being evaluated. Suitable options include
reviewed polygon/mask annotations or a documented external land-cover label
source with appropriate spatial and temporal alignment.

Do **not** use an NDVI threshold as the target label when evaluating an
NDVI-only baseline; that would make the comparison circular.

## Metrics

Report on the fixed test geography:

- IoU;
- Dice/F1;
- precision;
- recall.

Also report sample support and the number of labeled training blocks used at
each fraction.

## Uncertainty

At minimum:

- repeat label-block selection across multiple deterministic seeds;
- aggregate mean and spread across seeds;
- where sample size permits, bootstrap by geographic block rather than by
  individual pixel.

## Interpretation guardrails

A higher segmentation score means better agreement with the selected label
source at the benchmark resolution. It does not by itself establish biological
tree health, causal vegetation loss, or operational readiness.

The upstream I-JEPA dependency is CC BY-NC 4.0 and remains
research/prototype-only unless separate commercial licensing is obtained.
