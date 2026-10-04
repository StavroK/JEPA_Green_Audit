# Semantic perception baseline

## Decision

Add a remote-sensing-specific semantic branch to M3b before large-scale manual
labeling.

The first Sentinel-2 semantic baseline is **SatlasPretrain** rather than a
generic natural-image segmentation model.

Pinned upstream:

- repository: `allenai/satlas`
- commit: `c8b9aa5d4acdd3e4f58eb7cbb28ac18bb12c985f`
- Sentinel-2 model family: Swin-v2, single-image RGB or multispectral
- checkpoint: `sentinel2_swinb_si_rgb.pth`
- SHA256: `94c075a155fc489947dc091305586675649c99c7d8567fe2195792b58875511b`

## Why SatlasPretrain

SatlasPretrain is trained specifically on satellite/aerial imagery and its
low-resolution Sentinel-2 configuration includes a semantic **land_cover**
head. The corresponding taxonomy includes:

- water
- developed
- tree
- shrub
- grass
- crop
- bare
- snow
- wetland
- mangroves
- moss

It also includes a **tree_cover** regression task.

That is materially closer to the Urban Green Intelligence problem than asking
I-JEPA, DINOv2, or ImageNet ResNet18 to provide semantic labels through a simple
linear probe.

## Project taxonomy

For urban-green reporting, map model outputs to a smaller auditable taxonomy:

| Project class | Satlas source classes |
|---|---|
| tree_canopy | tree |
| other_vegetation | shrub, grass, crop, wetland, mangroves, moss |
| developed | developed |
| bare_ground | bare |
| water | water |
| uncertain | background / unsupported / low-confidence |

At 10 m Sentinel-2 resolution, **tree_canopy means tree-dominant land-cover
signal**, not delineated individual crowns.

Buildings, roads, parking and stadium roofs are all primarily represented by
the `developed` class at this resolution. Finer separation requires
higher-resolution imagery and a different semantic/object model.

## Role of each branch

- Semantic model: **what is present?**
- NDVI/spectral indices: **is there a vegetation-related spectral signal?**
- nDSM: **is there vertical structure?**
- I-JEPA: **has the learned visual representation changed?**

No branch should be promoted to sole ground truth.

## M3b experiment

Before manual labeling:

1. run the refined La Pastora Sentinel QC;
2. run Satlas semantic inference on both dates;
3. calculate class composition per AOI and per candidate review cell;
4. use semantic maps only as a QC aid when choosing the labeling grid;
5. create independent human labels from RGB imagery without showing model
   predictions during labeling;
6. benchmark semantic predictions against those labels afterward.

This preserves label independence.

## High-resolution track

For 1–2 m orthophotos, test an aerial segmentation model separately. SAMGeo is
a useful candidate for object-mask generation and vectorization, but its masks
must still be assigned semantic meaning and validated. It should not be used to
make unsupported tree-health claims.

## Acceptance criteria

M3b semantic integration is complete when:

- a pinned, reproducible Sentinel semantic model is runnable locally;
- tree/other-vegetation/developed/bare/water proportions are exported;
- semantic outputs are aligned to the same AOI/grid as other evidence;
- predictions are compared with independent human labels;
- limitations of 10 m tree-canopy inference are explicit;
- Fundidora and La Pastora are reported separately before any pooled result.


## Satlas validation gate

The first two local La Pastora inference runs produced an implausibly high
`developed` fraction (~89–91%) and near-zero `tree_canopy`, even after
replacing black padding with a real 512×512 Sentinel context tile.

Therefore Satlas semantic outputs are currently **unvalidated** and must not be
used as measurements, labels, or AOI-selection truth.

Before deciding whether to retain Satlas:

1. inspect TCI vs semantic maps side by side;
2. inspect per-pixel maximum probability and normalized entropy;
3. verify whether the model is confidently wrong or simply uncertain;
4. verify L1C-vs-L2A TCI domain compatibility;
5. if domain mismatch remains material, replace or supplement Satlas with a
   Sentinel-native semantic product/baseline rather than forcing this model.

NDVI and human RGB labels remain the trusted benchmark references for the
current experiment.


## Model decision after La Pastora QC

The La Pastora Satlas QC shows that the model is not suitable as the primary
semantic baseline for this project in its current configuration.

Observed behavior:
- 2025: ~89.3% developed, ~10.5% other vegetation, ~0.17% tree;
- 2026: ~90.7% developed, ~8.8% other vegetation, 0% tree;
- mean maximum class probability ~0.65-0.70;
- ~65-78% of AOI pixels have top-class probability >= 0.5.

The model is therefore not merely uncertain: it is moderately confident while
visually over-predicting developed land across a vegetation-rich scene.

### Revised semantic strategy

**Primary contemporary semantic baseline:** Google / WRI **Dynamic World V1**.

Reasons:
- 10 m near-real-time land-cover product derived from Sentinel-2 L1C;
- per-image predictions rather than only annual composites;
- class probabilities are available;
- classes map directly to the project needs:
  - trees
  - grass
  - flooded_vegetation
  - crops
  - shrub_and_scrub
  - built
  - bare
  - water
  - snow_and_ice
- outputs can be thresholded by top-class probability before use.

Dynamic World requires Earth Engine access. Keep that dependency optional and
do not make it a prerequisite for the zero-cost GitHub Pages demo.

**Zero-auth historical semantic cross-check:** Microsoft Planetary Computer
`io-lulc-annual-v02` (Impact Observatory 9-class annual LULC), currently
available through 2023. It is not contemporaneous with 2025/2026 imagery, so use
it only as an AOI plausibility / historical land-cover prior, never as current
truth.

**Satlas status:** retained as a documented negative/diagnostic experiment, not
as an operational semantic measurement source.
