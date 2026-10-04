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
- initial checkpoint candidate: `sentinel2_swinb_si_rgb.pth`

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
