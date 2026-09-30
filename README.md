# AI4GOOD Urban Green Intelligence — JEPA Green Audit

**Self-supervised urban vegetation monitoring using Joint-Embedding Predictive Architectures (JEPA).**

This repository is an AI4GOOD México prototype exploring whether JEPA-style self-supervised visual representations can reduce the amount of labeled geospatial data required for urban vegetation monitoring.

## What the prototype is designed to answer

- Where is urban vegetation being lost or gained?
- Which zones show meaningful changes between observation periods?
- Can self-supervised JEPA representations improve label efficiency for vegetation segmentation?
- Can learned representations complement conventional vegetation indices such as NDVI?
- How can the outputs become auditable indicators for municipalities, ESG teams, real-estate developers, and environmental consultants?

## MVP v0.1

The first version focuses on four capabilities:

1. **Select an area of interest** on an urban map.
2. **Estimate vegetation coverage** from geospatial imagery.
3. **Compare two observations** and flag canopy/vegetation change.
4. **Produce an interpretable change / audit score** that can later incorporate JEPA embedding distance.

The browser demo uses transparent sample data so it can run at zero infrastructure cost on GitHub Pages. The Python package contains the analytical building blocks for replacing sample values with real Sentinel-2/aerial imagery and JEPA embeddings.

## Architecture

```text
Satellite / aerial imagery
          |
          v
  preprocessing / tiling
          |
    +-----+------+
    |            |
    v            v
JEPA encoder   spectral indices
    |          (NDVI, etc.)
    +-----+------+
          |
          v
representation + feature fusion
          |
   +------+------+ 
   |             |
   v             v
segmentation   temporal change
   |             |
   +------+------+
          |
          v
 auditable geo indicators
          |
          v
 dashboard / API / reports
```

See [architecture/ARCHITECTURE.md](architecture/ARCHITECTURE.md).

## Research hypothesis

> A JEPA-based self-supervised encoder adapted to urban remote-sensing imagery can learn transferable representations that improve vegetation-monitoring performance and/or label efficiency compared with supervised-only baselines.

The benchmark plan compares:

- supervised CNN/ResNet baseline;
- supervised Vision Transformer baseline;
- DINO-style self-supervised baseline;
- JEPA encoder;
- NDVI-only;
- JEPA + NDVI feature fusion.

See [research/EXPERIMENT_PLAN.md](research/EXPERIMENT_PLAN.md).

## Quick start — browser demo

Open `docs/index.html` locally, or enable GitHub Pages for the repository. The included GitHub Actions workflow publishes the `docs/` directory.

The demo deliberately labels modeled values as **prototype/sample estimates** until connected to validated imagery and model outputs.

## Quick start — Python

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/demo_analysis.py
```

## Data strategy

### Phase 1 — open imagery
Use Sentinel-2 for neighborhood/park scale monitoring and multispectral indices. Ten-meter bands are useful for area-level vegetation analysis but are not suitable for reliable individual-tree crown delineation.

### Phase 2 — higher resolution
Introduce open aerial orthophotos, drone imagery, or appropriately licensed high-resolution imagery for street/tree-crown level analysis.

### Phase 3 — domain adaptation
Adapt a pretrained self-supervised vision encoder to unlabeled local imagery before fine-tuning downstream segmentation or classification heads.

See [data/README.md](data/README.md).

## Responsible interpretation

The project distinguishes:

- **vegetation presence** from biological tree health;
- **model output** from verified field observation;
- **change signal** from causal explanation;
- **remote-sensing estimate** from municipal inventory data.

The MVP uses the term **Vegetation Condition Signal** rather than claiming a medical/biological diagnosis of plant health.

See [MODEL_CARD.md](MODEL_CARD.md).

## Business applications

Potential users include:

- municipalities and metropolitan planning teams;
- environmental and ESG consultants;
- real-estate developers;
- industrial facilities and logistics parks;
- universities and urban-research groups.

The product is positioned around measurable outputs—coverage, loss/gain, trends, spatial inequality, and field-inspection prioritization—not around selling an AI model.

See [BUSINESS_CASE.md](BUSINESS_CASE.md).

## Status

**Prototype / research project.** Not yet a validated environmental monitoring product.

## Brand

Built as an **AI4GOOD México** applied-AI experiment.

https://www.ai4good.mx
