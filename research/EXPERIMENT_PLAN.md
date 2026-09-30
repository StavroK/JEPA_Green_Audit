# Experiment Plan

## Research question

Can JEPA-style self-supervised visual representations reduce the amount of labeled data required for useful urban vegetation monitoring?

## Hypotheses

**H1 — Label efficiency**  
A JEPA encoder adapted with unlabeled urban imagery will retain stronger downstream segmentation performance than a supervised-only baseline as labeled training data is reduced.

**H2 — Complementarity**  
JEPA embeddings plus spectral vegetation features will outperform either representation alone on at least one downstream monitoring task.

**H3 — Temporal sensitivity**  
Embedding distance between aligned observations will separate meaningful vegetation-change events from stable areas better than raw RGB difference alone.

## Baselines

- supervised CNN / ResNet;
- supervised Vision Transformer;
- DINO-family self-supervised encoder;
- JEPA encoder;
- NDVI-only baseline;
- JEPA + NDVI fusion.

## Label-efficiency protocol

Train each downstream model using:

- 100% labels;
- 50%;
- 25%;
- 10%;
- 5%;
- 1%.

Keep train/validation/test geographic partitions fixed to avoid spatial leakage.

## Tasks

### Task A — vegetation segmentation
Metrics:
- IoU;
- Dice/F1;
- precision;
- recall.

### Task B — temporal change
Classes:
- stable;
- vegetation gain;
- vegetation loss.

Metrics:
- macro F1;
- per-class recall;
- false-positive rate.

### Task C — representation change
Measure cosine distance between aligned embeddings and compare against verified change labels.

Metrics:
- ROC-AUC;
- PR-AUC;
- threshold calibration.

## Data split safeguards

Avoid random tile splitting when adjacent tiles are highly correlated. Prefer geographic holdouts by neighborhood, corridor, or municipality.

## Ablations

- RGB vs RGB+NIR;
- frozen encoder vs partial fine-tuning;
- generic pretrained encoder vs domain-adapted encoder;
- JEPA embedding only vs JEPA + NDVI;
- single-date vs temporal-pair features.

## Minimum credible result

The prototype is successful even if JEPA is not the best model, provided the experiment can clearly answer:

1. how much labeled data each approach needs;
2. where JEPA helps or fails;
3. how robust results are across geography and season;
4. what business workflow remains viable.
