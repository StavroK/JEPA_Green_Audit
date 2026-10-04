# M3 Fundidora benchmark summary

## Scope

This milestone evaluates vegetation-presence classification and representation
quality in Parque Fundidora with geographic holdouts. It deliberately separates
three questions:

1. How well do NDVI and learned RGB representations perform on the 2025/2026
   Sentinel-2 benchmark?
2. Does adding 2024 structural context from INEGI nDSM help?
3. How sensitive are learned RGB representations to spatial resolution when
   acquisition date, geography, and labels are held constant?

The benchmark concerns **visible vegetation presence / representation quality**.
It does not diagnose biological tree health and does not attribute causes of
vegetation change.

## 2025/2026 Sentinel-2 human-label benchmark

Human review produced 128 labels across two dates:

- 78 vegetation
- 50 non-vegetation
- 0 uncertain

Four contiguous geographic column folds were used. The fixed test split was
recognized as too vegetation-heavy and was therefore not used as the primary
scientific conclusion.

At 100% labels, geographic CV produced:

| Model | IoU | F1 |
|---|---:|---:|
| NDVI | 0.682 ± 0.091 | 0.808 ± 0.066 |
| I-JEPA | 0.600 ± 0.247 | 0.716 ± 0.220 |
| I-JEPA PCA | 0.557 ± 0.245 | 0.676 ± 0.253 |
| I-JEPA + NDVI | 0.600 ± 0.247 | 0.716 ± 0.220 |
| Always vegetation | 0.609 ± 0.258 | 0.725 ± 0.204 |

NDVI remained stronger than I-JEPA across the useful 100%, 50%, 25%, and 10%
label regimes. The apparent I-JEPA advantage at 5% and 1% is not considered
evidence of superior label efficiency because those settings collapse to the
minimum block regime and both models are weak relative to the trivial reference.

## 2024 structural context

INEGI G14C26A3 MDS and MDT were combined into a 1.5 m nDSM and aggregated to
the same human-review cells.

At 100% labels:

| Model | IoU | F1 |
|---|---:|---:|
| NDVI | 0.682 ± 0.091 | 0.808 ± 0.066 |
| nDSM | 0.647 ± 0.138 | 0.778 ± 0.095 |
| NDVI + nDSM | 0.657 ± 0.059 | 0.792 ± 0.042 |
| I-JEPA | 0.600 ± 0.247 | 0.716 ± 0.220 |
| I-JEPA + nDSM | 0.603 ± 0.242 | 0.721 ± 0.213 |
| I-JEPA + NDVI + nDSM | 0.603 ± 0.242 | 0.721 ± 0.213 |

The nDSM contains useful structural signal and reduces variability when paired
with NDVI, but it does not rescue the Sentinel RGB I-JEPA representation. The
2024 nDSM is treated as static structural context rather than exact 2025/2026
ground truth.

## Controlled 2007 spatial-resolution benchmark

A verified INEGI G14C26A3 RGB orthophoto (2007, 1 m) was used to isolate
resolution from temporal change.

The native AOI was divided into a 64 × 36 geographic grid. A deterministic
spatial sample of 384 cells was human-reviewed from the 1 m RGB only:

- 320 usable labels
- 105 vegetation
- 215 non-vegetation
- 64 uncertain and excluded

The exact same geographic cells and labels were reused after area-averaging the
same image to 2 m, 5 m, and 10 m.

### Frozen representations

| Resolution | Frozen ResNet18 F1 | Frozen I-JEPA F1 | Frozen DINOv2 F1 |
|---:|---:|---:|---:|
| 1 m | 0.652 ± 0.099 | **0.734 ± 0.096** | 0.592 ± 0.167 |
| 2 m | 0.636 ± 0.141 | **0.702 ± 0.093** | 0.555 ± 0.163 |
| 5 m | 0.588 ± 0.168 | **0.643 ± 0.137** | 0.605 ± 0.156 |
| 10 m | 0.601 ± 0.113 | **0.711 ± 0.097** | 0.577 ± 0.147 |

Frozen I-JEPA is the strongest frozen representation at every tested
resolution. All learned models exceed the always-vegetation reference
(F1 0.448 ± 0.207).

This result supports a narrower claim: **I-JEPA transfers well as a frozen
representation on this controlled high-resolution urban-vegetation task.**

### Stronger supervised comparator

ImageNet ResNet18 was also partially adapted by fine-tuning layer4 and the
classifier head inside each geographic training fold.

| Resolution | Fine-tuned ResNet18 F1 | Frozen I-JEPA F1 |
|---:|---:|---:|
| 1 m | **0.738 ± 0.124** | 0.734 ± 0.096 |
| 2 m | **0.713 ± 0.131** | 0.702 ± 0.093 |
| 5 m | **0.707 ± 0.125** | 0.643 ± 0.137 |
| 10 m | 0.705 ± 0.098 | **0.711 ± 0.097** |

Once modest task-specific supervised adaptation is allowed, the apparent JEPA
advantage largely disappears. Therefore the benchmark does **not** support a
claim that JEPA is universally superior to supervised learning.

## Resolution interpretation

The partially fine-tuned ResNet18 curve declines smoothly from 1 m to 10 m,
which is consistent with a modest spatial-resolution penalty.

Frozen I-JEPA is non-monotonic and rebounds at 10 m. The rebound occurs across
multiple geographic folds rather than one isolated fold. Plausible contributors
include area averaging, resizing each cell to 224 × 224 before encoding, and
representation-specific receptive-field behavior. The benchmark therefore does
not justify a simple statement that performance always falls as GSD becomes
coarser.

## Geographic imbalance and uncertainty

The four 2007 test folds have markedly different vegetation prevalence:

- fold 0: 9 / 73
- fold 1: 11 / 72
- fold 2: 41 / 88
- fold 3: 44 / 87

This contributes to the observed variance. Paired fold comparisons are preferred
over comparisons of aggregate means alone. The repository includes
scripts/analyze_2007_paired_effects.py to report paired ΔF1 and exhaustive
four-fold bootstrap intervals.

Because there are only four geographic folds and one site, those intervals
should be treated as descriptive uncertainty rather than strong population-level
significance evidence.

## What M3 supports

The evidence supports the following conclusions:

- NDVI is the strongest simple feature on the 2025/2026 Sentinel benchmark.
- 2024 nDSM provides useful structural information but does not materially
  improve the Sentinel I-JEPA result.
- Coarse/mixed Sentinel RGB likely contributed to weak frozen I-JEPA transfer.
- On controlled 2007 high-resolution RGB, frozen I-JEPA outperforms both frozen
  supervised ResNet18 and frozen DINOv2.
- With modest supervised task adaptation, ResNet18 becomes competitive with or
  better than frozen I-JEPA.
- The result is site-specific and should be replicated on a second, denser-green
  site before making broader claims.

## Remaining acceptance gap

Issue #3 originally asks for supervised, non-JEPA self-supervised, JEPA,
NDVI-only, and JEPA+NDVI comparisons across the label-efficiency fractions.

The current repository now contains supervised and non-JEPA SSL controls in the
controlled 2007 RGB benchmark, but those two controls have **not yet been run
through the original 2025/2026 Sentinel label-fraction experiment**.

For strict closure against the original issue wording, that is the remaining
gap. Multi-site replication belongs in a separate follow-on milestone rather
than expanding M3 indefinitely.
