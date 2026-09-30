"""Run a tiny deterministic example of the analytical primitives."""

import numpy as np

from jepa_green_audit import (
    cosine_embedding_change,
    coverage_change,
    ndvi,
    vegetation_coverage,
    vegetation_mask,
)
from jepa_green_audit.audit import audit_priority_score


def main() -> None:
    red_before = np.array([[0.20, 0.30], [0.18, 0.40]])
    nir_before = np.array([[0.60, 0.42], [0.55, 0.45]])
    red_after = np.array([[0.25, 0.32], [0.28, 0.42]])
    nir_after = np.array([[0.48, 0.40], [0.40, 0.44]])

    before = vegetation_coverage(vegetation_mask(ndvi(nir_before, red_before)))
    after = vegetation_coverage(vegetation_mask(ndvi(nir_after, red_after)))
    change = coverage_change(before, after)

    emb_before = np.array([0.8, 0.1, 0.3, 0.4])
    emb_after = np.array([0.5, 0.2, 0.2, 0.6])
    latent_change = cosine_embedding_change(emb_before, emb_after)

    loss = max(0.0, -change["delta_percentage_points"])
    priority = audit_priority_score(loss, latent_change)

    print("AI4GOOD JEPA Green Audit — demo analysis")
    print(change)
    print(f"Embedding change: {latent_change:.3f}")
    print(f"Audit priority: {priority:.1f}/100")


if __name__ == "__main__":
    main()
