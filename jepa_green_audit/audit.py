"""Transparent audit score for prototype prioritization."""

from __future__ import annotations


def audit_priority_score(
    vegetation_loss_pp: float,
    embedding_change: float,
    quality: float = 1.0,
) -> float:
    """Compute a transparent 0-100 prototype priority score.

    This is deliberately simple and interpretable. It is not a regulatory or
    biological risk score.

    Parameters
    ----------
    vegetation_loss_pp:
        Positive number representing absolute vegetation loss in percentage
        points. Gains should be passed as zero.
    embedding_change:
        Cosine distance, normally near 0 for similar representations.
    quality:
        Input quality factor from 0 to 1.
    """
    q = max(0.0, min(1.0, float(quality)))
    loss_component = min(max(float(vegetation_loss_pp), 0.0) / 10.0, 1.0)
    embedding_component = min(max(float(embedding_change), 0.0), 1.0)
    score = 100.0 * q * (0.65 * loss_component + 0.35 * embedding_component)
    return round(score, 1)
