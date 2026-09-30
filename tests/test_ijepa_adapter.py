import numpy as np

from jepa_green_audit.ijepa_encoder import (
    IJEPA_CHECKPOINT_NAME,
    IJEPA_EMBED_DIM,
    IJEPA_LICENSE,
    IJEPA_UPSTREAM_COMMIT,
    mean_pool_patch_tokens,
)


def test_ijepa_is_pinned():
    assert len(IJEPA_UPSTREAM_COMMIT) == 40
    assert IJEPA_CHECKPOINT_NAME == "IN1K-vit.h.14-300e.pth.tar"
    assert IJEPA_EMBED_DIM == 1280
    assert IJEPA_LICENSE == "CC BY-NC 4.0"


def test_mean_pool_patch_tokens():
    tokens = np.array(
        [[[1.0, 3.0], [3.0, 5.0]], [[2.0, 4.0], [6.0, 8.0]]],
        dtype=np.float32,
    )
    pooled = mean_pool_patch_tokens(tokens)
    np.testing.assert_allclose(pooled, [[2.0, 4.0], [4.0, 6.0]])
