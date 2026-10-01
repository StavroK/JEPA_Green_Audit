import numpy as np

from jepa_green_audit.ijepa_encoder import (
    IJEPA_CHECKPOINT_NAME,
    IJEPA_EMBED_DIM,
    IJEPA_LICENSE,
    IJEPA_PATCH_GRID_SIZE,
    IJEPA_UPSTREAM_COMMIT,
    mean_pool_patch_tokens,
    normalize_checkpoint_state_dict,
)


def test_ijepa_is_pinned():
    assert len(IJEPA_UPSTREAM_COMMIT) == 40
    assert IJEPA_CHECKPOINT_NAME == "IN1K-vit.h.14-300e.pth.tar"
    assert IJEPA_EMBED_DIM == 1280
    assert IJEPA_PATCH_GRID_SIZE == 16
    assert IJEPA_LICENSE == "CC BY-NC 4.0"


def test_mean_pool_patch_tokens():
    tokens = np.array(
        [[[1.0, 3.0], [3.0, 5.0]], [[2.0, 4.0], [6.0, 8.0]]],
        dtype=np.float32,
    )
    pooled = mean_pool_patch_tokens(tokens)
    np.testing.assert_allclose(pooled, [[2.0, 4.0], [4.0, 6.0]])


def test_normalize_checkpoint_state_dict_strips_ddp_prefix():
    state = {"module.pos_embed": 1, "module.blocks.0.norm1.weight": 2}
    normalized = normalize_checkpoint_state_dict(state)
    assert set(normalized) == {"pos_embed", "blocks.0.norm1.weight"}
    assert normalized["pos_embed"] == 1


def test_normalize_checkpoint_state_dict_preserves_plain_keys():
    state = {"pos_embed": 1}
    assert normalize_checkpoint_state_dict(state) == state
