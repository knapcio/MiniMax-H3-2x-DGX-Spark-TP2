"""insert_anchor_block puts the anchor rows, shape and ref block right before upstream's latent guide."""
import pytest

torch = pytest.importorskip("torch")
from h3_tp2.plugin import insert_anchor_block  # noqa: E402


def _args(gh=8, gw=12, gt=7, image_rows=100):
    fr = (gh // 2) * (gw // 2)
    rows = torch.cat((torch.full((image_rows, 96), 1.0), torch.full((gt * fr, 96), 3.0)))
    return {"visual_condition": rows, "visual_condition_shapes": [(1, 16, 24), (gt, gh, gw)],
            "ref_blocks": [{"kind": "image", "latent_h": 16, "latent_w": 24}, {"kind": "audio", "ref_audio_t": 9},
                           {"kind": "latent_guide", "latent_t": gt, "latent_h": gh, "latent_w": gw, "ref_audio_t": 11}]}


def test_anchor_goes_before_the_guide():
    a = _args()
    fr = 4 * 6
    anchor = torch.full((7 * fr, 96), 2.0)
    out = insert_anchor_block(a, anchor, 7)
    rows = out["visual_condition"]
    assert rows.shape[0] == 100 + 7 * fr + 7 * fr
    assert (rows[:100] == 1).all() and (rows[100:100 + 7 * fr] == 2).all() and (rows[100 + 7 * fr:] == 3).all()
    assert out["visual_condition_shapes"] == [(1, 16, 24), (7, 8, 12), (7, 8, 12)]
    kinds = [b["kind"] for b in out["ref_blocks"]]
    assert kinds == ["image", "audio", "video", "latent_guide"]
    assert out["ref_blocks"][2] == {"kind": "video", "ref_audio_t": 0, "latent_t": 7, "latent_h": 8, "latent_w": 12}


def test_anchor_rejects_wrong_layout():
    a = _args()
    a["ref_blocks"].pop()  # no guide last
    with pytest.raises(RuntimeError):
        insert_anchor_block(a, torch.zeros(7 * 24, 96), 7)
    with pytest.raises(RuntimeError):
        insert_anchor_block(_args(), torch.zeros(5 * 24, 96), 7)  # row count does not match anchor_v
