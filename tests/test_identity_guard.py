"""CPU tests of the identity guard's pure parts (h3_tp2/identity_guard.py) and of face_bank's scoring."""

import json
import os
import sys

import numpy as np
import pytest

from h3_tp2 import identity_guard as ig

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))


def test_config_parsing():
    assert ig.config_from_env({}) is None
    assert ig.config_from_env({"H3_TP2_IDENTITY_GUARD": "0"}) is None
    cfg = ig.config_from_env({"H3_TP2_IDENTITY_GUARD": "0.25", "H3_TP2_IDENTITY_MIN_SIM": "0.45", "H3_TP2_IDENTITY_FRAMES": "6",
                              "H3_TP2_IDENTITY_PY": "/x/python"})
    assert (cfg.max_unknown, cfg.min_sim, cfg.frames, cfg.python, cfg.min_face) == (0.25, 0.45, 6, "/x/python", 0.05)
    assert ig.config_from_env({"H3_TP2_IDENTITY_GUARD": "0.2"}).min_sim == 0.4
    assert cfg.script.endswith(os.path.join("scripts", "face_bank.py"))
    with pytest.raises(ValueError):
        ig.config_from_env({"H3_TP2_IDENTITY_GUARD": "1.5"})


def test_sample_indices_spread_over_the_window_and_include_the_last_frame():
    assert ig.sample_indices(277, 8, skip_first=22) == [22, 58, 95, 131, 167, 203, 240, 276]
    assert ig.sample_indices(277, 1) == [276]
    assert ig.sample_indices(5, 8) == [0, 1, 2, 3, 4]
    assert ig.sample_indices(0, 8) == []
    assert ig.sample_indices(10, 3, skip_first=50) == [9]


def test_verdict():
    assert ig.verdict({"faces": 0}, 0.2) == (True, 0.0)
    ok, score = ig.verdict({"faces": 10, "unknown_frac": 0.1}, 0.2)
    assert ok and score == pytest.approx(0.5)
    ok, score = ig.verdict({"faces": 10, "unknown_frac": 0.4}, 0.2)
    assert not ok and score == pytest.approx(2.0)


def test_face_bank_scoring_is_pure():
    pytest.importorskip("torch")
    import face_bank

    e0 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    e1 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    stranger = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    bank = np.stack([e0, e1])
    frames = [[(0.3, 0.2, e0), (0.7, 0.2, e1)], [(0.3, 0.2, e0), (0.7, 0.2, stranger)], [(0.5, 0.01, stranger)],
              [(0.3, 0.2, e0), (0.7, 0.2, stranger), (0.5, 0.2, stranger)]]
    out = face_bank.score_frames(bank, frames, min_sim=0.5, min_face=0.04)
    assert out["frames"] == 4 and out["faces"] == 7  # the 1 % wide face is ignored
    assert out["known"] == 4 and out["unknown"] == 3
    assert out["unknown_frames"] == 2 and out["unknown_frac"] == pytest.approx(0.5)  # frames, not faces
    assert out["bank_best"] == [1.0, 1.0]
    assert json.dumps(out)


def test_guard_scores_through_the_subprocess(tmp_path, monkeypatch):
    """A fake face_bank script: the guard writes the sampled frames, calls it and parses its JSON."""
    fake = tmp_path / "fake_bank.py"
    fake.write_text(
        "import json, sys\n"
        "if sys.argv[1] == 'build':\n"
        "    print(json.dumps({'faces': 2, 'sources': ['a@x=0.3', 'b@x=0.7']}))\n"
        "else:\n"
        "    frames = [a for a in sys.argv[2:] if a.endswith('.png')]\n"
        "    print(json.dumps({'frames': len(frames), 'faces': len(frames), 'unknown_frac': 0.5}))\n"
    )
    cfg = ig.Config(max_unknown=0.2, python=sys.executable, script=str(fake))
    g = ig.Guard(cfg)
    ref = tmp_path / "ref.png"
    pytest.importorskip("PIL")
    from PIL import Image

    Image.new("RGB", (64, 32)).save(ref)
    assert g.set_references([str(ref)]) == 2
    frames = np.zeros((40, 16, 24, 3), dtype=np.uint8)
    out = g.score(frames, skip_first=4)
    assert out["frames"] == 8 and out["indices"] == ig.sample_indices(40, 8, 4)
    assert ig.verdict(out, cfg.max_unknown) == (False, pytest.approx(2.5))
