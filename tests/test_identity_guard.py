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
    assert (cfg.grow_sim, cfg.grow_max) == (0.0, 24)
    grow = ig.config_from_env({"H3_TP2_IDENTITY_GUARD": "0.2", "H3_TP2_IDENTITY_GROW": "0.6", "H3_TP2_IDENTITY_GROW_MAX": "5"})
    assert (grow.grow_sim, grow.grow_max) == (0.6, 5)
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
    # the sampled frames of the last KEEP_WINDOWS windows stay on disk, older ones go
    assert sorted(os.listdir(out["dir"])) == [f"frame{i:05d}.png" for i in out["indices"]]
    dirs = [g.score(frames)["dir"] for _ in range(ig.KEEP_WINDOWS)]
    assert not os.path.isdir(out["dir"]) and all(os.path.isdir(d) for d in dirs)


def test_frames_to_uint8_never_rescales_uint8():
    """A float decode goes through the converter; a uint8 one is only re-laid out to (B, F, H, W, 3)."""
    calls = []

    def convert(x):
        calls.append(x.shape)
        return np.zeros((1, 4, 2, 2, 3), dtype=np.uint8)

    f = np.zeros((1, 3, 4, 2, 2), dtype=np.float32)
    assert ig.frames_to_uint8(f, convert).shape == (1, 4, 2, 2, 3) and calls == [(1, 3, 4, 2, 2)]

    class U8:
        dtype, ndim, shape = "torch.uint8", 5, (1, 3, 4, 2, 2)

        def permute(self, *dims):
            return ("permuted", dims)

    assert ig.frames_to_uint8(U8(), convert) == ("permuted", (0, 2, 3, 4, 1)) and len(calls) == 1
    already = np.zeros((1, 4, 2, 2, 3), dtype=np.uint8)
    assert ig.frames_to_uint8(already, convert) is already and len(calls) == 1


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_grow_bank_adds_matched_faces_fifo_and_keeps_the_references(tmp_path):
    bank = tmp_path / "bank.npz"
    np.savez(bank, emb=np.stack([_unit([1, 0, 0]), _unit([0, 1, 0])]), src=np.array(["ref0@x=0.3", "ref0@x=0.7"]))
    cand = tmp_path / "faces.npz"
    np.savez(cand, emb=np.stack([_unit([1, 0.2, 0]), _unit([0.5, 0.5, 0.7]), _unit([0, 1, 0.1])]),
             best=np.array([0.7, 0.5, 0.9], np.float32), src=np.array(["w001@x=0.3", "w001@x=0.5", "w001@x=0.7"]))
    assert ig.grow_bank(str(bank), str(cand), 0.6, 24) == (2, 4, 2)
    data = np.load(bank)
    assert list(data["src"]) == ["ref0@x=0.3", "ref0@x=0.7", "w001@x=0.3", "w001@x=0.7"]
    assert list(data["grown"]) == [False, False, True, True]
    # the cap drops the oldest grown faces only; nothing is added below the threshold
    assert ig.grow_bank(str(bank), str(cand), 0.6, 3) == (2, 5, 3)
    data = np.load(bank)
    assert list(data["src"]) == ["ref0@x=0.3", "ref0@x=0.7", "w001@x=0.7", "w001@x=0.3", "w001@x=0.7"]
    assert ig.grow_bank(str(bank), str(cand), 0.95, 3) == (0, 5, 3)


def test_principal_faces_is_pure():
    pytest.importorskip("torch")
    import face_bank

    bank = np.stack([_unit([1, 0, 0]), _unit([0, 1, 0])])
    found = [[(0.3, 0.08, _unit([1, 0.1, 0])), (0.9, 0.02, _unit([0, 0, 1]))], [(0.7, 0.06, _unit([0, 1, 0]))]]
    embs, best, src, widths, probs = face_bank.principal_faces(bank, found, 0.05, ["a.png", "b.png"])
    assert len(embs) == 2 and src == ["a.png@x=0.30", "b.png@x=0.70"] and widths == [0.08, 0.06] and probs == [1.0, 1.0]
    assert best[0] == pytest.approx(float(bank[0] @ _unit([1, 0.1, 0]))) and best[1] == pytest.approx(1.0)
    # a doubtful detection that matches nobody is not an unknown face (the brass scale); a confident one is
    found = [[(0.3, 0.10, _unit([1, 0, 0]), 0.999), (0.5, 0.15, _unit([0, 0, 1]), 0.93)], [(0.1, 0.07, _unit([0, 0, 1]), 0.999)]]
    out = face_bank.score_frames(bank, found, 0.4, 0.05)
    assert (out["faces"], out["unknown"], out["unknown_frames"]) == (3, 2, 1)
    assert face_bank.score_frames(bank, found, 0.4, 0.05, unknown_min_prob=0.9)["unknown_frames"] == 2


def test_guard_grows_the_bank_after_an_accepted_window(tmp_path):
    fake = tmp_path / "fake_bank.py"
    fake.write_text(
        "import json, sys\n"
        "import numpy as np\n"
        "if sys.argv[1] == 'build':\n"
        "    np.savez(sys.argv[sys.argv.index('--out') + 1], emb=np.eye(2, dtype=np.float32), src=np.array(['a', 'b']))\n"
        "    print(json.dumps({'faces': 2, 'sources': ['a', 'b']}))\n"
        "else:\n"
        "    frames = [a for a in sys.argv[2:] if a.endswith('.png')]\n"
        "    out = {'frames': len(frames), 'faces': 2, 'unknown_frac': 0.0}\n"
        "    if '--save-faces' in sys.argv:\n"
        "        f = sys.argv[sys.argv.index('--save-faces') + 1]\n"
        "        np.savez(f, emb=np.ones((2, 2), np.float32), best=np.array([0.8, 0.3], np.float32), src=np.array(['w@x=0.3', 'w@x=0.7']))\n"
        "        out['faces_file'] = f\n"
        "    print(json.dumps(out))\n"
    )
    pytest.importorskip("PIL")
    from PIL import Image

    ref = tmp_path / "ref.png"
    Image.new("RGB", (64, 32)).save(ref)
    g = ig.Guard(ig.Config(max_unknown=0.2, python=sys.executable, script=str(fake), grow_sim=0.6))
    assert g.set_references([str(ref)]) == 2
    out = g.score(np.zeros((40, 16, 24, 3), dtype=np.uint8))
    assert out["faces_file"].endswith("faces.npz") and ig.verdict(out, 0.2)[0]
    assert g.grow(out) == 1
    assert np.load(g.bank)["emb"].shape == (3, 2)
    # growth off: the scorer is not asked for the faces file and grow() is a no-op
    g2 = ig.Guard(ig.Config(max_unknown=0.2, python=sys.executable, script=str(fake)))
    g2.set_references([str(ref)])
    out2 = g2.score(np.zeros((40, 16, 24, 3), dtype=np.uint8))
    assert "faces_file" not in out2 and g2.grow(out2) == 0


def test_person_check_joins_the_face_check(tmp_path):
    """With person_sim set the guard builds a person bank from the cards, scores people too, takes the worse
    unknown fraction and grows both banks."""
    fake = tmp_path / "fake_bank.py"
    fake.write_text(
        "import json, sys\n"
        "import numpy as np\n"
        "if sys.argv[1] == 'build':\n"
        "    np.savez(sys.argv[sys.argv.index('--out') + 1], emb=np.eye(2, dtype=np.float32), src=np.array(['a', 'b']))\n"
        "    print(json.dumps({'faces': 2, 'sources': ['a', 'b']}))\n"
        "else:\n"
        "    frames = [a for a in sys.argv[2:] if a.endswith('.png')]\n"
        "    out = {'frames': len(frames), 'faces': 2, 'unknown_frac': 0.0}\n"
        "    if '--save-faces' in sys.argv:\n"
        "        f = sys.argv[sys.argv.index('--save-faces') + 1]\n"
        "        np.savez(f, emb=np.ones((1, 2), np.float32), best=np.array([0.8], np.float32), src=np.array(['w@x=0.3']))\n"
        "        out['faces_file'] = f\n"
        "    print(json.dumps(out))\n"
    )
    pfake = tmp_path / "fake_person.py"
    pfake.write_text(
        "import json, sys\n"
        "import numpy as np\n"
        "if sys.argv[1] == 'build':\n"
        "    cards = [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == '--cards']\n"
        "    np.savez(sys.argv[sys.argv.index('--out') + 1], emb=np.eye(len(cards), 3, dtype=np.float32),\n"
        "             src=np.array(cards), who=np.arange(len(cards), dtype=np.int16))\n"
        "    print(json.dumps({'cards': len(cards), 'sources': cards}))\n"
        "else:\n"
        "    frames = [a for a in sys.argv[2:] if a.endswith('.png')]\n"
        "    out = {'frames': len(frames), 'persons': 5, 'unknown_frac': 0.25, 'mean_best': 0.7, 'min_best': 0.4}\n"
        "    if '--save-persons' in sys.argv:\n"
        "        f = sys.argv[sys.argv.index('--save-persons') + 1]\n"
        "        np.savez(f, emb=np.ones((2, 3), np.float32), best=np.array([0.9, 0.5], np.float32), src=np.array(['w@x=0.3', 'w@x=0.7']))\n"
        "        out['persons_file'] = f\n"
        "    print(json.dumps(out))\n"
    )
    pytest.importorskip("PIL")
    from PIL import Image

    scene, card0, card1 = (tmp_path / n for n in ("scene.png", "card0.png", "card1.png"))
    for f in (scene, card0, card1):
        Image.new("RGB", (64, 32)).save(f)
    cfg = ig.Config(max_unknown=0.2, python=sys.executable, script=str(fake), grow_sim=0.6, person_sim=0.6,
                    person_script=str(pfake))
    g = ig.Guard(cfg)
    assert g.set_references([str(scene), str(card0), str(card1)]) == 2
    assert g.pbank_cards == 2 and list(np.load(g.pbank)["src"]) == [str(card0), str(card1)]  # the scene is not a card
    out = g.score(np.zeros((40, 16, 24, 3), dtype=np.uint8))
    assert out["persons"] == 5 and out["face_unknown_frac"] == 0.0 and out["unknown_frac"] == 0.25
    ok, score = ig.verdict(out, 0.2)
    assert not ok and score == pytest.approx(1.25)
    assert ig.verdict({"faces": 0, "persons": 3, "unknown_frac": 0.5}, 0.2) == (False, pytest.approx(2.5))
    # growth: one face at 0.8 and one figure at 0.9 (the 0.5 figure stays out); the person bank keeps its 'who'
    assert g.grow(out) == 2
    pb = np.load(g.pbank)
    assert pb["emb"].shape == (3, 3) and list(pb["who"]) == [0, 1, -1] and list(pb["grown"]) == [False, False, True]
    assert ig.config_from_env({"H3_TP2_IDENTITY_GUARD": "0.2", "H3_TP2_IDENTITY_PERSON_SIM": "0.55"}).person_sim == 0.55
