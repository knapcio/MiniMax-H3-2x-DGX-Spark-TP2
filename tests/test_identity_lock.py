"""CPU checks of the opt-in identity lock (h3_tp2/identity_lock.py) on fake latents and a fake diffuse."""
import types

import pytest

torch = pytest.importorskip("torch")
from h3_tp2 import cont_match, identity_lock as il  # noqa: E402

C, H, W = 4, 8, 12            # latent channels / cells (fake canvas)
RPF = (H // 2) * (W // 2)     # DiT rows per latent frame


def video_t(frames):
    return (frames - 5) // 17 * 5 + 2


def window_args(index, frames=277, overlap=22):
    """Args the plugin's edited continuation loop passes for window ``index`` (0-based) on an FL2VA request."""
    lt, at = video_t(frames), round(frames * 5 / 3)
    a = {"num_frames": frames, "latent_t": lt, "latent_h": H, "latent_w": W, "audio_t": at, "seed": 3101 + index,
         "temporal_offset": 0.0, "pad_seq_len": None, "locked_audio_rows": None}
    if index == 0:
        a.update(task="fl2va", keyframe_frame_indices=[0], visual_condition=torch.full((RPF, 96), 7.0),
                 visual_condition_shape=(1, H, W), visual_condition_shapes=None)
        return a
    ov, ova = video_t(overlap), round(overlap * 5 / 3)
    clean = torch.zeros(lt * RPF, 96); clean[:ov * RPF] = 1.0
    mask = torch.ones(lt * RPF); mask[:ov * RPF] = 0.0
    ca = torch.zeros(2 * at, 32); ma = torch.ones(2 * at)
    for c in range(2):
        ca[c * at:c * at + ova] = 2.0 + c; ma[c * at:c * at + ova] = 0.0
    a.update(task="t2va", keyframe_frame_indices=None, visual_condition=None, visual_condition_shape=None,
             visual_condition_shapes=None, ref_blocks=None, audio_condition=None, audio_condition_lengths=None,
             video_edit_clean_rows=clean, video_edit_mask_rows=mask, video_edit_restore_mask_rows=mask.clone(),
             audio_edit_clean_rows=ca, audio_edit_mask_rows=ma, audio_edit_restore_mask_rows=ma.clone())
    return a


def fake_diffuse(calls):
    def diffuse(**a):
        calls.append(a)
        g = torch.Generator().manual_seed(int(a["seed"]))
        return (torch.randn(1, C, a["latent_t"], H, W, generator=g), torch.randn(2, 32, a["audio_t"], generator=g))
    return diffuse


def cfg(**kw):
    base = dict(modes=frozenset(), extend=0, boxes=((0.25, 0.25, 0.5, 0.75),), strength=0.3, radius=1, feather=1,
                ramp=0.25, lowpass=True)
    base.update(kw)
    return il.Config(**base)


# ------------------------------------------------------------------------------------------------- configuration

def test_env_unset_is_inactive_and_install_is_a_noop():
    mod = types.SimpleNamespace(wrap_diffuse=cont_match.wrap_diffuse)
    assert not il.Config.from_env({}).active
    assert il.install(mod, environ={}) is False and mod.wrap_diffuse is cont_match.wrap_diffuse


def test_env_parsing_and_validation():
    c = il.Config.from_env({"H3_TP2_ID_LOCK": "end,stats", "H3_TP2_ID_END_EXTEND": "68",
                            "H3_TP2_ID_LOCK_BOXES": "0.306,0.151,0.400,0.376;0.612,0.232,0.703,0.440"})
    assert c.modes == {"end", "stats"} and c.extend == 68 and len(c.boxes) == 2
    for bad in ({"H3_TP2_ID_LOCK": "face"}, {"H3_TP2_ID_LOCK": "end", "H3_TP2_ID_END_EXTEND": "50"},
                {"H3_TP2_ID_LOCK": "blend"}, {"H3_TP2_ID_LOCK": "stats", "H3_TP2_ID_LOCK_BOXES": "0.5,0.1,0.4,0.3"}):
        with pytest.raises(ValueError):
            il.Config.from_env(bad)


# -------------------------------------------------------------------------------------------------- end anchor

def test_end_anchor_reuses_window_one_keyframe_as_last_keyframe():
    calls = []
    d = il.wrap_inner(fake_diffuse(calls), cfg(modes=frozenset({"end"})))
    d(**window_args(0)); d(**window_args(1))
    first, second = calls
    assert first["task"] == "fl2va" and first["keyframe_frame_indices"] == [0]
    assert second["task"] == "fl2va" and second["keyframe_frame_indices"] == [-1]
    assert second["visual_condition"] is first["visual_condition"]
    assert second["visual_condition_shapes"] == [(1, H, W)] and second["ref_blocks"] is None
    assert second["num_frames"] == 277 and torch.equal(second["video_edit_mask_rows"], window_args(1)["video_edit_mask_rows"])


def test_end_anchor_extension_pads_the_edit_rows_and_crops_the_result():
    calls = []
    d = il.wrap_inner(fake_diffuse(calls), cfg(modes=frozenset({"end"}), extend=68))
    d(**window_args(0))
    base = window_args(1)
    video, audio = d(**base)
    sent = calls[1]
    assert sent["num_frames"] == 345 and sent["latent_t"] == video_t(345) == 102
    assert sent["audio_t"] == base["audio_t"] + 113
    lt, at, at2 = base["latent_t"], base["audio_t"], sent["audio_t"]
    assert torch.equal(sent["video_edit_clean_rows"][:lt * RPF], base["video_edit_clean_rows"])
    assert (sent["video_edit_mask_rows"][lt * RPF:] == 1).all() and (sent["video_edit_clean_rows"][lt * RPF:] == 0).all()
    assert sent["video_edit_mask_rows"].shape[0] == 102 * RPF
    for c in range(2):   # channel-major audio rows keep their pinned prefix at the start of each channel
        assert torch.equal(sent["audio_edit_clean_rows"][c * at2:c * at2 + at], base["audio_edit_clean_rows"][c * at:(c + 1) * at])
        assert (sent["audio_edit_mask_rows"][c * at2 + at:(c + 1) * at2] == 1).all()
    assert video.shape[2] == lt and audio.shape[-1] == at
    full_v, full_a = fake_diffuse([])(**sent)
    assert torch.equal(video, full_v[:, :, :lt]) and torch.equal(audio, full_a[..., :at])


def test_extension_is_skipped_with_a_locked_soundtrack():
    calls = []
    d = il.wrap_inner(fake_diffuse(calls), cfg(modes=frozenset({"end"}), extend=68))
    d(**window_args(0))
    a = window_args(1); a["locked_audio_rows"] = torch.zeros(2 * a["audio_t"], 32)
    d(**a)
    assert calls[1]["num_frames"] == 277 and calls[1]["keyframe_frame_indices"] == [-1]


def test_toff0_only_touches_ref2va_windows():
    calls = []
    d = il.wrap_inner(fake_diffuse(calls), cfg(modes=frozenset({"toff0"})))
    a = window_args(0); a.update(task="ref2va", temporal_offset=425.0)
    b = window_args(1); b["temporal_offset"] = 425.0
    d(**a); d(**b)
    assert calls[0]["temporal_offset"] == 0.0 and calls[1]["temporal_offset"] == 425.0 and calls[1]["task"] == "t2va"


def test_end_anchor_needs_a_first_frame_request():
    calls = []
    d = il.wrap_inner(fake_diffuse(calls), cfg(modes=frozenset({"end"})))
    first = window_args(0); first["keyframe_frame_indices"] = [-1]
    d(**first); d(**window_args(1))
    assert calls[1]["task"] == "t2va"


# --------------------------------------------------------------------------------------------- latent face lock

def test_feather_mask_and_shift_helpers():
    m = il.feather_mask((2, 4, 3, 6), H, W, 1)
    assert (m[2:4, 3:6] == 1).all() and m[0, 0] == 0 and 0 < m[1, 4] < 1
    x = torch.arange(H * W, dtype=torch.float32).reshape(H, W)
    s = il._shift_map(x, 1, -2)
    assert s[1, 0] == x[0, 2] and (s[0] == 0).all() and (s[:, -2:] == 0).all()
    lp = il._lowpass(torch.arange(16.).reshape(1, 4, 4))
    assert torch.equal(lp[0, :2, :2], torch.full((2, 2), 2.5))


def test_alignment_finds_a_moved_face():
    g = torch.Generator().manual_seed(0)
    ref = torch.randn(C, H, W, generator=g)
    moved = il._shift_map(ref, 1, -1) + 0.01 * torch.randn(C, H, W, generator=g)
    cells = il.face_cells((0.25, 0.25, 0.5, 0.75), H, W)
    (dy, dx), score = il.align_shift(moved, ref, cells, 2)
    assert (dy, dx) == (1, -1) and score > 0.99


def test_block_ramp_is_constant_per_block_and_reaches_one():
    w = il.block_ramp(75, 7, 0.25)
    assert w.shape[0] == 75 and w[-1] == 1.0
    blocks = (torch.arange(7, 82) // 5)
    for b in blocks.unique():
        assert w[blocks == b].unique().numel() == 1
    assert w[0] < w[3]   # token 7 (block 1) < token 10 (block 2)


def _two_windows(c, perturb=0.5):
    g = torch.Generator().manual_seed(1)
    first = torch.randn(1, C, 82, H, W, generator=g)
    later = first.clone() + perturb * torch.randn(1, C, 82, H, W, generator=g)
    return first, later


@pytest.mark.parametrize("mode", ["stats", "blend"])
def test_face_lock_touches_only_free_tokens_inside_the_feathered_box(mode):
    c = cfg(modes=frozenset({mode}), strength=0.5, ramp=0.0)
    first, later = _two_windows(c)
    out = il.lock_faces(il.reference_tokens(first), later, 7, c, index=2, apply={mode})
    assert out.shape == later.shape and out.dtype == later.dtype
    assert torch.equal(out[:, :, :7], later[:, :, :7])                     # pinned prefix untouched
    reach = il.feather_mask(il.face_cells(c.boxes[0], H, W), H, W, c.feather + c.radius)
    assert torch.equal(out[..., reach == 0], later[..., reach == 0])       # far from the face untouched
    assert not torch.equal(out, later)


def test_blend_moves_faces_toward_the_reference_and_strength_zero_is_identity():
    first, later = _two_windows(None)
    ref = il.reference_tokens(first)
    cells = il.face_cells((0.25, 0.25, 0.5, 0.75), H, W)
    r0, r1, c0, c1 = cells
    def err(v):
        return sum(float((v[0, :, j, r0:r1, c0:c1] - ref["A" if j % 5 == 0 else "B"][:, r0:r1, c0:c1]).norm()) for j in range(7, 82))
    c0_ = cfg(modes=frozenset({"blend"}), strength=0.0, radius=0)
    assert torch.equal(il.lock_faces(ref, later, 7, c0_, index=2, apply={"blend"}), later)
    c1_ = cfg(modes=frozenset({"blend"}), strength=0.5, radius=0, lowpass=False, ramp=0.0)
    assert err(il.lock_faces(ref, later, 7, c1_, index=2, apply={"blend"})) < 0.6 * err(later)


def test_stats_matches_box_statistics_at_full_strength():
    first, later = _two_windows(None)
    later = later * 1.5 + 0.7
    ref = il.reference_tokens(first)
    c = cfg(modes=frozenset({"stats"}), strength=1.0, radius=0, ramp=0.0)
    out = il.lock_faces(ref, later, 7, c, index=2, apply={"stats"})
    r0, r1, c0, c1 = il.face_cells(c.boxes[0], H, W)
    b_tok = [j for j in range(7, 82) if j % 5]
    got = out[0][:, b_tok, r0:r1, c0:c1].mean(dim=(1, 2, 3))
    want = ref["B"][:, r0:r1, c0:c1].mean(dim=(1, 2))
    before = later[0][:, b_tok, r0:r1, c0:c1].mean(dim=(1, 2, 3))
    assert (got - want).abs().max() < 0.05 < (before - want).abs().max()


# ------------------------------------------------------------------------------------------------- composition

def test_install_composes_around_the_look_lock_wrapper_and_keeps_its_hooks():
    mod = types.SimpleNamespace(wrap_diffuse=cont_match.wrap_diffuse)
    env = {"H3_TP2_ID_LOCK": "end,blend", "H3_TP2_ID_LOCK_BOXES": "0.25,0.25,0.5,0.75", "H3_TP2_ID_END_EXTEND": "68"}
    assert il.install(mod, environ=env) is True and il.install(mod, environ=env) is False
    calls = []
    d = mod.wrap_diffuse(fake_diffuse(calls), apply=False, overlap_v_hint=7, retries=0, lock="first", lock_classes=True,
                         lock_ref="tail")
    assert hasattr(d, "snapshot_state") and hasattr(d, "restore_state")
    v1, _ = d(**window_args(0))
    v2, a2 = d(**window_args(1))
    assert v1.shape[2] == 82 and v2.shape[2] == 82 and a2.shape[-1] == round(277 * 5 / 3)
    assert calls[1]["keyframe_frame_indices"] == [-1] and calls[1]["num_frames"] == 345
