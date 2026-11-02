"""CPU checks for h3_tp2.step_cache (speed plan item 7) against upstream's denoise loop.

The stock reference is ``tests/fake_h3_upstream.py``: upstream's loop, branch, scheduler and latent-edit code copied
verbatim (its fingerprints are checked against ``step_cache.EXPECTED``), with a small fake DiT that has upstream's
``forward`` / ``_embed`` / refiner call structure.  Every comparison is ``torch.equal`` (bitwise), over 3 denoising
steps.  What the CPU cannot show: CUDA kernels on the real DiT (run ``H3_TP2_STEP_CACHE=verify`` once on the GPU and
compare the latents digest with a stock run of the same boot)."""

import logging
import types
import unittest
from unittest import mock

import torch

from h3_tp2 import step_cache as sc
from tests import fake_h3_upstream as fake

LOG = "vllm_omni.h3_tp2.step_cache"
N_TEXT, N_IMG, N_AUDIO, PAD = 5, 24, 10, 3
SEQ = N_TEXT + N_IMG + N_AUDIO + PAD  # 42, even: rank spans of a world of 2


def make_case(*, n_cond=0, n_audio_ref=0, edit=False, locked=False, sigmas_v=None, sigmas_a=None, seed=1):
    g = torch.Generator().manual_seed(seed)
    text_pos = torch.arange(0, N_TEXT)
    img_pos = torch.arange(N_TEXT, N_TEXT + N_IMG)
    audio_pos = torch.arange(N_TEXT + N_IMG, N_TEXT + N_IMG + N_AUDIO)
    update = torch.ones(N_IMG, dtype=torch.bool)
    update[:n_cond] = False
    packed = {
        "seq_len": SEQ,
        "img_pos": img_pos,
        "audio_pos": audio_pos,
        "update_mask": update,
        "text_pos": text_pos,
        "cu_seqlens": torch.tensor([0, SEQ - PAD, SEQ], dtype=torch.int32),
        "img_position_ids": torch.randint(0, 9, (SEQ, 3), generator=g),
    }
    if n_audio_ref:
        audio_update = torch.ones(N_AUDIO, dtype=torch.bool)
        audio_update[-n_audio_ref:] = False
        packed["audio_update_mask"] = audio_update
    n_target = N_IMG - n_cond
    case = {
        "packed": packed,
        "text": torch.randn(N_TEXT, 8, generator=g),
        "tags": torch.randint(0, 3, (SEQ,), generator=g),
        "video": torch.randn(N_IMG, 96, generator=g),
        "audio": torch.randn(N_AUDIO, 32, generator=g),
        "cond": torch.randn(n_cond, 96, generator=g) if n_cond else None,
        "audio_ref": torch.randn(n_audio_ref, 32, generator=g) if n_audio_ref else None,
        "sigmas_v": sigmas_v or [1.0, 0.6, 0.3, 0.0],
        "sigmas_a": sigmas_a or [1.0, 0.5, 0.2, 0.0],
        "locked": torch.randn(N_AUDIO - n_audio_ref, 32, generator=g) if locked else None,
        "edit": None,
    }
    if edit:
        case["edit"] = (
            torch.randn(n_target, 96, generator=g),
            torch.randn(n_target, 96, generator=g),
            torch.rand(n_target, generator=g),
            torch.rand(n_target, generator=g),
        )
    return case


def make_branch(mod, case):
    branch = mod.MiniMaxH3DenoiseBranch(
        packed={k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in case["packed"].items()},
        text_embeddings=case["text"].clone(),
        token_tags=case["tags"].clone(),
        device=torch.device("cpu"),
    )
    if case["locked"] is not None:
        branch.locked_audio_rows = case["locked"].clone()
    return branch


def run(mod, case, *, backend="CUDNN_ATTN", world=1, rank=0, branch_hook=None):
    model = mod.MiniMaxH3DiTModel(backend=backend, seed=0)
    model.world, model.rank = world, rank
    pipe = mod.MiniMaxH3Pipeline(model)
    branch = make_branch(mod, case)
    if branch_hook is not None:
        branch_hook(branch)
    video_edit = None
    if case["edit"] is not None:
        video_edit = mod.MiniMaxH3LatentEdit.from_rows(*(t.clone() for t in case["edit"]))
    mod.CANCEL_CALLS.clear()
    v, a = pipe.diffuse(
        branch=branch,
        initial_video_rows=case["video"].clone(),
        initial_audio_rows=case["audio"].clone(),
        keyframe_cond_rows=None if case["cond"] is None else case["cond"].clone(),
        audio_ref_rows=None if case["audio_ref"] is None else case["audio_ref"].clone(),
        video_edit=video_edit,
        sigmas_video=case["sigmas_v"],
        sigmas_audio=case["sigmas_a"],
    )
    return types.SimpleNamespace(v=v, a=a, model=model, outputs=model.outputs, cancels=list(mod.CANCEL_CALLS),
                                 stats=dict(sc._STATE.stats))


def patched_module(name, *, verify=False):
    mod = fake.load(name)
    done = sc.install(mod, mod, mod, verify=verify, check_source=False)
    assert done == ["rope", "refiner", "index"], done
    return mod


class Base(unittest.TestCase):
    def setUp(self):
        sc._STATE.warned.clear()
        sc._STATE.verify = False
        sc._STATE.stats = {}

    def assertSame(self, ref, got):
        self.assertTrue(torch.equal(ref.v, got.v), "video rows differ")
        self.assertTrue(torch.equal(ref.a, got.a), "audio rows differ")
        self.assertEqual(len(ref.outputs), len(got.outputs))
        for (rv, ra), (gv, ga) in zip(ref.outputs, got.outputs):
            self.assertTrue(torch.equal(rv, gv) and torch.equal(ra, ga), "a step's DiT output differs")
        self.assertEqual(ref.cancels, got.cancels)  # (d): the same cancellation checks, same order


class TestFingerprints(Base):
    def test_verbatim_copies_match_reviewed_upstream(self):
        self.assertEqual(sc.fingerprint(fake.minimax_h3_denoise_loop), sc.EXPECTED["denoise_loop"])
        self.assertEqual(sc.fingerprint(fake.MiniMaxH3DenoiseBranch.forward_kwargs), sc.EXPECTED["forward_kwargs"])
        self.assertEqual(sc.fingerprint(fake.MiniMaxH3DenoiseBranch.fill_timesteps), sc.EXPECTED["fill_timesteps"])

    def test_source_check_skips_changed_functions(self):
        mod = fake.load("fake_fp")
        with self.assertLogs(LOG, level="WARNING") as logs:
            done = sc.install(mod, mod, mod)
        self.assertEqual(done, ["index"])  # the fake DiT is not upstream's forward/_embed
        self.assertTrue(any("dit_forward changed" in m for m in logs.output))
        self.assertTrue(any("dit_embed changed" in m for m in logs.output))

    def test_fingerprint_ignores_blank_and_comment_lines_only(self):
        a = "def f(x):\n    # c\n\n    return x + 1\n"
        self.assertEqual(sc.fingerprint_source(a), sc.fingerprint_source("def f(x):\n        return x + 1"))
        self.assertNotEqual(sc.fingerprint_source(a), sc.fingerprint_source("def f(x):\n    return x + 2\n"))


class TestEndToEnd(Base):
    CASES = {
        "t2va": {},
        "fl2va_cond_and_audio_ref": {"n_cond": 6, "n_audio_ref": 3},
        "locked_audio": {"n_audio_ref": 2, "locked": True},
        "cond_timestep_ties": {"n_cond": 4, "sigmas_v": [0.002, 0.001, 0.0005, 0.0], "sigmas_a": [0.5, 0.25, 0.1, 0.0]},
        "video_edit": {"n_cond": 4, "edit": True},
    }

    def test_bit_identical_over_three_steps(self):
        for name, kw in self.CASES.items():
            with self.subTest(case=name):
                case = make_case(**kw)
                ref = run(fake.load(f"stock_{name}"), case)
                got = run(patched_module(f"patched_{name}"), case)
                self.assertSame(ref, got)
                self.assertEqual(len(got.outputs), 3)
                # stock: 3 refiner runs + 3 inline RoPE builds; cached: one of each
                self.assertEqual(ref.model.token_refiner.calls, 3)
                self.assertEqual(got.model.token_refiner.calls, 1)
                self.assertEqual(ref.model.inline_rope_builds, 3)
                self.assertEqual(got.model.inline_rope_builds, 0)
                self.assertEqual(got.stats.get("rope_built"), 1)
                self.assertEqual(got.stats.get("rope_reused"), 2)
                self.assertEqual(got.stats.get("refiner_reused"), 2)
                self.assertEqual(got.stats.get("index_loop"), 1)
                self.assertNotIn("idx_put_stock", got.stats)
                if kw.get("edit"):
                    self.assertEqual(got.stats.get("unique_stock"), 3)  # per-row target timesteps: torch.unique
                else:
                    self.assertEqual(got.stats.get("unique_host"), 3)

    def test_sequence_parallel_rank_span(self):
        case = make_case(n_cond=6, n_audio_ref=3)
        for rank in (0, 1):
            with self.subTest(rank=rank):
                ref = run(fake.load(f"stock_sp{rank}"), case, world=2, rank=rank)
                got = run(patched_module(f"patched_sp{rank}"), case, world=2, rank=rank)
                self.assertSame(ref, got)

    def test_verify_mode(self):
        case = make_case(n_cond=6, n_audio_ref=3)
        ref = run(fake.load("stock_verify"), case)
        got = run(patched_module("patched_verify", verify=True), case)
        self.assertSame(ref, got)
        self.assertFalse([k for k in got.stats if "mismatch" in k])
        self.assertEqual(got.model.token_refiner.calls, 3)  # verify recomputes on every hit

    def test_new_request_invalidates(self):
        mod = patched_module("patched_two_requests")
        first = run(mod, make_case(seed=1))
        second = run(mod, make_case(seed=2))  # another prompt / layout tensors: nothing carried over
        self.assertEqual(second.stats.get("rope_built"), 1)
        self.assertEqual(second.stats.get("refiner_run"), 1)
        self.assertSame(run(fake.load("stock_second"), make_case(seed=2)), second)
        self.assertFalse(torch.equal(first.v, second.v))
        self.assertEqual(sc._STATE.rope, {})
        self.assertEqual(sc._STATE.refiner, {})  # cleared when diffuse returns


class TestRefinerGuards(Base):
    def test_step_gated_backend_is_refused(self):
        case = make_case()
        ref = run(fake.load("stock_rain"), case, backend="RAINFUSION_ATTN")
        with self.assertLogs(LOG, level="WARNING") as logs:
            got = run(patched_module("patched_rain"), case, backend="RAINFUSION_ATTN")
        self.assertSame(ref, got)
        self.assertEqual(got.model.token_refiner.calls, 3)
        self.assertTrue(any("RAINFUSION_ATTN" in m for m in logs.output))

    def test_hooks_are_refused(self):
        mod = patched_module("patched_hooks")
        model = mod.MiniMaxH3DiTModel()
        model.token_refiner.blocks[0].register_forward_pre_hook(lambda m, a: None)
        ok, why = sc._refiner_step_independent(model)
        self.assertFalse(ok)
        self.assertIn("hooks", why)

    def test_quantised_kv_is_refused(self):
        model = fake.MiniMaxH3DiTModel()
        model.token_refiner.blocks[1].attn.attention._kv_cache_dtype = "fp8"
        self.assertFalse(sc._refiner_step_independent(model)[0])


class _ScopePipe:
    def diffuse(self, fn):
        return fn()


class TestChangedInputsWithinRequest(Base):
    """Direct forward calls inside one diffuse scope: changed inputs must miss the cache."""

    def setUp(self):
        super().setUp()
        self.stock = fake.load("stock_direct")
        self.mod = patched_module("patched_direct")
        sc._install_scope(types.SimpleNamespace(MiniMaxH3Pipeline=_ScopePipe))
        self.case = make_case(n_cond=6, n_audio_ref=3)

    def kwargs(self, mod):
        branch = make_branch(mod, self.case)
        return branch.forward_kwargs(
            video_rows=self.case["video"].clone(), audio_rows=self.case["audio"].clone(), t_video=0.3, t_audio=0.4,
            imgvid_cond_timestep=0.999, audio_ref_cond_timestep=1.0,
        )

    def check(self, ref_model, model, ref_kw, kw):
        with torch.inference_mode():
            rv, ra = ref_model(**ref_kw)
            gv, ga = model(**kw)
        self.assertTrue(torch.equal(rv, gv) and torch.equal(ra, ga))

    def test_changes(self):
        ref_model, model = self.stock.MiniMaxH3DiTModel(), self.mod.MiniMaxH3DiTModel()
        ref_kw, kw = self.kwargs(self.stock), self.kwargs(self.mod)

        def body():
            for _ in range(3):
                self.check(ref_model, model, ref_kw, kw)
            self.assertEqual((sc._STATE.stats["rope_built"], sc._STATE.stats["refiner_run"]), (1, 1))
            # a new img_position_ids tensor with other values
            ids = kw["img_position_ids"] + 1
            ref_kw["img_position_ids"], kw["img_position_ids"] = ids.clone(), ids
            self.check(ref_model, model, ref_kw, kw)
            self.assertEqual(sc._STATE.stats["rope_built"], 2)
            # the same tensor changed in place (version counter / CPU sample)
            ref_kw["img_position_ids"].add_(1)
            kw["img_position_ids"].add_(1)
            self.check(ref_model, model, ref_kw, kw)
            self.assertEqual(sc._STATE.stats["rope_built"], 3)
            # another sequence-parallel span
            ref_model.world = model.world = 2
            ref_model.rank = model.rank = 1
            self.check(ref_model, model, ref_kw, kw)
            self.assertEqual(sc._STATE.stats["rope_built"], 4)
            # new prompt rows, then the same rows changed in place, then another refiner max_seqlen
            text = kw["prompt_embeds"] * 2
            ref_kw["prompt_embeds"], kw["prompt_embeds"] = text.clone(), text
            self.check(ref_model, model, ref_kw, kw)
            ref_kw["prompt_embeds"].mul_(-1)
            kw["prompt_embeds"].mul_(-1)
            self.check(ref_model, model, ref_kw, kw)
            for k in (ref_kw, kw):
                k["refiner_packed_seq_params"] = dict(k["refiner_packed_seq_params"], max_seqlen_q=4)
            self.check(ref_model, model, ref_kw, kw)
            self.assertEqual(sc._STATE.stats["refiner_run"], 4)
            self.check(ref_model, model, ref_kw, kw)  # unchanged again: a hit
            self.assertEqual(sc._STATE.stats["refiner_run"], 4)

        _ScopePipe().diffuse(body)

    def test_outside_diffuse_is_stock(self):
        model = self.mod.MiniMaxH3DiTModel()
        kw = self.kwargs(self.mod)
        with torch.inference_mode():
            for _ in range(2):
                model(**kw)
        self.assertEqual(model.token_refiner.calls, 2)
        self.assertEqual(model.inline_rope_builds, 2)


class TestIndexLoop(Base):
    def test_unique_from_groups_matches_torch_unique(self):
        g = torch.Generator().manual_seed(3)
        for trial in range(200):
            groups = torch.randint(0, 5, (64,), generator=g)
            if trial % 3 == 0:
                groups[groups == 2] = 3  # an empty group
            idx = types.SimpleNamespace(groups=groups, present=sorted(set(groups.tolist())))
            choices = [0.0, 0.25, 0.999, 0.9995, 1.0, 1.0 + 1e-9]  # the last one rounds to 1.0f: a float32 tie
            values = [choices[int(i)] for i in torch.randint(0, len(choices), (5,), generator=g)]
            ts = torch.empty(64, dtype=torch.float32)
            for gid in range(5):
                ts[groups == gid] = values[gid]
            ref_u, ref_i = torch.unique(ts, sorted=True, return_inverse=True)
            u, i = sc._unique_from_groups(idx, values, torch.device("cpu"))
            self.assertTrue(torch.equal(ref_u, u) and torch.equal(ref_i, i), (values, idx.present))
            self.assertEqual((u.dtype, i.dtype), (ref_u.dtype, ref_i.dtype))

    def test_setup_failure_falls_back_to_stock_loop(self):
        case = make_case(n_cond=6, n_audio_ref=3)
        ref = run(fake.load("stock_boom"), case)
        mod = patched_module("patched_boom")
        with mock.patch.object(sc, "_Idx", side_effect=RuntimeError("boom")):
            with self.assertLogs(LOG, level="WARNING") as logs:
                got = run(mod, case)
                again = run(mod, case)
        self.assertSame(ref, got)
        self.assertSame(ref, again)
        self.assertNotIn("index_loop", got.stats)
        self.assertEqual(sum("stock loop" in m for m in logs.output), 1)  # logged once

    def test_layout_guard(self):
        good = make_branch(fake, make_case(n_cond=2))
        self.assertIsNotNone(sc._make_idx(good))
        bad = types.SimpleNamespace(**{k: getattr(good, k) for k in
                                       ("update_mask_dev", "audio_update_mask_dev", "img_pos_dev", "audio_pos_dev",
                                        "seq_len")})
        bad.img_pos_dev = bad.img_pos_dev.to(torch.int32)
        with self.assertLogs(LOG, level="WARNING"):
            self.assertIsNone(sc._make_idx(bad))
        bad.img_pos_dev = good.img_pos_dev[:-1]  # mask and positions of different length
        self.assertIsNone(sc._make_idx(bad))

    def test_put_falls_back_for_broadcast_values(self):
        dst = torch.zeros(6, 4)
        mask = torch.tensor([True, False, True, False, False, True])
        index = torch.nonzero(mask).view(-1)
        ref = dst.clone()
        ref[mask] = torch.tensor([1.0, 2.0, 3.0, 4.0])  # one row broadcast to the selected rows
        sc._put(dst, index, lambda: mask, torch.tensor([1.0, 2.0, 3.0, 4.0]))
        self.assertTrue(torch.equal(ref, dst))
        self.assertEqual(sc._STATE.stats.get("idx_put_stock"), 1)

    def test_index_state_is_removed_after_the_loop(self):
        mod = patched_module("patched_cleanup")
        seen = []
        case = make_case()
        run(mod, case, branch_hook=seen.append)
        self.assertNotIn("_h3_tp2_idx", seen[0].__dict__)
        self.assertIsNone(sc._branch_idx(seen[0]))


class TestInstall(Base):
    def test_idempotent(self):
        mod = patched_module("patched_twice")
        loop = mod.minimax_h3_denoise_loop
        fwd = mod.MiniMaxH3DiTModel.forward
        sc.install(mod, mod, mod, check_source=False)
        self.assertIs(mod.minimax_h3_denoise_loop, loop)
        self.assertIs(mod.MiniMaxH3DiTModel.forward, fwd)
        case = make_case()
        self.assertSame(run(fake.load("stock_twice"), case), run(mod, case))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
