"""Layout check for h3_tp2.ulysses_overlap: with all_to_all_single emulated on CPU for P ranks, the grouped
sequence->heads / heads->sequence exchanges reproduce the stock all_to_all_4D result head for head."""

import unittest
from unittest import mock

import torch

from h3_tp2 import ulysses_overlap as ov


class _Emu:
    """Emulates dist.all_to_all_single over P ranks: rank r's output = concat over source ranks of their r-th chunk."""

    def __init__(self, ws):
        self.ws = ws
        self.inputs = {}
        self.outputs = {}

    def run(self, per_rank_inputs):
        chunks = [t.chunk(self.ws, dim=0) for t in per_rank_inputs]
        return [torch.cat([chunks[src][r] for src in range(self.ws)], dim=0) for r in range(self.ws)]


def _stock_seq_to_heads(x, ws, emu_run):
    bs, shard_seqlen, hc, hs = x[0].shape
    shard_hc = hc // ws
    ins = [t.reshape(bs, shard_seqlen, ws, shard_hc, hs).transpose(0, 2).contiguous() for t in x]
    outs = emu_run(ins)
    return [o.reshape(shard_seqlen * ws, bs, shard_hc, hs).transpose(0, 1).contiguous() for o in outs]


def _stock_heads_to_seq(x, ws, emu_run):
    bs, seqlen, shard_hc, hs = x[0].shape
    shard_seqlen = seqlen // ws
    ins = [
        t.reshape(bs, ws, shard_seqlen, shard_hc, hs).transpose(0, 3).transpose(0, 1).contiguous().reshape(ws, shard_hc, shard_seqlen, bs, hs)
        for t in x
    ]
    outs = emu_run(ins)
    return [o.reshape(shard_hc * ws, shard_seqlen, bs, hs).transpose(0, 2).contiguous().reshape(bs, shard_seqlen, shard_hc * ws, hs) for o in outs]


def _run_grouped(fn, per_rank, ws, emu, recv_shape, res_shape):
    """Call the module function for every rank with all_to_all_single replaced by the emulation (rank-synchronous)."""
    results = [None] * ws
    staged = {}

    def fake_a2a(out, inp, group=None):
        staged[len(staged)] = (out, inp)

    with mock.patch.object(ov.dist, "all_to_all_single", side_effect=fake_a2a), mock.patch.object(ov.dist, "get_world_size", return_value=ws):
        # the function packs its input, "sends", then unpacks the receive buffer; emulate by running it twice:
        # first pass collects the packed inputs, second pass is fed the real receives
        inputs = []
        for r in range(ws):
            staged.clear()
            fn(per_rank[r], ws, None, torch.empty(recv_shape), torch.empty(res_shape))
            inputs.append(staged[0][1].clone())
        outs = emu.run(inputs)
        for r in range(ws):
            staged.clear()

            def fake_fill(out, inp, group=None, _o=outs[r]):
                out.copy_(_o)

            with mock.patch.object(ov.dist, "all_to_all_single", side_effect=fake_fill):
                results[r] = fn(per_rank[r], ws, None, torch.empty(recv_shape), torch.empty(res_shape))
    return results


class OverlapLayoutTest(unittest.TestCase):
    def test_grouped_matches_stock(self):
        torch.manual_seed(0)
        ws, groups, bs, s_loc, heads, d = 2, 2, 1, 6, 8, 4
        emu = _Emu(ws)
        q = [torch.randn(bs, s_loc, heads, d) for _ in range(ws)]
        # stock: all heads at once, per-head "attention" = scale by (global head index + 1), back
        stock_fwd = _stock_seq_to_heads(q, ws, emu.run)
        stock_att = []
        for r in range(ws):
            x = stock_fwd[r].clone()
            for h in range(x.shape[2]):
                x[:, :, h] *= r * (heads // ws) + h + 1
            stock_att.append(x)
        stock_out = _stock_heads_to_seq(stock_att, ws, emu.run)
        # grouped
        hg = heads // groups
        backs = [[] for _ in range(ws)]
        for g in range(groups):
            sl = slice(g * hg, (g + 1) * hg)
            hl = hg // ws
            fwd = _run_grouped(ov._a2a_seq_to_heads, [t[:, :, sl] for t in q], ws, emu, (ws, s_loc, bs, hl, d), (bs, ws * s_loc, hl, d))
            att = []
            for r in range(ws):
                x = fwd[r].clone()
                for h in range(x.shape[2]):
                    gh = g * hg + r * (hg // ws) + h  # global head this rank holds in group g
                    x[:, :, h] *= gh + 1
                att.append(x)
            back = _run_grouped(ov._a2a_heads_to_seq, att, ws, emu, (ws, hl, s_loc, bs, d), (bs, s_loc, hg, d))
            for r in range(ws):
                backs[r].append(back[r])
        for r in range(ws):
            out = torch.cat(backs[r], dim=2)
            self.assertEqual(out.shape, stock_out[r].shape)
            self.assertTrue(torch.equal(out, stock_out[r]), f"rank {r} layout differs")


if __name__ == "__main__":
    unittest.main()
