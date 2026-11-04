"""When the aiter backend sends MLA decode to aiter's gfx1250 paged Gluon kernel."""

import sys
import types
import unittest
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention import aiter_backend
from sglang.srt.layers.attention import aiter_mla_gluon as amg
from sglang.srt.mem_cache.kv.router import KVLayerRouter
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_MLA_MODULE = "aiter.ops.triton.attention.mla"
_KV_LORA_RANK = 512
_ROPE_DIM = 64
_PAGE_SIZE = 64
_HEADS = 8
_KERNEL_OUT = 3.0


def _fake_aiter(mla_decode_fwd):
    modules = {
        name: types.ModuleType(name)
        for name in (
            "aiter",
            "aiter.ops",
            "aiter.ops.triton",
            "aiter.ops.triton.attention",
        )
    }
    modules[_MLA_MODULE] = types.ModuleType(_MLA_MODULE)
    modules[_MLA_MODULE].mla_decode_fwd = mla_decode_fwd
    return modules


def _metadata(bs, paged):
    return aiter_backend.ForwardMetadata(
        kv_indptr=None,
        kv_indices=None,
        qo_indptr=torch.arange(bs + 1, dtype=torch.int32),
        kv_last_page_len=None,
        max_q_len=1,
        max_kv_len=100,
        paged_mla_block_tables=(
            torch.arange(bs, dtype=torch.int32)[:, None] + 1 if paged else None
        ),
        paged_mla_seq_lens=torch.full((bs,), 100, dtype=torch.int32),
    )


def _backend(kv_dtype=torch.bfloat16, bs=1, paged=True):
    """An AiterAttnBackend holding only what MLA decode reads."""
    kv_buffer = torch.zeros((1024, 1, _KV_LORA_RANK + _ROPE_DIM), dtype=kv_dtype)
    backend = aiter_backend.AiterAttnBackend.__new__(aiter_backend.AiterAttnBackend)
    backend.use_mla = True
    backend.input_dtype = torch.bfloat16
    backend.page_size = _PAGE_SIZE
    backend.dcp_world_size = 1
    backend.use_mla_dcp_asm = False
    pool = types.SimpleNamespace(start_layer=0, layer_transfer_counter=None)
    view = types.SimpleNamespace(get_key_buffer=lambda view_layer_id: kv_buffer)
    pool.layer_router = KVLayerRouter.from_layer_maps(
        pool, [("full", view, {lid: lid for lid in range(8)})]
    )
    backend.token_to_kv_pool = pool
    backend.device = torch.device("cpu")
    backend.kv_cache_dtype = kv_dtype
    backend.k_scale = backend.v_scale = torch.tensor([1.0])
    backend.use_mla_auto_kv_splits = False
    backend.forward_metadata = _metadata(bs, paged)
    return backend


def _layer(k_scale=None):
    return types.SimpleNamespace(
        layer_id=0,
        qk_head_dim=_KV_LORA_RANK + _ROPE_DIM,
        v_head_dim=_KV_LORA_RANK,
        tp_q_head_num=_HEADS,
        scaling=1.0,
        k_scale=None if k_scale is None else torch.tensor(k_scale),
        v_scale=None,
        logit_cap=0.0,
    )


def _q(layer, n_tokens):
    return torch.zeros((n_tokens, _HEADS * layer.qk_head_dim), dtype=torch.bfloat16)


class _RecordingKernel:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        kwargs["out"].fill_(_KERNEL_OUT)


class TestSupportsPagedMlaGluonDecode(CustomTestCase):
    def _supports(self, gfx1250=True, mla_gluon=True, **overrides):
        kwargs = dict(
            input_dtype=torch.bfloat16,
            page_size=_PAGE_SIZE,
            dcp_world_size=1,
            qk_rope_head_dim=_ROPE_DIM,
        )
        kwargs.update(overrides)
        with mock.patch.object(amg, "is_gfx1250_supported", return_value=gfx1250):
            with envs.SGLANG_AITER_MLA_GLUON.override(mla_gluon):
                return amg.supports_paged_mla_gluon_decode(**kwargs)

    def test_gfx1250_bf16_page_64_runs_the_paged_kernel(self):
        self.assertTrue(self._supports())

    def test_settings_the_kernel_cannot_serve_decline(self):
        """Only page size 64 is validated; fp16 activations would be run as bf16,
        DCP needs aiter's own CP path, a 0 rope dim is a kernel shape error, and
        SGLANG_AITER_MLA_GLUON=0 turns every aiter Gluon MLA decode off."""
        for kwargs in (
            dict(page_size=1),
            dict(page_size=16),
            dict(page_size=32),
            dict(page_size=128),
            dict(dcp_world_size=2),
            dict(input_dtype=torch.float16),
            dict(qk_rope_head_dim=0),
            dict(gfx1250=False),
            dict(mla_gluon=False),
        ):
            with self.subTest(**kwargs):
                self.assertFalse(self._supports(**kwargs))


class TestPagedMlaBlockTables(CustomTestCase):
    def test_page_ids_follow_each_requests_pages(self):
        """A row lists the request's pages in order, one per page_size tokens,
        and is wide enough for the longest sequence."""
        pages = [[5, 2, 9], [1, 7, 3]]
        req_to_token = torch.cat(
            [
                torch.cat([p * _PAGE_SIZE + torch.arange(_PAGE_SIZE) for p in row]).to(
                    torch.int32
                )[None]
                for row in pages
            ]
        )
        for max_kv_len, width in ((64, 1), (65, 2), (128, 2), (129, 3)):
            with self.subTest(max_kv_len=max_kv_len):
                tables = amg.paged_mla_block_tables(
                    req_to_token=req_to_token,
                    req_pool_indices=torch.tensor([1, 0]),
                    max_kv_len=max_kv_len,
                    page_size=_PAGE_SIZE,
                )
                torch.testing.assert_close(
                    tables,
                    torch.tensor([pages[1][:width], pages[0][:width]]).to(torch.int32),
                )


class TestUsesPagedMlaGluonDecode(CustomTestCase):
    def test_only_single_token_decode_runs_the_paged_kernel(self):
        """Verify and draft extend carry several query tokens per sequence; they
        must keep running mla_decode_fwd."""
        backend = aiter_backend.AiterAttnBackend.__new__(aiter_backend.AiterAttnBackend)
        for name, enabled, mode, spec_info, expected in (
            ("decode", True, ForwardMode.DECODE, None, True),
            ("disabled", False, ForwardMode.DECODE, None, False),
            ("spec_decode", True, ForwardMode.DECODE, object(), False),
            ("idle", True, ForwardMode.IDLE, None, False),
            ("extend", True, ForwardMode.EXTEND, None, False),
            ("target_verify", True, ForwardMode.TARGET_VERIFY, object(), False),
            ("draft_extend", True, ForwardMode.DRAFT_EXTEND_V2, object(), False),
        ):
            with self.subTest(name):
                backend.use_paged_mla_gluon_decode = enabled
                self.assertIs(
                    backend._uses_paged_mla_gluon_decode(mode, spec_info), expected
                )


class TestAiterGfx1250MlaDecodeDispatch(CustomTestCase):
    def setUp(self):
        self.kernel = _RecordingKernel()
        self.asm_out = torch.full((1, _HEADS, _KV_LORA_RANK), 7.0)
        for patcher in (
            mock.patch.dict(sys.modules, _fake_aiter(self.kernel)),
            mock.patch.object(
                aiter_backend, "prefer_mla_gluon_decode", lambda **_: False
            ),
            mock.patch.object(
                aiter_backend.AiterAttnBackend,
                "_mla_decode_fwd_with_head_pad",
                lambda *args, **kwargs: self.asm_out,
            ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def _decode(self, backend, layer, bs):
        return backend._forward_mla_decode(_q(layer, bs), layer, None, None)

    def test_paged_metadata_returns_the_paged_kernel_output(self):
        backend = _backend(bs=2)
        o = self._decode(backend, _layer(), 2)

        self.assertEqual(tuple(o.shape), (2, _HEADS, _KV_LORA_RANK))
        self.assertEqual(o.dtype, torch.bfloat16)
        self.assertTrue(torch.all(o == _KERNEL_OUT))
        (call,) = self.kernel.calls
        fm = backend.forward_metadata
        self.assertIs(call["block_tables"], fm.paged_mla_block_tables)
        self.assertIs(call["seqused_k"], fm.paged_mla_seq_lens)
        self.assertIs(call["cu_seqlens_q"], fm.qo_indptr)
        self.assertEqual(
            tuple(call["kv_buffer"].shape),
            (1024 // _PAGE_SIZE, _PAGE_SIZE, 1, _KV_LORA_RANK + _ROPE_DIM),
        )
        self.assertEqual(call["kv_lora_rank"], _KV_LORA_RANK)
        self.assertEqual(call["qk_rope_head_dim"], _ROPE_DIM)

    def test_no_paged_metadata_runs_the_asm_decode(self):
        """Batches without a paged page table must reach the ASM decode."""
        backend = _backend(paged=False)
        self.assertIs(self._decode(backend, _layer(), 1), self.asm_out)
        self.assertEqual(self.kernel.calls, [])

    def test_kv_descale_matches_the_asm_decode(self):
        """The MLA store writes K unscaled; only an fp8 cache takes the k_scale."""
        for name, kv_dtype, layer_k_scale, expected in (
            ("bf16", torch.bfloat16, 2.0, None),
            ("fp8_layer_scale", amg.fp8_dtype, 2.0, 2.0),
            ("fp8_backend_scale", amg.fp8_dtype, None, 1.0),
        ):
            with self.subTest(name):
                backend = _backend(kv_dtype=kv_dtype)
                layer = _layer(k_scale=layer_k_scale)
                backend.forward_decode(
                    _q(layer, 1),
                    None,
                    None,
                    layer,
                    types.SimpleNamespace(forward_mode=ForwardMode.DECODE),
                    save_kv_cache=False,
                )
                kv_descale = self.kernel.calls[-1]["kv_descale"]
                if expected is None:
                    self.assertIsNone(kv_descale)
                else:
                    self.assertEqual(kv_descale.item(), expected)


class TestGfx950MlaGluonOnGfx1250(CustomTestCase):
    def test_gfx950_mla_gluon_is_never_selected_on_gfx1250(self):
        """gfx1250 must not pick (or report as enabled) the gfx950 kernel;
        other arches keep picking it."""
        mla_gluon = object()
        modules = {
            name: types.ModuleType(name)
            for name in (
                "triton",
                "triton.experimental",
                "triton.experimental.gluon",
                "triton.experimental.gluon.language",
                "aiter",
                "aiter.ops",
                "aiter.ops.triton",
                "aiter.ops.triton.gluon",
                "aiter.ops.triton.gluon.mla_gluon",
            )
        }
        modules["triton.experimental.gluon.language"].PaddedSharedLayout = (
            lambda cga_layout=None: None
        )
        modules["aiter.ops.triton.gluon.mla_gluon"].mla_gluon = mla_gluon
        self.addCleanup(amg._gluon_fn.cache_clear)
        with mock.patch.dict(sys.modules, modules):
            with envs.SGLANG_AITER_MLA_GLUON.override(True):
                for gfx1250, expected in ((True, None), (False, mla_gluon)):
                    with self.subTest(gfx1250=gfx1250):
                        amg._gluon_fn.cache_clear()
                        with mock.patch.object(
                            amg, "is_gfx1250_supported", return_value=gfx1250
                        ):
                            self.assertIs(amg._gluon_fn(), expected)


if __name__ == "__main__":
    unittest.main()
