"""gfx1250 fp8 MLA prefill through aiter's FlyDSL a8w8 kernel.

``AiterAttnBackend._mla_prefill_varlen`` casts q/k/v to fp8 with scale 1 and
calls the FlyDSL varlen entry when ``use_gfx1250_fp8_prefill_attn`` is set.
Guards: the cast reaches FlyDSL with the caller's arguments; a shape FlyDSL
turns down, a non-bf16 query or the flag off stays on bf16
``flash_attn_varlen_func``; and on gfx1250 the a8w8 kernel matches an fp32
reference built from the same fp8-rounded inputs: causal with and without a
prefix, the non-causal prefix-chunk call with a request that has no keys, and
both with and without the LSE that the chunked-prefix merge reads.
"""

import types
import unittest
from unittest import mock

import torch

from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
from sglang.srt.layers.attention import aiter_backend
from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
from sglang.srt.utils.common import is_gfx1250_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd")

# Kimi-K2 TP8 MLA prefill in MHA form: 8 heads, nope 128 + rope 64, v 128.
_HEADS = 8
_QK_DIM = 192
_V_DIM = 128


def _qkv(num_q: int, num_k: int, dtype=torch.bfloat16, device="cpu"):
    g = torch.Generator(device="cpu").manual_seed(0)
    q = torch.randn(num_q, _HEADS, _QK_DIM, generator=g).to(device, dtype)
    k = torch.randn(num_k, _HEADS, _QK_DIM, generator=g).to(device, dtype)
    v = torch.randn(num_k, _HEADS, _V_DIM, generator=g).to(device, dtype)
    return q, k, v


class TestGfx1250Fp8PrefillRouting(CustomTestCase):
    def setUp(self):
        self.cu = torch.tensor([0, 4], dtype=torch.int32)
        self.args = (self.cu, self.cu, 4, 4)
        self.kwargs = {"softmax_scale": 0.1, "causal": True}

    def _run(self, enabled, flydsl_result, q_dtype=torch.bfloat16):
        q, k, v = _qkv(4, 4, dtype=q_dtype)
        flydsl = mock.Mock(return_value=flydsl_result)
        bf16 = mock.Mock(return_value="bf16")
        backend = types.SimpleNamespace(
            use_gfx1250_fp8_prefill_attn=enabled, flydsl_varlen_fwd=flydsl
        )
        with mock.patch.object(
            aiter_backend, "flash_attn_varlen_func", bf16, create=True
        ):
            out = AiterAttnBackend._mla_prefill_varlen(
                backend, q, k, v, *self.args, **self.kwargs
            )
        return out, flydsl, bf16, (q, k, v)

    def test_casts_to_fp8_and_calls_flydsl(self):
        out, flydsl, bf16, _ = self._run(True, "fp8")
        self.assertEqual(out, "fp8")
        bf16.assert_not_called()
        (q8, k8, v8, *args), kwargs = flydsl.call_args
        self.assertEqual({t.dtype for t in (q8, k8, v8)}, {fp8_dtype})
        self.assertEqual(tuple(args), self.args)
        self.assertEqual(kwargs, self.kwargs)

    def test_flydsl_decline_falls_back_to_bf16(self):
        out, flydsl, bf16, (q, k, v) = self._run(True, None)
        self.assertEqual(out, "bf16")
        flydsl.assert_called_once()
        (q0, k0, v0, *args), kwargs = bf16.call_args
        self.assertIs(q0, q)
        self.assertIs(k0, k)
        self.assertIs(v0, v)
        self.assertEqual(tuple(args), self.args)
        self.assertEqual(kwargs, self.kwargs)

    def test_flag_off_stays_bf16(self):
        out, flydsl, _, _ = self._run(False, "fp8")
        self.assertEqual(out, "bf16")
        flydsl.assert_not_called()

    def test_fp16_query_stays_on_input_dtype(self):
        out, flydsl, _, _ = self._run(True, "fp8", q_dtype=torch.float16)
        self.assertEqual(out, "bf16")
        flydsl.assert_not_called()


def _has_a8w8_kernel() -> bool:
    try:
        from aiter.ops.flydsl.kernels.fmha_gfx1250 import (  # noqa: F401
            fmha_fwd_prefill_a8w8_m32x8,
        )
    except ImportError:
        return False
    return True


def _reference(q, k, v, cu_q, cu_k, scale, causal):
    """fp32 attention, causal mask bottom-right aligned; returns out and LSE [H, T].

    Rows with no visible key give output 0 and LSE -inf, as the kernel does.
    """
    outs, lses = [], []
    for i in range(len(cu_q) - 1):
        qs, ks = q[cu_q[i] : cu_q[i + 1]].float(), k[cu_k[i] : cu_k[i + 1]].float()
        vs = v[cu_k[i] : cu_k[i + 1]].float()
        sq, sk = qs.shape[0], ks.shape[0]
        s = torch.einsum("qhd,khd->hqk", qs, ks) * scale
        if causal:
            keep = torch.ones(sq, sk, dtype=torch.bool, device=q.device).tril(sk - sq)
            s = s.masked_fill(~keep, float("-inf"))
        lses.append(torch.logsumexp(s, dim=-1))
        p = s.softmax(dim=-1).nan_to_num(0.0)
        outs.append(torch.einsum("hqk,khd->qhd", p, vs))
    return torch.cat(outs), torch.cat(lses, dim=-1)


@unittest.skipUnless(
    torch.cuda.is_available() and is_gfx1250_supported() and _has_a8w8_kernel(),
    "needs gfx1250 and an aiter with the FlyDSL a8w8 prefill kernel",
)
class TestGfx1250Fp8PrefillKernel(CustomTestCase):
    # Provisional bounds, to be tightened from the error measured on gfx1250.
    # The kernel rounds P to e4m3 before PV, about 3-4% RMS on the output,
    # which a reference on the same fp8 q/k/v does not model.
    OUT_REL_ERR = 5e-2
    LSE_TOL = 1e-2

    def _check(self, q_lens, k_lens, causal=True, return_lse=True):
        from aiter.ops.flydsl.fmha_kernels import flydsl_flash_attn_varlen_fwd

        cu_q = [0]
        cu_k = [0]
        for lq, lk in zip(q_lens, k_lens):
            cu_q.append(cu_q[-1] + lq)
            cu_k.append(cu_k[-1] + lk)
        q, k, v = _qkv(cu_q[-1], cu_k[-1], device="cuda")
        scale = _QK_DIM**-0.5
        backend = types.SimpleNamespace(
            use_gfx1250_fp8_prefill_attn=True,
            flydsl_varlen_fwd=flydsl_flash_attn_varlen_fwd,
        )
        result = AiterAttnBackend._mla_prefill_varlen(
            backend,
            q,
            k,
            v,
            torch.tensor(cu_q, dtype=torch.int32, device="cuda"),
            torch.tensor(cu_k, dtype=torch.int32, device="cuda"),
            max(q_lens),
            max(k_lens),
            softmax_scale=scale,
            causal=causal,
            return_lse=return_lse,
        )
        if return_lse:
            out, lse = result[:2]
        else:
            self.assertIs(type(result), torch.Tensor)
            out = result
        ref_out, ref_lse = _reference(
            *(t.to(fp8_dtype) for t in (q, k, v)), cu_q, cu_k, scale, causal
        )
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(tuple(out.shape), (cu_q[-1], _HEADS, _V_DIM))
        rel_err = (out.float() - ref_out).norm() / ref_out.norm()
        self.assertLessEqual(rel_err.item(), self.OUT_REL_ERR)
        if return_lse:
            torch.testing.assert_close(
                lse.float(), ref_lse, atol=self.LSE_TOL, rtol=self.LSE_TOL
            )

    def test_no_prefix(self):
        self._check([256, 100], [256, 100])

    def test_with_prefix(self):
        self._check([128, 64], [384, 200])

    def test_prefix_chunk_with_empty_kv(self):
        # _forward_extend_prefix_chunk: non-causal, and a request can have no
        # keys in this chunk.
        self._check([128, 64, 32], [256, 0, 100], causal=False)

    def test_without_lse(self):
        self._check([256, 100], [256, 100], return_lse=False)


if __name__ == "__main__":
    unittest.main()
