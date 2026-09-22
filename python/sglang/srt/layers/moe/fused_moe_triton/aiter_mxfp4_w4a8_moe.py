# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/vllm-project/vllm/pull/46516
#
# MXFP4-weight / FP8-activation (W4A8) fused MoE for AMD gfx1250 (RDNA / gfx12).
#
# gfx1250's aiter CK/ASM ``fused_moe`` produces garbage for the GPT-OSS MXFP4
# W4A8 layout, so this path routes through aiter's *triton* ``moe_gemm_a8w4``
# kernel instead (the same kernel gfx950 uses). Two gfx1250-specific quirks are
# handled, mirroring the vLLM enablement:
#   1. The in-kernel TDM gather fails to compile on gfx1250, so we disable the
#      TDM routing path and gather activation rows into expert-sorted order in
#      torch (passing ``gather_indx=None`` to the GEMM).
#   2. The gfx1250 ``moe_gemm_a8w4`` reads a CDNA4-swizzled MX scale as garbage,
#      so the weight scale is kept unswizzled and ``swizzle_mx_scale=None`` is
#      passed to the kernel.

from __future__ import annotations

import torch

_TDM_DISABLED = False


def _is_gfx1250() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 5)


def _import_aiter_w4a8():
    """Import the aiter triton routing + a8w4 GEMM entry points.

    Returns ``(routing, moe_gemm_a8w4, downcast_to_static_fp8)`` or ``None`` if
    the installed aiter build does not expose the triton W4A8 path.
    """
    try:
        try:
            import aiter.ops.triton.moe.moe_routing.routing as _routing_mod
        except ImportError:
            import aiter.ops.triton.moe_routing.routing as _routing_mod

        from aiter.ops.triton.moe.moe_op_gemm_a8w4 import moe_gemm_a8w4
        from aiter.ops.triton.moe.quant_moe import downcast_to_static_fp8
    except ImportError:
        return None

    global _TDM_DISABLED
    if not _TDM_DISABLED:
        # gfx1250: the in-kernel TDM gather emitted by the routing sort / GEMM
        # fails to compile (``TDM gather dst must be 2D``). Force the non-TDM
        # path; we gather activations manually below.
        _routing_mod.is_tdm_avail = lambda: False
        _TDM_DISABLED = True

    return (
        _routing_mod.routing,
        moe_gemm_a8w4,
        downcast_to_static_fp8,
    )


def _interleave_gate_up(t: torch.Tensor) -> torch.Tensor:
    """Convert a SEPARATED ``[gate_0..gate_{I-1}, up_0..up_{I-1}]`` first dim
    (after the expert dim) into the INTERLEAVED ``[gate_0, up_0, gate_1, up_1,
    ...]`` order that ``moe_gemm_a8w4``'s fused SwiGLU expects (gate on the
    even lanes, up on the odd lanes)."""
    e, two_i = t.shape[0], t.shape[1]
    i = two_i // 2
    rest = t.shape[2:]
    t = t.view(e, 2, i, *rest)
    perm = (0, 2, 1) + tuple(range(3, t.dim()))
    return t.permute(*perm).reshape(e, two_i, *rest).contiguous()


def prepare_w4a8_gfx1250_weights(
    w13_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w13_weight_bias: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    w2_weight_bias: torch.Tensor,
    interleave_w13: bool = True,
):
    """Reshape SGLang's loaded Quark W4A8 MoE buffers into the ``[E, K, N]``
    (contraction-major) packed layout consumed by ``moe_gemm_a8w4``.

    Input (SGLang / HF Quark layout, per expert), output-channel major:
        w13_weight        [E, 2I, H//2]   uint8 (2 FP4 packed along H)
        w13_weight_scale  [E, 2I, H//32]  uint8 (e8m0), gate/up SEPARATED
        w13_weight_bias   [E, 2I]         fp32
        w2_weight         [E, H,  I//2]   uint8
        w2_weight_scale   [E, H,  I//32]  uint8 (e8m0)
        w2_weight_bias    [E, H]          fp32

    Output (moe_gemm_a8w4 layout), contraction (K) major, gate/up INTERLEAVED
    for w13:
        w13  [E, H//2, 2I]   w13_scale [E, H//32, 2I]   w13_bias [E, 2I]
        w2   [E, I//2, H]    w2_scale  [E, I//32, H]    w2_bias  [E, H]
    """
    # Interleave gate/up on w13 (output dim) so the fused SwiGLU picks gate on
    # even lanes and up on odd lanes. GLM W4A4 uses SEPARATED SiLU-and-mul
    # instead, so callers pass ``interleave_w13=False``. The interleave output
    # is contiguous, so the subsequent transpose(1, 2) yields a *column-major*
    # [E, K, N] view (stride(-2) == 1), which ``moe_gemm_a8w4`` requires for
    # MXFP weights. Without interleave, make w13 contiguous first so the
    # transpose is still unit-strided on K.
    if interleave_w13:
        w13_weight = _interleave_gate_up(w13_weight)
        w13_weight_scale = _interleave_gate_up(w13_weight_scale)
        w13_weight_bias = _interleave_gate_up(w13_weight_bias)
    else:
        w13_weight = w13_weight.contiguous()
        w13_weight_scale = w13_weight_scale.contiguous()
        w13_weight_bias = w13_weight_bias.contiguous()

    # Transpose to contraction-major [E, K(packed), N] *without* making it
    # contiguous, so the K dimension stays unit-strided (column-major).
    w13_weight = w13_weight.transpose(1, 2)
    w13_weight_scale = w13_weight_scale.transpose(1, 2)
    w2_weight = w2_weight.contiguous().transpose(1, 2)
    w2_weight_scale = w2_weight_scale.contiguous().transpose(1, 2)

    return (
        w13_weight,
        w13_weight_scale,
        w13_weight_bias.contiguous(),
        w2_weight,
        w2_weight_scale,
        w2_weight_bias.contiguous(),
    )


def aiter_w4a8_gfx1250_forward(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    topk: int,
    w13_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w13_weight_bias: torch.Tensor,
    a13_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    w2_weight_bias: torch.Tensor,
    a2_scale: torch.Tensor,
    gemm1_alpha: float,
    gemm1_limit: float,
    renormalize: bool = True,
    apply_router_weight_on_input: bool = False,
) -> torch.Tensor:
    """MXFP4 W4A8 GPT-OSS MoE forward for gfx1250 via aiter triton
    ``moe_gemm_a8w4``.

    ``w*`` / ``w*_scale`` / ``w*_bias`` must already be in the
    ``moe_gemm_a8w4`` layout produced by :func:`prepare_w4a8_gfx1250_weights`.
    ``a13_scale`` / ``a2_scale`` are the static per-tensor FP8 activation scales
    for gate_up_proj and down_proj respectively.
    """
    imported = _import_aiter_w4a8()
    if imported is None:
        raise RuntimeError(
            "aiter triton W4A8 MoE (moe_gemm_a8w4) is required for the gfx1250 "
            "GPT-OSS MXFP4 path but was not found in the installed aiter build."
        )
    routing, moe_gemm_a8w4, downcast_to_static_fp8 = imported

    assert hidden_states.dtype == torch.bfloat16

    # aiter routing on the raw router logits. renormalize=True (GPT-OSS)
    # corresponds to applying softmax to the top-k selection inside the kernel
    # (sm_first=False).
    routing_data, gather_idx, scatter_idx = routing(
        router_logits, topk, sm_first=not renormalize
    )
    gammas = routing_data.gate_scal

    # gfx1250: the in-kernel gather is broken, so we pass gather_indx=None to
    # moe_gemm_a8w4 and perform the gather ourselves.
    gather_src = gather_idx.to(torch.long) // topk
    x = hidden_states[gather_src]
    if apply_router_weight_on_input:
        # Router weights must be applied in bf16 before quantization.
        x = x * gammas[:, None].to(x.dtype)
    x_fp8 = downcast_to_static_fp8(x, a13_scale)

    # GEMM1: FP8 activations x MXFP4 weights, fused SwiGLU, requantize the
    # intermediate to FP8 using the down_proj activation scale (a2_scale) so
    # GEMM2 can consume it directly.
    intermediate_cache1 = moe_gemm_a8w4(
        x_fp8,
        w13_weight,
        None,
        w13_weight_scale,
        a13_scale,
        a2_scale,
        w13_weight_bias,
        routing_data,
        gather_indx=None,
        scatter_indx=None,
        gammas=None,
        swizzle_mx_scale=None,
        out_dtype=x_fp8.dtype,
        apply_swiglu=True,
        alpha=gemm1_alpha,
        limit=gemm1_limit,
    )

    # GEMM2: down projection, scatter back to token order and apply the router
    # weights (gammas) unless they were already applied on the input.
    intermediate_cache3 = moe_gemm_a8w4(
        intermediate_cache1,
        w2_weight,
        None,
        w2_weight_scale,
        a2_scale,
        None,
        w2_weight_bias,
        routing_data,
        gather_indx=None,
        scatter_indx=scatter_idx,
        gammas=None if apply_router_weight_on_input else gammas,
        swizzle_mx_scale=None,
        out_dtype=torch.bfloat16,
    )

    return intermediate_cache3.contiguous()


_FP8_E4M3_MAX = 448.0
_FUSED_ROUTING_NK_LIMIT = 4096


def _fp8_per_tensor_scale(x: torch.Tensor) -> torch.Tensor:
    return (x.detach().float().abs().amax().clamp(min=1e-12) / _FP8_E4M3_MAX).to(
        torch.float32
    )


def _routing_from_topk_torch(
    topk_weights: torch.Tensor, topk_ids: torch.Tensor, n_expts_tot: int
):
    ids = topk_ids.reshape(-1).to(torch.int32)
    scal = topk_weights.reshape(-1)
    topk_indx = torch.argsort(ids, stable=True).to(torch.int32)
    n_gates = ids.numel()
    gate_indx = torch.empty(n_gates, dtype=torch.int32, device=ids.device)
    gate_indx[topk_indx.long()] = torch.arange(
        n_gates, device=ids.device, dtype=torch.int32
    )
    gate_scal = scal[topk_indx.long()]
    hist = torch.bincount(ids.long(), minlength=n_expts_tot).to(torch.int32)
    if hist.numel() > n_expts_tot:
        hist = hist[:n_expts_tot]
    elif hist.numel() < n_expts_tot:
        hist = torch.nn.functional.pad(hist, (0, n_expts_tot - hist.numel()))
    return hist, topk_indx, gate_indx, gate_scal


def _expt_data_from_hist(hist, n_expts_tot: int, n_gates: int, block_m: int):
    from aiter.ops.triton.moe.moe_routing.routing import ExptData

    device = hist.device
    token_offs_raw = torch.empty(n_expts_tot + 1, dtype=torch.int32, device=device)
    token_offs_raw[0] = 0
    token_offs_raw[1:] = torch.cumsum(hist, dim=0).to(torch.int32)

    n_tiles = (hist + (block_m - 1)) // block_m
    token_offs_pad = torch.empty(n_expts_tot + 1, dtype=torch.int32, device=device)
    token_offs_pad[0] = 0
    token_offs_pad[1:] = torch.cumsum(n_tiles, dim=0).to(torch.int32)

    if n_gates <= n_expts_tot:
        max_n_tiles = max(n_gates, 1)
    else:
        max_n_tiles = max(
            n_expts_tot - 1 - ((n_expts_tot - n_gates - 1) // block_m), 1
        )

    block_pid_map = torch.full((max_n_tiles,), -1, dtype=torch.int32, device=device)
    n_filled = int(n_tiles.sum().item())
    if n_filled > 0:
        experts = torch.arange(n_expts_tot, device=device, dtype=torch.int32)
        expert_ids = torch.repeat_interleave(experts, n_tiles.to(torch.int64))
        starts = token_offs_pad[:-1].repeat_interleave(n_tiles.to(torch.int64))
        packed = torch.arange(n_filled, device=device, dtype=torch.int32)
        local_b = packed - starts
        dest = (starts + local_b).long()
        block_pid_map[dest] = (local_b << 16) + expert_ids
    return ExptData(hist, token_offs_raw, token_offs_pad, block_pid_map)


def routing_from_sglang_topk(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    n_expts_tot: int,
):
    """Build aiter ``RoutingData`` from SGLang's already-computed top-k.

    Do not re-run aiter ``routing()`` on DeepSeek/GLM logits: that would ignore
    grouped top-k, fused shared-expert slots, and ``routed_scaling_factor``.
    """
    import triton
    from aiter.ops.triton.moe.moe_routing.routing import RoutingData

    topk_ids = topk_ids.contiguous()
    topk_weights = topk_weights.contiguous()
    n_tokens, n_expts_act = topk_ids.shape
    n_gates = n_tokens * n_expts_act

    tokens_per_expt = max(1, n_gates // n_expts_tot)
    block_m = max(16, min(triton.next_power_of_2(tokens_per_expt), 128))
    if _is_gfx1250():
        # gfx1250 quirk 3: moe_gemm_a8w4 only produces correct results at
        # block_m == 16. Larger tiles, which the heuristic above picks as soon
        # as a forward carries enough tokens, return garbage (rel RMS 0.19 at
        # block_m=32, NaN/Inf at 64 and 128) and wreck generation quality for
        # any batch beyond a handful of tokens.
        block_m = 16

    if n_gates <= _FUSED_ROUTING_NK_LIMIT:
        try:
            from aiter.ops.triton.fusions.fused_routing_from_topk import (
                fused_routing_from_topk,
            )

            hist, topk_indx, gate_indx, gate_scal = fused_routing_from_topk(
                topk_weights, topk_ids.to(torch.int32), n_expts_tot
            )
        except Exception:
            hist, topk_indx, gate_indx, gate_scal = _routing_from_topk_torch(
                topk_weights, topk_ids, n_expts_tot
            )
    else:
        hist, topk_indx, gate_indx, gate_scal = _routing_from_topk_torch(
            topk_weights, topk_ids, n_expts_tot
        )

    expt_data = _expt_data_from_hist(hist, n_expts_tot, n_gates, block_m)
    routing_data = RoutingData(
        block_m, gate_scal, hist, n_expts_tot, n_expts_act, expt_data
    )
    return routing_data, topk_indx, gate_indx


def aiter_w4a8_gfx1250_forward_from_topk(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w13_weight_bias: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    w2_weight_bias: torch.Tensor,
    apply_router_weight_on_input: bool = False,
    apply_swiglu: bool = False,
    gemm1_alpha: float = 1.0,
    gemm1_limit: float = 1e9,
    swiglu_add_residual: bool = False,
) -> torch.Tensor:
    """MXFP4 W4A8 MoE for gfx1250 using existing SGLang top-k (GLM / DeepSeek).

    GEMM1 writes bf16, then SiLU-and-mul (SEPARATED gate/up, matching GLM) or
    fused SwiGLU (INTERLEAVED, matching GPT-OSS). GEMM2 uses a dynamically
    computed per-tensor FP8 scale. Gather is done in torch (TDM gather is
    broken on gfx1250).
    """
    imported = _import_aiter_w4a8()
    if imported is None:
        raise RuntimeError(
            "aiter triton W4A8 MoE (moe_gemm_a8w4) is required for the gfx1250 "
            "MXFP4 path but was not found in the installed aiter build."
        )
    _, moe_gemm_a8w4, downcast_to_static_fp8 = imported

    if hidden_states.dtype != torch.bfloat16:
        hidden_states = hidden_states.to(torch.bfloat16)

    n_expts_tot = w13_weight.shape[0]
    routing_data, gather_idx, scatter_idx = routing_from_sglang_topk(
        topk_weights, topk_ids, n_expts_tot
    )
    gammas = routing_data.gate_scal
    topk = topk_ids.shape[-1]

    gather_src = gather_idx.to(torch.long) // topk
    x = hidden_states[gather_src]
    if apply_router_weight_on_input:
        x = x * gammas[:, None].to(x.dtype)

    a13_scale = _fp8_per_tensor_scale(x)
    x_fp8 = downcast_to_static_fp8(x, a13_scale)

    intermediate = moe_gemm_a8w4(
        x_fp8,
        w13_weight,
        None,
        w13_weight_scale,
        a13_scale,
        None,
        w13_weight_bias,
        routing_data,
        gather_indx=None,
        scatter_indx=None,
        gammas=None,
        swizzle_mx_scale=None,
        out_dtype=torch.bfloat16,
        apply_swiglu=apply_swiglu,
        alpha=gemm1_alpha,
        limit=gemm1_limit,
        swiglu_add_residual=swiglu_add_residual,
    )

    if not apply_swiglu:
        d = intermediate.shape[-1] // 2
        intermediate = torch.nn.functional.silu(intermediate[..., :d]) * intermediate[
            ..., d:
        ]

    a2_scale = _fp8_per_tensor_scale(intermediate)
    x2 = downcast_to_static_fp8(intermediate.contiguous(), a2_scale)

    output = moe_gemm_a8w4(
        x2,
        w2_weight,
        None,
        w2_weight_scale,
        a2_scale,
        None,
        w2_weight_bias,
        routing_data,
        gather_indx=None,
        scatter_indx=scatter_idx,
        gammas=None if apply_router_weight_on_input else gammas,
        swizzle_mx_scale=None,
        out_dtype=torch.bfloat16,
    )
    return output.contiguous()

