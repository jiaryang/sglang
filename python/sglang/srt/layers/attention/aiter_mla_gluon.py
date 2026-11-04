"""aiter Gluon MLA decode wrappers for the aiter attention backend.

Two independent kernels, both turned off by ``SGLANG_AITER_MLA_GLUON=0``:

- gfx950: ``aiter.ops.triton.gluon.mla_gluon`` for low head-count fp8 MLA
  (e.g. Kimi K3 TP8, 12 heads/GPU) over token-level kv_indices. Needs aiter with
  ROCm/aiter #4480 (batch>1 ``bh16bn128``) and #4555 (decode CUDA graph KV splits)
  and Triton >= 3.7 (Gluon ``cga_layout``); otherwise the caller zero-pads heads
  for ``mla_decode_fwd``.
- gfx1250: ``aiter.ops.triton.attention.mla.mla_decode_fwd``, which runs aiter's
  gfx1250 Gluon MLA kernels, over paged block tables with a bf16 query. Needs
  aiter with ROCm/aiter#5935.
"""

from __future__ import annotations

import functools
import inspect
import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
from sglang.srt.environ import envs
from sglang.srt.utils import is_gfx1250_supported

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention

logger = logging.getLogger(__name__)


# ===== gfx950: token-indexed mla_gluon =====


@functools.lru_cache(maxsize=1)
def _gluon_fn():
    """aiter mla gluon entry point return None if disabled"""
    if not envs.SGLANG_AITER_MLA_GLUON.get():
        logger.info("aiter mla gluon is disabled manually.")
        return None
    if is_gfx1250_supported():
        logger.info("aiter mla gluon is disabled on gfx1250, which uses paged Gluon.")
        return None
    try:
        import triton
        import triton.experimental.gluon.language as gl
        from aiter.ops.triton.gluon.mla_gluon import mla_gluon
    except ImportError as exc:
        logger.info("aiter mla gluon import error message: %s", exc)
        return None
    # mla_gluon builds its shared layouts with cga_layout, added in Triton 3.7;
    # older Triton fails at compile time with an opaque error instead.
    if "cga_layout" not in inspect.signature(gl.PaddedSharedLayout).parameters:
        logger.info(
            "aiter mla gluon is disabled due to triton %s has no Gluon cga_layout (need >= 3.7)",
            getattr(triton, "__version__", "unknown"),
        )
        return None
    return mla_gluon


def log_mla_gluon_capability(log: logging.Logger | None = None) -> None:
    """Report whether Gluon MLA decode is valid; the reason is logged by _gluon_fn."""
    ready = _gluon_fn() is not None
    (log or logger).info(
        "aiter mla gluon is %s",
        "enabled" if ready else "disabled; falling back to zero-pad mla_decode_fwd",
    )


def prefer_mla_gluon_decode(
    *, head_pad_mode: str, num_head: int, kv_cache_dtype: torch.dtype
) -> bool:
    return (
        head_pad_mode == "zero"
        and num_head == 12
        and kv_cache_dtype == fp8_dtype
        and _gluon_fn() is not None
    )


def mla_gluon_decode(
    *,
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    layer: RadixAttention,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    sm_scale: float,
    min_kv_seq_len: int,
    kv_scale: float = 1.0,
    qlen: int = 1,
    use_2d_view: bool = False,
    return_lse: bool = False,
):
    """Run Gluon MLA decode for fused Q [num_tokens, H, 576] and MLA KV pool.
    Returns [num_tokens, H, v_head_dim] (or ``(out, lse)`` when ``return_lse``),
    or None when Gluon is unavailable.
    """
    mla_gluon = _gluon_fn()
    if mla_gluon is None:
        return None

    num_head = layer.tp_q_head_num
    kv_lora_rank = layer.v_head_dim
    qk_rope_head_dim = layer.qk_head_dim - kv_lora_rank
    batch_size = q.shape[0] // qlen

    q_nope, q_pe = torch.split(q, [kv_lora_rank, qk_rope_head_dim], dim=-1)
    if qlen > 1:
        # Splitting the leading dim is a stride change, so these stay views of
        # the non-contiguous torch.split outputs.
        q_nope = q_nope.view(batch_size, qlen, num_head, kv_lora_rank)
        q_pe = q_pe.view(batch_size, qlen, num_head, qk_rope_head_dim)
        o = q.new_empty((batch_size, qlen, num_head, kv_lora_rank))
    else:
        o = q.new_empty((batch_size, num_head, kv_lora_rank))

    extra_kwargs = {}
    if return_lse:
        extra_kwargs["return_lse"] = True

    result = mla_gluon(
        q_nope,
        q_pe,
        k_buffer.view(-1, layer.qk_head_dim),
        o,
        kv_indices,
        kv_indptr,
        sm_scale,
        k_pe=None,
        kv_pe_offset=kv_lora_rank,
        use_2d_view=use_2d_view,
        kv_scale=kv_scale,
        min_kv_seq_len=min_kv_seq_len,
        **extra_kwargs,
    )
    # Hand back the caller's flat [num_tokens, H, v] layout either way.
    out = o.flatten(0, 1) if qlen > 1 else o
    if not return_lse:
        return out
    # mla_gluon writes the output into `o` and returns it alongside the lse.
    _, lse = result
    return out, lse


# ===== gfx1250: paged mla_decode_fwd =====


def supports_paged_mla_gluon_decode(
    *,
    input_dtype: torch.dtype,
    page_size: int,
    dcp_world_size: int,
    qk_rope_head_dim: int,
) -> bool:
    """Whether single-token MLA decode runs ``paged_mla_gluon_decode``."""
    return (
        envs.SGLANG_AITER_MLA_GLUON.get()
        and is_gfx1250_supported()
        and input_dtype == torch.bfloat16
        and qk_rope_head_dim > 0
        # The only page size aiter's test_mla_decode_fwd covers for bf16/fp8 KV.
        and page_size == 64
        and dcp_world_size == 1
    )


def paged_mla_block_tables(
    *,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    max_kv_len: int,
    page_size: int,
) -> torch.Tensor:
    """[bs, cdiv(max_kv_len, page_size)] page ids of each request.

    The paged allocator keeps a page's slots contiguous, so a page id is the
    slot of the page's first token divided by the page size.
    """
    return req_to_token[req_pool_indices, :max_kv_len:page_size] // page_size


def paged_mla_gluon_decode(
    *,
    q: torch.Tensor,
    o: torch.Tensor,
    kv_buffer: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    page_size: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    sm_scale: float,
    kv_descale: Optional[torch.Tensor],
) -> None:
    """Decode one query token per sequence into ``o`` [bs, H, kv_lora_rank].

    q is [bs, H, kv_lora_rank + qk_rope_head_dim] and kv_buffer the layer's MLA
    key buffer [num_slots, 1, kv_lora_rank + qk_rope_head_dim]. The kernel reads
    only the first cdiv(seq_len, page_size) entries of each block_tables row.
    """
    from aiter.ops.triton.attention.mla import mla_decode_fwd

    mla_decode_fwd(
        q=q,
        kv_buffer=kv_buffer.view(
            -1, page_size, kv_buffer.shape[-2], kv_buffer.shape[-1]
        ),
        out=o,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seq_lens,
        # 0: the kernel sizes its KV split from a fixed guess, which keeps
        # CUDA graph replay valid for any sequence length.
        max_seqlen_kv=0,
        block_tables=block_tables,
        softmax_scale=sm_scale,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        causal=True,
        q_descale=None,
        kv_descale=kv_descale,
    )
