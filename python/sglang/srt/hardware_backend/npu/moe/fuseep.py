"""Ascend FuseEP fused dispatch+GEMM+combine forward path.

Follows the mega_moe shape: a free-function bypass invoked from
``FusedMoE.forward`` when ``--moe-a2a-backend ascend_fuseep`` is set, plus a
weight-postprocess helper that NPU quant_methods call from their
``process_weights_after_loading`` when the same backend is selected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.distributed import get_moe_ep_group
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.utils import npu_format_cast
from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPBuffer
from sglang.srt.layers.moe.utils import DeepEPMode
from sglang.srt.runtime_context import get_exec

if TYPE_CHECKING:
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.topk import TopKOutput


_PARAMS_BYTES = 2  # bf16 — Ascend's Dispatch & Combine does not support fp16

# fused_deep_moe quant_mode slot: the A5 aclnnFusedDeepMoe host derives the
# FP4 vs FP8 vs int8 GMM from the gmm1 weight dtype itself, so the value is
# not a user-facing switch — test_fused_deep_moe_a5.py pins the same fixed
# compatibility value 0 (FUSED_COMPAT_QUANT_MODE). 1 = int8 (W8A8) for the
# int8 weight contract.
_FUSEEP_QUANT_MODE_INT8 = 1
_FUSEEP_QUANT_MODE_COMPAT = 0


def _is_fp4_moe_layer(layer: FusedMoE) -> bool:
    """Identify FP4 experts by their quant method.

    Weight dtype is not a reliable discriminator across the load lifecycle
    (checkpoint bytes are uint8-packed before repacking), so the quant method
    is checked instead.
    """
    from sglang.srt.hardware_backend.npu.quantization.fp4_moe_methods import (
        NPUW4A4Fp4MoEMethod,
    )

    return isinstance(layer.quant_method, NPUW4A4Fp4MoEMethod)


def _get_fuseep_buffer(layer: FusedMoE):
    DeepEPBuffer.set_dispatch_mode_as_low_latency()
    return DeepEPBuffer.get_deepep_buffer(
        get_moe_ep_group().device_group,
        layer.hidden_size,
        _PARAMS_BYTES,
        DeepEPMode.LOW_LATENCY,
        envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get(),
        layer.num_experts,
    )


def forward_fuseep(
    layer: FusedMoE,
    hidden_states: torch.Tensor,
    topk_output: TopKOutput,
) -> torch.Tensor:
    if _is_fp4_moe_layer(layer):
        # FP4 scales ship as raw E8M0 exponent bytes (uint8, [E, N, K/32]).
        # Re-interpret and pair them as [E, N, K/64, 2] — the layout the A5
        # fused op's B-scale tensor assumes (MakeMxScaleLayout with a 2-wide
        # pair dim, matching npu_dynamic_mx_quant(axis=1) output — see
        # test_fused_deep_moe_a5.py). The deep_ep wrapper's `.float()` then
        # decodes the power-of-two value instead of casting the byte as an
        # integer. int8 layers keep their dequant scales in float dtypes and
        # need no reinterpret.
        w13_scale = (
            layer.w13_weight_scale_inv.view(torch.float8_e8m0fnu)
            .reshape(*layer.w13_weight_scale_inv.shape[:-1], -1, 2)
        )
        w2_scale = (
            layer.w2_weight_scale_inv.view(torch.float8_e8m0fnu)
            .reshape(*layer.w2_weight_scale_inv.shape[:-1], -1, 2)
        )
        quant_mode = _FUSEEP_QUANT_MODE_COMPAT
    else:
        w13_scale = layer.w13_weight_scale
        w2_scale = layer.w2_weight_scale
        quant_mode = _FUSEEP_QUANT_MODE_INT8

    buf = _get_fuseep_buffer(layer)
    hidden_states, _ = buf.fused_deep_moe(
        hidden_states,
        topk_idx=topk_output.topk_ids,
        topk_weights=topk_output.topk_weights,
        gmm1_permuted_weight=layer.w13_weight,
        gmm1_permuted_weight_scale=w13_scale,
        gmm2_weight=layer.w2_weight,
        gmm2_weight_scale=w2_scale,
        num_max_dispatch_tokens_per_rank=(
            envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get()
        ),
        num_experts=layer.num_experts,
        quant_mode=quant_mode,
        fuse_mode=get_exec().moe.fuseep_mode,
    )
    return hidden_states


def _permute_w13_weight_scale(w: torch.Tensor, tile_n: int) -> torch.Tensor:
    if tile_n % 2 != 0:
        raise ValueError(f"tile_n must be even, got {tile_n}")

    *dims, n = w.shape
    if n % tile_n != 0:
        raise ValueError(f"Last dimension {n} must be divisible by tile_n {tile_n}")

    w_reshaped = w.reshape(*dims, 2, n // tile_n, tile_n // 2)
    perm_order = list(range(len(dims))) + [-2, -3, -1]
    return w_reshaped.permute(perm_order).reshape(*dims, n)


def repack_fp4_weight_for_fused_deep_moe(w: torch.Tensor) -> torch.Tensor:
    """Repack checkpoint FP4 weights into the aclnnFusedDeepMoe (A5) layout.

    The msmodelslim checkpoint stores packed FP4 as ``[E, N, K/2]`` uint8 with
    two K values per byte (even k in the low nibble) — the layout the A5
    ``npu_quant_matmul(x2_dtype=float4_e2m1fn_x2)`` consumes. The A5 fused op
    instead consumes a logical ``[K, N]`` FP4 matrix packed along N
    (``[E, K, N/2]``, even n in the low nibble) — the layout
    ``npu_dynamic_mx_quant(axis=1)`` produces. This conversion is a lossless
    nibble-level transpose. Block scales (checkpoint ``[E, N, K/32]`` uint8,
    i.e. ``[E, N, K/64, 2]`` in the fused op's pair view) are already in the
    kernel's N-major layout and need no change.
    """
    from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
        _get_float4_e2m1fn_x2_dtype,
    )

    if w.dim() != 3:
        raise ValueError(
            f"expected [E, N, K/2] packed FP4 weight, got {tuple(w.shape)}"
        )
    num_experts, n, k_half = w.shape
    if n % 2 != 0:
        raise ValueError(f"N dimension must be even for packed FP4, got {n}")

    # val(e, n, k): even k in the low nibble of w[e, n, k//2], odd k in high.
    # Split by source nibble, then interleave pairs along N:
    #   out[e, k, j] = val(e, 2j, k) | val(e, 2j+1, k) << 4
    even_n = w[:, 0::2, :]  # [E, N/2, K/2]
    odd_n = w[:, 1::2, :]  # [E, N/2, K/2]
    out = torch.empty(
        (num_experts, k_half * 2, n // 2), dtype=torch.uint8, device=w.device
    )
    # k even -> low nibbles; k odd -> high nibbles (planes transposed to
    # [E, K/2, N/2] to match the strided out slices)
    out[:, 0::2, :] = ((even_n & 0x0F) | ((odd_n & 0x0F) << 4)).transpose(1, 2)
    out[:, 1::2, :] = ((even_n >> 4) | ((odd_n >> 4) << 4)).transpose(1, 2)
    return out.contiguous().view(_get_float4_e2m1fn_x2_dtype())


def process_fp4_fuseep_weights(layer: torch.nn.Module, weight_prefix: str) -> None:
    """Apply the FuseEP FP4 weight layout for a single weight group.

    Invoked by ``NPUW4A4Fp4MoEMethod.maybe_process_fp4_fuseep_weights`` for
    both ``"w13"`` and ``"w2"``.

    The weight is repacked from the checkpoint's K-packed ``[E, N, K/2]``
    bytes to the fused op's N-packed ``[E, K, N/2]`` fp4x2 layout. The scale
    (``{prefix}_weight_scale_inv``) stays as the checkpoint's ``[E, N, K/32]``
    uint8 — byte-identical to the fused op's N-major ``[N, K/64, 2]`` B-scale
    view; ``forward_fuseep`` pairs the last dim when handing the scales to
    deep_ep.
    """
    # FP4 experts run through the fused_deep_moe op (fuseep_mode 1), whose A5
    # kernel consumes the N-packed fp4x2 layout above. fuseep_mode 2
    # (dispatch_ffn_combine) only implements the int8 weight contract in the
    # fused op — fail fast at load time instead of hitting the C++ assert on
    # the first forward.
    from sglang.srt.runtime_context import get_exec

    if get_exec().moe.fuseep_mode != 1:
        raise NotImplementedError(
            "FP4 experts on ascend_fuseep require --fuseep-mode 1 "
            "(dispatch_gmm_combine_decode); mode "
            f"{get_exec().moe.fuseep_mode} (dispatch_ffn_combine) is "
            "int8-only in the fused op."
        )

    weight = getattr(layer, f"{weight_prefix}_weight")
    weight.data = repack_fp4_weight_for_fused_deep_moe(weight.data)


def _reshape_w13_weight(
    weight: torch.Tensor, dim: int, chunk_size: int = 64
) -> torch.Tensor:
    # Achieving greater computing power through reshape on Ascend.
    original_shape = weight.shape
    if dim < 0:
        dim += len(original_shape)

    if original_shape[dim] % (2 * chunk_size) != 0:
        raise ValueError(
            f"Dimension {dim} size {original_shape[dim]} must be divisible by "
            f"{2 * chunk_size}"
        )

    new_shape = (
        *original_shape[:dim],
        2,
        original_shape[dim] // (2 * chunk_size),
        chunk_size,
        *original_shape[dim + 1 :],
    )

    weight = weight.view(new_shape)
    weight = weight.transpose(dim, dim + 1).contiguous()
    return weight.view(*original_shape[:dim], -1, *original_shape[dim + 1 :])


def _release_weight_cache(weight: torch.Tensor) -> torch.Tensor:
    # .contiguous() introduces additional memory overhead; release with resize_(0)
    origin_weight = weight.data.transpose(1, 2)
    new_weight = origin_weight.contiguous()
    origin_weight.untyped_storage().resize_(0)
    return new_weight


def _scale_from_float_to_int64(scale: torch.Tensor) -> torch.nn.Parameter:
    import numpy as np

    converted = torch.from_numpy(
        np.frombuffer(
            scale.cpu().to(torch.float32).numpy().tobytes(), dtype=np.int32
        ).astype(np.int64)
    ).to(scale.device)
    return torch.nn.Parameter(converted, requires_grad=False)


def process_fuseep_weights(layer: torch.nn.Module, weight_prefix: str) -> None:
    """Apply the Ascend FuseEP-specific weight layout for a single weight group.

    Invoked by ``maybe_apply_fuseep_weights`` for both ``"w13"`` and ``"w2"``.
    """
    if get_exec().moe.fuseep_mode == 1:
        # -- The fused MoE optimization mode "1": dispatch_gmm_combine_decode --
        if weight_prefix == "w13":
            cpu_w13 = layer.w13_weight.data.transpose(1, 2).cpu()
            layer.w13_weight.data = _reshape_w13_weight(cpu_w13, -1).npu()
            w13_scale = layer.w13_weight_scale.data.squeeze(-1).contiguous()
            w13_scale = _permute_w13_weight_scale(w13_scale, 128)
            layer.w13_weight_scale = torch.nn.Parameter(
                w13_scale.to(torch.float32), requires_grad=False
            )
            layer.w13_weight.data = npu_format_cast(layer.w13_weight.data)
        else:  # weight_prefix == "w2"
            layer.w2_weight.data = npu_format_cast(layer.w2_weight.data)
            w2_scale = layer.w2_weight_scale.data.squeeze(-1).contiguous()
            layer.w2_weight_scale = torch.nn.Parameter(
                w2_scale.to(torch.float32), requires_grad=False
            )
    elif get_exec().moe.fuseep_mode == 2:
        # -- The fused MoE optimization mode "2": dispatch_ffn_combine --
        if weight_prefix == "w13":
            w13_weight = _release_weight_cache(layer.w13_weight)
            layer.w13_weight.data = npu_format_cast(w13_weight)
            layer.w13_weight_scale.data = layer.w13_weight_scale.data.view(
                layer.w13_weight_scale.data.shape[0], -1
            )
            layer.w13_weight_scale = _scale_from_float_to_int64(
                layer.w13_weight_scale.data
            )
        else:  # weight_prefix == "w2"
            w2_weight = _release_weight_cache(layer.w2_weight)
            layer.w2_weight.data = npu_format_cast(w2_weight)
            w2_scale = layer.w2_weight_scale.data.squeeze(-1).contiguous()
            layer.w2_weight_scale = torch.nn.Parameter(
                w2_scale.to(torch.float32), requires_grad=False
            )
            layer.w2_weight_scale = _scale_from_float_to_int64(
                layer.w2_weight_scale.data
            )

    # -- offsets (exist or not, same logic for both prefixes) ---------------
    offset_attr = f"{weight_prefix}_weight_offset"
    if hasattr(layer, offset_attr):
        setattr(
            layer,
            offset_attr,
            torch.nn.Parameter(
                getattr(layer, offset_attr).data.squeeze(-1).contiguous(),
                requires_grad=False,
            ),
        )
