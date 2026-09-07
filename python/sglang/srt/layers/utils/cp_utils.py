from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.dp_attention import (
    attn_cp_all_gather_into_tensor,
    is_allocation_symmetric,
)
from sglang.srt.runtime_context import (
    get_parallel,
    uses_mla_backend,
)


"""Import-only shims for deprecated platform backends awaiting CP refactoring.

The legacy CP algorithms have been removed. These names keep the retained
NPU/MUSA attention backends importable for non-CP inference; calling them fails.
"""


@dataclass
class ContextParallelMetadata:
    # Layout lists have length bs * cp_segment_num (= bs * 2 * cp_size).
    split_list: List[int] = None
    zigzag_index: List[int] = None
    cp_reverse_index: List[int] = None
    reverse_split_len: List[int] = None

    # Per-rank-aggregate lists have length cp_size.
    # max_rank_len is a list of cp_size copies of max(per_rank_actual_token),
    # kept as a list for torch.split() bucket sizes.
    per_rank_actual_token: List[int] = None
    max_rank_len: List[int] = None

    # Per-sequence FlashAttention tensors (shape [bs] or [bs+1]).
    kv_len_prev_tensor: torch.Tensor = None  # [bs] int32 CUDA
    kv_len_next_tensor: torch.Tensor = None  # [bs] int32 CUDA
    actual_seq_q_prev_tensor: torch.Tensor = None  # [bs] int32 CUDA
    actual_seq_q_next_tensor: torch.Tensor = None  # [bs] int32 CUDA
    cu_seqlens_q_prev_tensor: torch.Tensor = None  # [bs+1] int32 CUDA
    cu_seqlens_q_next_tensor: torch.Tensor = None  # [bs+1] int32 CUDA

    # Scalars derived from the per-sequence lists above.
    total_q_prev_tokens: int = 0
    total_q_next_tokens: int = 0
    max_seqlen_q_prev: int = 0
    max_seqlen_q_next: int = 0

    # Per-seq CPU lists (useful for NSA indexer and diagnostics).
    kv_len_prev_list: List[int] = None
    kv_len_next_list: List[int] = None
    actual_seq_q_prev_list: List[int] = None
    actual_seq_q_next_list: List[int] = None

    # Aggregate sum of extend_seq_lens across the batch.
    total_seq_lens: int = 0
    bs: int = 1



def _deprecated_platform_cp():
    raise ValueError(
        "Prefill CP on HIP/NPU/MUSA is deprecated; CP support will be refactored soon."
    )


def is_prefill_context_parallel_enabled():
    return get_parallel().enable_prefill_context_parallel


def is_prefill_cp_in_seq_split():
    return (
        is_prefill_context_parallel_enabled()
        and get_parallel().prefill_cp_mode == "in-seq-split"
    )


def is_mla_prefill_cp_enabled() -> bool:
    return get_parallel().enable_prefill_context_parallel and uses_mla_backend()


def mla_use_prefill_cp(forward_batch, mla_enable_prefill_cp=None):
    if mla_enable_prefill_cp is None:
        mla_enable_prefill_cp = is_mla_prefill_cp_enabled()
    return (
        forward_batch.attn_cp_metadata is not None
        and mla_enable_prefill_cp
        and forward_batch.forward_mode.is_context_parallel_extend()
    )


def can_cp_split(seq_len: int, cp_size: int, forward_batch):
    # Base conditions: CP must be enabled, size > 1, and this must be a
    # CP-extend (prefill) step. The seq_len // (cp_size * 2) check ensures
    # the load-balancing split into 2 * cp_size blocks is non-degenerate.
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    cur_cp_seq_len = seq_len // (cp_size * 2)
    if not (
        cur_cp_seq_len != 0
        and cp_size > 1
        # prepare_context_parallel_metadata hard-codes bs_per_cp_group = 1;
        # guard explicitly to avoid silent mis-partitioning under continuous batching.
        and forward_batch.forward_mode.is_context_parallel_extend()
        # is_context_parallel_extend() returns True for MIXED (prefill+decode
        # in one step), but the zigzag split only makes sense on pure extend.
        and forward_batch.forward_mode != ForwardMode.MIXED
        and is_prefill_context_parallel_enabled()
    ):
        return False

    # Per-sequence guards for bs > 1. Every sequence must be long enough for
    # the 2*cp_size-way split. A sub-threshold request reaching this point
    # means the scheduler failed to filter it out and a silent non-CP
    # fallback would have masked the bug -- raise instead. Per-sequence
    # radix-cache prefix is supported: prefix is baked into kv_len_prev/next
    # via prefix_offsets[s] inside prepare_context_parallel_metadata.
    extend_lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
    if extend_lens is None:
        return True

    cp_min = cp_size * 2
    for L in extend_lens:
        if L < cp_min:
            # A sub-threshold request cannot be zigzag-split into 2*cp_size
            # blocks; fall back to a normal (non-CP) prefill for this batch
            # instead of failing. Happens e.g. when a radix-cache prefix hit
            # leaves only a few unique extend tokens.
            return False

    return True


def _get_strategy_for_metadata(forward_batch):
    """Return the CP strategy only for metadata created by the strategy API.

    DSV4 CP-v1 still builds the legacy ``ContextParallelMetadata`` below.  It
    intentionally does not carry the CP-v2 logical/physical padding contract,
    so passing it to a target-branch strategy would mix the two protocols.
    """
    from sglang.srt.layers.cp.base import (
        BaseContextParallelMetadata,
        get_cp_strategy,
    )

    metadata = getattr(forward_batch, "attn_cp_metadata", None)
    if not isinstance(metadata, BaseContextParallelMetadata):
        return None
    return get_cp_strategy()


def cp_split_and_rebuild_data(forward_batch, input_: torch.Tensor):
    from sglang.srt.layers.attention.dsa.utils import (
        dsa_cp_round_robin_split_data,
        is_dsa_prefill_cp_round_robin_split,
    )

    strategy = _get_strategy_for_metadata(forward_batch)
    if strategy is not None:
        return strategy.shard_hidden_states(input_, forward_batch)

    if is_dsa_prefill_cp_round_robin_split():
        cp_size = get_parallel().attn_cp_size
        assert input_.shape[0] % cp_size == 0, (
            f"Expect input shape 0 can divided by cp size, but got input shape {input_.shape}, cp size {cp_size}"
        )
        return dsa_cp_round_robin_split_data(input_)

    input_list = list(
        torch.split(input_, forward_batch.attn_cp_metadata.split_list, dim=0)
    )
    result = torch.cat(
        [input_list[i] for i in forward_batch.attn_cp_metadata.zigzag_index], dim=0
    )
    return result


def cp_all_gather_reorganized_into_tensor(input_tensor, cp_size, forward_batch, stream):
    """
    Allgather communication for context_parallel(kv_cache, index_k, hidden_states).
    Handles tensors with arbitrary trailing dimensions, including DSV4 mHC
    hidden states shaped as [num_tokens, hc_mult, hidden_size].
    This implementation mainly consists of three parts:
    Step 1, padding the input shape to unify the shape for allgather communication (the shape must be the same).
    Step 2, synchronized allgather communication.
    Step 3, removing the padding and reassembling the data according to the actual tokens.
    """
    max_len = forward_batch.attn_cp_metadata.max_rank_len[0]
    pad_size = max_len - input_tensor.shape[0]
    if pad_size > 0:
        padding = [0, 0] * (input_tensor.ndim - 1) + [0, pad_size]
        input_tensor = F.pad(input_tensor, padding, mode="constant", value=0)
    group = get_parallel().attn_cp_group
    with use_symmetric_memory(group, disabled=not is_allocation_symmetric()):
        input_tensor_full = torch.empty(
            max_len * cp_size,
            *input_tensor.shape[1:],
            device=input_tensor.device,
            dtype=input_tensor.dtype,
        )

    group.all_gather_into_tensor(input_tensor_full, input_tensor)

    outputs_list_max = list(
        torch.split(
            input_tensor_full, forward_batch.attn_cp_metadata.max_rank_len, dim=0
        )
    )
    outputs = torch.cat(
        [
            outputs_list_max[index][:per_rank_len]
            for index, per_rank_len in enumerate(
                forward_batch.attn_cp_metadata.per_rank_actual_token
            )
        ],
        dim=0,
    )

    return outputs


def cp_all_gather_reorganized_into_tensor_kv_cache(
    input_tensor, cp_size, forward_batch, stream
):
    """
    Allgather communication for context_parallel KV cache.
    Handles multi-dimensional tensors (e.g., [seq_len, num_heads, head_dim]).
    """
    max_len = forward_batch.attn_cp_metadata.max_rank_len[0]
    pad_size = max_len - input_tensor.shape[0]
    if pad_size > 0:
        # Pad the first dimension (seq_len). F.pad expects padding in reverse dimension order.
        # For n dimensional tensor, we need 2*n values: (last_dim_left, last_dim_right, ..., first_dim_left, first_dim_right)
        # To pad only the first dimension: [0, 0] * (ndim - 1) + [0, pad_size]
        padding = [0, 0] * (input_tensor.ndim - 1) + [0, pad_size]
        input_tensor = F.pad(input_tensor, padding, mode="constant", value=0)

    # Create output tensor with proper shape for all dimensions
    group = get_parallel().attn_cp_group
    with use_symmetric_memory(group, disabled=not is_allocation_symmetric()):
        input_tensor_full = torch.empty(
            max_len * cp_size,
            *input_tensor.shape[1:],
            device=input_tensor.device,
            dtype=input_tensor.dtype,
        )

    group.all_gather_into_tensor(input_tensor_full, input_tensor)

    outputs_list_max = list(
        torch.split(
            input_tensor_full, forward_batch.attn_cp_metadata.max_rank_len, dim=0
        )
    )
    outputs = torch.cat(
        [
            outputs_list_max[index][:per_rank_len]
            for index, per_rank_len in enumerate(
                forward_batch.attn_cp_metadata.per_rank_actual_token
            )
        ],
        dim=0,
    )

    return outputs


def cp_all_gather_rerange_output(input_tensor, cp_size, forward_batch, stream):
    """
    # for in-seq-split
    |   +-----------before allgather------------+|
    |   | dp_atten_tp0: block0, block7 |
    |   | dp_atten_tp1: block1, block6 |
    |   | dp_atten_tp2: block2, block5 |
    |   | dp_atten_tp3: block3, block4 |
    |
    |   +----------before rerange---------------+|
    | block0 | block7 | block1 | block6 | block2 | block5 | block3 | block4 |
    |
    |   +--------------result-------------------+
    | block0 | block1 | block2 | block3 | block4 | block5 | block6 | block7 |
    |   +-------------------------+

    # for round-robin-split
    |   +-----------before allgather------------+|
    | dp_atten_tp0: token0, token4, token8, token12, token16, ... |
    | dp_atten_tp1: token1, token5, token9, token13, token17, ... |
    | dp_atten_tp2: token2, token6, token10, token14, token18, ... |
    | dp_atten_tp3: token3, token7, token11, token15, token19, ... |
    |
    |   +--------------result-------------------+
    | token0, token1, token2, token3, token4, token5, token6, token7, ...
    |   +-------------------------+
    """
    from sglang.srt.layers.attention.dsa.utils import (
        is_dsa_prefill_cp_round_robin_split,
    )

    strategy = _get_strategy_for_metadata(forward_batch)
    if strategy is not None:
        return strategy.gather_hidden_states(input_tensor, forward_batch, stream)

    if is_dsa_prefill_cp_round_robin_split():
        with use_symmetric_memory(
            get_parallel().attn_cp_group, disabled=not is_allocation_symmetric()
        ):
            output_tensor = input_tensor.new_empty(
                (input_tensor.shape[0] * cp_size, *input_tensor.shape[1:]),
            )
        attn_cp_all_gather_into_tensor(
            output_tensor,
            input_tensor,
        )
        out_shape = output_tensor.shape
        output_tensor = (
            output_tensor.view(cp_size, -1, *out_shape[1:])
            .transpose(0, 1)
            .reshape(out_shape)
        )
        return output_tensor

    # TODO: Do we need to remove the padding here?
    output_tensor = cp_all_gather_reorganized_into_tensor(
        input_tensor,
        cp_size,
        forward_batch,
        stream,
    )
    outputs_list = list(
        torch.split(
            output_tensor, forward_batch.attn_cp_metadata.reverse_split_len, dim=0
        )
    )
    output_tensor = torch.cat(
        [outputs_list[i] for i in forward_batch.attn_cp_metadata.cp_reverse_index],
        dim=0,
    )
    return output_tensor


def cp_all_gather_rerange_kv_cache(input_tensor, cp_size, forward_batch, stream):
    """
    Allgather and reorganize KV cache from all ranks in context parallel group.

    # for in-seq-split
    |   +-----------before allgather------------+|
    |   | dp_atten_tp0: block0, block7 |
    |   | dp_atten_tp1: block1, block6 |
    |   | dp_atten_tp2: block2, block5 |
    |   | dp_atten_tp3: block3, block4 |
    |
    |   +----------before rerange---------------+|
    | block0 | block7 | block1 | block6 | block2 | block5 | block3 | block4 |
    |
    |   +--------------result-------------------+
    | block0 | block1 | block2 | block3 | block4 | block5 | block6 | block7 |
    |   +-------------------------+
    """
    strategy = _get_strategy_for_metadata(forward_batch)
    if strategy is not None:
        return strategy.gather_kv_cache(input_tensor, forward_batch, stream)

    output_tensor = cp_all_gather_reorganized_into_tensor_kv_cache(
        input_tensor,
        cp_size,
        forward_batch,
        stream,
    )
    outputs_list = list(
        torch.split(
            output_tensor, forward_batch.attn_cp_metadata.reverse_split_len, dim=0
        )
    )
    output_tensor = torch.cat(
        [outputs_list[i] for i in forward_batch.attn_cp_metadata.cp_reverse_index],
        dim=0,
    )
    # No need to reshape - output_tensor already has the correct shape [seq_len, ...]
    return output_tensor


def cp_allgather_and_save_kv_cache(forward_batch, layer, k, v, cp_size, swa_loc=None):
    _deprecated_platform_cp()
