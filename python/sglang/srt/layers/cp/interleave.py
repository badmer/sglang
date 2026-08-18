# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Interleave context parallel strategy shell.

For ``cp_size = 4``, each rank owns every fourth token:

    cp0: token0, token4, token8,  token12, token16, ...
    cp1: token1, token5, token9,  token13, token17, ...
    cp2: token2, token6, token10, token14, token18, ...
    cp3: token3, token7, token11, token15, token19, ...

After all-gather, tokens are restored to the original order:

    token0, token1, token2, token3, token4, token5, token6, token7, ...
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, List, Optional

import torch
import torch.nn.functional as F

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.cp.base import (
    BaseContextParallelMetadata,
    ContextParallelStrategy,
    ContextParallelStrategyKind,
    CPAttentionBackendKind,
)
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
from sglang.srt.runtime_context import get_parallel


@dataclass
class InterleaveContextParallelMetadata(BaseContextParallelMetadata):
    per_rank_actual_token: Optional[List[int]] = None
    max_rank_len: Optional[List[int]] = None


class InterleaveCPStrategy(ContextParallelStrategy):
    name = "interleave"
    kind = ContextParallelStrategyKind.INTERLEAVE

    def can_apply(self, num_tokens: int, forward_batch) -> bool:
        if self.cp_size <= 1 or num_tokens < self.cp_size:
            return False
        forward_mode = getattr(forward_batch, "forward_mode", None)
        return forward_mode is None or forward_mode.is_context_parallel_extend()

    def build_metadata(
        self,
        num_tokens: int,
        seqs_len: Optional[List[int]],
        extend_seqs_len: Optional[List[int]] = None,
    ) -> InterleaveContextParallelMetadata:
        total = int(num_tokens)
        per_rank_actual_token = [
            total // self.cp_size + int(total % self.cp_size > rank)
            for rank in range(self.cp_size)
        ]
        max_rank_len = [max(per_rank_actual_token, default=0)] * self.cp_size
        return InterleaveContextParallelMetadata(
            total_seq_lens=total,
            bs=len(extend_seqs_len or seqs_len or [num_tokens]),
            per_rank_actual_token=per_rank_actual_token,
            max_rank_len=max_rank_len,
        )

    def shard_hidden_states(self, x: Any, forward_batch) -> Any:
        return x[self.cp_rank :: self.cp_size].contiguous()

    def shard_position_ids(self, positions: Any, forward_batch) -> Any:
        return positions[self.cp_rank :: self.cp_size].contiguous()

    def gather_hidden_states(
        self, x: Any, forward_batch, stream: Optional[Any] = None
    ) -> Any:
        return self._all_gather_interleaved(x, forward_batch, stream)

    def gather_kv_cache(
        self, x: Any, forward_batch, stream: Optional[Any] = None
    ) -> Any:
        return self._all_gather_interleaved(x, forward_batch, stream)

    def local_q_indices(self, num_tokens: int, forward_batch) -> Any:
        device = getattr(getattr(forward_batch, "input_ids", None), "device", None)
        if device is None:
            device = torch.device("cpu")
        return torch.arange(
            self.cp_rank, int(num_tokens), self.cp_size, device=device, dtype=torch.long
        )

    def run_attention(
        self,
        q: Any,
        forward_batch,
        device: Any,
        attn_fn,
        attention_backend: CPAttentionBackendKind = CPAttentionBackendKind.FLASH_ATTENTION,
    ) -> Any:
        raise NotImplementedError(
            "Interleave attention dispatch will land in a follow-up PR"
        )

    def materialize_full_kv(
        self, forward_batch, layer: Any, k: Any, v: Any, swa_loc: Optional[Any] = None
    ) -> None:
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )
        key_cache_full = self.gather_kv_cache(k.contiguous(), forward_batch)
        value_cache_full = self.gather_kv_cache(v.contiguous(), forward_batch)
        get_token_to_kv_pool().set_kv_buffer(
            layer,
            KVWriteLoc(cache_loc, swa_loc),
            key_cache_full,
            value_cache_full,
            layer.k_scale,
            layer.v_scale,
        )

    def _all_gather_interleaved(self, x: torch.Tensor, forward_batch, stream):
        meta = forward_batch.attn_cp_metadata
        max_len = meta.max_rank_len[0]
        pad_size = max_len - x.shape[0]
        if pad_size > 0:
            padding = [0, 0] * (x.ndim - 1) + [0, pad_size]
            x = F.pad(x, padding, mode="constant", value=0)

        group = get_parallel().attn_cp_group
        ctx = (
            use_symmetric_memory(group, disabled=not is_allocation_symmetric())
            if x.is_cuda
            else nullcontext()
        )
        with ctx:
            gathered = torch.empty(
                max_len * self.cp_size,
                *x.shape[1:],
                device=x.device,
                dtype=x.dtype,
            )
        group.all_gather_into_tensor(gathered, x)

        total = int(meta.total_seq_lens)
        if total == 0:
            return gathered[:0]

        per_rank = meta.per_rank_actual_token
        if all(int(n) == int(per_rank[0]) for n in per_rank):
            # Equal per-rank lengths: one interleave copy restores the
            # original token order (out[i*cp + r] = rank r's i-th token).
            return (
                gathered.view(self.cp_size, int(per_rank[0]), *x.shape[1:])
                .transpose(0, 1)
                .reshape(int(per_rank[0]) * self.cp_size, *x.shape[1:])
            )

        chunks = torch.split(gathered, meta.max_rank_len, dim=0)
        trimmed = [
            chunks[rank][:per_rank_len]
            for rank, per_rank_len in enumerate(per_rank)
        ]
        if not trimmed:
            return x.new_empty((0, *x.shape[1:]))
        flat = torch.cat(trimmed, dim=0)

        logical = torch.arange(total, device=x.device, dtype=torch.long)
        rank_ids = logical % self.cp_size
        rank_offsets = logical // self.cp_size
        prefix = [0]
        for n in per_rank[:-1]:
            prefix.append(prefix[-1] + int(n))
        prefix_tensor = torch.tensor(prefix, device=x.device, dtype=torch.long)
        source = prefix_tensor[rank_ids] + rank_offsets
        return flat.index_select(0, source)
