"""Layer-sharded DSV4 KV pool for NPU prefill context parallelism.

``LayerSplitDSV4NPUTokenToKVPool`` splits the DeepSeek-V4 NPU KV/indexer cache
layers across context-parallel (CP) ranks so each rank only materializes the
layers it owns, cutting per-rank KV memory on PD prefill workers. When a rank
reads a layer owned by another CP rank, the CP group runs a chunked all-gather
into a per-family remote scratch buffer: the owner stages each chunk into a
fresh staging tensor (never pool storage) and every rank copies the owner's
slot. All-gather is the only collective that delivers correct payload bytes on
the ZBAL-interposed torch.distributed of this stack; broadcast corrupts them.

Compress-state pools are not sharded: the compressor runs on every rank for
every layer (its internal CP all-gather requires the whole group), so every
rank holds identical per-layer state and only the KV/indexer buffers are
sharded.

Layer split is enabled only for DSV4 PD prefill workers under prefill-CP (see
``sglang.srt.layers.cp.utils.is_glm_dsa_cache_layer_split_enabled``).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.dsv4.dsv4_layer_split_plan import (
    DSV4LayerShardPlan,
)
from sglang.srt.hardware_backend.npu.dsv4.dsv4_memory_pool import (
    DeepSeekV4SingleKVPool,
    DSV4NPUTokenToKVPool,
    NPUDeepSeekV4IndexerPool,
    NPUDeepSeekV4SingleKVPool,
)
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)


def _num_pages(size: int, page_size: int) -> int:
    """Physical pages covering ``size`` tokens at ``page_size`` tokens/page."""
    return (size + page_size + 1) // page_size


def _probe_stats(tensor: torch.Tensor) -> Tuple[str, int]:
    """``(checksum, NaN count)`` of ``tensor``; non-finite sums report verbatim."""
    flat = tensor.reshape(-1)
    if not flat.numel():
        return "na", 0
    values = flat.float()
    nan = int(torch.isnan(values).sum().item())
    total = values.sum().item()
    checksum = str(int(total)) if math.isfinite(total) else repr(total)
    return checksum, nan


_LS_DEBUG = envs.SGLANG_DSV4_LS_DEBUG.get()
# The compressor can write on a stream the current-stream sync does not order
# against; only a device-wide sync makes the probe's pre/post reads ground
# truth across streams.
_LS_SYNC = envs.SGLANG_DSV4_LS_SYNC.get()
# Owner reads stage chunks of this many bytes through a fresh staging tensor
# so no collective operand is pool-resident multi-MB memory (ZBAL/VMM), the
# operand class that corrupts on this stack.
_LS_CHUNK_BYTES = envs.SGLANG_DSV4_LS_CHUNK_BYTES.get()
assert _LS_CHUNK_BYTES > 0, "SGLANG_DSV4_LS_CHUNK_BYTES must be positive"
# Async prefix prefetch above this many pages falls back to the reader's sync
# transfer: a large radix prefix makes every layer enqueue a multi-chunk
# prefix all-gather whose queue depth on the comm stream hangs the EP group
# (observed 432 pages x 14 chunks per layer).
_PREFIX_ASYNC_MAX_PAGES = 64
# Unconsumed prefix launches allowed in flight before further launches are
# skipped (same fallback): bounds the comm-stream queue depth.
_PREFIX_MAX_INFLIGHT = 2 * 5
_LS_DEBUG_VERBOSE = False
_LS_FINGERPRINT_BLOCKS = 16
# marks a (family, layer) whose async transfer was never launched
_NOT_LAUNCHED = object()


def _fingerprint_equal(lo: torch.Tensor, hi: torch.Tensor) -> bool:
    """NaN-safe exact comparison: identical buffers with NaNs must compare equal."""
    return bool(torch.equal(torch.nan_to_num(lo), torch.nan_to_num(hi)))


class LayerSplitNPUDeepSeekV4SingleKVPool(NPUDeepSeekV4SingleKVPool):
    """NPU bf16 KV pool allocating full buffers for owned layers only.

    Non-owned layers get 0-row placeholders so ``kv_buffer`` stays index
    aligned; their content is materialized on read via the parent pool's
    owner all-gather scratch buffer.
    """

    def __init__(self, *args, layer_owned_fn: Callable[[int], bool], **kwargs):
        self._layer_owned_fn = layer_owned_fn
        super().__init__(*args, **kwargs)

    def _num_pages_for(self, local_layer_idx: int) -> int:
        full_pages = _num_pages(self.size, self.kernel_page_size)
        return full_pages if self._layer_owned_fn(local_layer_idx) else 0

    def create_buffer(self, *, num_pages: int):
        # The NPU base derives the page count from self.size; honor num_pages
        # so non-owned layers are 0-row, not narrowed full storage.
        assert self.store_dtype == torch.bfloat16, (
            "LayerSplitDSV4NPUTokenToKVPool requires a bf16 KV cache; other "
            "store dtypes fall back to a layout that cannot express 0-row "
            f"non-owned layers (got {self.store_dtype})."
        )
        kv_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.kv_cache_total_dim = kv_dim
        return torch.zeros(
            num_pages,
            self.kernel_page_size,
            1,
            kv_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.kv_buffer = [
                self.create_buffer(num_pages=self._num_pages_for(i))
                for i in range(self.layer_num)
            ]


class LayerSplitNPUDeepSeekV4IndexerPool(NPUDeepSeekV4IndexerPool):
    """NPU c4-indexer pool allocating owned layers only.

    The packed CUDA ``index_k_with_scale_buffer`` (NSA compat, unused for NPU
    reads) and the dedicated int8 K / fp16 scale buffers all follow ownership.
    """

    def __init__(self, *args, layer_owned_fn: Callable[[int], bool], **kwargs):
        self._layer_owned_fn = layer_owned_fn
        super().__init__(*args, **kwargs)

    def _owned_pages(self, local_layer_idx: int) -> int:
        full_pages = _num_pages(self.size, self._kernel_page_size)
        return full_pages if self._layer_owned_fn(local_layer_idx) else 0

    def _create_buffer(self):
        kp = self._kernel_page_size
        page_bytes = self.page_size * self.get_bytes_per_token()
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.index_k_with_scale_buffer = [
                torch.zeros(
                    self._owned_pages(i),
                    page_bytes,
                    dtype=self.index_k_with_scale_buffer_dtype,
                    device=self.device,
                )
                for i in range(self.layer_num)
            ]
            self.index_k_buffer = [
                torch.zeros(
                    self._owned_pages(i), kp, 1, self.index_head_dim,
                    dtype=torch.int8, device=self.device,
                )
                for i in range(self.layer_num)
            ]
            self.index_scale_buffer = [
                torch.zeros(
                    self._owned_pages(i), kp, 1, 1,
                    dtype=torch.float16, device=self.device,
                )
                for i in range(self.layer_num)
            ]


# Buffer families served through owner all-gather scratch copies. index_k and
# index_scale exist only on c4 layers.
_REMOTE_FAMILIES = ("swa", "c4", "c128", "index_k", "index_scale")


class LayerSplitDSV4NPUTokenToKVPool(DSV4NPUTokenToKVPool):
    """DSV4 NPU KV pool that shards layers across CP ranks.

    Reads of non-owned layers are served from a per-family remote scratch
    buffer filled by an owner all-gather; writes only land on owner ranks and
    invalidate the local scratch copy so the next read re-gathers.
    """

    def __init__(self, *args, layer_shard_rank: int, layer_shard_size: int, **kwargs):
        assert (
            layer_shard_rank is not None and layer_shard_size > 1
        ), "LayerSplitDSV4NPUTokenToKVPool requires layer_shard_size > 1"
        self.layer_shard_rank = layer_shard_rank
        self.layer_shard_size = layer_shard_size
        self.layer_shard_enabled = True
        # Built on the first _make_kv_pool call inside super().__init__: the
        # plan needs layer_num / ratios / stage range the base sets before it.
        self._shard_plan: Optional[DSV4LayerShardPlan] = None
        super().__init__(*args, **kwargs)
        assert (
            not self._unified_kv
        ), "Layer split does not support the unified-KV layout yet"
        self._init_remote_buffers()
        plan = self._get_shard_plan()
        # Global (absolute) layer range owned by this rank, read by the PD
        # bootstrap (disaggregation/prefill.py) to advertise the shard window.
        self.layer_shard_start = plan.shard_start
        self.layer_shard_end = plan.shard_end
        logger.info(
            "DSV4 layer shard plan (continuous): layer_num=%d, shard_size=%d, "
            "rank=%d, global=[%d,%d), owned c4 range=%s, owned c128 range=%s, "
            "partitions=%s",
            self.layer_num,
            self.layer_shard_size,
            self.layer_shard_rank,
            plan.shard_start,
            plan.shard_end,
            plan.owned_bucket_range("c4"),
            plan.owned_bucket_range("c128"),
            plan.partition_summary(),
        )

    # ---- ownership plan ----------------------------------------------------

    def _get_shard_plan(self) -> DSV4LayerShardPlan:
        if self._shard_plan is None:
            self._shard_plan = DSV4LayerShardPlan(
                rank=self.layer_shard_rank,
                shard_size=self.layer_shard_size,
                num_layers=self.layer_num,
                stage_start=self._stage_start,
                ratios=self.compression_ratios[self._stage_start : self._stage_end],
            )
        return self._shard_plan

    def _is_layer_owned(self, layer_id: int) -> bool:
        return self._get_shard_plan().is_layer_owned(layer_id)

    def _layer_owner_rank(self, layer_id: int) -> int:
        return self._get_shard_plan().owner_rank(layer_id)

    def _owned_fn_for_bucket(self, bucket: str) -> Callable[[int], bool]:
        plan = self._get_shard_plan()
        ids = plan.bucket_layer_ids(bucket)
        return lambda local_idx: plan.is_stage_local_owned(ids[local_idx])

    # ---- sub-pool factories ------------------------------------------------

    def _make_kv_pool(
        self,
        *,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        global_page_size: int,
        cls: type = DeepSeekV4SingleKVPool,
    ) -> LayerSplitNPUDeepSeekV4SingleKVPool:
        assert cls is DeepSeekV4SingleKVPool, (
            "enable_hisparse is incompatible with --enable-dsa-cache-layer-split "
            f"(got c4 pool class {cls.__name__})."
        )
        # Full/SWA use the global page size, C4 its native page, C128 its own;
        # mirrors the NPU base _make_kv_pool.
        if page_size * 4 == global_page_size:
            bucket, kernel_page_size = "c4", page_size
        elif page_size * 128 == global_page_size:
            bucket, kernel_page_size = "c128", self.c128_page_size
        else:
            bucket, kernel_page_size = "swa", global_page_size
        return LayerSplitNPUDeepSeekV4SingleKVPool(
            size,
            page_size,
            dtype,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            kernel_page_size=kernel_page_size,
            layer_owned_fn=self._owned_fn_for_bucket(bucket),
        )

    def _make_indexer_pool(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        index_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
    ) -> LayerSplitNPUDeepSeekV4IndexerPool:
        # Indexer shares C4 addresses and therefore uses the same native page.
        return LayerSplitNPUDeepSeekV4IndexerPool(
            size,
            page_size,
            dtype,
            index_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            kernel_page_size=page_size,
            layer_owned_fn=self._owned_fn_for_bucket("c4"),
        )

    # ---- remote scratch + owner all-gather --------------------------------

    def _init_remote_buffers(self) -> None:
        # One full-layer scratch per family, allocated on every rank: a rank
        # with no owned layer of a bucket still receives its gathers.
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            device = self.device
            kv_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
            dtype = self.swa_kv_pool.store_dtype

            def scratch(sub_pool, page_size: int, rows: int, cols: int, dt: torch.dtype):
                return torch.zeros(
                    _num_pages(sub_pool.size, page_size),
                    page_size,
                    rows,
                    cols,
                    dtype=dt,
                    device=device,
                )

            indexer = self.c4_indexer_kv_pool
            self._remote_buffers = {
                "swa": scratch(
                    self.swa_kv_pool, self.swa_kv_pool.kernel_page_size, 1, kv_dim, dtype
                ),
                "c4": scratch(
                    self.c4_kv_pool, self.c4_kv_pool.kernel_page_size, 1, kv_dim, dtype
                ),
                "c128": scratch(
                    self.c128_kv_pool,
                    self.c128_kv_pool.kernel_page_size,
                    1,
                    kv_dim,
                    dtype,
                ),
                "index_k": scratch(
                    indexer, indexer._kernel_page_size, 1, self.indexer_head_dim, torch.int8
                ),
                "index_scale": scratch(
                    indexer, indexer._kernel_page_size, 1, 1, torch.float16
                ),
            }
        # family -> layer_id of the last materialized remote copy.
        self._remote_layer_cache: Dict[str, Optional[int]] = {
            family: None for family in _REMOTE_FAMILIES
        }
        # Active-page plan of the current forward (set by the attention
        # backend once per forward), split by write stability:
        # - prefix pages were written by earlier chunks: stable for the whole
        #   forward, prefetched up front (prefetch_prefix)
        # - fresh pages are written by this forward's per-layer stores: they
        #   can only transfer after each layer's write (prefetch_read)
        self._staging_prefix: Dict[str, torch.Tensor] = {}
        self._staging_fresh: Dict[str, torch.Tensor] = {}
        self._staging_selected: Dict[str, torch.Tensor] = {}
        self._staging_plan: Optional[Dict[str, torch.Tensor]] = None
        # Async remote reads: family -> (layer_id, event, keepalive) for the
        # fresh transfer launched on the comm stream and not yet consumed;
        # prefix transfers are tracked per (family, layer_id) in
        # _prefix_events (event or None when the portion was empty).
        # (family, layer) -> launch state: an Event once launched, None when
        # the portion was empty, False when backpressure/threshold skipped it
        # (the reader then sync-transfers those rows), absent = hook never ran.
        self._prefix_events: Dict[Tuple[str, int], Any] = {}
        self._prefix_inflight = 0
        self._fresh_pending: Dict[str, Optional[Tuple[int, Any, tuple]]] = {
            family: None for family in _REMOTE_FAMILIES
        }
        self._prefetch_enabled = envs.SGLANG_DSV4_LS_PREFETCH.get()
        self._comm_stream: Optional[torch.npu.Stream] = None
        self._require_prefetched_reads = False
        # Shared chunk-staging operand; int8 is the smallest family dtype, so
        # byte capacity == element capacity.
        self._staging = torch.zeros(
            _LS_CHUNK_BYTES, dtype=torch.int8, device=self.device
        )
        # Read issue counter for the symmetric-participation probe
        # (SGLANG_DSV4_LS_DEBUG=1): CP partners must log identical sequences.
        self._read_seq = 0
        self._read_mismatch = 0
        self._self_test_pending = _LS_DEBUG

    # index_k / index_scale share the c4 page-id space and the c4 plan.

    def begin_forward_staging(
        self,
        tables: Dict[str, Optional[torch.Tensor]],
        prefix_lens: Optional[List[int]] = None,
    ) -> None:
        """Record the forward's active pages per family (CP-group union),
        split into prefix/fresh portions by each request's cached-prefix
        length.

        Called once per prefill forward by the attention backend with the
        per-family page tables; both CP ranks call symmetrically. Prefix
        pages were written by earlier chunks (stable all forward) and are
        prefetched up front; fresh pages are written by this forward's
        per-layer stores and transfer only after each layer's write. Reads of
        non-owned layers scatter the pages at their original page ids, so
        consumer page tables need no remap; pages outside the union are never
        addressed by this forward's attention.
        """
        spans = {
            "swa": self.swa_kv_pool.kernel_page_size,
            "c4": self.c4_kv_pool.kernel_page_size * 4,
            "c128": self.c128_kv_pool.kernel_page_size * 128,
        }
        group = get_parallel().attn_cp_group
        for family, table in tables.items():
            remote = self._remote_buffers.get(family)
            if remote is None or table is None or table.numel() == 0:
                continue
            max_pages = remote.shape[0]
            span = spans.get(family)
            flat = table.reshape(-1).to(torch.long)
            in_range = (flat >= 0) & (flat < max_pages)
            if prefix_lens is not None and span is not None:
                lens = torch.as_tensor(prefix_lens, dtype=torch.long, device=flat.device)
                prefix_pages_per_row = (lens + span - 1) // span
                is_prefix = (
                    torch.arange(table.shape[1], device=flat.device)[None, :]
                    < prefix_pages_per_row[:, None]
                ).reshape(-1)
            else:
                is_prefix = torch.zeros_like(flat, dtype=torch.bool)
            masks = []
            for want_prefix in (True, False):
                pick = is_prefix if want_prefix else ~is_prefix
                mask = torch.zeros(max_pages, dtype=torch.int32, device=flat.device)
                hit = in_range & pick
                mask.index_add_(
                    0,
                    flat[hit],
                    torch.ones(int(hit.sum()), dtype=torch.int32, device=flat.device),
                )
                masks.append(mask)
            stacked = torch.stack(masks)
            torch.distributed.all_reduce(stacked, group=group.device_group)
            self._staging_prefix[family] = torch.nonzero(
                stacked[0], as_tuple=False
            ).flatten()
            self._staging_fresh[family] = torch.nonzero(
                stacked[1], as_tuple=False
            ).flatten()
            self._staging_selected[family] = torch.nonzero(
                stacked[0] + stacked[1], as_tuple=False
            ).flatten()
        # index_k / index_scale share the c4 page-id space: mirror the c4
        # plans under their own names so every family resolves directly
        for alias, target in (("index_k", "c4"), ("index_scale", "c4")):
            if target in self._staging_prefix:
                self._staging_prefix[alias] = self._staging_prefix[target]
                self._staging_fresh[alias] = self._staging_fresh[target]
                self._staging_selected[alias] = self._staging_selected[target]
        # per-forward state: prior entries (stale layer/plan keys) are dropped.
        # A remote-read cache hit from the previous forward would suppress this
        # forward's transfers while the plan above has moved on — reset it too.
        self._prefix_events = {}
        self._prefix_inflight = 0
        self._remote_layer_cache = {family: None for family in _REMOTE_FAMILIES}
        self._staging_plan = tables
        if _LS_DEBUG:
            if prefix_lens:
                logger.warning(
                    "LSPLAN-PLEN rank=%d prefix_lens=%s",
                    self.layer_shard_rank,
                    prefix_lens,
                )
            for family in self._staging_selected:
                pre = self._staging_prefix[family]
                fresh = self._staging_fresh[family]
                logger.warning(
                    "LSPLAN rank=%d family=%s pages=%d (prefix=%d fresh=%d)/%d",
                    self.layer_shard_rank,
                    family,
                    int(pre.numel() + fresh.numel()),
                    int(pre.numel()),
                    int(fresh.numel()),
                    self._remote_buffers[family].shape[0],
                )
            if not self._staging_selected:
                logger.warning("LSPLAN rank=%d empty plan", self.layer_shard_rank)

    def _self_test_delivery(self) -> None:
        """One-shot delivery check before the first read, exercising the same
        chunked all-gather the production path uses: the shard's first rank
        fills a pattern, all ranks gather, and the fingerprints must match."""
        group = get_parallel().attn_cp_group
        scratch = self._remote_buffers["swa"]
        elem = scratch.element_size()
        probe_elems = min(
            scratch.numel(), _LS_CHUNK_BYTES // elem
        )
        if self.layer_shard_rank == 0:
            scratch.reshape(-1)[:probe_elems].copy_(
                torch.arange(
                    1, probe_elems + 1, dtype=torch.float32, device=self.device
                ).to(scratch.dtype)
            )
        torch.npu.synchronize()
        send = self._staging[: probe_elems * elem].view(scratch.dtype)
        send.copy_(scratch.reshape(-1)[:probe_elems])
        gathered = torch.empty(
            (group.world_size, probe_elems), dtype=scratch.dtype, device=scratch.device
        )
        torch.distributed.all_gather_into_tensor(
            gathered, send, group=group.device_group
        )
        values = gathered[0].float()
        stats = torch.stack([values.sum(), values.isnan().sum().float()])
        lo = stats.clone()
        hi = stats.clone()
        torch.distributed.all_reduce(
            lo, op=torch.distributed.ReduceOp.MIN, group=group.device_group
        )
        torch.distributed.all_reduce(
            hi, op=torch.distributed.ReduceOp.MAX, group=group.device_group
        )
        logger.warning(
            "LSSELF rank=%d equal=%d sum=%.6g nan=%.0f",
            self.layer_shard_rank,
            int(_fingerprint_equal(lo, hi)),
            stats[0].item(),
            stats[1].item(),
        )
        scratch.zero_()
        torch.npu.synchronize()

    def _read_layer_buffer(self, family: str, layer_id: int) -> torch.Tensor:
        """This rank's buffer for ``family``/``layer_id``, remote layers included.

        The scratch holds one layer per family, so the result must be consumed
        before the next read overwrites it; the collective is issued on the
        current stream, so it orders after the owner's staging writes. All
        ranks issue these in the same order (a skipped call hangs the group)
        and invalidation rides the write calls every rank issues.
        """
        local = self._local_family_buffer(family, layer_id)
        remote = self._remote_buffers[family]
        if self._remote_layer_cache[family] == layer_id:
            if _LS_DEBUG and family in ("c4", "c128"):
                logger.warning(
                    "LSDBG-HIT rank=%d %s:%d", self.layer_shard_rank, family, layer_id
                )
            return local if local is not None else remote

        if self._self_test_pending:
            # Deferred to the first read: at pool-init time the ZBAL process
            # -group adaptor is not initialized yet and the collective raises
            # "Check failed: init" (zbal_pytorch_process_group.cpp).
            self._self_test_pending = False
            try:
                self._self_test_delivery()
            except Exception as exc:  # noqa: BLE001
                logger.warning("LSSELF failed: %s", exc)
        target = local if local is not None else remote
        stream = torch.npu.current_stream()
        covered = False
        selected = None
        if self._staging_plan is not None:
            # Consume this layer's async transfers: the prefix portion
            # (launched up front / by the previous layer) and the fresh
            # portion (launched by this layer's write paths). A stale fresh
            # pending (different layer) is waited on so its scratch writes
            # cannot race the sync fallback below.
            prefix_ev = self._prefix_events.pop((family, layer_id), _NOT_LAUNCHED)
            # False = skipped by the launch-side backpressure/threshold; the
            # sync fallback below re-gathers the full union, so the prefix
            # portion counts as intentionally uncovered (never a guard raise).
            prefix_skipped = prefix_ev is False
            fresh = self._fresh_pending.get(family)
            self._fresh_pending[family] = None
            selected = self._staging_selected.get(family)
            if not prefix_skipped and (
                prefix_ev is not None and prefix_ev is not _NOT_LAUNCHED
            ):
                self._prefix_inflight -= 1
                stream.wait_event(prefix_ev)
            if fresh is not None:
                stream.wait_event(fresh[1])
                covered = fresh[0] == layer_id
            covered = covered and (
                prefix_skipped or prefix_ev is not _NOT_LAUNCHED
            )
            if (
                self._require_prefetched_reads
                and self._prefetch_enabled
                and local is None
                and not covered
                and prefix_ev is _NOT_LAUNCHED
            ):
                raise RuntimeError(
                    f"layer split: non-owned read {family}:{layer_id} without "
                    "a prefetched transfer — the async read hooks were lost"
                )
            if covered:
                self._remote_layer_cache[family] = layer_id
                self._read_seq += 1
                if _LS_DEBUG:
                    # Whole-buffer fingerprints are meaningless under staging:
                    # untouched scratch pages stay stale by design.
                    logger.warning(
                        "LSDBG-STAGING rank=%d seq=%d %s:%d pages=%d/%d",
                        self.layer_shard_rank,
                        self._read_seq,
                        family,
                        layer_id,
                        int(selected.numel()),
                        remote.shape[0],
                    )
                return target
        # attn_cp_group must have no other submitters while layer split is on:
        # the duplicate attn_cp_overlap group (bootstrap.py) owns the side
        # -stream CP collectives, and HCCL pairs this group's collectives by
        # submission order only.
        group = get_parallel().attn_cp_group
        if _LS_DEBUG and _LS_SYNC:
            torch.npu.synchronize()
        pre = _probe_stats(target) if _LS_DEBUG else None
        if selected is not None and selected.numel() < remote.shape[0]:
            self._read_via_allgather_selected(
                family, layer_id, local, remote, selected, group
            )
            self._remote_layer_cache[family] = layer_id
            self._read_seq += 1
            if _LS_DEBUG:
                # Whole-buffer fingerprints are meaningless under staging:
                # untouched scratch pages stay stale by design.
                logger.warning(
                    "LSDBG-STAGING rank=%d seq=%d %s:%d pages=%d/%d",
                    self.layer_shard_rank,
                    self._read_seq,
                    family,
                    layer_id,
                    int(selected.numel()),
                    remote.shape[0],
                )
            return target
        if _LS_DEBUG and _LS_SYNC:
            torch.npu.synchronize()
        pre = _probe_stats(target) if _LS_DEBUG else None
        self._read_via_allgather_chunks(family, layer_id, local, remote, group)
        self._remote_layer_cache[family] = layer_id
        self._read_seq += 1
        if _LS_DEBUG:
            if _LS_SYNC:
                torch.npu.synchronize()
            post = _probe_stats(target)
            # CP partners align offline by seq: on the owner, pre is the
            # compressor-written source content; on receivers, post is the
            # copy that arrived. NaN counts separate a NaN source from a
            # corrupted copy, pre/post sums separate faithfulness from loss.
            logger.warning(
                "LSDBG rank=%d seq=%d %s:%d src=%d owned=%d pre=%s/%d post=%s/%d",
                self.layer_shard_rank,
                self._read_seq,
                family,
                layer_id,
                self._layer_owner_rank(layer_id),
                int(local is not None),
                pre[0],
                pre[1],
                post[0],
                post[1],
            )
            # Device-truth check: order-insensitive fingerprints (min/max are
            # exact reductions, the NaN count is integral) over fixed buffer
            # blocks, compared via MIN/MAX all_reduce so neither host-side
            # D2H reads of VMM-mapped pages nor fp32 sum-order differences
            # between ranks can skew the verdict. Per-block results localize
            # where a partial copy stops matching. Only a 1-element boolean
            # reaches the host.
            values = target.float().reshape(-1)
            blocks = values.chunk(_LS_FINGERPRINT_BLOCKS)
            cols = []
            for block in blocks:
                cols.append(block.min())
                cols.append(block.max())
            cols.append(values.isnan().sum().float())
            stats = torch.stack(cols)
            lo = stats.clone()
            hi = stats.clone()
            torch.distributed.all_reduce(lo, op=torch.distributed.ReduceOp.MIN, group=group.device_group)
            torch.distributed.all_reduce(hi, op=torch.distributed.ReduceOp.MAX, group=group.device_group)
            equal = _fingerprint_equal(lo, hi)
            if not equal:
                bad_cols = (
                    ~torch.eq(torch.nan_to_num(lo), torch.nan_to_num(hi))
                ).nonzero().flatten().tolist()
                bad_blocks = sorted({c // 2 for c in bad_cols if c < 2 * len(blocks)})
                block_elems = max(1, blocks[0].numel())
                logger.warning(
                    "LSSEQ rank=%d seq=%d %s:%d equal=0 bad_blocks=%s/%d "
                    "block_bytes=%dKB first_bad_off=%dKB",
                    self.layer_shard_rank,
                    self._read_seq,
                    family,
                    layer_id,
                    bad_blocks[:8],
                    len(blocks),
                    block_elems * values.element_size() // 1024,
                    min(bad_blocks) * block_elems * values.element_size() // 1024,
                )
                self._read_mismatch += 1
            elif _LS_DEBUG_VERBOSE:
                logger.warning(
                    "LSSEQ rank=%d seq=%d %s:%d equal=1",
                    self.layer_shard_rank,
                    self._read_seq,
                    family,
                    layer_id,
                )
        return target

    def _read_via_allgather_chunks(
        self,
        family: str,
        layer_id: int,
        local: Optional[torch.Tensor],
        remote: torch.Tensor,
        group,
    ) -> None:
        """Materialize ``family``/``layer_id`` via chunked all-gather.

        On this stack the ZBAL-interposed broadcast corrupts the delivered
        bytes while all-gather (the compressor's per-layer KV rerange) delivers
        correctly. Every chunk is staged into the fresh staging tensor, all
        ranks gather it, and the non-owner copies the owner's slot into its
        scratch.
        """
        flat_src = local.reshape(-1) if local is not None else None
        flat_dst = remote.reshape(-1)
        elem = flat_dst.element_size()
        total = flat_dst.numel()
        step = max(1, _LS_CHUNK_BYTES // elem)
        owner_slot = self._layer_owner_rank(layer_id)
        world = group.world_size
        for start in range(0, total, step):
            end = min(start + step, total)
            width = end - start
            send = self._staging[: width * elem].view(flat_dst.dtype)
            if flat_src is not None:
                send.copy_(flat_src[start:end])
            gathered = torch.empty(
                (world, width), dtype=flat_dst.dtype, device=flat_dst.device
            )
            group.all_gather_into_tensor(gathered, send)
            if flat_src is None:
                flat_dst[start:end].copy_(gathered[owner_slot])




    def prefetch_prefix(self, layer_id: int) -> None:
        """Launch this layer's prefix-portion transfers on the comm stream.

        Prefix pages were written by earlier chunks and stay stable for the
        whole forward, so this is safe any time before the layer's reads;
        calling it at the end of the previous layer's forward overlaps the
        transfer with that layer's compute. Idempotent per forward.
        """
        if not self._prefetch_enabled or self._staging_plan is None:
            return
        if layer_id >= self.layer_num:
            return
        ratio = self.layer_mapping[layer_id].compress_ratio
        if ratio == 0:
            families = ("swa",)
        elif ratio == 128:
            families = ("swa", "c128")
        else:
            families = ("swa", "c4", "index_k", "index_scale")
        group = get_parallel().attn_cp_group
        if self._comm_stream is None:
            self._comm_stream = torch.npu.Stream()
        comm = self._comm_stream
        comm.wait_stream(torch.npu.current_stream())
        keepalive: list = []
        launched: list = []
        with torch.npu.stream(comm):
            for family in families:
                key = (family, layer_id)
                if key in self._prefix_events:
                    continue
                sel = self._staging_prefix.get(family)
                if sel is None or sel.numel() == 0:
                    self._prefix_events[key] = None
                    continue
                if (
                    sel.numel() > _PREFIX_ASYNC_MAX_PAGES
                    or self._prefix_inflight >= _PREFIX_MAX_INFLIGHT
                ):
                    # Too big / too deep to prefetch safely: the reader sync-
                    # transfers these rows instead (False = allowed fallback).
                    self._prefix_events[key] = False
                    continue
                local = self._local_family_buffer(family, layer_id)
                remote = self._remote_buffers[family]
                self._launch_allgather_selected(
                    local, remote, sel, layer_id, group, keepalive,
                    tag=f"prefix:{family}",
                )
                self._prefix_inflight += 1
                launched.append((key, len(keepalive)))
            event = torch.npu.Event()
            event.record(comm)
        for key, _ in launched:
            self._prefix_events[key] = event

    def prefetch_read(self, layer_id: int) -> None:
        """Launch this layer's fresh-portion transfers on the comm stream.

        Called from the layer's write paths: the fresh pages (this chunk's
        output pages) can only transfer after their writes land, and the
        layer's reads wait on the recorded event.
        """
        if not self._prefetch_enabled or self._staging_plan is None:
            return
        ratio = self.layer_mapping[layer_id].compress_ratio
        # every layer reads its swa-window buffer; compressed layers add
        # their sparse families on top
        if ratio == 0:
            families = ("swa",)
        elif ratio == 128:
            families = ("swa", "c128")
        else:
            families = ("swa", "c4", "index_k", "index_scale")
        group = get_parallel().attn_cp_group
        if self._comm_stream is None:
            self._comm_stream = torch.npu.Stream()
        comm = self._comm_stream
        comm.wait_stream(torch.npu.current_stream())
        keepalive: list = []
        launched: list = []
        with torch.npu.stream(comm):
            for family in families:
                if self._remote_layer_cache[family] == layer_id:
                    continue
                sel = self._staging_fresh.get(family)
                if sel is None or sel.numel() == 0:
                    continue
                if (
                    self._fresh_pending.get(family) is not None
                    and self._fresh_pending[family][0] == layer_id
                ):
                    continue
                local = self._local_family_buffer(family, layer_id)
                remote = self._remote_buffers[family]
                self._launch_allgather_selected(
                    local, remote, sel, layer_id, group, keepalive,
                    tag=f"fresh:{family}",
                )
                launched.append(family)
            event = torch.npu.Event()
            event.record(comm)
            for family in launched:
                self._fresh_pending[family] = (layer_id, event, tuple(keepalive))

    def _read_via_allgather_selected(
        self,
        family: str,
        layer_id: int,
        local: Optional[torch.Tensor],
        remote: torch.Tensor,
        selected: torch.Tensor,
        group,
    ) -> None:
        """Synchronously transfer the forward's active pages of
        ``family``/``layer_id`` on the current stream. Both CP ranks agree on
        ``selected`` (the group-wide union from ``begin_forward_staging``);
        the non-owner scatters the gathered pages at their ORIGINAL page ids,
        so consumer page tables need no remap."""
        self._launch_allgather_selected(
            local, remote, selected, layer_id, group, keepalive=[]
        )

    def _invalidate_wait_pending(self, family: str) -> None:
        """Order a pending async transfer before a write overwrites the same
        family scratch: the write must land after the transfer, not during."""
        pending = self._fresh_pending.get(family)
        if pending is not None:
            self._fresh_pending[family] = None
            torch.npu.current_stream().wait_event(pending[1])

    def prepare_forward(self, require_prefetched_reads: bool) -> None:
        """Reset the per-forward async-read requirement.

        The attention backend sets it True once the prefill-CP page tables are
        ready: from then on a non-owned read must hit a prefetched transfer
        (or the cache) — falling back to a synchronous read means the prefetch
        hooks were lost, and the stale-scratch race they prevent would be
        silently reintroduced.
        """
        self._require_prefetched_reads = require_prefetched_reads

    def _launch_allgather_selected(
        self,
        local: Optional[torch.Tensor],
        remote: torch.Tensor,
        selected: torch.Tensor,
        layer_id: int,
        group,
        keepalive: list,
        tag: str = "",
    ) -> None:
        """Enqueue the selected-page all-gather (+ receiver scatter) on the
        CURRENT stream context — call inside the target stream context."""
        if _LS_DEBUG:
            logger.warning(
                "LSLAUNCH rank=%d tag=%s layer=%d pages=%d chunks=%d",
                self.layer_shard_rank,
                tag,
                layer_id,
                int(selected.numel()),
                len(selected.split(max(1, _LS_CHUNK_BYTES // (
                    remote.reshape(remote.shape[0], -1).shape[1]
                    * remote.element_size())))),
            )
        pages = remote.shape[0]
        flat_dst = remote.reshape(pages, -1)
        row_elems = flat_dst.shape[1]
        elem = flat_dst.element_size()
        step_pages = max(1, _LS_CHUNK_BYTES // (row_elems * elem))
        owner_slot = self._layer_owner_rank(layer_id)
        flat_src = None if local is None else local.reshape(pages, -1)
        for sel in selected.split(step_pages):
            n = int(sel.numel())
            send = (
                self._staging[: n * row_elems * elem]
                .view(flat_dst.dtype)
                .view(n, row_elems)
            )
            if flat_src is not None:
                send.copy_(flat_src.index_select(0, sel))
            gathered = torch.empty(
                (group.world_size, n, row_elems),
                dtype=flat_dst.dtype,
                device=flat_dst.device,
            )
            group.all_gather_into_tensor(gathered, send)
            keepalive.append(gathered)
            if flat_src is None:
                flat_dst.index_copy_(0, sel, gathered[owner_slot])

    def _invalidate_family(self, families: Tuple[str, ...], layer_id: int) -> None:
        for family in families:
            self._invalidate_wait_pending(family)
            if self._remote_layer_cache[family] == layer_id:
                self._remote_layer_cache[family] = None

    def _local_family_buffer(
        self, family: str, layer_id: int
    ) -> Optional[torch.Tensor]:
        """This rank's own buffer for ``family``/``layer_id``, or None when the
        layer belongs to another CP rank."""
        if not self._is_layer_owned(layer_id):
            return None
        if family == "swa":
            return self.swa_kv_pool.kv_buffer[layer_id]
        item = self.layer_mapping[layer_id]
        if family == "c4":
            assert item.compress_ratio == 4
            return self.c4_kv_pool.kv_buffer[item.compress_layer_id]
        if family == "c128":
            assert item.compress_ratio == 128
            return self.c128_kv_pool.kv_buffer[item.compress_layer_id]
        if family == "index_k":
            assert item.compress_ratio == 4
            return self.c4_indexer_kv_pool.index_k_buffer[item.compress_layer_id]
        assert family == "index_scale" and item.compress_ratio == 4
        return self.c4_indexer_kv_pool.index_scale_buffer[item.compress_layer_id]

    # ---- KV reads: owner all-gather -----------------------------------------

    def get_swa_buffer(
        self, layer_id: int, loc: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        kv = self._read_layer_buffer("swa", layer_id)
        if loc is not None:
            kv = kv.flatten(0, 1)[loc]
        return kv

    def get_compress_buffer(
        self,
        layer_id: int,
        from_indexer: bool = False,
        loc: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        item = self.layer_mapping[layer_id]
        if item.compress_ratio == 0:
            return None
        if _LS_DEBUG and not from_indexer:
            logger.warning(
                "LSDBG-CALL rank=%d %s:%d",
                self.layer_shard_rank,
                "c4" if item.compress_ratio == 4 else "c128",
                layer_id,
            )
        if from_indexer:
            assert item.compress_ratio == 4, "indexer only on c4 layers"
            kv = self._read_layer_buffer("index_k", layer_id)
        elif item.compress_ratio == 4:
            kv = self._read_layer_buffer("c4", layer_id)
        else:
            kv = self._read_layer_buffer("c128", layer_id)
        if loc is not None:
            kv = kv.flatten(0, 1)[loc]
        return kv

    def get_compress_dequant_scale_buffer(
        self, layer_id: int, from_indexer: bool
    ) -> torch.Tensor:
        assert from_indexer, "only indexer compress pool has dequant scale"
        return self._read_layer_buffer("index_scale", layer_id)

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        item = self.layer_mapping[layer_id]
        if item.compress_ratio == 0:
            return self._read_layer_buffer("swa", layer_id)
        if item.compress_ratio == 4:
            return self._read_layer_buffer("c4", layer_id)
        return self._read_layer_buffer("c128", layer_id)

    def get_swa_raw_buffer(self, layer_id: int) -> torch.Tensor:
        return self._read_layer_buffer("swa", layer_id)

    # ---- KV writes: owned-only, invalidate remote copies --------------------

    def _log_write_loc(self, kind: str, layer_id: int, loc: torch.Tensor) -> None:
        if _LS_DEBUG and loc is not None and loc.numel():
            owned = self._is_layer_owned(layer_id)
            logger.warning(
                "LSLOC rank=%d %s:%d owned=%d n=%d min=%d max=%d",
                self.layer_shard_rank,
                kind,
                layer_id,
                int(owned),
                loc.numel(),
                int(loc.min()),
                int(loc.max()),
            )

    def set_swa_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache: torch.Tensor,
    ) -> None:
        self._invalidate_family(("swa",), layer_id)
        self._log_write_loc("swa", layer_id, loc)
        if not self._is_layer_owned(layer_id):
            return
        super().set_swa_buffer(layer_id, loc, cache)

    def set_compress_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        kv: torch.Tensor,
        kv_scale: Optional[torch.Tensor],
        from_indexer: bool,
    ) -> None:
        self._log_write_loc(
            "index" if from_indexer else "cmp", layer_id, loc
        )
        ratio = self.layer_mapping[layer_id].compress_ratio
        if ratio == 4:
            self._invalidate_family(("c4", "index_k", "index_scale"), layer_id)
        elif ratio == 128:
            self._invalidate_family(("c128",), layer_id)
        if not self._is_layer_owned(layer_id):
            return
        super().set_compress_buffer(layer_id, loc, kv, kv_scale, from_indexer)

    # ---- PD transfer: report owned layers only ------------------------------

    def get_contiguous_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        """Main PD buffers: [c4 KV, index K, index scale], each section sliced
        to this rank's owned c4 layers."""
        buffers = (
            self._owned_bucket_buffers(self.c4_kv_pool.kv_buffer, "c4")
            + self._owned_bucket_buffers(self.c4_indexer_kv_pool.index_k_buffer, "c4")
            + self._owned_bucket_buffers(
                self.c4_indexer_kv_pool.index_scale_buffer, "c4"
            )
        )
        return (
            [buf.data_ptr() for buf in buffers],
            [buf.nbytes for buf in buffers],
            [buf[0].nbytes for buf in buffers],
        )

    def get_state_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        """SWA component (owned only): SWA KV + c4 attn/indexer states."""
        plan = self._get_shard_plan()
        swa_start, swa_end = plan.owned_stage_local_range()
        data_ptrs: List[int] = []
        data_lens: List[int] = []
        item_lens: List[int] = []
        for buf in self.swa_kv_pool.kv_buffer[swa_start:swa_end]:
            data_ptrs.append(buf.data_ptr())
            data_lens.append(buf.nbytes)
            item_lens.append(buf[0].nbytes)
        for pools in (self.compress_state_pools, self.indexer_compress_state_pools):
            # compress_state_pools is absolute-layer-indexed; layer split
            # requires pp_size == 1, so stage-local ids coincide.
            for idx in plan.owned_stage_local_ids("c4"):
                state = pools[idx].kv_score_buffer.kv_score
                data_ptrs.append(state.data_ptr())
                data_lens.append(state.nbytes)
                item_lens.append(state[0].nbytes * pools[idx].ring_size)
        return data_ptrs, data_lens, item_lens

    def get_c128_kv_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        buffers = self._owned_bucket_buffers(self.c128_kv_pool.kv_buffer, "c128")
        return (
            [buf.data_ptr() for buf in buffers],
            [buf.nbytes for buf in buffers],
            [buf[0].nbytes for buf in buffers],
        )

    def get_c128_state_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        data_ptrs: List[int] = []
        data_lens: List[int] = []
        item_lens: List[int] = []
        for idx in self._get_shard_plan().owned_stage_local_ids("c128"):
            pool = self.compress_state_pools[idx]
            state = pool.kv_score_buffer.kv_score
            data_ptrs.append(state.data_ptr())
            data_lens.append(state.nbytes)
            item_lens.append(state[0].nbytes * pool.ring_size)
        return data_ptrs, data_lens, item_lens

    def _owned_bucket_buffers(
        self, buffers: List[torch.Tensor], bucket: str
    ) -> List[torch.Tensor]:
        start, end = self._get_shard_plan().owned_bucket_range(bucket)
        return buffers[start:end]

    # ---- PD transfer: layer ids parallel to the owned buf lists -------------

    def _owned_global_bucket_ids(self, bucket: str) -> List[int]:
        plan = self._get_shard_plan()
        return [self._stage_start + i for i in plan.owned_stage_local_ids(bucket)]

    def get_kv_layer_ids(self) -> List[int]:
        return self._owned_global_bucket_ids("c4") * 3

    def get_state_layer_ids(self) -> List[int]:
        plan = self._get_shard_plan()
        start, end = plan.owned_stage_local_range()
        owned_c4 = self._owned_global_bucket_ids("c4")
        return (
            [self._stage_start + i for i in range(start, end)]
            + owned_c4 * 2
        )

    def get_c128_layer_ids(self) -> List[int]:
        return self._owned_global_bucket_ids("c128")
