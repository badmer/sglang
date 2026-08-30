"""NPU interleaved RoPE cos/sin cache for DeepSeek-V4 on Ascend.

One Dsv4NpuRoPE per freqs_cis (singleton by id). Tables are built once at
init and registered as buffers on the shared rotary_emb, so model.to() moves
them and a captured aclgraph sees stable tensors; decode only does index_select.

mscale: cos/sin stored in freqs_cis must already be pre-multiplied by the YARN
mscale at precompute time (see precompute_freqs_cis). We just read what's stored.
"""

import logging
import sys
import traceback
from typing import Optional

import torch

# Temporary debug instrumentation for the rope-memo verification; remove
# afterwards. WARNING level so it survives log-level filtering:
#   [rope PRIME] per-forward prime marker: module count, forward_batch id,
#                mode/spec class, positions tensor id, primed freqs_cis ids
#   [rope MISS]  every memo miss, with the reason and the sglang call chain
#   [rotary]     rotary launches tagged with cos/sin tensor ids + call chain,
#                so profile kernels can be bound to Python call sites; shared
#                memo entries show up as repeated identical id pairs
#   [rope CMP]   the compressor's direct get_cos_sin (bypasses the memo),
#                counted so profile gathers can be attributed to it
_MISS_N = 0
_PRIME_N = 0
_ROTARY_N = 0
_CMP_N = 0
_GCS_N = 0
_GCS_SITES: dict[str, int] = {}
_MEMO_LOG = logging.getLogger(__name__)


def _sglang_call_sites(depth: int = 4) -> str:
    frames = [f for f in traceback.extract_stack() if "sglang" in (f.filename or "")]
    frames = frames[-depth:]
    return " <- ".join(
        f"{f.filename.rsplit('/', 1)[-1]}:{f.lineno}:{f.name}" for f in frames
    )


def _fb_tag(forward_batch) -> str:
    # "fb-id/spec-class": EAGLE draft forwards are told apart from target
    # verify / plain decode by the spec_info subclass carried on the batch.
    spec = forward_batch.spec_info
    return f"{id(forward_batch)}/{type(spec).__name__ if spec is not None else 'nospec'}"


# Temporary diagnostics for the per-forward rope memo, classified so one log
# line identifies WHY a lookup missed: prime never ran on this forward_batch
# (miss_no_memo), dtype/config key mismatch (miss_key), or the positions
# object differs from the primed one (miss_positions). Remove once deployment
# is verified.


def _note_gcs_call(dtype, cache_dtype, inverse: bool) -> None:
    # Temporary: count EVERY get_cos_sin invocation by its (out-of-module)
    # callsite with exact totals, dumped periodically. Unlike the sampled
    # MISS/CMP/rotary lines this is exhaustive, so any surviving per-layer
    # gather path shows up with its true count. sys._getframe is ~us; the
    # internal wrappers (prime/rope_cos_sin/cmp_cos_sin) are skipped so the
    # site is the model/backend code that originated the gather.
    global _GCS_N
    _GCS_N += 1
    f = sys._getframe(2)
    if f.f_code.co_filename == __file__:
        f = sys._getframe(3)
    site = f"{f.f_code.co_filename.rsplit('/', 1)[-1]}:{f.f_lineno}"
    key = f"{site}|d={str(dtype).replace('torch.', '')},c={str(cache_dtype).replace('torch.', '')},inv={int(inverse)}"
    _GCS_SITES[key] = _GCS_SITES.get(key, 0) + 1
    if _GCS_N % 1000 == 0:
        _MEMO_LOG.warning("[rope GCS total=%d] %s", _GCS_N, _GCS_SITES)


class Dsv4NpuRoPE:
    """Interleaved cos/sin tables, layout [c0,c0,c1,c1,...] / [s0,s0,s1,s1,...]."""

    # id(freqs_cis) -> instance. freqs_cis is a module buffer, lives with the model.
    _instances: dict[int, "Dsv4NpuRoPE"] = {}

    def __init__(
        self, freqs_cis: torch.Tensor, rotary_emb: Optional[object] = None
    ) -> None:
        self.freqs_cis = freqs_cis
        # cos/sin registered as buffers on this module (None -> fall back to _tables).
        self.rotary_emb = rotary_emb
        # contiguous real/imag halves of complex freqs_cis [max_pos, rope_dim/2];
        # .real/.imag are strided views, materialize once to avoid per-call
        # StridedSlice from aclnnIndex over the strided views.
        self._real_imag: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        self._tables: dict[
            tuple[torch.dtype, torch.device], tuple[torch.Tensor, torch.Tensor]
        ] = {}

    @classmethod
    def for_freqs(
        cls, freqs_cis: torch.Tensor, rotary_emb: Optional[object] = None
    ) -> "Dsv4NpuRoPE":
        # rotary_emb is only used at creation; callers sharing a warmed-up freqs_cis may omit it.
        inst = cls._instances.get(id(freqs_cis))
        if inst is None or inst.freqs_cis is not freqs_cis:
            inst = cls(freqs_cis, rotary_emb)
            cls._instances[id(freqs_cis)] = inst
        return inst

    def _contig_real_imag(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._real_imag is None:
            self._real_imag = (
                self.freqs_cis.real.contiguous(),
                self.freqs_cis.imag.contiguous(),
            )
        return self._real_imag

    @staticmethod
    def _buffer_names(dtype: torch.dtype) -> tuple[str, str]:
        suffix = str(dtype).replace("torch.", "").replace(".", "_")
        return (
            f"_npu_interleaved_rope_cos_cache_{suffix}",
            f"_npu_interleaved_rope_sin_cache_{suffix}",
        )

    def _register_or_set_buffer(self, name: str, tensor: torch.Tensor) -> None:
        owner = self.rotary_emb
        if hasattr(owner, "register_buffer"):
            if name in getattr(owner, "_buffers", {}):
                setattr(owner, name, tensor)
            else:
                owner.register_buffer(name, tensor, persistent=False)
        else:
            setattr(owner, name, tensor)

    def ensure_tables(
        self, dtype: torch.dtype, *, allow_build: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Returns [max_pos, rope_dim] tables. Call once at init (allow_build=True);
        # decode uses allow_build=False (no repeat_interleave inside the captured graph).
        expected_shape = (self.freqs_cis.shape[0], self.freqs_cis.shape[1] * 2)

        if self.rotary_emb is not None:
            cos_name, sin_name = self._buffer_names(dtype)
            cos = getattr(self.rotary_emb, cos_name, None)
            sin = getattr(self.rotary_emb, sin_name, None)
            if (
                cos is not None
                and sin is not None
                and tuple(cos.shape) == expected_shape
                and tuple(sin.shape) == expected_shape
                and cos.dtype == dtype
                and sin.dtype == dtype
                and cos.device == self.freqs_cis.device
                and sin.device == self.freqs_cis.device
            ):
                return cos, sin
        else:
            cached = self._tables.get((dtype, self.freqs_cis.device))
            if cached is not None:
                cos, sin = cached
                if (
                    tuple(cos.shape) == expected_shape
                    and tuple(sin.shape) == expected_shape
                ):
                    return cached

        if not allow_build:
            raise RuntimeError(
                "NPU interleaved RoPE cache is missing in a no-build path. "
                "Initialize it before forward to keep decode free of repeat_interleave."
            )

        real_contig, imag_contig = self._contig_real_imag()
        cos = real_contig.repeat_interleave(2, dim=-1).to(dtype=dtype).contiguous()
        sin = imag_contig.repeat_interleave(2, dim=-1).to(dtype=dtype).contiguous()

        if self.rotary_emb is not None:
            cos_name, sin_name = self._buffer_names(dtype)
            self._register_or_set_buffer(cos_name, cos)
            self._register_or_set_buffer(sin_name, sin)
        else:
            self._tables[(dtype, self.freqs_cis.device)] = (cos, sin)
        return cos, sin

    def get_cos_sin(
        self,
        positions: torch.Tensor,
        dtype: torch.dtype,
        *,
        view_4d: bool = False,
        inverse: bool = False,
        allow_build: bool = True,
        cache_dtype: Optional[torch.dtype] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # positions: [T]. Returns [T, rope_dim], or [T, 1, 1, rope_dim] if view_4d.
        # Position-gathered tensors are forward-local; do not cache across forwards
        # or MTP decode reuses the previous step's RoPE when only positions change.
        _note_gcs_call(dtype, cache_dtype or dtype, inverse)
        cache_dtype = dtype if cache_dtype is None else cache_dtype
        cos_cache, sin_cache = self.ensure_tables(cache_dtype, allow_build=allow_build)
        cos = cos_cache.index_select(0, positions)
        sin = sin_cache.index_select(0, positions)
        if inverse:
            sin = -sin
        if cos.dtype != dtype:
            cos = cos.to(dtype)
            sin = sin.to(dtype)
        if view_4d:
            rope_dim = cos.shape[-1]
            cos = cos.view(-1, 1, 1, rope_dim)
            sin = sin.view(-1, 1, 1, rope_dim)
        return cos, sin

    @staticmethod
    def apply_rotary_mul_inplace(
        q_rope: torch.Tensor,
        kv_rope: Optional[torch.Tensor],
        cos4: torch.Tensor,
        sin4: torch.Tensor,
        qk_nope_dim: int = 0,
    ) -> None:
        # q_rope: [T, n_heads, head_dim]; cos4/sin4: [T, 1, 1, rope_dim];
        # kv_rope: [T, 1, head_dim] or None. Prefer the NPU kernel: torch accumulates
        # bf16 muls in bf16 while the kernel uses fp32; drift compounds and flips argmax.
        global _ROTARY_N
        if q_rope.numel() == 0:
            # Idle/empty forward: skip the empty kernel launch too.
            return
        _ROTARY_N += 1
        if _ROTARY_N <= 40 or _ROTARY_N % 200 == 0:
            _MEMO_LOG.warning(
                "[rotary #%d]%s q%s cos_id=%d sin_id=%d at %s",
                _ROTARY_N,
                "+kv" if kv_rope is not None else "",
                tuple(q_rope.shape),
                id(cos4),
                id(sin4),
                _sglang_call_sites(),
            )
        rope_dim = cos4.shape[-1]
        torch.ops.custom.inplace_partial_rotary_mul(
            q_rope.unsqueeze(1),
            cos4,
            sin4,
            rotary_mode="interleave",
            partial_slice=[qk_nope_dim, qk_nope_dim + rope_dim],
        )
        if kv_rope is not None:
            if kv_rope.dim() == 3:
                kv_view = kv_rope.unsqueeze(1)
            else:
                kv_view = kv_rope.view(-1, 1, 1, rope_dim)
            torch.ops.custom.inplace_partial_rotary_mul(
                kv_view,
                cos4,
                sin4,
                rotary_mode="interleave",
                partial_slice=[qk_nope_dim, qk_nope_dim + rope_dim],
            )


# Per-forward memo of position-gathered (cos, sin), stashed on the ForwardBatch
# under this attribute by prime_rope_cos_sin (the single writer).
_ROPE_MEMO_ATTR = "_dsv4_npu_rope_memo"
_CMP_MEMO_ATTR = "_dsv4_npu_rope_cmp_memo"


def prime_rope_cos_sin(attn_modules, forward_batch, positions) -> None:
    # Materialize: callers may pass a generator, and the debug log below must
    # not consume it before the priming loop (that bug left the memo empty and
    # every lookup fell to the per-layer fallback path).
    attn_modules = list(attn_modules)
    if forward_batch.forward_mode.is_idle():
        # Idle draft forwards still run the (zero-token) layer loop, so keep
        # empty memos for the lookups -- every gather here is a no-op launch
        # (57% of D-node primes under PD were idle; layer lookups hit the
        # numel==0 fast path in rope_cos_sin).
        setattr(forward_batch, _ROPE_MEMO_ATTR, {})
        setattr(forward_batch, _CMP_MEMO_ATTR, {})
        return
    memo: dict = {}
    for attn in attn_modules:
        freqs_cis = attn.freqs_cis
        fwd_key = (id(freqs_cis), torch.bfloat16, False)
        if fwd_key in memo:
            continue
        cos, sin = Dsv4NpuRoPE.for_freqs(
            freqs_cis, getattr(attn, "rotary_emb", None)
        ).get_cos_sin(
            positions,
            torch.bfloat16,
            view_4d=True,
            inverse=False,
            allow_build=False,
            cache_dtype=torch.bfloat16,
        )
        memo[fwd_key] = (positions, cos, sin)
        memo[(id(freqs_cis), torch.bfloat16, True)] = (positions, cos, -sin)
    setattr(forward_batch, _ROPE_MEMO_ATTR, memo)
    # Compressor gathers (cmp_cos_sin) share this forward's lifecycle: prime
    # is the single writer that clears them, so a reused ForwardBatch (EAGLE
    # draft loop) can never serve a stale gather even when the cmp-positions
    # tensor object is reused.
    setattr(forward_batch, _CMP_MEMO_ATTR, {})
    global _PRIME_N
    _PRIME_N += 1
    # One line per forward is cheap and the entries' tensor ids are what the
    # parser matches rotary cos_ids against -- sparse sampling here would make
    # most rotary lines look unmatched. Dense for the first 5000 forwards,
    # then thin out.
    if _PRIME_N <= 5000 or _PRIME_N % 200 == 0:
        # entries carries every gathered tensor id (per freqs: cos/sin of the
        # forward pair, inv_s of the derived inverse pair) so rotary lines can
        # be matched back to their prime -- an unmatched rotary cos_id means a
        # gather outside the memo.
        entries = sorted(
            (
                f"f{k[0]}/c{id(v[1])}/s{id(v[2])}"
                if not k[2]
                else f"f{k[0]}/inv_s{id(v[2])}"
            )
            for k, v in memo.items()
        )
        _MEMO_LOG.warning(
            "[rope PRIME #%d] modules=%d fb=%s mode=%s bs=%s positions_id=%d "
            "entries=%s",
            _PRIME_N,
            len(attn_modules),
            _fb_tag(forward_batch),
            forward_batch.forward_mode.name,
            forward_batch.batch_size,
            id(positions),
            entries,
        )


def rope_cos_sin(
    freqs_cis: torch.Tensor,
    rotary_emb,
    forward_batch,
    positions: torch.Tensor,
    dtype: torch.dtype,
    *,
    inverse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if positions.numel() == 0 or forward_batch.forward_mode.is_idle():
        # Idle/empty forwards run a (possibly padded) layer loop whose output
        # is discarded: hand back uninitialized tensors instead of gathering.
        # numel==0 covers empty-token forwards (draft idle); is_idle covers
        # DP-padded idle forwards whose positions are non-empty -- those
        # would otherwise miss the empty prime memo and gather per layer.
        rope_dim = freqs_cis.shape[-1] * 2
        empty = positions.new_empty((positions.shape[0], 1, 1, rope_dim), dtype=dtype)
        return empty, empty
    memo = getattr(forward_batch, _ROPE_MEMO_ATTR, None)
    entry = memo.get((id(freqs_cis), dtype, inverse)) if memo is not None else None
    if entry is not None and entry[0] is positions:
        return entry[1], entry[2]
    global _MISS_N
    _MISS_N += 1
    if _MISS_N <= 20 or _MISS_N % 500 == 0:
        if memo is None:
            reason = "no_memo_on_forward_batch"
        elif entry is None:
            reason = f"key_not_found(dtype={dtype},inverse={inverse})"
        else:
            reason = "positions_object_mismatch"
        _MEMO_LOG.warning(
            "[rope MISS #%d] %s fb=%s mode=%s freqs_id=%d positions_id=%d "
            "primed_freqs=%s primed_positions=%s at %s",
            _MISS_N,
            reason,
            _fb_tag(forward_batch),
            forward_batch.forward_mode.name,
            id(freqs_cis),
            id(positions),
            sorted({k[0] for k in memo}) if memo is not None else None,
            {k: id(v[0]) for k, v in memo.items()} if memo is not None else None,
            _sglang_call_sites(),
        )
    # bf16 tables are ensured at layer init; gathering in the activation dtype
    # skips the fp32-gather + cast pair. Bit-identical values: rounding the
    # table once equals rounding each gathered element.
    cache_dtype = dtype if dtype == torch.bfloat16 else torch.float32
    return Dsv4NpuRoPE.for_freqs(freqs_cis, rotary_emb).get_cos_sin(
        positions,
        dtype,
        view_4d=True,
        inverse=inverse,
        allow_build=False,
        cache_dtype=cache_dtype,
    )


def note_cmp_gather(ratio: int, positions, cos, sin, cached: bool, mode: str) -> None:
    # Temporary: the compressor's forward_compress gathers cos/sin directly
    # (own positions, fp32); count hits vs misses so the cmp-memo effect is
    # measurable in the serving log, split by forward mode.
    global _CMP_N
    _CMP_N += 1
    if _CMP_N <= 40 or _CMP_N % 500 == 0:
        _MEMO_LOG.warning(
            "[rope CMP #%d] ratio=%d cached=%s mode=%s positions_id=%d cos_id=%d "
            "sin_id=%d at %s",
            _CMP_N,
            ratio,
            "hit" if cached else "miss",
            mode,
            id(positions),
            id(cos),
            id(sin),
            _sglang_call_sites(),
        )


def cmp_cos_sin(
    freqs_cis: torch.Tensor,
    rotary_emb,
    forward_batch,
    ratio: int,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    # The compressor path gathers over fm.positions_cmp_padding_c{ratio} in
    # fp32; one gather per (freqs_cis, ratio) per forward, shared by every
    # layer's compressor and the indexer's inner compressor. The memo dict is
    # created/cleared only by prime_rope_cos_sin: when prime did not run
    # (TBO children) fall back to a per-layer gather rather than risk caching
    # across forward boundaries. Third return value: whether the memo hit.
    memo = getattr(forward_batch, _CMP_MEMO_ATTR, None)
    if positions.numel() == 0:
        empty = positions.new_empty((0, freqs_cis.shape[-1] * 2), dtype=torch.float32)
        return empty, empty, False
    if memo is None:
        cos, sin = Dsv4NpuRoPE.for_freqs(freqs_cis, rotary_emb).get_cos_sin(
            positions,
            torch.float32,
            view_4d=False,
            allow_build=False,
        )
        return cos, sin, False
    key = (id(freqs_cis), ratio)
    entry = memo.get(key)
    if entry is not None and entry[0] is positions:
        return entry[1], entry[2], True
    cos, sin = Dsv4NpuRoPE.for_freqs(freqs_cis, rotary_emb).get_cos_sin(
        positions,
        torch.float32,
        view_4d=False,
        allow_build=False,
    )
    memo[key] = (positions, cos, sin)
    return cos, sin, False
