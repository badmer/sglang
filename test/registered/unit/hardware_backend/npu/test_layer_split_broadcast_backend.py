"""Multi-NPU verification of torch.distributed broadcast for the layer-split pool.

The DSV4 NPU layer-split pool reads non-owned layers via chunked all-gather
because an earlier investigation concluded broadcast corrupts payload bytes on
this stack. That observation came from a server process group using the "zbal"
torch backend (selected by SGLANG_ZBAL_LOCAL_MEM_SIZE > 0, see
``platforms/device_mixin.py``), not from HCCL itself.

This test checks broadcast byte-exactness under BOTH backends so the pool's
collective choice is grounded in current evidence:

* basic       - single broadcast per buffer across a size sweep.
* queue       - all broadcasts enqueued back-to-back with no intermediate sync
                (the strict order shape that failed historically), one sync at
                the end, every buffer verified bitwise.
* alternating - the pool's consume pattern: one shared scratch per rank,
                ownership alternating per layer, the receiver copies the
                scratch aside before the next broadcast overwrites it.
* allgather   - the same queue pattern via all_gather as the current baseline.

Run directly on 2+ NPUs:
    ASCEND_RT_VISIBLE_DEVICES=0,1 python -m pytest \\
        test/registered/unit/hardware_backend/npu/test_layer_split_broadcast_backend.py

Set DSV4_LSBB_SKIP_ZBAL=1 to skip the zbal-backend phases.
"""

import os
import unittest

import torch
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=600, suite="per-commit-2-npu-a2")

WORLD = int(os.environ.get("DSV4_LSBB_WORLD", "2"))
PORT = int(os.environ.get("DSV4_LSBB_PORT", "29831"))
PHASE_TIMEOUT = int(os.environ.get("DSV4_LSBB_TIMEOUT", "300"))
SIZES_INT32 = [1024, 256 * 1024, 4 * 1024 * 1024]  # 4KB / 1MB / 16MB
N_LAYERS = 8


def _pattern(buf: torch.Tensor, seed: int) -> None:
    """Deterministic int32 payload: seed-dependent arange with wrapping."""
    flat = buf.view(torch.int32).reshape(-1)
    base = torch.arange(flat.numel(), dtype=torch.int32, device=buf.device)
    flat.copy_((base * 2654435761 + seed) % (2**31))


def _child(rank: int, world: int, port: int, backend: str, pattern: str) -> None:
    import torch_npu  # noqa: F401

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    if backend == "zbal":
        # register_backend("zbal", ...) runs at import; it must precede
        # init_process_group. switch_to_allocator must precede any NPU
        # allocation, mirroring the server's early init_zbal_on_npu.
        import zbal  # noqa: F401

        from sglang.srt.hardware_backend.npu.utils import init_zbal

        assert init_zbal(world, rank, rank) == 1
    torch.npu.set_device(rank)

    from sglang.srt.distributed import init_distributed_environment

    init_distributed_environment(
        world_size=world,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend=backend,
    )

    device = f"npu:{rank}"
    torch.distributed.barrier()
    if pattern == "basic":
        _run_basic(device)
    elif pattern == "queue":
        _run_queue(device, mode="broadcast")
    elif pattern == "alternating":
        _run_alternating(device)
    elif pattern == "sidestream":
        _run_sidestream(device)
    elif pattern == "allgather":
        _run_queue(device, mode="allgather")
    torch.distributed.barrier()
    torch.npu.synchronize()
    print(f"[rank {rank}] {backend}/{pattern} OK")


def _run_basic(device: str) -> None:
    for n_int32 in SIZES_INT32:
        for owner in range(WORLD):
            buf = torch.zeros(n_int32, dtype=torch.int32, device=device)
            if torch.distributed.get_rank() == owner:
                _pattern(buf, seed=owner * 977 + n_int32)
            torch.distributed.broadcast(buf, src=owner)
            torch.npu.synchronize()
            want = torch.empty_like(buf)
            _pattern(want, seed=owner * 977 + n_int32)
            assert torch.equal(buf, want), (n_int32, owner)


def _run_queue(device: str, mode: str) -> None:
    # Per-layer buffers enqueued back-to-back: one sync only at the end, then
    # every layer must hold its owner's exact bytes.
    n = 256 * 1024  # 1 MB per int32 view
    bufs = [
        torch.full((n,), 0x7EEE7EEE, dtype=torch.int32, device=device)
        for _ in range(N_LAYERS)
    ]
    for layer in range(N_LAYERS):
        src_rank = layer % WORLD
        if mode == "broadcast":
            if torch.distributed.get_rank() == src_rank:
                _pattern(bufs[layer], seed=layer * 31 + 7)
            torch.distributed.broadcast(bufs[layer], src=src_rank)
        else:
            send = torch.empty(n, dtype=torch.int32, device=device)
            if torch.distributed.get_rank() == src_rank:
                _pattern(send, seed=layer * 31 + 7)
            gathered = torch.empty((WORLD, n), dtype=torch.int32, device=device)
            torch.distributed.all_gather_into_tensor(gathered, send)
            bufs[layer].copy_(gathered[src_rank])
    torch.npu.synchronize()
    torch.distributed.barrier()
    for layer in range(N_LAYERS):
        want = torch.empty(n, dtype=torch.int32, device=device)
        _pattern(want, seed=layer * 31 + 7)
        assert torch.equal(bufs[layer], want), layer


def _run_alternating(device: str) -> None:
    # The pool's consume shape: ONE scratch; ownership alternates; the
    # receiver copies the scratch aside before the next broadcast overwrites
    # it, so ordering of broadcast vs D2D copy is what is under test.
    n = 256 * 1024
    scratch = torch.full((n,), 0x7EEE7EEE, dtype=torch.int32, device=device)
    keep = []
    for layer in range(N_LAYERS):
        owner = layer % WORLD
        if torch.distributed.get_rank() == owner:
            _pattern(scratch, seed=layer * 131 + 3)
        work = torch.distributed.broadcast(scratch, src=owner, async_op=True)
        if work is not None:
            work.wait()
        keep.append(scratch.clone())
    torch.npu.synchronize()
    torch.distributed.barrier()
    for layer in range(N_LAYERS):
        want = torch.empty(n, dtype=torch.int32, device=device)
        _pattern(want, seed=layer * 131 + 3)
        assert torch.equal(keep[layer], want), layer


def _run_sidestream(device: str) -> None:
    # The async-prefetch shape: each broadcast is enqueued on a side stream
    # right after its owner-side write; the consumer waits an event recorded
    # on the side stream before reading the scratch from the main stream.
    n = 256 * 1024
    side = torch.npu.Stream()
    scratch = torch.full((n,), 0x7EEE7EEE, dtype=torch.int32, device=device)
    keep = []
    cur = torch.npu.current_stream()
    for layer in range(N_LAYERS):
        owner = layer % WORLD
        side.wait_stream(cur)
        with torch.npu.stream(side):
            if torch.distributed.get_rank() == owner:
                _pattern(scratch, seed=layer * 217 + 5)
            work = torch.distributed.broadcast(scratch, src=owner, async_op=True)
            event = torch.npu.Event()
            event.record(side)
        cur.wait_event(event)
        if work is not None:
            work.wait()
        keep.append(scratch.clone())
        cur.synchronize()
    torch.distributed.barrier()
    for layer in range(N_LAYERS):
        want = torch.empty(n, dtype=torch.int32, device=device)
        _pattern(want, seed=layer * 217 + 5)
        assert torch.equal(keep[layer], want), layer


def _run_phase(phase: str, backend: str, pattern: str, world: int, port: int) -> int:
    import time

    ctx = mp.spawn(
        _child,
        args=(world, port, backend, pattern),
        nprocs=world,
        join=False,
    )
    start = time.monotonic()
    while not ctx.join(timeout=1.0):
        if time.monotonic() - start > PHASE_TIMEOUT:
            for p in ctx.processes:
                if p.is_alive():
                    p.terminate()
            for p in ctx.processes:
                p.join()
            print(f"phase {phase}: TIMEOUT (treated as hang)")
            return 124
    codes = [p.exitcode for p in ctx.processes]
    if any(c != 0 for c in codes):
        print(f"phase {phase}: exitcodes={codes}")
        return 1
    return 0


class TestLayerSplitBroadcastBackend(CustomTestCase):
    def _check_world(self):
        if torch.npu.device_count() < 2:
            self.skipTest("needs >= 2 NPUs")

    def _run(self, backend: str, pattern: str, port: int):
        rc = _run_phase(f"{backend}/{pattern}", backend, pattern, WORLD, port)
        self.assertEqual(rc, 0, f"{backend}/{pattern} failed (rc={rc})")

    def test_basic_hccl(self):
        self._check_world()
        self._run("hccl", "basic", PORT)

    def test_queue_hccl(self):
        self._check_world()
        self._run("hccl", "queue", PORT + 1)

    def test_alternating_hccl(self):
        self._check_world()
        self._run("hccl", "alternating", PORT + 2)

    def test_sidestream_hccl(self):
        self._check_world()
        self._run("hccl", "sidestream", PORT + 7)

    def test_queue_allgather_reference(self):
        self._check_world()
        self._run("hccl", "allgather", PORT + 3)

    def test_basic_zbal(self):
        if os.environ.get("DSV4_LSBB_SKIP_ZBAL"):
            self.skipTest("DSV4_LSBB_SKIP_ZBAL set")
        self._check_world()
        os.environ["SGLANG_ZBAL_LOCAL_MEM_SIZE"] = "62084"
        os.environ["ZBAL_NPU_ALLOC_CONF"] = "use_vmm_for_static_memory:True"
        try:
            self._run("zbal", "basic", PORT + 4)
            self._run("zbal", "queue", PORT + 5)
            self._run("zbal", "alternating", PORT + 6)
        finally:
            os.environ.pop("SGLANG_ZBAL_LOCAL_MEM_SIZE", None)
            os.environ.pop("ZBAL_NPU_ALLOC_CONF", None)


if __name__ == "__main__":
    unittest.main()
