"""CP-v2 metadata validation for the NPU DSV4 flow (2 NPUs, no server).

Enables SGLANG_ENABLE_CP_V2, builds interleave strategy metadata through the
same prepare_cp_forward entry the eager runner uses, and verifies the invariants
the ascend backend relies on: per-rank actual tokens partition the batch, local
index selections across ranks cover every token exactly once, and the pad path
yields a consistent physical token space.

Run directly on 2 NPUs:
    ASCEND_RT_VISIBLE_DEVICES=0,1 python -m pytest \\
        test/registered/unit/hardware_backend/npu/test_dsv4_cp_v2_metadata.py
"""

import os
import unittest

import torch
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=120, suite="per-commit-2-npu-a2")

PORT = int(os.environ.get("DSV4_CP_V2_PORT", "29971"))


def _run(rank: int, world: int, port: int):
    import torch_npu  # noqa: F401

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ["SGLANG_ENABLE_CP_V2"] = "1"
    torch.npu.set_device(rank)

    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.layers.cp.base import init_cp_strategy
    from sglang.srt.layers.cp.utils import is_cp_v2_active, prepare_cp_forward
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    init_distributed_environment(
        world_size=world,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="hccl",
    )
    initialize_model_parallel(
        tensor_model_parallel_size=world,
        attention_context_model_parallel_size=world,
    )
    # Publish the parallel config bag the CP padding path reads.
    from sglang.srt.runtime_context import _ConfigBag, get_context

    parallel = _ConfigBag("parallel")
    parallel._set("attn_cp_size", world)
    parallel._set("attn_cp_rank", rank)
    parallel._set("enable_prefill_context_parallel", False)
    parallel._set("enable_dsa_prefill_context_parallel", True)
    parallel._set("dsa_prefill_cp_mode", "round-robin-split")
    ctx = get_context()
    ctx._config_bags = {"parallel": parallel}
    ctx.parallel._config = parallel
    init_cp_strategy(
        enable_prefill_cp=True, cp_size=world, cp_strategy="interleave"
    )

    # Two requests: 5 and 3 extend tokens (rank-local interleave shards them).
    extend_lens = [5, 3]
    num_tokens = sum(extend_lens)
    fb = type("FB", (), {})()
    fb.forward_mode = ForwardMode.EXTEND
    fb.input_ids = torch.arange(num_tokens)
    fb.seq_lens_cpu = torch.tensor([5, 3], dtype=torch.int64)
    fb.extend_seq_lens_cpu = list(extend_lens)
    fb.attn_cp_metadata = None
    fb.global_num_tokens_cpu = None
    fb.out_cache_loc = torch.arange(num_tokens)

    assert is_cp_v2_active(fb), "CP-v2 must be active for the test batch"
    prepare_cp_forward(fb)

    meta = fb.attn_cp_metadata
    assert meta is not None
    padded = sum(meta.per_rank_actual_token)
    assert padded >= num_tokens, (padded, num_tokens)
    assert len(meta.per_rank_actual_token) == world

    from sglang.srt.layers.cp.interleave import InterleaveCPStrategy

    strategy = InterleaveCPStrategy(cp_size=world)
    local_q = strategy.local_q_indices(num_tokens, fb).to(torch.long)
    # Union of both ranks' selections must cover every token exactly once.
    gathered = [None] * world
    torch.distributed.all_gather_object(gathered, local_q.tolist())
    union = sorted(i for rank_ids in gathered for i in rank_ids if i < num_tokens)
    assert union == list(range(num_tokens)), f"cover mismatch: {union}"

    # Out cache loc truncated to logical tokens by prepare_cp_forward.
    assert fb.out_cache_loc.shape[0] == num_tokens
    print(f"[rank {rank}] OK: padded={padded} local={sorted(local_q.tolist())}")
    torch.distributed.barrier()


class TestDSV4CPV2Metadata(CustomTestCase):
    def test_cp_v2_metadata(self):
        world = min(
            int(os.environ.get("DSV4_CP_V2_WORLD", "2")), torch.npu.device_count()
        )
        if world < 2:
            self.skipTest("CP-v2 metadata test needs >= 2 NPUs")
        mp.spawn(_run, args=(world, PORT), nprocs=world, join=True)


if __name__ == "__main__":
    unittest.main()
