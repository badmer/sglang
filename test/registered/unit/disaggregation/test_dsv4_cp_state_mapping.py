"""Regression coverage for DSV4 PD state transfer with Prefill CP."""

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from sglang.srt.disaggregation.ascend.conn import AscendStateType
from sglang.srt.disaggregation.base.conn import StateIndexMap
from sglang.srt.disaggregation.common.utils import pack_int_lists
from sglang.srt.disaggregation.mooncake.conn import (
    TransferInfo,
    pair_state_index_maps,
    split_state_index_maps,
    validate_state_map_cp_routing,
)
from sglang.srt.hardware_backend.npu.dsv4.dsv4_common_hooks import (
    dsv4_state_payloads,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestStateIndexMap(unittest.TestCase):
    def test_prefill_cp_rank_pairs_only_its_owned_logical_pages(self):
        prefill = StateIndexMap(
            logical_indices=np.array([0, 3], dtype=np.int32),
            physical_indices=np.array([100, 103], dtype=np.int32),
        )
        decode = StateIndexMap(
            logical_indices=np.array([0, 1, 2, 3], dtype=np.int32),
            physical_indices=np.array([10, 11, 12, 13], dtype=np.int32),
        )

        src_indices, dst_indices = prefill.pair_with(decode)

        np.testing.assert_array_equal(src_indices, np.array([100, 103]))
        np.testing.assert_array_equal(dst_indices, np.array([10, 13]))

    def test_missing_decode_logical_page_is_not_silently_truncated(self):
        prefill = StateIndexMap(
            logical_indices=np.array([0, 3], dtype=np.int32),
            physical_indices=np.array([100, 103], dtype=np.int32),
        )
        decode = StateIndexMap(
            logical_indices=np.array([0, 1], dtype=np.int32),
            physical_indices=np.array([10, 11], dtype=np.int32),
        )

        with self.assertRaisesRegex(ValueError, "logical state pages.*3"):
            prefill.pair_with(decode)

    def test_empty_decode_map_reports_the_missing_prefill_logical_page(self):
        prefill = StateIndexMap(
            logical_indices=np.array([2], dtype=np.int32),
            physical_indices=np.array([102], dtype=np.int32),
        )
        decode = StateIndexMap(
            logical_indices=np.array([], dtype=np.int32),
            physical_indices=np.array([], dtype=np.int32),
        )

        with self.assertRaisesRegex(ValueError, "logical state pages.*2"):
            prefill.pair_with(decode)

    def test_four_prefill_cp_maps_pair_against_one_decode_map_without_overlap(self):
        decode = StateIndexMap(
            logical_indices=np.arange(8, dtype=np.int32),
            physical_indices=np.arange(10, 18, dtype=np.int32),
        )
        prefill_maps = [
            StateIndexMap(
                logical_indices=np.array(logical_pages, dtype=np.int32),
                physical_indices=np.array(physical_pages, dtype=np.int32),
            )
            for logical_pages, physical_pages in (
                ([0, 7], [100, 107]),
                ([1, 6], [101, 106]),
                ([2, 5], [102, 105]),
                ([3, 4], [103, 104]),
            )
        ]

        paired_dst_pages = [page_map.pair_with(decode)[1] for page_map in prefill_maps]

        np.testing.assert_array_equal(
            np.sort(np.concatenate(paired_dst_pages)), np.arange(10, 18)
        )


class TestDSV4StatePayloadLogicalPages(unittest.TestCase):
    def test_swa_payload_keeps_logical_page_keys_when_zero_pages_are_dropped(self):
        req_to_token_pool = SimpleNamespace(
            req_to_token_c4=torch.zeros((1, 2), dtype=torch.int32),
            req_to_token_swa=torch.tensor(
                [[2, 2, 0, 0, 6, 6, 0, 0]], dtype=torch.int32
            ),
        )

        payloads = dsv4_state_payloads(
            req_to_token_pool,
            req_pool_idx=0,
            seq_len=8,
            page_size=2,
            window_size=8,
            include_logical_pages=True,
        )
        page_map = payloads[AscendStateType.DSV4_SWA]()

        self.assertIsInstance(page_map, StateIndexMap)
        np.testing.assert_array_equal(page_map.logical_indices, np.array([0, 2]))
        np.testing.assert_array_equal(page_map.physical_indices, np.array([1, 3]))

    def test_128k_swa_uses_global_tail_page_during_cp_transfer(self):
        seq_len = 131072
        page_size = 128
        swa_table = torch.zeros((1, seq_len), dtype=torch.int32)
        swa_table[0, seq_len - page_size] = 7 * page_size
        req_to_token_pool = SimpleNamespace(
            req_to_token_c4=torch.zeros((1, 1), dtype=torch.int32),
            req_to_token_swa=swa_table,
        )

        payloads = dsv4_state_payloads(
            req_to_token_pool,
            req_pool_idx=0,
            seq_len=seq_len,
            page_size=page_size,
            window_size=page_size,
            include_logical_pages=True,
            transfer_cp_rank=0,
            transfer_cp_size=4,
        )
        page_map = payloads[AscendStateType.DSV4_SWA]()

        np.testing.assert_array_equal(
            page_map.logical_indices, np.array([1023], dtype=np.int32)
        )
        np.testing.assert_array_equal(
            page_map.physical_indices, np.array([7], dtype=np.int32)
        )

    def test_c4_pages_are_partitioned_once_across_four_transfer_cp_ranks(self):
        seq_len = 131072
        page_size = 128
        c4_tokens = seq_len // 4
        c4_table = torch.zeros((1, c4_tokens), dtype=torch.int32)
        c4_table[0, ::page_size] = (
            torch.arange(1, c4_tokens // page_size + 1, dtype=torch.int32) * page_size
        )
        req_to_token_pool = SimpleNamespace(
            req_to_token_c4=c4_table,
            req_to_token_swa=torch.zeros((1, 1), dtype=torch.int32),
        )

        page_maps = []
        for cp_rank in range(4):
            payloads = dsv4_state_payloads(
                req_to_token_pool,
                req_pool_idx=0,
                seq_len=seq_len,
                page_size=page_size,
                window_size=page_size,
                include_logical_pages=True,
                transfer_cp_rank=cp_rank,
                transfer_cp_size=4,
            )
            page_maps.append(payloads[AscendStateType.DSV4_C4]())

        combined_logical = np.concatenate(
            [page_map.logical_indices for page_map in page_maps]
        )
        combined_physical = np.concatenate(
            [page_map.physical_indices for page_map in page_maps]
        )
        np.testing.assert_array_equal(combined_logical, np.arange(256, dtype=np.int32))
        np.testing.assert_array_equal(
            combined_physical, np.arange(1, 257, dtype=np.int32)
        )

    def test_rejects_invalid_transfer_cp_topology(self):
        req_to_token_pool = SimpleNamespace(
            req_to_token_c4=torch.zeros((1, 1), dtype=torch.int32),
        )

        for cp_rank, cp_size in ((0, 0), (-1, 4), (4, 4)):
            with self.subTest(cp_rank=cp_rank, cp_size=cp_size):
                with self.assertRaises(ValueError):
                    dsv4_state_payloads(
                        req_to_token_pool,
                        req_pool_idx=0,
                        seq_len=1,
                        page_size=128,
                        window_size=128,
                        transfer_cp_rank=cp_rank,
                        transfer_cp_size=cp_size,
                    )


class TestMooncakeStateMapWire(unittest.TestCase):
    def test_metadata_encoder_keeps_legacy_and_logical_state_entries_parallel(self):
        physical_indices, logical_indices = split_state_index_maps(
            [
                np.array([5], dtype=np.int32),
                StateIndexMap(
                    logical_indices=np.array([2, 4], dtype=np.int32),
                    physical_indices=np.array([20, 40], dtype=np.int32),
                ),
            ]
        )

        self.assertEqual(physical_indices[0].tolist(), [5])
        self.assertEqual(physical_indices[1].tolist(), [20, 40])
        self.assertEqual(logical_indices[0].tolist(), [])
        self.assertEqual(logical_indices[1].tolist(), [2, 4])

    def test_decode_rejects_dsv4_state_map_when_a_prefill_cp_rank_is_not_routed(self):
        state_map = StateIndexMap(
            logical_indices=np.array([0], dtype=np.int32),
            physical_indices=np.array([10], dtype=np.int32),
        )

        with self.assertRaisesRegex(RuntimeError, "ALL_CP_RANKS_TRANSFER.*CP1"):
            validate_state_map_cp_routing(
                [state_map], target_cp_ranks=[0], prefill_cp_size=2
            )

    def test_prefill_pairs_its_local_pages_against_full_decode_map(self):
        src_indices, dst_indices = pair_state_index_maps(
            StateIndexMap(
                logical_indices=np.array([0, 3], dtype=np.int32),
                physical_indices=np.array([100, 103], dtype=np.int32),
            ),
            dst_indices=[10, 11, 12, 13],
            dst_logical_indices=[0, 1, 2, 3],
        )

        np.testing.assert_array_equal(src_indices, np.array([100, 103]))
        np.testing.assert_array_equal(dst_indices, np.array([10, 13]))

    def test_transfer_info_decodes_logical_state_page_frame(self):
        info = TransferInfo.from_zmq(
            [
                b"7",
                b"127.0.0.1",
                b"8998",
                b"session",
                np.array([1, 2], dtype=np.int32).tobytes(),
                b"9",
                pack_int_lists([[30, 31]], "i"),
                b"1",
                b"0",
                pack_int_lists([[4, 7]], "i"),
            ]
        )

        self.assertEqual(info.dst_state_indices, [[30, 31]])
        self.assertEqual(info.dst_state_logical_indices, [[4, 7]])


if __name__ == "__main__":
    unittest.main()
