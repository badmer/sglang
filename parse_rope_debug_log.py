#!/usr/bin/env python3
"""Summarize the temporary rope-memo debug logs into a one-screen verdict.

Input: serving log(s) containing the WARNING lines emitted by dsv4_rope.py:
    [rope PRIME #N] modules=.. fb=.. mode=.. bs=.. positions_id=..
                    entries=[f<freqs>/c<cos_id>/s<sin_id>, f.../inv_s<id>]
    [rope MISS  #N] <reason> fb=.. mode=.. freqs_id=.. positions_id=..
                    primed_freqs=.. primed_positions=.. at <chain>
    [rotary     #N][+kv] q(shape) cos_id=.. sin_id=.. at <chain>
    [rope CMP   #N] ratio=.. cached=hit|miss mode=.. positions_id=..
                    cos_id=.. sin_id=.. at <chain>

The #N counters are authoritative (log lines themselves are rate-limited).

Sections:
  汇总        -- counters per class, split by forward mode
  共享健康度  -- rotary cos/sin ids matched back to primed ids
  调用点      -- rotary / MISS sites (chain's innermost sglang frame)
  时间分布    -- first/last timestamp per class + per-minute buckets, to
                 tell capture-phase (startup) activity from steady state

Usage: python3 parse_rope_debug_log.py <log1> [log2 ...]
"""

import ast
import collections
import re
import sys

MISS_RE = re.compile(r"\[rope MISS #(\d+)\] (\S+)")
PRIME_RE = re.compile(r"\[rope PRIME #(\d+)\]")
ROTARY_RE = re.compile(
    r"\[rotary #(\d+)\](\[\+kv\])? q\(([^)]*)\) cos_id=(\d+) sin_id=(\d+) at (.*)"
)
CMP_RE = re.compile(r"\[rope CMP #(\d+)\] ratio=(\d+) cached=(hit|miss)")
# [rope GCS total=N] {'file.py:line|d=..,c=..,inv=..': count, ...} -- exact
# per-callsite get_cos_sin totals, one counter per rank; the last dump per
# rank is that rank's final cumulative value.
GCS_RE = re.compile(r"\[rope GCS total=(\d+)\] (\{.*\})")
# New primes carry entries=[f<id>/c<id>/s<id>, f<id>/inv_s<id>]; old ones only
# freqs_ids=[...]. Only the cos/sin/inv_s ids feed the matching set (the f id
# is a module buffer, not a gathered tensor).
ENTRY_ITEM_RE = re.compile(r"f\d+/(?:c(\d+)/s(\d+)|inv_s(\d+))")
ENTRIES_RE = re.compile(r"entries=\[([^\]]*)\]")
TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def last_site(chain: str) -> str:
    # Drop the instrumentation's own frames (dsv4_rope.py) so the reported
    # site is the real caller.
    parts = [p.strip() for p in chain.split("<-") if "dsv4_rope.py" not in p]
    return parts[-1] if parts else chain.strip()


def bucket_min(ts: str) -> str:
    return ts[:16]  # YYYY-MM-DD HH:MM


class Stats:
    def __init__(self) -> None:
        self.total = 0          # authoritative #N max
        self.lines = 0          # sampled log lines
        self.by_mode = collections.Counter()
        self.first_ts = None
        self.last_ts = None
        self.by_minute = collections.Counter()


def note(st: Stats, ts, mode: str = "?") -> None:
    st.lines += 1
    if mode != "?":
        st.by_mode[mode] += 1
    if ts:
        if st.first_ts is None or ts < st.first_ts:
            st.first_ts = ts
        if st.last_ts is None or ts > st.last_ts:
            st.last_ts = ts
        st.by_minute[bucket_min(ts)] += 1


def main() -> int:
    prime = Stats()
    rotary = Stats()
    miss = Stats()
    cmp_hit = Stats()
    cmp_miss = Stats()

    miss_by_reason_site = collections.Counter()
    rotary_sites = collections.Counter()
    rotary_pairs = collections.Counter()
    rotary_with_kv = 0
    rotary_matched = 0
    rotary_unmatched = collections.Counter()
    primed_ids: set[int] = set()
    log_first_ts = None
    log_last_ts = None
    # rank -> (total, sites dict); last dump per rank wins (cumulative).
    gcs_last: dict[str, tuple[int, dict]] = {}

    for path in sys.argv[1:]:
        with open(path, errors="ignore") as f:
            for line in f:
                m_ts = TS_RE.search(line)
                ts = m_ts.group(1) if m_ts else None
                if ts:
                    if log_first_ts is None or ts < log_first_ts:
                        log_first_ts = ts
                    if log_last_ts is None or ts > log_last_ts:
                        log_last_ts = ts

                m = PRIME_RE.search(line)
                if m:
                    prime.total = max(prime.total, int(m.group(1)))
                    mode = re.search(r" mode=(\S+)", line)
                    note(prime, ts, mode.group(1) if mode else "?")
                    ent = ENTRIES_RE.search(line)
                    if ent:
                        for item in ent.group(1).split(","):
                            m2 = ENTRY_ITEM_RE.search(item)
                            if m2:
                                for num in m2.groups():
                                    if num:
                                        primed_ids.add(int(num))
                    continue

                m = MISS_RE.search(line)
                if m:
                    miss.total = max(miss.total, int(m.group(1)))
                    mode = re.search(r" mode=(\S+)", line)
                    note(miss, ts, mode.group(1) if mode else "?")
                    chain = line.split(" at ", 1)[1].strip() if " at " in line else "?"
                    miss_by_reason_site[(m.group(2), last_site(chain))] += 1
                    continue

                m = ROTARY_RE.search(line)
                if m:
                    rotary.total = max(rotary.total, int(m.group(1)))
                    note(rotary, ts)
                    cos_id, sin_id = int(m.group(4)), int(m.group(5))
                    rotary_pairs[(cos_id, sin_id)] += 1
                    rotary_sites[last_site(m.group(6))] += 1
                    if m.group(2):
                        rotary_with_kv += 1
                    if primed_ids and cos_id in primed_ids:
                        rotary_matched += 1
                    else:
                        rotary_unmatched[last_site(m.group(6))] += 1
                    continue

                m = CMP_RE.search(line)
                if m:
                    st = cmp_hit if m.group(3) == "hit" else cmp_miss
                    st.total = max(st.total, int(m.group(1)))
                    mode = re.search(r" mode=(\S+)", line)
                    note(st, ts, mode.group(1) if mode else "?")
                    continue

                m = GCS_RE.search(line)
                if m:
                    rank = re.search(r"DP\d+", line[:60])
                    try:
                        sites = ast.literal_eval(m.group(2))
                    except (ValueError, SyntaxError):
                        sites = {}
                    gcs_last[rank.group(0) if rank else "?"] = (
                        int(m.group(1)),
                        sites,
                    )

    cmp_total = max(cmp_hit.total, cmp_miss.total)

    def mode_row(st: Stats, label: str) -> str:
        if not st.by_mode:
            return ""
        top = ", ".join(f"{k}:{v}" for k, v in st.by_mode.most_common(5))
        return f"\n  {label:<14}: {top}"

    print("=" * 66)
    print("rope 调试日志汇总")
    print("=" * 66)
    print(f"primes 总数     : {prime.total}  (采样 {prime.lines} 条)"
          f"{mode_row(prime, '按 mode')}")
    print(f"rotary 启动总数 : {rotary.total}  (采样 {rotary.lines} 条,"
          f"带 kv {rotary_with_kv}; {prime.total and rotary.total // prime.total}"
          f"/forward)")
    print(f"MISS 总数       : {miss.total}{mode_row(miss, '按 mode')}")
    print(f"CMP 总数        : {cmp_total}  (compressor 路径 gather)")
    if cmp_hit.lines or cmp_miss.lines:
        rate = cmp_hit.lines / max(cmp_hit.lines + cmp_miss.lines, 1)
        print(f"  采样命中      : hit {cmp_hit.lines} / miss {cmp_miss.lines}"
              f" ({rate:.0%} hit)")
        print(f"  miss 按 mode  : {dict(cmp_miss.by_mode) or '(采样内无 miss)'}")

    if prime.by_mode.get("IDLE"):
        print(f"  IDLE prime    : {prime.by_mode['IDLE']} 条采样 -- 空转 forward"
              " 也在 prime,可另行关闭")

    print()
    print("-" * 66)
    print("get_cos_sin 按调用点精确计数 (GCS, 跨 rank 求和, 最终累计值)")
    print("-" * 66)
    if gcs_last:
        merged: collections.Counter = collections.Counter()
        for _total, sites in gcs_last.values():
            for k, v in sites.items():
                merged[k] += v
        total_calls = sum(merged.values())
        ranks = len(gcs_last)
        print(f"  共 {total_calls} 次 / {ranks} 个 rank"
              f" (平均 {total_calls // max(ranks, 1)}/rank)")
        for k, n in merged.most_common(12):
            print(f"    {n:7d}  {k}")
        per_fwd = total_calls // max(ranks, 1) / max(prime.total, 1)
        print(f"  平均每 forward {per_fwd:.1f} 次 get_cos_sin;"
            " 修复后预期 <=5(prime 2 + cmp miss 2 + 余量)。")
        print("  出现非 prime/cmp 的调用点(尤其 fp32 c=float32)= 残留直连路径。")
    else:
        print("  无 GCS dump 行(每 1000 次调用输出一条,运行时间过短?)。")

    print()
    print("-" * 66)
    print("共享健康度 (rotary cos_id 关联回 prime 记录的 tensor id)")
    print("-" * 66)
    if rotary.lines and primed_ids:
        pct = rotary_matched / max(rotary.lines, 1)
        print(f"  prime 记录 id 数: {len(primed_ids)};"
              f" rotary 采样 {rotary.lines} 条中 {rotary_matched} ({pct:.0%})"
              " 匹配 prime")
        if prime.lines * 2 < prime.total:
            print("  (prime 行数远小于 forward 数时未匹配率会被高估;"
                "prime 对前 5000 个 forward 全量记录)")
        if rotary_unmatched:
            print("  未匹配(= memo 外的现场 gather)调用点:")
            for site, n in rotary_unmatched.most_common(6):
                print(f"    {n:5d}  {site}")
    else:
        print("  无 entries= 数据(旧日志)或无 rotary 采样,跳过。")

    print()
    print("-" * 66)
    print("调用点 top (rotary / MISS)")
    print("-" * 66)
    for site, n in rotary_sites.most_common(10):
        print(f"    {n:5d}  rotary  {site}")
    for (reason, site), n in miss_by_reason_site.most_common(8):
        print(f"    {n:5d}  MISS {reason}  @  {site}")

    print()
    print("-" * 66)
    print("时间分布 (区分启动捕获期 / 稳态; 回放期无 Python 日志)")
    print("-" * 66)
    if log_first_ts:
        print(f"  日志时间跨度  : {log_first_ts} -> {log_last_ts}")
        for name, st in (("prime", prime), ("MISS", miss), ("CMP-miss", cmp_miss)):
            if st.first_ts:
                print(f"  {name:<10} 首次/末次: {st.first_ts} / {st.last_ts}"
                      f"  (采样 {st.lines} 条)")
        if cmp_miss.by_minute:
            print("  CMP-miss 按分钟 (前 8 桶):")
            for b, n in sorted(cmp_miss.by_minute.items())[:8]:
                print(f"    {b}  {'#' * min(n, 40)} {n}")

    print()
    print("=" * 66)
    print("结论")
    print("=" * 66)
    if miss.total == 0:
        print("结果 B:rope_cos_sin 零未命中,memo 生效。")
        if cmp_miss.lines:
            mr = cmp_miss.lines / max(cmp_miss.lines + cmp_hit.lines, 1)
            if mr < 0.1:
                print(f"  CMP miss 占比 {mr:.0%}(每 forward 首层写入,设计内)。")
            else:
                print(f"  注意:CMP miss 占比 {mr:.0%},远超首层写入比例;"
                    "确认 prime 是否覆盖该 forward(TBO 不缓存)。")
        else:
            print("  CMP 采样内零 miss(或无 CMP 日志)。")
        if rotary.lines and primed_ids:
            pct = rotary_matched / max(rotary.lines, 1)
            if pct >= 0.95:
                print(f"  rotary {pct:.0%} 匹配 prime 的 tensor -> 层间共享成立。")
            else:
                print(f"  注意:仅 {pct:.0%} rotary 匹配 prime,见上面未匹配调用点。")
        print("  若 profile 仍有 gather 而 MISS=0:残留只可能是 prime 本身"
            "(每 forward 一簇)、CMP miss(每 forward 2 次)、或 graph 捕获期"
            "固化 -- 用上面时间分布判断 miss 是否集中在启动阶段。")
    else:
        print(f"结果 A:{miss.total} 次未命中。按 reason 定位:")
        print("    no_memo_on_forward_batch -> prime 没在该 forward_batch 上跑")
        print("    key_not_found            -> freqs_cis 对象或 dtype 不一致")
        print("    positions_object_mismatch-> 层内 positions 与 prime 的不是同一对象")
        print("  miss 集中在启动阶段(graph 捕获 warmup)时,回放每次都会重放"
            "这些 fallback,稳态日志反而安静 -- 看时间分布确认。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
