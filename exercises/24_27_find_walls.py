"""Find the three walls from a concurrency sweep (lessons 25 and 27).

Steps:
1. Saturation: walk neighbouring rows; the first row whose next doubling adds
   less than 5% throughput is where the GPU stops scaling.
2. SLO wall: find the last row that passes the P99 TPOT target and the first
   that fails; interpolate between them in log2(concurrency), since the sweep
   doubles each row.
3. Memory wall: requests that fit at their final length = pool ÷ length.
4. Capacity: the lowest wall wins (lesson 27: c* = min of the walls).
"""

import math

SWEEP = [
    # (concurrency, total tok/s, P99 TPOT ms)
    (1, 150, 6.5),
    (2, 290, 6.6),
    (4, 560, 6.8),
    (8, 1100, 7.2),
    (16, 2000, 8.1),
    (32, 3400, 9.6),
    (64, 4600, 13.8),
    (128, 5000, 26.0),
    (256, 5150, 52.0),
]


def find_walls(points, tpot_slo_ms, pool_tokens, tokens_per_request):
    """Return (walls, limiting_wall, capacity).

    points: list of (concurrency, tok_s, p99_tpot_ms), concurrency doubling each row.
    tpot_slo_ms: P99 TPOT target in ms (e.g. 10 ms = 100 tok/s per user).
    pool_tokens: KV cache size in tokens (vllm:cache_config_info kv_cache_size_tokens).
    tokens_per_request: final length of one request (prompt + output).
    """
    # 1. Saturation: fractional throughput gain per doubling, e.g. 5,150 / 5,000 - 1 = 3%.
    saturation = None
    for (c, tok, _), (c_next, tok_next, _) in zip(points, points[1:]):
        gain = tok_next / tok - 1
        if gain < 0.05:
            saturation = c
            break

    # 2. SLO wall: the pair where this row passes and the next fails.
    slo = None
    for (c, _, t), (c_next, _, t_next) in zip(points, points[1:]):
        if t <= tpot_slo_ms < t_next:
            frac = (tpot_slo_ms - t) / (t_next - t)  # 0 = at c, 1 = at c_next
            log_c = math.log2(c) + frac * (math.log2(c_next) - math.log2(c))
            slo = math.floor(2**log_c)  # back from log2 to users, rounded down
            break

    # 3. Memory: requests that fit when every one has grown to its final length.
    memory = pool_tokens // tokens_per_request

    # 4. Capacity: smallest wall, skipping walls the sweep never hit (None).
    walls = {k: v for k, v in {"saturation": saturation, "slo": slo, "memory": memory}.items()
             if v is not None}
    limiting = min(walls, key=walls.get)
    return walls, limiting, walls[limiting]


if __name__ == "__main__":
    print(find_walls(SWEEP, 10, 532_768, 2_000))  # chat-like: SLO wall at 34
    print(find_walls(SWEEP, 50, 532_768, 7_000))  # agent-like: memory wall at 76
