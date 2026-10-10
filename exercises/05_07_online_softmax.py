"""Exercise 05-07 · Online softmax (the trick inside FlashAttention).

Problem
  softmax(x)_i = e^(x_i - m) / l, where m = max(x) and l = sum_j e^(x_j - m).
  Normally you need the WHOLE row first to know m and l. FlashAttention can't hold a long row,
  so it sees the scores in BLOCKS and must get the exact same answer.

Trick: keep two running numbers while walking the blocks
  m = biggest score seen so far (never goes down)
  l = sum of e^(score - m) over everything seen so far
  When a block brings a bigger max, everything already in l was measured against the OLD max,
  so rescale it:   l = l * e^(m_old - m_new)   then add the new block's terms.

Worked example: blocks [2, 1] then [3, 0]
  block 1: m = 2, l = e^0 + e^-1                         ≈ 1.368
  block 2: m_new = 3, rescale 1.368 * e^(2-3) ≈ 0.503, add e^0 + e^-3 ≈ 1.050  ->  l ≈ 1.553
  full row check: e^-1 + e^-2 + e^0 + e^-3               ≈ 1.553   (same)

Part A: compute (m, l) block by block, then the softmax weights.
Part B (the real FlashAttention move): also keep a running weighted sum of VALUES, so you get
        the attention output sum_i softmax_i * v_i in ONE pass, without ever storing the weights.
        It gets rescaled exactly like l.
"""
import math


def online_softmax(scores, block_size):
    """Part A: return the softmax weights, computing m and l block by block."""
    m = float("-inf")   # running max: starts below any score
    l = 0.0             # running sum of e^(score - m) for every score seen so far

    for start in range(0, len(scores), block_size):
        block = scores[start:start + block_size]

        # Step 1 · New running max. Compare with the OLD max too: it must never go down,
        # otherwise the rescale factor below would exceed 1 and could overflow in FP16.
        m_new = max(m, max(block))

        # Step 2 · l was summed relative to the old max. Multiplying by e^(m - m_new) re-expresses
        # every old term relative to the new max: e^(x - m) * e^(m - m_new) = e^(x - m_new).
        # Then add this block's own terms, already relative to m_new.
        l = l * math.exp(m - m_new) + sum(math.exp(x - m_new) for x in block)

        # Step 3 · From now on, everything is measured against the new max.
        m = m_new

    # Step 4 · With the final m and l (same as the full row's), each weight is e^(x - m) / l.
    return [math.exp(x - m) / l for x in scores]


def online_attention(scores, values, block_size):
    """Part B: return sum_i softmax(scores)_i * values_i in one pass (values are plain numbers here).

    Same idea as Part A, plus acc = sum of e^(score - m) * value. At the end, acc / l is the
    weighted average of the values: the attention output. The weights are never stored.
    """
    m = float("-inf")   # running max
    l = 0.0             # running sum of e^(score - m)            (the softmax denominator)
    acc = 0.0           # running sum of e^(score - m) * value    (the softmax numerator, times values)

    for start in range(0, len(scores), block_size):
        block_s = scores[start:start + block_size]
        block_v = values[start:start + block_size]

        # New running max (never decreases)
        m_new = max(m, max(block_s))

        # Both l and acc were summed against the old max, so BOTH need the same rescale.
        scale = math.exp(m - m_new)

        # Denominator: rescale old terms, add this block's e^(s - m_new)
        l = l * scale + sum(math.exp(s - m_new) for s in block_s)

        # Numerator: rescale old terms, add this block's e^(s - m_new) * v
        acc = acc * scale + sum(math.exp(s - m_new) * v for s, v in zip(block_s, block_v))

        m = m_new

    # Weighted average of the values = numerator / denominator
    return acc / l


def naive_softmax(scores):
    m = max(scores)
    e = [math.exp(x - m) for x in scores]
    return [v / sum(e) for v in e]


if __name__ == "__main__":
    s = [2, 1, 3, 0]
    print(online_softmax(s, 2))                      # must match the next line
    print(naive_softmax(s))
    v = [10, 20, 30, 40]
    print(online_attention(s, v, 2))                 # must match the next line (24.71)
    print(sum(w * x for w, x in zip(naive_softmax(s), v)))
    big = [12, 5, 40, -3, 39, 7]                     # e^40 overflows FP16; the trick never computes it
    print(online_attention(big, [1, 2, 3, 4, 5, 6], 2), sum(w * x for w, x in zip(naive_softmax(big), [1, 2, 3, 4, 5, 6])))
