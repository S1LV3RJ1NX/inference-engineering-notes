# Revision 05-07 · Attention Math + Transformer Architecture

*Interactive revision of [lesson 05](../notes/05-dot-products-softmax-attention-math.md) and [lesson 07](../notes/07-transformer-architecture-refresher.md): each question as asked, the answer, where I slipped, and the one-line takeaway.*

## Q1 · Derive KV bytes per token, and what GQA buys <span class="badge ok">solid</span>

*Covers: 05, 07, 19*

> [!NOTE]
> Llama 3 8B: 32 layers, 8 KV heads, head dim 128, BF16. How many bytes of KV cache per token? Llama 2 7B is the same but with 32 KV heads (no GQA): what's its KV per token, and what does that do to users per GPU?

**How I solved it (powers of two)**

1. KV/token = 2 (K and V) × 32 (layers) × 8 (KV heads) × 128 (head dim) × 2 (bytes, BF16).
2. In powers of two: 2¹ × 2⁵ × 2³ × 2⁷ × 2¹ = 2¹⁷ = 131,072 bytes = **128 KiB**.
3. Llama 2 7B swaps 8 heads (2³) for 32 (2⁵): 2¹⁹ = **512 KiB**, 4× more.
4. Same H100 and ~53 GB cache pool (lesson 18): 4× the bytes per token means **4× fewer tokens**, ~100k instead of ~400k, so ~25 users at 4k context instead of ~100.

> [!WARNING]
> Gave the exponents without naming the factors. In an interview, say what each one is: K and V, layers, KV heads, head dim, bytes.

> [!IMPORTANT]
> KV/token = 2 × layers × KV heads × head dim × bytes. GQA (8 KV heads instead of 32) cuts it 4×, so 4× more users fit.

> [!TIP]
> Powers of two make these products instant: 32 = 2⁵, 8 = 2³, 128 = 2⁷, and 2¹⁰ = 1 KiB.

## Q2 · Parameters in one transformer block <span class="badge mid">partial</span>

*Covers: 07, 12*

> [!NOTE]
> Llama 3 8B: h = 4,096, MLP size i = 14,336, KV width 8 × 128 = 1,024. Q and O are h × h; K and V are h × 1,024; the MLP has three h × i matrices. How many parameters per block, and which part dominates?

**Answer** (asked for the worked answer; h = 2¹², KV width = 2¹⁰, i = 3.5 × 2¹²)

| Part | Matrices | Count |
|---|---|---|
| Q + O | 2 × h × h = 2 × 2¹² × 2¹² = 2²⁵ | ≈ 33.6M |
| K + V | 2 × h × 1,024 = 2 × 2¹² × 2¹⁰ = 2²³ | ≈ 8.4M |
| **Attention** | | **≈ 42M** |
| **MLP** (gate, up, down) | 3 × h × i = 3 × 2¹² × 3.5 × 2¹² = 10.5 × 2²⁴ | **≈ 176M** |
| **Block** | | **≈ 218M** (MLP ≈ 81%) |

- Check: 32 × 218M ≈ 6.98B, plus embedding + LM head 2 × 128,256 × 4,096 ≈ 1.05B → **8.03B**.
- **Why it matters:** decode time = bytes ÷ bandwidth and the MLP holds ~81% of each block's bytes, so it's the first target for quantization and for MoE. Speeding up attention alone barely helps (Amdahl).

> [!IMPORTANT]
> Block ≈ 2h² + 2h·(KV width) + 3h·i. MLPs hold ~70% of all weights (~81% per block), so they dominate decode's byte reads.

> [!TIP]
> Re-drill the shape of each matrix: Q, O are h × h; K, V are h × (KV heads × head dim); the MLP is three h × i.

## Q3 · Is MoE cheap to host? (Mixtral 8x7B) <span class="badge mid">partial</span>

*Covers: 07, 18*

> [!NOTE]
> "Mixtral 8x7B only uses ~13B parameters per token, so host it like a 13B model on one 80 GB H100 in BF16." Mixtral stores 46.7B parameters, 12.9B active per token. Right or wrong, for memory and for compute?

**Answer**

- **Memory follows total parameters:** 46.7B × 2 bytes ≈ **93 GB** in BF16 > 80 GB, so **it doesn't fit**. Options: FP8 (~47 GB, leaving ~72 − 47 − 3 ≈ 22 GB for KV cache) or 2 GPUs.
- **Compute follows active parameters:** 2 × 12.9B ≈ **26 GFLOPs per token**, like a 13B model. That part is true.
- **Nuance:** at batch 1, decode reads only the active experts' weights, so it's fast like a 13B. In a big batch, tokens route to different experts, so nearly all 93 GB gets read each step and the advantage shrinks.

> [!WARNING]
> Stated the concept (memory = total, compute = active) but didn't finish with the verdict: 93 GB doesn't fit in 80 GB.

> [!IMPORTANT]
> MoE splits memory from compute: host all experts (total params), compute only the active ones. Cheap per token is not cheap to host.

> [!TIP]
> Always end a "can we?" question with the number and a yes/no.

## Q4 · Why ÷ √d, and why subtract the max in softmax? <span class="badge mid">partial</span>

*Covers: 05, 13*

> [!NOTE]
> Attention computes q · k ÷ √d before softmax. What goes wrong without ÷ √d (d = 128)? Softmax kernels subtract the row's maximum before e^x: why, and does it change the result?

**Answer**

- **÷ √d:** a dot product of d random unit-variance numbers has variance d, so scores spread by √128 ≈ 11. Dividing by √d brings the spread back to ~1. Without it, softmax saturates into **nearly one-hot**: one token takes all the weight, attention stops blending, and gradients vanish in training.
- **Subtract the max:** prevents **overflow**: FP16 tops out at 65,504, and e¹² ≈ 162,755 already overflows.
- **Same result:** e^(x − m) = e^x · e^(−m), and the e^(−m) cancels between numerator and denominator. Only differences between scores matter (lesson 13).

> [!WARNING]
> Said √d "centers the mean": it rescales the spread, not the mean. Didn't give the consequence (one-hot softmax) or say that subtracting the max leaves the weights unchanged.

> [!IMPORTANT]
> ÷ √d keeps scores at unit spread so softmax doesn't go one-hot; subtracting the max avoids FP16 overflow without changing the weights.

> [!TIP]
> For every "why do we do X?" give both halves: what breaks without it, and what it costs (here: nothing).

## Exercise · Online softmax and one-pass attention <span class="badge mid">partial</span>

*Covers: 05*

> [!NOTE]
> FlashAttention sees scores in blocks. Compute exact softmax weights block by block (Part A), then the attention output Σ softmax_i × v_i in one pass without storing the weights (Part B).

**The idea:** keep a running max `m` and a running sum `l` of e^(score − m). When a block raises the max, rescale everything summed so far by e^(m_old − m_new), because e^(x − m_old) × e^(m_old − m_new) = e^(x − m_new). For attention, keep `acc` = Σ e^(score − m) × value, rescaled the same way; the output is `acc / l`.

**Worked example:** blocks [2, 1] then [3, 0]. After block 1: m = 2, l = e⁰ + e⁻¹ ≈ 1.368. Block 2: m = 3, l = 1.368 × e⁻¹ + e⁰ + e⁻³ ≈ 0.503 + 1.050 = 1.553, the same as the full row.

```python
def online_attention(scores, values, block_size):
    m, l, acc = float("-inf"), 0.0, 0.0      # running max, denominator, numerator × values
    for start in range(0, len(scores), block_size):
        block_s = scores[start:start + block_size]
        block_v = values[start:start + block_size]
        m_new = max(m, max(block_s))         # never decreases
        scale = math.exp(m - m_new)          # re-express old sums against the new max
        l = l * scale + sum(math.exp(s - m_new) for s in block_s)
        acc = acc * scale + sum(math.exp(s - m_new) * v for s, v in zip(block_s, block_v))
        m = m_new
    return acc / l

online_attention([2, 1, 3, 0], [10, 20, 30, 40], 2)   # 24.71, same as the naive version
```

> [!WARNING]
> Part A: wrote `m_new = max(block)`, forgetting the old max; the running max must never decrease (the test passed only because the last block held the biggest score). Part B: needed the parallel spelled out, then forgot to rescale `acc` (`acc + …` instead of `acc * scale + …`), giving 31.78 instead of 24.71.

> [!IMPORTANT]
> Online softmax: running max + running sum, rescale old sums by e^(m_old − m_new) whenever the max rises. Anything summed against the max (denominator and numerator) gets the same rescale.

> [!TIP]
> Every running quantity measured "relative to m" must be rescaled together when m changes.
