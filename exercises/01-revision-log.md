# Revision 01 · What Is Inference Engineering

*Interactive revision of [lesson 01](../notes/01-what-is-inference-engineering.md): each question as asked, how I reasoned it out (my words, written in full), where I slipped, and the one-line takeaway.*

## Q1 · Can one user get 300 tokens/s? <span class="badge ok">solid</span>

*Covers: 01, 16*

> [!NOTE]
> We serve Llama 3 8B in BF16 on one H100, with only one user online. The PM wants to advertise "300 tokens/s streaming". Can we promise it?

**How I solved it**

1. The model has about 8 billion parameters, and BF16 stores each in 2 bytes, so the weights are about **16 GB**.
2. An H100's memory bandwidth is about **3.35 TB/s**.
3. Every decode step reads all the weights once: 16 × 10⁹ bytes ÷ 3.35 × 10¹² bytes/s = (16 ÷ 3.35) × 10⁻³ s ≈ **4.8 ms**.
4. In one second that's 1 ÷ 4.8 ms ≈ 200 steps, one token each: about **200 tokens/s at most**.
5. 300 is above that ceiling, so the answer is **no**.

**What to add next time**

- Name the limiting resource out loud: **memory bandwidth**, not compute. The math units are mostly idle.
- 200 tokens/s is a **ceiling**, not the expected speed. Real batch-1 runs land 3–4× slower (lesson 16), and the ceiling drops as the KV cache grows.
- To reach 300, read fewer bytes: 8-bit weights are 8 GB → ~2.4 ms → a ceiling of ~415 tokens/s.

> [!IMPORTANT]
> Decode time per step ≈ bytes read ÷ memory bandwidth.

## Q2 · 64 users at once: step time and speed <span class="badge mid">partial</span>

*Covers: 01, 16*

> [!NOTE]
> Same GPU and model, but 64 users are chatting at once, each with about 4,096 tokens of context in their KV cache (128 KB per token). How long is one decode step, what is the total tokens/s, and what does each user get?

**How I solved it (with hints)**

1. I couldn't recall the formula, so I went back to the principle from Q1: **time per step ≈ bytes read in that step ÷ bandwidth**. The only real question is which bytes get read.
2. **Weights:** batched, so the 16 GB is read **once per step for everyone**, not once per user.
3. **KV cache:** each user's new token attends to that user's own past tokens, so every user's whole cache is read: 4,096 × 128 KB ≈ 0.5 GB each, × 64 ≈ **32 GB** (about 34 GB in decimal units).
4. **Total read per step:** 16 + 32 ≈ 48 GB, ÷ 3.35 TB/s ≈ **14 ms** (the lessons get 15 ms with exact sizes).
5. **Per user:** 14 ms = 0.014 s, so 1 ÷ 0.014 ≈ **71 tokens/s**.
6. **Total:** each step gives all 64 users a token, so total = batch × (1 ÷ step time) = batch ÷ step time: 64 ÷ 0.014 s = 71 × 64 ≈ **4,500 tokens/s**.
7. **Versus batch 1:** the GPU makes 4,500 ÷ 200 ≈ **22× more**, while each user slows from 200 to 71, only **~2.8× slower**.

> [!WARNING]
> First wrote KV as 128 KB × 64 ≈ 8 MB: 128 KB is **one token**, each user has 4,096. Then wrote 1 ÷ 14 ≈ 7 tokens/s: 14 ms is 0.014 s, so it's 71. Sanity check: a step ~3× longer than batch 1's should give ~⅓ of 200, not ~1/30.

The 16 GB weight read is shared by everyone; only the KV cache is per user. The price of batching is KV memory (32 GB here), growing with batch × context.

> [!IMPORTANT]
> Step time ≈ (weights + batch × context × KV per token) ÷ bandwidth. Total tokens/s = batch ÷ step time; each user gets 1 ÷ step time.

> [!TIP]
> Per-request KV = tokens in the conversation × 128 KB. Always sanity-check ms → tokens/s against batch 1 (4.8 ms → 200).

## Q3 · Cost per million tokens <span class="badge mid">partial</span>

*Covers: 01*

> [!NOTE]
> The H100 rents for $3/hour. What does 1 million output tokens cost at ~200 tokens/s (one user), and at ~4,500 tokens/s (64 users batched)?

**Step-by-step answer.** Cost = price per hour × hours of GPU needed.

| Step | One user (200 tok/s) | 64 users (4,500 tok/s) |
|---|---|---|
| Time for 1M tokens | 1,000,000 ÷ 200 = **5,000 s** | 1,000,000 ÷ 4,500 ≈ **222 s** |
| Convert to hours | 5,000 ÷ 3,600 ≈ **1.39 h** | 222 ÷ 3,600 ≈ **0.06 h** |
| Cost = $3/h × hours | **$4.17** | **$0.18** |

Ratio: 4.17 ÷ 0.18 ≈ **22× cheaper**, the same factor as the throughput gain, because the GPU's hourly price doesn't change.

> [!WARNING]
> Divided 1,000,000 ÷ 200 as 500 instead of 5,000, and divided $3 by the hours instead of multiplying (2 hours at $3/h is $6, not $1.50).

> [!IMPORTANT]
> Cost per 1M tokens = $ per GPU-hour ÷ (tokens/s × 3,600) × 10⁶. Cost per token falls exactly as fast as throughput rises.

> [!TIP]
> Write units next to every number (s, h, $/h) so an upside-down formula shows itself. Sanity check: "about an hour, so about $3?"

## Q4 · Why is prefill of 500 tokens only ~2× one decode token? <span class="badge mid">partial</span>

*Covers: 01, 15*

> [!NOTE]
> Prefill pushes a 500-token prompt through in ~8 ms; one decode token takes ~4.8 ms. Why so close? Which resource limits each phase?

**My explanation (concept, correct).** In prefill all the prompt tokens are available up front, so nothing waits and the work runs in parallel: the compute units stay busy, so prefill is compute-intensive. In decode only one token at a time flows through all the weights, so the GPU spends most of its time waiting for data: memory-intensive.

**Gap:** I couldn't show *with numbers* that decode is memory-bound.

**Step-by-step proof:** compare math time with byte time; the bigger one sets the speed.

1. **Facts:** each parameter does 2 FLOPs per token (multiply + add). H100 ≈ 989 TFLOPS and 3.35 TB/s.
2. **Decode math:** 2 × 8B = 16 × 10⁹ FLOPs. GPU: 989 TFLOPS = 989 × 10¹² FLOP/s ≈ 1,000 × 10¹² = 10¹⁵ FLOP/s. Divide: 16 × 10⁹ ÷ 10¹⁵ = 16 × 10⁻⁶ s = **16 µs**.
3. **Decode bytes:** 16 × 10⁹ bytes ÷ 3.35 × 10¹² bytes/s = (16 ÷ 3.35) × 10⁻³ s = **4.8 ms** (3.35 × 5 ≈ 16.75, so 16 ÷ 3.35 ≈ 4.8).
4. 4.8 ms ÷ 16 µs ≈ **300×**: decode spends ~300× more time waiting for bytes than doing math. Math units idle >99%: **memory-bound**.
5. **Ridge point:** 989 × 10¹² FLOP/s ÷ 3.35 × 10¹² B/s; tera and seconds cancel → 989 ÷ 3.35 ≈ **295 FLOP/byte** (3.35 × 300 ≈ 1,000). Batch-1 decode does ~1.
6. **Prefill 500 tokens:** weights still read once (4.8 ms), but math = 500 × 16 µs = **8 ms**. Math > bytes: **compute-bound**, ≈ 8 ms.

> [!WARNING]
> Said "decode is 300× slower": it's "300× more time on bytes than on math".

> [!IMPORTANT]
> Time ≈ max(math time, byte time). Batch-1 decode: bytes win by ~300× (memory-bound). Prefill: one weight read shared by 500 tokens, so math wins (compute-bound).

> [!TIP]
> Powers-of-ten division: giga ÷ tera = 10⁻³, so 16 G ÷ 1,000 T = 16 µs. Say "max(math, bytes)" out loud.

## Exercise · Napkin calculator in code <span class="badge mid">partial</span>

*Covers: 01, 16*

> [!NOTE]
> Write `decode_step(params_b, bytes_per_param, batch, ctx_tokens, ...)` returning step time (ms), total tokens/s, per-user tokens/s, and whether it's memory- or compute-bound.

**Gap:** I knew the hand math but couldn't structure it as code: where to start, in what order.

**The structure to reuse** (same chain of thought as by hand):

0. **Read the inputs:** what each means physically (params in billions, bytes per param = precision, batch = users per step, ctx = tokens in each user's cache).
1. **Normalize** to plain units: params × 10⁹, TB/s × 10¹², TFLOPS × 10¹².
2. **Bytes per step:** weights once + batch × ctx × KV per token (KB × 1024) → ÷ bandwidth = byte time.
3. **Math per step:** 2 × params × batch FLOPs → ÷ FLOPs/s = math time.
4. **Bottleneck:** step = max(byte time, math time); bound = whichever was bigger.
5. **Speeds:** per user = 1 ÷ step; total = per user × batch.

```python
def decode_step(params_b, bytes_per_param, batch, ctx_tokens,
                kv_kb_per_token=128, bw_tbs=3.35, tflops=989):
    # Step 1 · Normalize to plain units
    params = params_b * 10**9          # parameters
    bw = bw_tbs * 10**12               # bytes per second
    flops = tflops * 10**12            # FLOPs per second

    # Step 2 · Bytes: weights once (shared) + every user's whole cache
    weight_bytes = params * bytes_per_param
    kv_bytes = batch * ctx_tokens * kv_kb_per_token * 1024
    t_bytes = (weight_bytes + kv_bytes) / bw          # seconds

    # Step 3 · Math: 2 FLOPs per param per token, one token per user
    flops_needed = 2 * params * batch
    t_math = flops_needed / flops                     # seconds

    # Step 4 · Bottleneck: the slower one sets the pace
    step_s = max(t_bytes, t_math)
    bound = "memory" if t_bytes >= t_math else "compute"

    # Step 5 · Speeds: each step gives every user one token
    per_user_tok_s = 1 / step_s
    total_tok_s = per_user_tok_s * batch
    return step_s * 1000, total_tok_s, per_user_tok_s, bound

decode_step(8, 2, 1, 0)       # (4.78 ms, 209 tok/s, 209, 'memory')
decode_step(8, 2, 64, 4096)   # (15.0 ms, 4,257 tok/s, 66.5, 'memory')
```

> [!WARNING]
> First wrote `t_math = t_bytes / tflops` (a time divided by a speed: meaningless units). Fix: FLOPs needed ÷ FLOPs per second. Also returned seconds instead of ms (× 1000).

> [!IMPORTANT]
> When stuck on code, write the hand-solution steps as comments first, then fill one line under each.

## Q5 · At what batch does decode become compute-bound? <span class="badge ok">solid</span>

*Covers: 01*

> [!NOTE]
> With zero context, raise the batch until `decode_step` says 'compute'. Where does it flip, and which number does that match?

**My answer:** around 300, the ridge point. **Checked with the code:** batch 295 → memory, 296 → compute; ridge = 989 ÷ 3.35 ≈ 295.

**Why batch = ridge:** in BF16 one step does 2 × P × B FLOPs and reads 2 × P bytes of weights, so FLOPs per byte = **B**. Decode turns compute-bound once B passes the ridge.

**Caveat:** with real context every user adds KV bytes too, so the flip comes much later or never: batch 296 at 4k context is still memory-bound.
