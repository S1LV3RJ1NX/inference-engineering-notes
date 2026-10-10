# Revision 01-02 · What Is Inference Engineering + The Inference Stack

*Interactive revision of [lesson 01](../notes/01-what-is-inference-engineering.md) and [lesson 02](../notes/02-the-inference-stack.md): each question as asked, how I reasoned it out (my words, written in full), where I slipped, and the one-line takeaway.*

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

## Q5 · At what batch does decode become compute-bound? <span class="badge ok">solid</span>

*Covers: 01*

> [!NOTE]
> With zero context, raise the batch size. At what batch does decode flip from memory-bound to compute-bound, and which number does that match?

**Answer:** around **batch 300**, the ridge point. Ridge = 989 ÷ 3.35 ≈ 295, so batch 295 is still memory-bound and 296 is compute-bound.

**Why batch = ridge:** in BF16 one step does 2 × P × B FLOPs and reads 2 × P bytes of weights, so FLOPs per byte = **B**. Decode turns compute-bound once B passes the ridge.

**Caveat:** with real context every user adds KV bytes too, so the flip comes much later or never: batch 296 at 4k context is still memory-bound.

> [!IMPORTANT]
> Batch-1 decode does ~1 FLOP/byte; the ridge is ~300. The weight read is shared, so arithmetic intensity ≈ batch size.

## Q6 · Autoscaling on the wrong signal <span class="badge mid">partial</span>

*Covers: 02, 18*

> [!NOTE]
> Four replicas of Llama 3 8B sit behind an autoscaler that adds a replica when CPU utilization passes 70%. During a spike, users wait 5+ seconds for the first token, but CPU shows 12% on every server, so nothing scales. What's wrong, and what should the autoscaler watch instead?

**Answer**

- **CPU is the wrong signal.** The GPU does the work, so the CPU sits mostly idle even when a replica is overloaded.
- **Watch two engine signals instead:**
  1. **Queue length** (requests waiting, or time spent waiting): the 5 seconds is mostly time in the engine's queue, because prefill itself takes milliseconds.
  2. **KV cache usage:** each user holds its own cache, so a full cache means no room for more users. Use the engine's metric, not raw GPU memory: engines reserve ~90% of GPU memory at startup, so `nvidia-smi` always looks "full".
- **Scale early, on the trend:** a new replica must load 16 GB of weights before it serves anyone.

> [!WARNING]
> Said "GPU memory utilization" at first (always ~90% because of preallocation). Missed the queue signal until the hint.

> [!IMPORTANT]
> Autoscale on queue length and KV cache usage, not CPU. Scale early, because new replicas take seconds to load.

> [!TIP]
> When a latency number looks huge, ask where the time goes: compute (prefill/decode) or waiting (queue).

## Q7 · How many users fit on one GPU? <span class="badge ok">solid</span>

*Covers: 02, 18, 19*

> [!NOTE]
> One H100 (80 GB) serves Llama 3 8B in BF16 (16 GB of weights). Ignoring runtime overhead, how many tokens of KV cache fit (128 KB per token), and how many 4,096-token conversations can run at once?

**How I solved it**

1. Memory left for the cache: 80 − 16 = **64 GB** (the weights take 16).
2. Tokens: 64 × 10⁹ bytes ÷ 128 × 10³ bytes = 0.5 × 10⁶ ≈ **500k tokens** (488k with 128 KB = 131,072 bytes).
3. Users: 500 × 10³ ÷ 4,096 ≈ **122 users** (~119 with the exact numbers).

> [!WARNING]
> Wrote the division as 16 × 10⁹ ÷ 128 × 10³: 16 GB is the weights. The cache gets 80 − 16 = 64 GB (the final number was right).

**Caveat:** this is the naive upper bound. Runtime, activations and headroom take ~11 GB more, so a real engine gets ~400k tokens (lesson 18).

> [!IMPORTANT]
> KV capacity = (HBM − weights − overhead) ÷ KV per token. Users = capacity ÷ tokens per conversation.

## Q8 · Kernel launch overhead <span class="badge mid">partial</span>

*Covers: 02, 17, 18*

> [!NOTE]
> One decode step launches ~300 kernels at ~4 µs of CPU time each; the ideal step (bytes ÷ bandwidth) is 4.8 ms. How much time goes to launches, as a share of the step, and how do you cut it?

**Answer**

- Launch time: 300 × 4 µs = 1,200 µs = **1.2 ms**, which is 1.2 ÷ 4.8 = **25%** of the ideal step.
- **Fix 1, kernel fusion:** merge several kernels into one, so there are fewer launches and fewer round trips to memory.
- **Fix 2, CUDA graphs:** record the step's launches once and replay them with a single launch, so ~300 CPU launches become ~1. Needs fixed tensor shapes (lesson 17: preallocated KV cache) and costs 1–5 GB of memory (lesson 18).

> [!WARNING]
> Said 40% (it's 25%: 4.8 ÷ 4 = 1.2). Offered "reduce bytes travelled" as a fix: that speeds up the bandwidth part, not launch overhead. Couldn't recall CUDA graphs.

> [!IMPORTANT]
> Launch overhead = kernels × µs per launch. Cut it with fusion (fewer kernels) or CUDA graphs (one launch per step).

> [!TIP]
> Percentages: divide the part by the whole, then sanity-check with a simple fraction (¼, ⅓, ½). Remember the pair "fusion + CUDA graphs".

## Q9 · Cold start of a new replica <span class="badge ok">solid</span>

*Covers: 02*

> [!NOTE]
> During a spike the autoscaler launches a new replica, which must load 16 GB of weights from storage at ~2 GB/s. How long until it serves, what does that mean for when to trigger, and how can you shorten it?

**Answer**

- **Load time:** 16 GB ÷ 2 GB/s = **8 s** of weight loading alone. Add container start, GPU init and CUDA graph capture, and real cold starts are often tens of seconds.
- **When to trigger:** early, on a rising trend, e.g. **KV cache usage at 70–80%** or **queue length growing**, so the new replica is ready before the current ones saturate.
- **How to shorten it:**
  - Faster source: keep weights in **host RAM**, or on a **local NVMe cache** instead of network storage.
  - Fewer bytes: **FP8 weights** (8 GB) halve the load to ~4 s.
  - Don't load at spike time: keep a **warm standby pool** of replicas already loaded.

> [!WARNING]
> First said "trigger at a certain threshold" without naming it; the threshold must be early and on engine signals.

> [!IMPORTANT]
> Cold start ≈ weight bytes ÷ load bandwidth. Trigger early on queue/KV trends; shorten with faster storage, fewer bytes, or warm replicas.

> [!TIP]
> For any "time = bytes ÷ speed" problem, the levers are always: fewer bytes, faster speed, or do it ahead of time.

## Exercise · Round-robin vs prefix-aware router <span class="badge ok">solid</span>

*Covers: 02*

> [!NOTE]
> 12 requests, each a shared system prompt (A, B, C cycling, 1,000 tokens) plus a fresh 50-token user message, go to 4 replicas that cache prompts they've prefilled. How many prompt tokens does the fleet prefill under round-robin (`i % 4`) vs prefix-aware routing (`prefix_id % 4`)?

**Setup I needed clarified first:** `policy` exists to compare two routers with one function; `prefix_id` is which system prompt (not which replica); under round-robin the cycles `i % 4` and `i % 3` don't line up, so every replica sees A, B and C and caches nothing useful.

**The structure:**

1. **State:** one set per replica, holding the prompt ids it has cached.
2. **Pick the replica:** round-robin `i % n`; prefix-aware `prefix_id % n`.
3. **Cost:** `user_len` always, plus `prefix_len` only if that replica hasn't cached the prompt.
4. **Update:** add the prompt to that replica's set.
5. **Output:** sum of costs.

```python
def total_prefill_tokens(requests, n_replicas, policy):
    cached = [set() for _ in range(n_replicas)]          # Step 1 · state per replica
    total = 0
    for i, (prefix_id, prefix_len, user_len) in enumerate(requests):
        if policy == "round_robin":                       # Step 2 · pick the replica
            replica = i % n_replicas
        else:
            replica = prefix_id % n_replicas
        cost = user_len + (0 if prefix_id in cached[replica] else prefix_len)   # Step 3
        cached[replica].add(prefix_id)                    # Step 4 · update state
        total += cost
    return total                                          # Step 5

requests = [(i % 3, 1000, 50) for i in range(12)]
total_prefill_tokens(requests, 4, "round_robin")   # 12,600 (12 prompt prefills)
total_prefill_tokens(requests, 4, "prefix")        # 3,600  (3 prompt prefills)
```

**Result:** prefix-aware routing prefills **3.5× fewer** tokens.

**Caveat (interview follow-up):** pure prefix routing left replica 3 idle while 0–2 took all the load. Real inference-aware routers combine both: send to the replica holding the prefix unless it's overloaded, otherwise the least-loaded one.

> [!IMPORTANT]
> Routing by shared prefix turns repeated prefill into cache hits; balance it against load so no replica idles or overloads.
