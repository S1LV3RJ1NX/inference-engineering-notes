# Revision 03-04 · Levels of Inference Engineering + Math for Memory, Throughput & Speedups

*Interactive revision of [lesson 03](../notes/03-levels-of-inference-engineering.md) and [lesson 04](../notes/04-math-for-memory-throughput-speedups.md): each question as asked, the answer, where I slipped, and the one-line takeaway.*

## Q1 · API bill with and without prompt caching <span class="badge mid">partial</span>

*Covers: 03*

> [!NOTE]
> Each request sends the same 3,000-token system instructions, a 200-token user question, and gets a 300-token answer. DeepSeek-R1 prices per million tokens: input $0.55, cached input $0.14, output $2.19. What do 1,000 requests cost without caching, and with the system instructions cached?

**Answer**

| | No cache | System prompt cached |
|---|---|---|
| System, 3,000 tokens | 3,000 × 0.55 = 1,650 | 3,000 × 0.14 = 420 |
| Question, 200 tokens | 200 × 0.55 = 110 | 200 × 0.55 = 110 |
| Answer, 300 output tokens | 300 × 2.19 = 657 | 300 × 2.19 = 657 |
| Sum (tokens × $/M) | 2,417 | 1,187 |
| **1,000 requests** = sum ÷ 10⁶ × 10³ | **$2.42** | **$1.19** |

Caching the shared prefix halves the bill; output (657) is now over half of what's left, so capping output length is the next lever.

> [!WARNING]
> Priced the 300 answer tokens at $0.14 (cached input) instead of $2.19 (output), and didn't divide by 10⁶ (prices are per million tokens).

> [!IMPORTANT]
> Cost = Σ tokens × price, per token type (cached input, input, output). Output costs ~4× input; cache the stable prefix and cap output.

> [!TIP]
> Before multiplying, label each token count with its type: cached input, fresh input, or output.

## Q2 · Personalization that kills the cache <span class="badge ok">solid</span>

*Covers: 03, 11, 14*

> [!NOTE]
> A teammate puts the user's name and the current timestamp at the very start of the prompt, before the 3,000-token system instructions. Overnight, the cached-input share drops to almost zero. Why, and what's the fix?

**Answer**

- The first tokens now differ for every request, so no request shares a prefix with an earlier one: nothing can be served at the cached-input price, and the bill jumps back toward $2.42 per 1k.
- **Why prefixes:** with causal attention, each token's keys and values depend on every token before it. Change token 0 and every later token's K/V changes, so nothing after the first difference is reusable.
- **Fix:** stable content first (system instructions), variable content last (name, timestamp, user question).

> [!IMPORTANT]
> Prompt caches match prefixes: put stable text first and anything that varies at the end.

## Q3 · Does Llama 3 70B fit on one H100? <span class="badge ok">solid</span>

*Covers: 04, 18, 19*

> [!NOTE]
> Serve Llama 3 70B on one 80 GB H100. How big are the weights in BF16, FP8 and INT4, and does each one "fit" for serving users?

**Answer** (budget 72 GB = 90% of 80; ~3 GB runtime + activations; 320 KiB KV per token)

| Precision | Weights = 70B × bytes | Verdict |
|---|---|---|
| BF16 (2 B) | **140 GB** | Doesn't load (> 80 GB) |
| FP8 (1 B) | **70 GB** | Loads, but 72 − 70 − 3 < 0: **no KV cache**, can't serve |
| INT4 (0.5 B + scales) | **~35 GB** | 72 − 35 − 3 ≈ 34 GB KV ÷ 320 KiB ≈ **~100k tokens** (~25 users at 4k) |

> [!WARNING]
> Called FP8 "rarely fits": the precise answer is "loads but can't serve", because there's no room left for the KV cache.

> [!IMPORTANT]
> "Fits" means weights + runtime + KV cache at peak, not just weights. Fitting the weights with nothing left for cache is useless.

## Q4 · A 3× faster attention kernel (Amdahl) <span class="badge mid">partial</span>

*Covers: 04*

> [!NOTE]
> Attention takes 30% of a decode step. A teammate makes the attention kernel 3× faster and claims "decode is now 3× faster." What's the real speedup, and the best possible if attention became free?

**How I solved it (time accounting)**

1. Today a step = **100 units**: 30 attention + 70 everything else.
2. **3× faster attention:** 30 ÷ 3 = **10** units; the other 70 don't change. New step = 10 + 70 = **80**. Speedup = 100 ÷ 80 = **1.25×**.
3. **Attention free:** 0 + 70 = 70 units. Speedup = 100 ÷ 70 ≈ **1.43×**, the ceiling.
4. Formula: S = 1 ÷ ((1 − p) + p/s) = 1 ÷ (0.7 + 0.3/3) = 1 ÷ 0.8 = 1.25; with s → ∞, S → 1 ÷ 0.7 ≈ 1.43.

> [!WARNING]
> First gave "70%" for the ceiling (a speedup is a ratio of times: 100 ÷ 70 = 1.43×). Then gave 100 ÷ 70 for the 3× case too: the 30 units shrink to 10, not to 0.

> [!IMPORTANT]
> Speedup = old time ÷ new time. Only the part you touched shrinks; the untouched (1 − p) caps the total at 1 ÷ (1 − p).

> [!TIP]
> Always redo it as "100 units → how many units now?" before reaching for the formula.

## Exercise · Amdahl calculator for several optimizations <span class="badge ok">solid</span>

*Covers: 04*

> [!NOTE]
> Write `overall_speedup(parts)` where each part is (fraction of the original step, speedup of that part), non-overlapping. Return the true end-to-end speedup, which catches the "adding speedups" trap.

**The structure:** old time = 1.0 → each touched part now takes fraction ÷ speedup → the untouched fraction keeps its full time → new time = the sum → speedup = old ÷ new.

```python
def overall_speedup(parts):
    old_time = 1.0                                                            # Step 1
    touched_new = sum(fraction / speedup for fraction, speedup in parts.values())  # Step 2
    untouched = 1 - sum(fraction for fraction, _ in parts.values())           # Step 3
    new_time = touched_new + untouched                                        # Step 4
    return old_time / new_time                                                # Step 5

overall_speedup({"attention": (0.3, 3)})                   # 1.25
overall_speedup({"attention": (0.3, 3), "mlp": (0.5, 2)})  # 1.82 (not 5, not 6)
overall_speedup({"attention": (0.3, 1e9)})                 # 1.43 (the ceiling)
```

**Why 1.82:** new time = 0.3/3 + 0.5/2 + 0.2 = 0.1 + 0.25 + 0.2 = 0.55, and 1 ÷ 0.55 ≈ 1.82. Optimizing 80% of the step still leaves 0.55 of the original time.

> [!IMPORTANT]
> Never add or multiply speedups. Add up the new times, then divide old time by new time.
