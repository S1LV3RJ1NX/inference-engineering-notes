# Revision 06-08-09 · Benchmark Statistics, Streaming & the GPU Dashboard

*Interactive revision of [lesson 06](../notes/06-probability-sampling-benchmark-statistics.md), [lesson 08](../notes/08-generate-and-stream-your-first-tokens.md) and [lesson 09](../notes/09-watch-gpu-memory-utilization-power.md): each question as asked, the answer, where I slipped, and the one-line takeaway.*

## Q1 · Tail latency in a 20-call agent <span class="badge miss">missed</span>

*Covers: 06*

> [!NOTE]
> One agent task makes 20 sequential LLM calls. The API's SLO is P99 = 2 s (1% of calls take longer). What fraction of tasks hit at least one slow call, and how should per-call targets be set for agents?

**How to solve it: flip to "nothing goes wrong"**

1. One call is fast with probability 0.99.
2. All 20 fast: 0.99²⁰ ≈ **0.82**.
3. At least one slow: 1 − 0.82 ≈ **18% of tasks**. Shortcut for small q: k × q = 20 × 1% ≈ 20%.
4. **Per-call target:** to keep slow tasks to 1%, each call may be slow only ~1% ÷ 20 = 0.05% of the time, i.e. a **P99.95** per-call target. A per-call P99 behaves like a task-level P82.

> [!WARNING]
> Said "1% of users hit a slow call": 1% is per call, and each task makes 20 calls. Also read P99 as "90% of cases" (it's 99%). Proposed keeping the per-call target at P99, which is far too loose for agents.

> [!IMPORTANT]
> P(at least one slow in k calls) = 1 − (1 − q)ᵏ ≈ k·q. Multi-call tasks amplify the tail, so per-call targets must be much tighter (≈ q_task ÷ k).

**Computing 0.99²⁰ by hand**

- Rough: (1 − x)ⁿ ≈ 1 − n·x = 1 − 0.2 = 0.80.
- Better: (1 − x)ⁿ ≈ e^(−n·x) = e^(−0.2) ≈ 1 − 0.2 + 0.02 = **0.82** (using eʸ ≈ 1 + y + y²/2).
- Exact: square repeatedly: 0.99² = 0.9801 → ⁴ ≈ 0.9606 → ⁸ ≈ 0.9227 → ¹⁶ ≈ 0.8515; then ¹⁶ × ⁴ ≈ 0.8515 × 0.9606 ≈ 0.818.

> [!TIP]
> For "at least one" questions, compute the chance of "none" and subtract from 1. Remember (1 − x)ⁿ ≈ e^(−n·x): 50 calls at P99 → e^(−0.5) ≈ 0.61 → ~39% of tasks see a slow call.

## Q2 · Is a 3.5% speedup real? <span class="badge miss">missed</span>

*Covers: 06*

> [!NOTE]
> 5 runs before average 20.0 ms TPOT; 5 runs after average 19.3 ms; run-to-run σ ≈ 0.8 ms. "3.5% faster, ship it." Is it real?

**How to think about it: wobble, not formulas**

1. **One run wobbles** by σ ≈ 0.8 ms around the true value.
2. **Averaging calms it, slowly:** an average of n runs wobbles σ ÷ √n = 0.8 ÷ √5 ≈ 0.8 ÷ 2.2 ≈ **0.36 ms** (the standard error).
3. **A difference of two averages wobbles more:** both are uncertain, and two independent wobbles combine as √2 ≈ 1.4×: 0.36 × 1.4 ≈ **0.5 ms**.
4. **"Pretty sure" ≈ 2 wobbles** (~95% confidence): noise band ≈ 2 × 0.5 ≈ **±1.0 ms**.
5. **Compare:** observed difference = 20.0 − 19.3 = **0.7 ms**, inside ±1.0 ms. **Could be luck: not proven, don't ship yet.**
6. **To settle it:** 4× the runs (20 each) halves the band to ±0.5 ms; if the 0.7 ms gap holds, it's real.

> [!WARNING]
> Didn't know how to start. The missing mental model: one run wobbles σ, an average wobbles σ/√n, a difference wobbles ×√2, and only a gap beyond ~2× that is real.

> [!IMPORTANT]
> Noise band for a difference ≈ 2 × √2 × σ/√n. A speedup smaller than the band isn't proven; 4× the runs halves the band.

> [!TIP]
> Re-derive it as a story: one run → average of n → difference of two → ×2 for confidence.

## Q3 · A benchmark claiming 2,000 tokens/s <span class="badge mid">partial</span>

*Covers: 08, 01*

> [!NOTE]
> A script times `model.generate()` for Llama 3 8B, batch 1, one H100, and reports 2,000 tokens/s. Why must it be wrong, and what are two likely bugs?

**Answer**

- **Impossible:** batch-1 decode is capped by **memory bandwidth**: 16 GB ÷ 3.35 TB/s ≈ 4.8 ms per token → ≤ ~200 tokens/s. 2,000 is 10× over.
- **Bug 1, no GPU sync:** CUDA is asynchronous; Python queues work and returns. Stopping the timer without `torch.cuda.synchronize()` (or CUDA events) times the queuing, not the GPU, so the rate looks far too high.
- **Bug 2, counting the prompt:** dividing prompt + generated tokens by the time instead of generated tokens only (1.5× too high in lesson 08's example).
- **Fix:** synchronize before reading the clock, count only generated tokens, and report TTFT and TPOT separately.

> [!WARNING]
> Said the ceiling comes from the arithmetic units (TFLOPS): it's memory bandwidth. Couldn't name the two bugs (no GPU sync, prompt counted as output).

> [!IMPORTANT]
> Any batch-1 decode number above weights ÷ bandwidth is a measurement bug: check GPU sync and what tokens were counted.

> [!TIP]
> Sanity-check every benchmark against the floor before believing it.

## Q4 · "GPU-util is 100%, we need more GPUs" <span class="badge mid">partial</span>

*Covers: 09*

> [!NOTE]
> One user chats with Llama 3 8B on an H100. Dashboard: GPU-util 100%, power 350 W of a 700 W cap, decode 6 ms per token. Is the GPU maxed out? Is half the power cap a problem? How do you measure real usage?

**Answer**

- **GPU-util is a duty cycle:** the fraction of time any kernel ran. One kernel on 1 of 132 SMs reads 100%. It says nothing about how busy the GPU is.
- **Half power is normal:** memory-bound decode leaves the math units waiting, so the board draws less. Real power problems show as power-cap or thermal throttling in clock event reasons.
- **Measure MBU** = achieved bandwidth ÷ peak bandwidth: 16 GB ÷ 6 ms ≈ 2.67 TB/s, ÷ 3.35 TB/s ≈ **80%**. MFU = (15 GFLOP ÷ 6 ms) ÷ 989 TFLOPS ≈ **0.25%**.
- **Verdict:** not maxed out. One user at ~80% bandwidth with idle compute, so batching more users is nearly free. Don't buy GPUs.

> [!WARNING]
> Said half power means "half the arithmetic units are used, so kernels need work": it's expected for memory-bound decode. Divided by the ridge instead of peak bandwidth.

> [!IMPORTANT]
> Judge decode by MBU (achieved ÷ peak bandwidth), not GPU-util or power. Low power and 100% util are both normal for one-user decode.

> [!TIP]
> MBU = bytes per step ÷ step time ÷ peak bandwidth; MFU = FLOPs per step ÷ step time ÷ peak FLOPS.

## Exercise · See benchmark noise by simulation <span class="badge ok">solid</span>

*Covers: 06*

> [!NOTE]
> Simulate long-tailed latencies. (A) Estimate P99 from n = 100, 1,000, 10,000 samples, 200 times each, and measure how much the estimates wobble. (B) Brute-force the 20-call agent: what fraction of tasks hit a slow call?

```python
def p99_spread(n, repeats=200):
    estimates = [p99([latency() for _ in range(n)]) for _ in range(repeats)]   # 200 benchmarks
    mean = statistics.mean(estimates)
    rel_spread = statistics.stdev(estimates) / mean
    return mean, rel_spread

def slow_task_fraction(calls=20, q=0.01, tasks=100_000):
    slow_tasks = 0
    for _ in range(tasks):
        if any(random.random() < q for _ in range(calls)):   # slow if ANY call is slow
            slow_tasks += 1
    return slow_tasks / tasks
```

| Samples per benchmark | P99 estimate | Wobble |
|---|---|---|
| 100 | 2.97 s | ±16% |
| 1,000 | 3.17 s | ±6% |
| 10,000 | 3.20 s | ±2% |

Slow tasks: **18.3%**, matching 1 − 0.99²⁰ ≈ 18%.

**What it shows:**
- Wobble shrinks like √n: 100× the samples → ~8× less wobble (the σ/√n from Q2).
- Small samples **under-report the tail**: at n = 100 only ~1 sample lands beyond P99, so the estimate averages 2.97 s vs the true ~3.20 s. A 100-request benchmark flatters your tail.

> [!IMPORTANT]
> P99 needs thousands of samples; with ~100 it's both noisy and biased low.
