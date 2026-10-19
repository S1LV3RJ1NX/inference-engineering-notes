# Inference Engineering notes

My study notes from the [Inference Engineering course on fanout.sh](https://fanout.sh/inference-eng), written while preparing for inference engineering interviews.

Each lesson has a Markdown version under [`notes/`](notes/) and a printable PDF under [`pdf/`](pdf/). Every note contains:

- **A TL;DR** of the lesson's core claims.
- **Concept sections** summarized in my own words, with every number and calculation re-checked.
- **Original figures** (charts, timelines, roofline plots) drawn from the lesson's numbers to make the ideas easier to remember.

To revise, use the [Revision](#revision) section: interview-style questions solved from first principles, with coding exercises, per lesson group. (Earlier versions of the notes ended with interview cheat sheets; they're preserved at the [`with-cheat-sheets`](https://github.com/S1LV3RJ1NX/inference-engineering-notes/tree/with-cheat-sheets) tag.)

## Lessons

| # | Lesson | Notes | PDF |
|---|---|---|---|
| 01 | What Is Inference & Inference Engineering? | [md](notes/01-what-is-inference-engineering.md) | [pdf](pdf/01-what-is-inference-engineering.pdf) |
| 02 | The Inference Stack: Model to GPU to Production | [md](notes/02-the-inference-stack.md) | [pdf](pdf/02-the-inference-stack.pdf) |
| 03 | Levels of Inference Engineering | [md](notes/03-levels-of-inference-engineering.md) | [pdf](pdf/03-levels-of-inference-engineering.pdf) |
| 04 | Math for Memory, Throughput & Speedups | [md](notes/04-math-for-memory-throughput-speedups.md) | [pdf](pdf/04-math-for-memory-throughput-speedups.pdf) |
| 05 | Dot Products, Softmax & Attention Math | [md](notes/05-dot-products-softmax-attention-math.md) | [pdf](pdf/05-dot-products-softmax-attention-math.pdf) |
| 06 | Probability, Sampling & Benchmark Statistics | [md](notes/06-probability-sampling-benchmark-statistics.md) | [pdf](pdf/06-probability-sampling-benchmark-statistics.pdf) |
| 07 | Transformer Architecture Refresher | [md](notes/07-transformer-architecture-refresher.md) | [pdf](pdf/07-transformer-architecture-refresher.pdf) |
| 08 | Generate & Stream Your First Tokens | [md](notes/08-generate-and-stream-your-first-tokens.md) | [pdf](pdf/08-generate-and-stream-your-first-tokens.pdf) |
| 09 | Watch GPU Memory, Utilization & Power | [md](notes/09-watch-gpu-memory-utilization-power.md) | [pdf](pdf/09-watch-gpu-memory-utilization-power.pdf) |
| 10 | Trace the Transformer Forward Pass | [md](notes/10-trace-the-transformer-forward-pass.md) | [pdf](pdf/10-trace-the-transformer-forward-pass.pdf) |
| 11 | Q, K, V & Causal Attention During Inference | [md](notes/11-qkv-causal-attention-during-inference.md) | [pdf](pdf/11-qkv-causal-attention-during-inference.pdf) |
| 12 | MLP, RMSNorm & Residual Connections | [md](notes/12-mlp-rmsnorm-residual-connections.md) | [pdf](pdf/12-mlp-rmsnorm-residual-connections.pdf) |
| 13 | Logits, Sampling & the Next Token | [md](notes/13-logits-sampling-next-token.md) | [pdf](pdf/13-logits-sampling-next-token.pdf) |
| 14 | Why Autoregressive Generation Recomputes Work | [md](notes/14-why-autoregressive-generation-recomputes-work.md) | [pdf](pdf/14-why-autoregressive-generation-recomputes-work.pdf) |
| 15 | Understand & Measure the Prefill Phase | [md](notes/15-understand-measure-prefill-phase.md) | [pdf](pdf/15-understand-measure-prefill-phase.pdf) |
| 16 | Understand & Measure the Decode Phase | [md](notes/16-understand-measure-decode-phase.md) | [pdf](pdf/16-understand-measure-decode-phase.pdf) |
| 17 | Build a KV Cache | [md](notes/17-build-a-kv-cache.md) | [pdf](pdf/17-build-a-kv-cache.pdf) |
| 18 | Calculate Model & Runtime Memory | [md](notes/18-calculate-model-runtime-memory.md) | [pdf](pdf/18-calculate-model-runtime-memory.pdf) |
| 19 | Calculate KV Memory, Context & Concurrency Limits | [md](notes/19-kv-memory-context-concurrency-limits.md) | [pdf](pdf/19-kv-memory-context-concurrency-limits.pdf) |
| 20 | TTFT, TPOT, ITL, E2E & Throughput | [md](notes/20-ttft-tpot-itl-e2e-throughput.md) | [pdf](pdf/20-ttft-tpot-itl-e2e-throughput.pdf) |
| 21 | Concurrency, Request Rate & Arrival Patterns | [md](notes/21-concurrency-request-rate-arrival-patterns.md) | [pdf](pdf/21-concurrency-request-rate-arrival-patterns.pdf) |
| 22 | P50, P95, P99 & Variance | [md](notes/22-p50-p95-p99-variance.md) | [pdf](pdf/22-p50-p95-p99-variance.pdf) |
| 23 | Goodput, SLOs & Failure Rates | [md](notes/23-goodput-slos-failure-rates.md) | [pdf](pdf/23-goodput-slos-failure-rates.pdf) |
| 24 | Establish the Single-Request Baseline | [md](notes/24-establish-single-request-baseline.md) | [pdf](pdf/24-establish-single-request-baseline.pdf) |
| 25 | Sweep Concurrency & Request Rate | [md](notes/25-sweep-concurrency-request-rate.md) | [pdf](pdf/25-sweep-concurrency-request-rate.pdf) |
| 26 | Sweep Prompt & Output Length | [md](notes/26-sweep-prompt-output-length.md) | [pdf](pdf/26-sweep-prompt-output-length.pdf) |
| 27 | Find Saturation, SLO Failure & OOM Boundaries | [md](notes/27-find-saturation-slo-oom-boundaries.md) | [pdf](pdf/27-find-saturation-slo-oom-boundaries.pdf) |
| 28 | Why Plain PyTorch Serving Breaks Down | [md](notes/28-why-plain-pytorch-serving-breaks-down.md) | [pdf](pdf/28-why-plain-pytorch-serving-breaks-down.pdf) |
| 29 | OpenAI-Compatible APIs & Streaming | [md](notes/29-openai-compatible-apis-streaming.md) | [pdf](pdf/29-openai-compatible-apis-streaming.pdf) |
| 30 | Read Engine Metrics & Request State | [md](notes/30-read-engine-metrics-request-state.md) | [pdf](pdf/30-read-engine-metrics-request-state.pdf) |
| 31 | Follow a Request Through vLLM | [md](notes/31-follow-a-request-through-vllm.md) | [pdf](pdf/31-follow-a-request-through-vllm.pdf) |
| 32 | One-by-One Serving vs Static Batching | [md](notes/32-one-by-one-serving-vs-static-batching.md) | [pdf](pdf/32-one-by-one-serving-vs-static-batching.pdf) |
| 33 | Iteration-Level Scheduling | [md](notes/33-iteration-level-scheduling.md) | [pdf](pdf/33-iteration-level-scheduling.pdf) |
| 34 | Continuous Batching | [md](notes/34-continuous-batching.md) | [pdf](pdf/34-continuous-batching.pdf) |
| 35 | Token Budgets, Max Sequences & Max Batched Tokens | [md](notes/35-token-budgets-max-seqs-max-batched-tokens.md) | [pdf](pdf/35-token-budgets-max-seqs-max-batched-tokens.pdf) |

## Revision

After writing the notes I drilled them with interview-style questions and small coding exercises, one lesson group at a time. Each group has a **revision PDF** under [`revision/`](revision/) (the questions, my worked reasoning, where I slipped, and a one-line takeaway per question) built from a Markdown log in [`exercises/`](exercises/), plus any Python exercises. Cover the question, answer it yourself, then check.

| Group | Lessons | Revision | Log | Coding exercise |
|---|---|---|---|---|
| 01-02 | 01, 02 | [pdf](revision/01-02-revision.pdf) | [md](exercises/01-02-revision-log.md) | [router](exercises/02_prefill_tokens.py) |
| 03-04 | 03, 04 | [pdf](revision/03-04-revision.pdf) | [md](exercises/03-04-revision-log.md) | [Amdahl](exercises/03_04_amdahl.py) |
| 05-07 | 05, 07 | [pdf](revision/05-07-revision.pdf) | [md](exercises/05-07-revision-log.md) | [online softmax](exercises/05_07_online_softmax.py) |
| 06-08-09 | 06, 08, 09 | [pdf](revision/06-08-09-revision.pdf) | [md](exercises/06-08-09-revision-log.md) | [tail simulation](exercises/06_08_09_tail_sim.py) |
| 10-13 | 10, 11, 12, 13 | [pdf](revision/10-13-revision.pdf) | [md](exercises/10-13-revision-log.md) | [sampler](exercises/10_13_sampler.py) |
| 14-17 | 14, 15, 16, 17 | [pdf](revision/14-17-revision.pdf) | [md](exercises/14-17-revision-log.md) | [toy KV cache](exercises/14_17_kv_cache.py) |
| 18-19 | 18, 19 | [pdf](revision/18-19-revision.pdf) | [md](exercises/18-19-revision-log.md) | [paged KV allocator](exercises/18_19_paged_kv.py) |
| 20-23 | 20, 21, 22, 23 | [pdf](revision/20-23-revision.pdf) | [md](exercises/20-23-revision-log.md) | none |
| 24-27 | 24, 25, 26, 27 | [pdf](revision/24-27-revision.pdf) | [md](exercises/24-27-revision-log.md) | [find_walls](exercises/24_27_find_walls.py) |
| 28-31 | 28, 29, 30, 31 | [pdf](revision/28-31-revision.pdf) | [md](exercises/28-31-revision-log.md) | none |

## About

These are unofficial, personal notes and are not affiliated with or endorsed by fanout.sh. They summarize, not replace, the course: the videos go much deeper, and each note links back to its lesson. If the notes help you, take [the course](https://fanout.sh/inference-eng).

Spotted a mistake? Open an issue.
