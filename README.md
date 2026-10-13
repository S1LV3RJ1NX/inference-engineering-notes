# Inference Engineering notes

My study notes from the [Inference Engineering course on fanout.sh](https://fanout.sh/inference-eng), written while preparing for inference engineering interviews.

Each lesson has a Markdown version under [`notes/`](notes/) and a printable PDF under [`pdf/`](pdf/). Every note contains:

- **A TL;DR** of the lesson's core claims.
- **Concept sections** summarized in my own words, with every number and calculation re-checked.
- **Original figures** (charts, timelines, roofline plots) drawn from the lesson's numbers to make the ideas easier to remember.
- **An interview cheat sheet** on its own pages: numbers to memorize, formulas, comparisons, symptom-to-fix tables and rapid-fire Q&A.

For self-testing there's also a [Revision](#revision) section: interview-style questions and coding exercises per lesson group.

**Interview tomorrow?** Read the [master cheat sheet](pdf/00-interview-cheat-sheet.pdf) ([md](notes/00-interview-cheat-sheet.md)). It collects every lesson's cheat sheet in one file.

## Lessons

| # | Lesson | Notes | PDF |
|---|---|---|---|
| 00 | Interview cheat sheet (all lessons) | [md](notes/00-interview-cheat-sheet.md) | [pdf](pdf/00-interview-cheat-sheet.pdf) |
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

## About

These are unofficial, personal notes and are not affiliated with or endorsed by fanout.sh. They summarize, not replace, the course: the videos go much deeper, and each note links back to its lesson. If the notes help you, take [the course](https://fanout.sh/inference-eng).

Spotted a mistake? Open an issue.
