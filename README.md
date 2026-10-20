# Inference Engineering notes

My study notes from the [Inference Engineering course on fanout.sh](https://fanout.sh/inference-eng), written while preparing for inference engineering interviews.

Each lesson has a Markdown version under [`notes/`](notes/) and a printable PDF under [`pdf/`](pdf/). Every note contains:

- **A TL;DR** of the lesson's core claims.
- **Concept sections** summarized in my own words, with every number and calculation re-checked.
- **Original figures** (charts, timelines, roofline plots) drawn from the lesson's numbers to make the ideas easier to remember.

To revise, use the [Revision](#revision) section: six themed interview packs of mock-interview questions with model answers, plus coding exercises. (Earlier versions of the notes ended with interview cheat sheets; they're preserved at the [`with-cheat-sheets`](https://github.com/S1LV3RJ1NX/inference-engineering-notes/tree/with-cheat-sheets) tag.)

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
| 36 | Queueing, Fairness, Cancellation & Preemption | [md](notes/36-queueing-fairness-cancellation-preemption.md) | [pdf](pdf/36-queueing-fairness-cancellation-preemption.pdf) |
| 37 | Why Contiguous KV Allocation Breaks | [md](notes/37-why-contiguous-kv-allocation-breaks.md) | [pdf](pdf/37-why-contiguous-kv-allocation-breaks.pdf) |
| 38 | KV Blocks, Fragmentation & Block Tables | [md](notes/38-kv-blocks-fragmentation-block-tables.md) | [pdf](pdf/38-kv-blocks-fragmentation-block-tables.pdf) |
| 39 | PagedAttention | [md](notes/39-pagedattention.md) | [pdf](pdf/39-pagedattention.pdf) |
| 40 | Allocate, Share & Reclaim KV Blocks | [md](notes/40-allocate-share-reclaim-kv-blocks.md) | [pdf](pdf/40-allocate-share-reclaim-kv-blocks.pdf) |
| 41 | Preemption, Recomputation & KV Pressure | [md](notes/41-preemption-recomputation-kv-pressure.md) | [pdf](pdf/41-preemption-recomputation-kv-pressure.pdf) |
| 42 | Tune GPU Memory & KV Capacity | [md](notes/42-tune-gpu-memory-kv-capacity.md) | [pdf](pdf/42-tune-gpu-memory-kv-capacity.pdf) |
| 43 | Find the SLO-Safe Configuration & Headroom | [md](notes/43-find-slo-safe-configuration-headroom.md) | [pdf](pdf/43-find-slo-safe-configuration-headroom.pdf) |
| 44 | Reproduce the Long-Prompt Interference Problem | [md](notes/44-reproduce-long-prompt-interference.md) | [pdf](pdf/44-reproduce-long-prompt-interference.pdf) |
| 45 | Chunked Prefill & Choosing the Chunk Size | [md](notes/45-chunked-prefill-choosing-chunk-size.md) | [pdf](pdf/45-chunked-prefill-choosing-chunk-size.pdf) |
| 46 | Find Repeated Prefix Computation | [md](notes/46-find-repeated-prefix-computation.md) | [pdf](pdf/46-find-repeated-prefix-computation.pdf) |
| 47 | Automatic Prefix Caching | [md](notes/47-automatic-prefix-caching.md) | [pdf](pdf/47-automatic-prefix-caching.pdf) |
| 48 | Shared Prompts, RAG Context & Multi-Turn Reuse | [md](notes/48-shared-prompts-rag-multi-turn-reuse.md) | [pdf](pdf/48-shared-prompts-rag-multi-turn-reuse.pdf) |
| 49 | RadixAttention, Eviction & Cache Churn | [md](notes/49-radixattention-eviction-cache-churn.md) | [pdf](pdf/49-radixattention-eviction-cache-churn.pdf) |
| 50 | Precision, Memory & Hardware Support | [md](notes/50-precision-memory-hardware-support.md) | [pdf](pdf/50-precision-memory-hardware-support.pdf) |
| 51 | BF16, FP8, INT8 & 4-Bit Inference | [md](notes/51-bf16-fp8-int8-4-bit-inference.md) | [pdf](pdf/51-bf16-fp8-int8-4-bit-inference.pdf) |
| 52 | AWQ, GPTQ & SmoothQuant | [md](notes/52-awq-gptq-smoothquant.md) | [pdf](pdf/52-awq-gptq-smoothquant.pdf) |
| 53 | Quantized GEMM & Dequantization Overhead | [md](notes/53-quantized-gemm-dequantization-overhead.md) | [pdf](pdf/53-quantized-gemm-dequantization-overhead.pdf) |
| 54 | Measure Quality Regression | [md](notes/54-measure-quality-regression.md) | [pdf](pdf/54-measure-quality-regression.pdf) |
| 55 | Stop Guessing: Build a Profiling Workflow | [md](notes/55-build-a-profiling-workflow.md) | [pdf](pdf/55-build-a-profiling-workflow.pdf) |
| 56 | PyTorch Profiler & Memory Snapshots | [md](notes/56-pytorch-profiler-memory-snapshots.md) | [pdf](pdf/56-pytorch-profiler-memory-snapshots.pdf) |
| 57 | Nsight Systems | [md](notes/57-nsight-systems.md) | [pdf](pdf/57-nsight-systems.pdf) |
| 58 | Nsight Compute | [md](notes/58-nsight-compute.md) | [pdf](pdf/58-nsight-compute.pdf) |
| 59 | Profile Prefill vs Decode | [md](notes/59-profile-prefill-vs-decode.md) | [pdf](pdf/59-profile-prefill-vs-decode.pdf) |

## Revision

After writing the notes I drilled them with interview-style questions, then merged every drill into **six themed interview packs** under [`revision/`](revision/). Each card is a mock-interview question followed by a 60-second answer, the key numbers, the likely follow-up and the trap I actually fell into. Read the question, answer it out loud, then check. Each pack ends with the coding exercises from [`exercises/`](exercises/) that back it up.

| Pack | Theme | Lessons | PDF | Markdown |
|---|---|---|---|---|
| 01 | Speed physics | 01–04, 06, 09, 15, 16, 24, 25, 28, 44 | [pdf](revision/01-speed-physics.pdf) | [md](revision/01-speed-physics.md) |
| 02 | Memory and capacity | 03–05, 07, 16–19, 25, 42, 50 | [pdf](revision/02-memory-capacity.pdf) | [md](revision/02-memory-capacity.md) |
| 03 | Inside the serving engine | 19, 26, 28, 30, 32–41, 44, 45 | [pdf](revision/03-engine-internals.pdf) | [md](revision/03-engine-internals.md) |
| 04 | Measure and operate | 01, 06, 08, 15, 16, 20–25, 27, 30, 43 | [pdf](revision/04-measure-operate.pdf) | [md](revision/04-measure-operate.md) |
| 05 | Prefix caching and quantization | 03, 04, 06, 11, 14, 22, 40, 46–54 | [pdf](revision/05-optimization-levers.pdf) | [md](revision/05-optimization-levers.md) |
| 06 | Model internals | 05, 10–14, 17 | [pdf](revision/06-model-internals.pdf) | [md](revision/06-model-internals.md) |

## About

These are unofficial, personal notes and are not affiliated with or endorsed by fanout.sh. They summarize, not replace, the course: the videos go much deeper, and each note links back to its lesson. If the notes help you, take [the course](https://fanout.sh/inference-eng).

Spotted a mistake? Open an issue.
