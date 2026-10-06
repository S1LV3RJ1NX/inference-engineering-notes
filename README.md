# Inference Engineering notes

My study notes from the [Inference Engineering course on fanout.sh](https://fanout.sh/inference-eng), written while preparing for inference engineering interviews.

Each lesson has a Markdown version under [`notes/`](notes/) and a printable PDF under [`pdf/`](pdf/). Every note contains:

- **A TL;DR** of the lesson's core claims.
- **Concept sections** summarized in my own words, with every number and calculation re-checked.
- **Original figures** (charts, timelines, roofline plots) drawn from the lesson's numbers to make the ideas easier to remember.
- **An interview cheat sheet** on its own pages: numbers to memorize, formulas, comparisons, symptom-to-fix tables and rapid-fire Q&A.

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

## About

These are unofficial, personal notes and are not affiliated with or endorsed by fanout.sh. They summarize, not replace, the course: the videos go much deeper, and each note links back to its lesson. If the notes help you, take [the course](https://fanout.sh/inference-eng).

Spotted a mistake? Open an issue.
