# Interview cheat sheet: all lessons

*Every lesson's cheat sheet in one place. ◆ = my addition beyond the video; everything else is from the lessons.*

<div class="lesson-start"></div>

## 01 · What Is Inference & Inference Engineering?

[Full notes](01-what-is-inference-engineering.md)


### Say it in 30 seconds

> Inference is running a trained model token by token for every request, so fleet-wide it costs more than training. Decode is memory-bandwidth-bound: every step streams all the weights to make one token per sequence. The core trade is latency vs throughput vs cost, and batching is lever one. The job: measure, find the bottleneck, change one thing, repeat.

### Only numbers worth memorizing

- **H100:** 3.35 TB/s HBM, ◆ ~989 TFLOPS BF16. **Llama 3 8B:** 16 GB in BF16, ~128 KB KV per token (taught in 02).
- **Software matters:** batching gives ~15× cheaper tokens; vLLM gets up to 24× naive HF throughput.

### Derive, don't memorize

Let *P* = parameters, *B* = batch size, *BW* = memory bandwidth.

#### 1. Decode speed = bandwidth ÷ bytes per step

Every decode step streams all the weights once, so time per step is bytes moved over bytes per second.

$$t_{\mathrm{step}} \approx \frac{2P\ \mathrm{bytes}}{\mathrm{BW}} = \frac{16\ \mathrm{GB}}{3.35\ \mathrm{TB/s}} \approx 4.8\ \mathrm{ms} \quad\Rightarrow\quad \leq 200\ \mathrm{tok/s\ per\ user}$$

> [!TIP]
> Halve the bytes (8-bit weights) and the ceiling doubles. That's why precision is a speed lever.

#### 2. Memory-bound or compute-bound? Compare FLOPs per byte to the ridge ◆

Each weight does a multiply and an add (2 FLOPs) per token and costs 2 bytes to read. One read is shared by all *B* sequences:

$$\frac{\mathrm{FLOPs}}{\mathrm{byte}} \approx \frac{2P \cdot B}{2P} = B \qquad\mathrm{vs}\qquad \mathrm{ridge} = \frac{989\ \mathrm{TFLOPS}}{3.35\ \mathrm{TB/s}} \approx 300$$

Batch 1 gives ~1, far below 300, so decode is memory-bound and batching is nearly free speed. Prefill puts ~500 prompt tokens through one read, above the ridge, so it's compute-bound:

$$t_{\mathrm{prefill}} \approx \frac{N_{\mathrm{prompt}} \cdot 2P}{\mathrm{peak\ FLOPS}} = \frac{500 \times 16\ \mathrm{GFLOP}}{989\ \mathrm{TFLOPS}} \approx 8\ \mathrm{ms}$$

#### 3. Cost = what you pay per hour ÷ what you make per hour

$$\frac{\$}{1\mathrm{M\ tokens}} = \frac{\$\ \mathrm{per\ GPU\ hour}}{\mathrm{tok/s} \times 3600} \times 10^{6} \qquad \frac{3}{200 \times 3600} \times 10^{6} \approx 4.17 \qquad \frac{3}{3000 \times 3600} \times 10^{6} \approx 0.28$$

#### 4. Why batching isn't free: every sequence brings its own KV cache ◆

Each layer stores one K and one V vector per KV head for every past token:

$$\mathrm{KV/token} = 2 \times L \times H_{\mathrm{kv}} \times d_{\mathrm{head}} \times \mathrm{bytes} = 2 \times 32 \times 8 \times 128 \times 2 = 128\ \mathrm{KB}$$

A step reads the weights once plus every sequence's cache (context length *C*), so steps slow down as the batch grows:

$$t_{\mathrm{step}}(B) \approx \frac{2P + B \cdot C \cdot \mathrm{KV/token}}{\mathrm{BW}}$$

### Prefill vs decode

| | Prefill | Decode |
|---|---|---|
| Work | Whole prompt in one parallel pass | One token per sequence per step, serial |
| Bottleneck | Compute (FLOPs) | Memory bandwidth (bytes) |
| User sees it as | TTFT | Gap between tokens (ITL / TPOT) |
| Grows with | Prompt length | Output length, and batch × context (KV reads) |

### Symptom → lever (for "how would you fix it?" questions)

| Symptom | Likely cause | First levers |
|---|---|---|
| High $/token, GPU underused | Batch too small | Scheduling: batch more requests per step |
| Batch capped by out-of-memory | KV cache fills memory | Memory: pack KV tightly, reuse shared prefixes |
| Slow tokens even at low load | Too many weight bytes per step | Precision: 8/4-bit weights |
| High TTFT | Long prompts or queueing | Shorter prompts, more replicas (routing fixes: see 02) |
| Model doesn't fit one GPU | Size | Parallelism across GPUs, or precision |

### Rapid-fire Q&A

| Question | Crisp answer |
|---|---|
| Why does batching help, and what does it cost? | One weight read serves all *B* sequences. The cost is KV memory and slower steps (derivation 4). |
| ◆ Max tok/s for 70B BF16 on 8×H100? | Derivation 1: 140 GB ÷ (8 × 3.35 TB/s) ≈ 5.2 ms/step, so ≤ ~190 tok/s before communication overhead. |
| Cost too high: where do you start? | Measure (utilization, batch size, TTFT/ITL) before changing anything; then check off-GPU wins such as a smaller model, shorter prompts, caching. |

> [!WARNING]
>
> - "Just call `generate()`" or "buy a faster GPU": both ignore the scheduler, memory and setup.
> - Trading away quality: a faster but worse answer is not a win.

<div class="lesson-break"></div>

## 02 · The Inference Stack: Model to GPU to Production

[Full notes](02-the-inference-stack.md)


### Say it in 30 seconds

> A request crosses about eight layers. The model defines the math; kernels, CUDA and the GPU set how fast it runs, bounded by bytes ÷ bandwidth. The inference engine batches requests every step and manages KV cache memory. The API server, router and autoscaler pick the replica and the replica count. Nearly all time is GPU time; the upper layers decide how well it's used.

### Only numbers worth memorizing

- **H100:** 80 GB HBM, 132 SMs. **Llama 3.1 8B:** 32 layers. Routing ~1 ms; decode ~5 ms/token.

### Derive, don't memorize

Every derivation here is the same idea: **time = bytes ÷ bandwidth**, or **capacity = bytes ÷ bytes-per-item**. Per-step time and KV per token: see 01, derivations 1 and 4.

#### 1. KV capacity = memory left after the weights ÷ cache per token ◆

$$N_{\mathrm{tokens}} \approx \frac{\mathrm{HBM} - \mathrm{weights}}{\mathrm{KV/token}} = \frac{80 - 16\ \mathrm{GB}}{128\ \mathrm{KB}} \approx 488\mathrm{k} \approx 119\ \mathrm{requests} \times 4\mathrm{k}$$

> [!TIP]
> That token budget, not compute, caps the batch. It's why engines manage KV in blocks and why quantizing weights or KV frees room for more users.

#### 2. Launch overhead = kernels × launch cost, against the step time ◆

$$\frac{N_{\mathrm{kernels}} \times t_{\mathrm{launch}}}{t_{\mathrm{step}}} = \frac{300 \times 4\ \mu\mathrm{s}}{4.8\ \mathrm{ms}} \approx 25\%$$

> [!TIP]
> Illustrative numbers, but the shape is real: at small batch, CPU launch time is a big slice of each step. Fusion (and ◆ CUDA graphs) cut it.

#### 3. Cold start = weight bytes ÷ load bandwidth ◆

$$t_{\mathrm{cold}} \approx \frac{16\ \mathrm{GB}}{2\ \mathrm{GB/s}} = 8\ \mathrm{s}$$

> [!TIP]
> Assumes ~2 GB/s from local disk; from network storage it's often slower. Scaling reacts late, so autoscalers must scale before the queue explodes.

### Layer → question → what to watch

| Layer | Answers | Tools | Watch / lever |
|---|---|---|---|
| Model + framework | What to compute | Weights, recipe, tokenizer, PyTorch | Model size, precision |
| Kernels + CUDA + GPU | How fast the math runs | cuBLAS, FlashAttention, CUDA | Bytes moved, fusion, launches |
| Inference engine | Who runs each step; where memory goes | vLLM, SGLang, TensorRT-LLM | Batch size, KV usage |
| API server | Speaks HTTP, tokenizes, streams | OpenAI-compatible server | Streaming latency |
| Router + autoscaler | Which replica; how many | Prefix-aware router | Queue length, cache hits |

### Symptom → layer to investigate

| Symptom | Layer | First fix |
|---|---|---|
| GPU idle between kernels, CPU busy | Kernels | Fuse kernels; cut launches |
| Same math slower than expected | Kernels | Better kernels (cuBLAS, FlashAttention) |
| Repeated system prompts inflate TTFT | Router | Prefix-aware routing |
| Queue grows while CPU looks idle | Autoscaler | Scale on queue length and KV use |
| New replicas arrive too late | Fleet | Derivation 3: scale earlier, load faster |

### Rapid-fire Q&A

| Question | Crisp answer |
|---|---|
| What does an inference engine add over PyTorch? | A per-step scheduler (batching) and a KV cache memory manager. |
| Why not autoscale on CPU? | The work is on the GPU; the CPU idles. Watch queue length and KV use. |
| Why is a round-robin load balancer wasteful? | It scatters shared prompts, so each replica re-prefills them. |
| What's a replica? | One engine on its GPU(s), behind an API server. |
| Where does a request's time go? | Mostly GPU (prefill, then ~5 ms per decode step); upper layers decide utilization. |

> [!WARNING]
>
> - Treating LLM serving like a stateless web app: round-robin routing and CPU-based autoscaling.
> - Assuming the model file determines speed: it only defines the math.

<div class="lesson-break"></div>

## 03 · Levels of Inference Engineering

[Full notes](03-levels-of-inference-engineering.md)


### Say it in 30 seconds

> Inference work sits on four levels: the user picks and calls models, the operator makes a model fit and stay up, the optimizer gets more tokens per GPU within latency targets, and the engine builder changes the engine itself. Each level is easier if you understand the one below. The key skill is finding which level a problem lives on; cheap fixes on lower levels often beat kernel work.

### Only numbers worth memorizing

- **DeepSeek-R1 (Feb 2025):** $0.55/M input, $2.19/M output (~4×), $0.14/M cached input.
- **Llama 3 70B:** 140 GB at 16-bit, ~35 GB at 4-bit. **Batching measured:** MPT-7B/A100 0.9 → 12.5 req/s (~14×).

### Derive, don't memorize

#### 1. API bill = tokens × price, per token type

Each token type has its own price, and a cached prefix bills at the cached rate:

$$\mathrm{cost} = N_{\mathrm{cached}} \cdot p_{\mathrm{cached}} + N_{\mathrm{in}} \cdot p_{\mathrm{in}} + N_{\mathrm{out}} \cdot p_{\mathrm{out}} \quad\Rightarrow\quad \$2.42 \rightarrow \$1.19\ \mathrm{per\ 1k\ requests}$$

> [!TIP]
> Put stable text first (caches match prefixes ◆, so any change early in the prompt invalidates everything after it), and cap output length: output tokens cost ~4×.

#### 2. Fit check = params × bytes per param vs GPU memory

$$\mathrm{weights} = P \times \frac{\mathrm{bits}}{8}: \quad 70\mathrm{B} \times 2 = 140\ \mathrm{GB} > 80 \quad\Rightarrow\quad 2\ \mathrm{GPUs\ or\ 4\ bit}\ (35\ \mathrm{GB})$$

> [!TIP]
> "Fits" means weights plus KV cache at peak, not just weights. KV capacity: see 02, derivation 1.

### The four levels

| Level | Question | Controls | Example win |
|---|---|---|---|
| 1 · User | Which model, called how? | Model, prompt, output length, streaming, caching | Stable prefix first: bill ÷ 2 |
| 2 · Operator | Does it fit and stay up? | GPUs, precision, startup, failover, rollouts | Day/night fleet sizing |
| 3 · Optimizer | More per GPU within latency? | Batching, quantization, prefix caching, speculative decoding, PD split | ~14× from batching |
| 4 · Engine builder | What should the engine do differently? | Schedulers, kernels, memory layout | Orca, FlashAttention, PagedAttention |

### Symptom → which level

| Symptom | Level | First fix |
|---|---|---|
| API bill high, same instructions every request | 1 | Identical block at the front for cache hits |
| Model won't load on one GPU | 2 | Split across GPUs or quantize |
| Fits at launch, out of memory at peak | 2 | Budget KV for peak concurrency |
| Attention kernel dominates the profile | 4 | Kernel work (after levels 1–3 are exhausted) |

### Rapid-fire Q&A

| Question | Crisp answer |
|---|---|
| Why do output tokens cost ~4× input? | Decode is serial and memory-bound; prefill is parallel (see 01, derivation 2). |
| Why split prefill and decode onto separate machines? | ◆ Prefill is compute-bound, decode memory-bound; each group is tuned for its own job. |
| What did Orca, FlashAttention, PagedAttention contribute? | Per-step scheduling (continuous batching); no big attention matrix in HBM; paged KV cache (vLLM). |
| Is the engine builder the "senior" level? | No: levels aren't seniority, and most savings come from levels 1–3. |

> [!WARNING]
>
> - Jumping to level 4 (kernels) before checking model choice, prompts and caching.
> - Sizing memory for launch traffic instead of peak.
