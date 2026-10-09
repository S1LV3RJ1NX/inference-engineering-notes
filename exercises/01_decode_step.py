"""Exercise 01 · Napkin calculator for one decode step.

How to think about it (the same chain of thought as solving it by hand):

  0. Read the inputs: what does each one mean physically?
  1. Normalize: convert every input to plain units (bytes, FLOPs, seconds).
  2. Bytes: what must be read from memory in one step? -> time to read them.
  3. Math: how many FLOPs does one step do? -> time to compute them.
  4. Bottleneck: the two overlap, so the slower one sets the step time.
  5. Speeds: turn the step time into tokens/s per user and in total.
"""


def decode_step(params_b, bytes_per_param, batch, ctx_tokens,
                kv_kb_per_token=128, bw_tbs=3.35, tflops=989):
    """Return (step_ms, total_tok_s, per_user_tok_s, bound) where bound is 'memory' or 'compute'.

    Inputs (step 0: what they mean):
      params_b         model size in billions of parameters (Llama 3 8B -> 8)
      bytes_per_param  precision: BF16 = 2, FP8 = 1, INT4 = 0.5
      batch            how many users decode together in one step
      ctx_tokens       tokens already in each user's conversation (their KV cache length)
      kv_kb_per_token  KV cache per token in KB (Llama 3 8B: 128)
      bw_tbs           GPU memory bandwidth in TB/s (H100: 3.35)
      tflops           GPU peak math in TFLOPS (H100 BF16: 989)
    """

    # Step 1 · Normalize to plain units so no mixed powers of ten creep in.
    params = params_b * 10**9          # parameters
    bw = bw_tbs * 10**12               # bytes per second
    flops = tflops * 10**12            # FLOPs per second

    # Step 2 · Bytes read in one step.
    #   Weights: read ONCE per step, shared by every user in the batch.
    #   KV cache: every user's whole cache is read (their new token attends to all their past tokens).
    weight_bytes = params * bytes_per_param
    kv_bytes = batch * ctx_tokens * kv_kb_per_token * 1024
    t_bytes = (weight_bytes + kv_bytes) / bw                     # seconds

    # Step 3 · Math in one step.
    #   Each parameter does 2 FLOPs (multiply + add) per token, and each user gets one token.
    flops_needed = 2 * params * batch
    t_math = flops_needed / flops                                # seconds

    # Step 4 · Bottleneck: whichever takes longer sets the pace.
    step_s = max(t_bytes, t_math)
    bound = "memory" if t_bytes >= t_math else "compute"

    # Step 5 · Speeds: every step gives each user one token.
    per_user_tok_s = 1 / step_s
    total_tok_s = per_user_tok_s * batch

    return step_s * 1000, total_tok_s, per_user_tok_s, bound      # step time in ms


if __name__ == "__main__":
    print(decode_step(8, 2, 1, 0))       # expect ~4.8 ms, ~200 tok/s, 'memory'
    print(decode_step(8, 2, 64, 4096))   # expect ~15 ms, ~4,300 tok/s total
