"""Exercise 14-17 · Toy KV cache vs naive recompute (one attention head).

The big picture
  To produce token t, attention needs:
    - token t's QUERY  (what am I looking for?)          -> used once, never stored
    - every earlier token's KEY and VALUE (what do they offer / hand over?)
  Because of the causal mask, token j's key and value depend only on tokens 0..j, so once computed
  they never change. The naive loop recomputes them every step anyway (lesson 14: ~213x waste for
  512 + 256 tokens). The KV cache computes each token's k, v ONCE and keeps them.

Setup
  Token vectors x[0], x[1], ... each of size d = 8. One attention head with matrices Wq, Wk, Wv
  (random here; in a real model they're learned). For token t:
      q = x[t] @ Wq
      scores_j = q · k_j / sqrt(d)   for every j <= t      (causal: itself and earlier tokens only)
      out = softmax(scores) @ V[0..t]

What this file shows
  1. naive_outputs: recompute K, V for the whole prefix every step.
  2. cached_outputs: preallocated buffers + a position pointer; each k, v computed once.
  3. They give identical outputs (caching changes HOW we compute, not WHAT).
  4. Bug 2 from lesson 17: if attention reads the zero-filled empty slots, the answer changes,
     because q · 0 = 0 and e^0 = 1, so every empty slot steals attention weight.
"""
import numpy as np

# The mental model. Each token is a list of 8 numbers (a vector). @ means "multiply and add things together". 
# Two shapes to keep in mind:
# vector @ matrix → vector. x[t] @ Wq: one token (8 numbers) times an 8×8 matrix gives 8 new numbers.
# (n rows) @ matrix → (n rows). Doing it for n tokens at once gives n new rows.
rng = np.random.default_rng(0)
d, max_len = 8, 32 # each token = 8 numbers; cache can hold 32 tokens                                      
Wq, Wk, Wv = (rng.normal(size=(d, d)) for _ in range(3)) # three 8×8 matrices of random numbers

# Wq, Wk, Wv are the "learned" projection matrices from the lessons. 
# Here they're random, because we only care that cached and naive give the same answer, 
# not a meaningful one. A token's query is x[t] @ Wq, its key is x[t] @ Wk, its value is x[t] @ Wv.


def attend(q, K, V):
    """One query against keys K and values V (each row = one position).

    `@` rule: (rows x INNER) @ (INNER x cols) = (rows x cols); the matching INNER cancels.
      K @ q : (n x 8) @ (8,)  -> (n,)   n scores, one per key   (q acts as a column on the right)
      w @ V : (n,)   @ (n x 8) -> (8,)  one blended value       (w acts as a row on the left)
    scores = how well q matches each key (scaled by sqrt(d), lesson 05),
    softmax turns scores into weights that sum to 1,
    output = weighted average of the value rows.
    """
    scores = K @ q / np.sqrt(d)   # K is (n × 8), q is (8,). K @ q gives n dot products: how well q matches each key
    w = np.exp(scores - scores.max())   # exponentiate; subtracting the max avoids overflow (lesson 05)
    w /= w.sum()                        # divide by the total so the weights add up to 1 (softmax)
    return w @ V                        # weighted average of the value rows: w is (n,), V is (n × 8) → (8,)


def naive_outputs(x, prompt_len):
    """No cache: at every step, recompute K and V for the WHOLE prefix, then use only the last output."""
    outs = []
    for t in range(prompt_len - 1, len(x)):
        prefix = x[: t + 1]                             # tokens 0..t
        K, V = prefix @ Wk, prefix @ Wv                 # recomputed every step: this is the waste
        outs.append(attend(x[t] @ Wq, K, V))            # newest token's query against all of them
    return np.array(outs)


def cached_outputs(x, prompt_len, read_all_slots=False):
    """With a cache: each token's k, v is computed exactly once and written into a fixed buffer."""

    # Step 1 · State.
    #   Preallocate the full buffer up front (lesson 17): one row per position, max_len rows.
    #   Shapes never change, which is what CUDA graphs and compilers need.
    #   The buffer starts full of ZEROS; `pos` says how many rows are actually filled.
    K_cache = np.zeros((max_len, d))
    V_cache = np.zeros((max_len, d))
    pos = 0

    # Step 2 · Prefill.
    #   The whole prompt is known up front, so compute all its keys and values in one shot
    #   (one big matrix multiply: this is why prefill is compute-efficient, lesson 15)
    #   and write them into rows 0 .. prompt_len-1. The pointer jumps to the end of the prompt.
    K_cache[:prompt_len] = x[:prompt_len] @ Wk
    V_cache[:prompt_len] = x[:prompt_len] @ Wv
    pos = prompt_len

    def read():
        # Step 4 · Which rows attention may see.
        #   Correct: only the filled rows 0 .. pos-1.
        #   Bug 2 (read_all_slots=True): all max_len rows, including the zero rows. Each zero key gives
        #   score 0 -> weight e^0 = 1, so empty slots steal weight and pull the output toward 0.
        n = max_len if read_all_slots else pos
        return K_cache[:n], V_cache[:n]

    # The prompt's last position predicts the first new token: prefill produces the first output.
    outs = [attend(x[prompt_len - 1] @ Wq, *read())]

    # Step 3 · Decode: one new token per step.
    for t in range(prompt_len, len(x)):
        # Compute ONLY the new token's key and value (one small vector each), write them at slot pos.
        # Everything already in the cache is reused as-is: causal masking means it can't have changed.
        K_cache[pos] = x[t] @ Wk
        V_cache[pos] = x[t] @ Wv
        pos += 1                                        # one more filled slot

        # The new token's query is used once and thrown away: queries are never cached.
        outs.append(attend(x[t] @ Wq, *read()))
    return np.array(outs)


if __name__ == "__main__":
    x = rng.normal(size=(12, d))                         # 12 tokens: 5 prompt + 7 generated
    a, b = naive_outputs(x, 5), cached_outputs(x, 5)
    print("cached == naive:", np.allclose(a, b))          # True: the cache is exact
    bug = cached_outputs(x, 5, read_all_slots=True)
    print("reading empty slots, max error:", float(np.abs(bug - a).max()))   # ~4: bug 2 in action
