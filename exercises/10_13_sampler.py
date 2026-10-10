"""Exercise 10-13 · Build the sampler (temperature, top-p, min-p).

Toy vocabulary from lesson 13, prompt "The capital of France is":
  tokens = Paris, the, a, located, Lyon
  logits = 5.0, 3.2, 2.9, 2.1, 0.5
  softmax at T = 1  -> 74%, 12%, 9%, 4%, <1%

Pipeline (usual order):
  1. Greedy shortcut: temperature == 0 -> all probability on the argmax.
  2. Temperature: divide every logit by T (T < 1 sharpens, T > 1 flattens; never reorders).
  3. Softmax: subtract the max, exponentiate, divide by the sum.
  4. Cut the tail:
       top-p: sort by probability, keep the smallest set whose running sum reaches top_p.
       min-p: keep tokens with p >= min_p * (largest p).
     A token survives only if BOTH rules keep it.
  5. Rescale: survivors divided by their total, so they sum to 1; cut tokens get 0.
  6. Draw: pick an index at random, weighted by the final probabilities.

Expected (see __main__):
  T = 1, no cut       -> 0.74, 0.12, 0.09, 0.04, 0.01
  T = 0.5             -> Paris ~0.96
  T = 1, top_p = 0.9  -> running sum 74, 86, 95 passes 90 at the 3rd token -> 0.78, 0.13, 0.10, 0, 0
  T = 1, min_p = 0.1  -> cutoff 0.1 x 0.74 = 0.074 -> same 3 tokens kept
  T = 0               -> 1, 0, 0, 0, 0
"""
import math
import random


def sampler_probs(logits, temperature=1.0, top_p=1.0, min_p=0.0):
    """Return the final probability for each token after the whole pipeline."""
    n = len(logits)

    # Step 1 · Greedy shortcut
    if temperature == 0:
        best = max(range(n), key=lambda i: logits[i])
        return [1.0 if i == best else 0.0 for i in range(n)]

    # Step 2 · Temperature
    scaled = [logits[i] / temperature for i in range(n)]

    # Step 3 · Softmax (subtract the max first)
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    probs = [x / sum(exps) for x in exps]

    # Step 4a · top-p: walk tokens from most to least likely, keep until the running sum reaches top_p
    order = sorted(range(n), key=lambda i: probs[i], reverse=True)
    keep = set()
    running = 0.0
    for i in order:
        running += probs[i]
        keep.add(i)
        if running >= top_p:
            break

    # Step 4b · min-p: drop survivors below min_p × the top probability
    keep = [i for i in keep if probs[i] >= min_p * max(probs)]

    # Step 5 · Rescale survivors to sum to 1; everything else 0
    total = sum(probs[i] for i in keep)
    return [probs[i] / total if i in keep else 0.0 for i in range(n)]


def sample(logits, temperature=1.0, top_p=1.0, min_p=0.0):
    """Step 6 · Draw one token index according to the final probabilities."""
    probs = sampler_probs(logits, temperature, top_p, min_p)
    return random.choices(range(len(probs)), weights=probs)[0]


if __name__ == "__main__":
    z = [5.0, 3.2, 2.9, 2.1, 0.5]
    fmt = lambda p: [round(x, 2) for x in p]
    print(fmt(sampler_probs(z)))                  # [0.74, 0.12, 0.09, 0.04, 0.01]
    print(fmt(sampler_probs(z, temperature=0.5))) # Paris ~0.96
    print(fmt(sampler_probs(z, top_p=0.9)))       # [0.78, 0.13, 0.1, 0.0, 0.0]
    print(fmt(sampler_probs(z, min_p=0.1)))       # [0.78, 0.13, 0.1, 0.0, 0.0]
    print(fmt(sampler_probs(z, temperature=0)))   # [1.0, 0.0, 0.0, 0.0, 0.0]
    random.seed(0)
    draws = [sample(z, top_p=0.9) for _ in range(10_000)]
    print([round(draws.count(i) / 10_000, 2) for i in range(5)])   # ≈ the top_p line above
