"""Exercise 06-08-09 · See benchmark noise instead of computing it.

Part A · How many samples does P99 need?
  Latencies are long-tailed (most requests fast, a few very slow). Draw n latencies, take their
  P99, and repeat 200 times. If the 200 estimates are spread out, n was too small.
  Lesson 06 says: you need ~100 samples BEYOND the percentile for ~10% precision,
  so P99 needs ~100 / 0.01 = 10,000 samples. With n = 100, only ~1 sample is beyond P99.

Part B · Check the 20-call agent answer (Q1) by brute force.
  Simulate many tasks of 20 calls; each call is slow with probability 1%.
  Count the fraction of tasks with at least one slow call. Expect ~18%.

How to think about it:
  A1. One "benchmark" = draw n latencies, return their 99th percentile.
  A2. Repeat the benchmark 200 times -> 200 P99 estimates.
  A3. Spread = how far apart they are (e.g. relative std = std / mean).
  B1. One task = 20 calls; it's "slow" if ANY call is slow.
  B2. Fraction of slow tasks over many tasks.
"""
import random
import statistics


def latency():
    """One request's latency in seconds: lognormal = long right tail (median ~1 s)."""
    return random.lognormvariate(0.0, 0.5)


def p99(samples):
    """99th percentile: sort, take the value 99% of the way up."""
    s = sorted(samples)
    return s[int(0.99 * (len(s) - 1))]


def p99_spread(n, repeats=200):
    """Part A: run `repeats` benchmarks of n samples each; return (mean P99, relative spread)."""
    # A1 + A2 · 200 P99 estimates, each from n fresh latencies
    estimates = [p99([latency() for _ in range(n)]) for _ in range(repeats)]

    # A3 · Mean and relative spread (std / mean) of those estimates
    mean = statistics.mean(estimates)
    rel_spread = statistics.stdev(estimates) / mean
    return mean, rel_spread


def slow_task_fraction(calls=20, q=0.01, tasks=100_000):
    """Part B: fraction of tasks where at least one of `calls` calls is slow (probability q each)."""
    slow_tasks = 0
    for _ in range(tasks):
        # B1 · A task is slow if ANY of its calls is slow (random.random() < q)
        if any(random.random() < q for _ in range(calls)):
            slow_tasks += 1
    # B2 · Fraction
    return slow_tasks / tasks


if __name__ == "__main__":
    random.seed(0)
    for n in (100, 1_000, 10_000):
        mean, spread = p99_spread(n)
        print(f"n = {n:>6}: P99 ≈ {mean:.2f} s, spread ±{spread:.0%}")   # spread shrinks ~√n: big at 100, ~few % at 10k
    print(f"slow tasks: {slow_task_fraction():.1%}")                      # expect ~18% (1 - 0.99^20)
