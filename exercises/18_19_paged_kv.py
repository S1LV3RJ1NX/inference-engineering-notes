"""Exercise 18-19 · Simulate a paged KV allocator vs reserving the max length.

Why this matters (lessons 17 and 19)
  Every running request keeps a KV cache on the GPU. The pool is fixed (~400k tokens on an H100 for
  Llama 3 8B). How you hand that pool out decides how many users run at once:
    - Reserve the max length for every request (lesson 17's preallocated buffer): simple and safe,
      but a request that ends after 150 tokens still holds memory for 512. Most of it sits empty.
    - Paged (vLLM's PagedAttention): split the pool into small BLOCKS (here 16 tokens) and give a
      request a new block only when it actually needs one. Many more requests fit, but the engine is
      betting they stay short. If too many grow at once and the pool runs dry, it must PREEMPT.

Setup (small numbers so the simulation runs instantly)
  - Pool = 64 blocks × 16 tokens = 1,024 tokens.
  - 20 requests, each (prompt_len, output_len): prompt 100 tokens, outputs between 50 and 400.
  - Each step, every running request generates ONE token (that's what a decode step does).

Result
  reserve -> peak 2 running,  0 preemptions, 2,187 steps
  paged   -> peak 9 running, 50 preemptions, 1,265 steps
  Paged runs 4.5x more requests at once and finishes 1.7x sooner, but 50 times a request got evicted
  and had to start over: those users saw a stall.

How the code is built (the chain of thought)
  State to track:
    waiting  - queue of request ids not yet running (FIFO, preempted ones go back to the FRONT)
    running  - id -> [tokens it holds so far (prompt + generated), blocks it holds]
    free     - blocks left in the pool
  Each step:
    1. Admit:  while the first waiting request's starting blocks fit in `free`, move it to running.
    2. Grow:   every running request adds one token. If that token needs a block it doesn't hold
               (paged only), take one from `free`, or preempt someone if `free` is empty.
    3. Finish: a request that has generated all its output tokens returns its blocks to `free`.
  Everything is counted in BLOCKS, never tokens: convert with need(tokens) = ceil(tokens / 16).
"""
import math


def simulate(requests, pool_blocks, block=16, policy="paged", max_len=512):
    # Blocks needed to hold `toks` tokens. 100 tokens -> 7 blocks (7 × 16 = 112 ≥ 100).
    need = lambda toks: math.ceil(toks / block)

    waiting = list(range(len(requests)))             # request ids in arrival order
    running = {}                                     # id -> [tokens_so_far, blocks_held]
    free = pool_blocks                               # blocks not owned by anyone
    peak = preempt = steps = done = 0                # stats to report

    while done < len(requests):
        steps += 1

        # Step 1 · Admit from the front of the queue while it fits.
        #   'reserve': a request takes blocks for its whole max length up front (512 tokens = 32 blocks)
        #              and never asks again.
        #   'paged':   it takes only what its prompt needs now (100 tokens = 7 blocks).
        #   Stop at the first request that doesn't fit (FIFO: no skipping ahead).
        while waiting:
            rid = waiting[0]
            p, _ = requests[rid]
            want = need(max_len) if policy == "reserve" else need(p)   # BLOCKS, not tokens
            if want > free:
                break
            waiting.pop(0)
            free -= want
            running[rid] = [p, want]                 # it starts holding its prompt's tokens
        peak = max(peak, len(running))

        # Step 2 · Decode step: every running request generates one token.
        for rid in list(running):                    # list(): we may delete entries while looping
            if rid not in running:                   # preempted earlier in this same step
                continue
            toks, held = running[rid]
            p, out = requests[rid]
            toks += 1                                # the new token

            # Paged only: does the new token spill past the blocks it holds?
            #   7 blocks hold tokens 1..112, so token 113 is the first that needs block 8.
            #   Comparing need(toks) with held is exact; checking toks % 16 would be off by one.
            if policy == "paged" and need(toks) > held:
                if free == 0:
                    # Pool is dry: preempt. Evict the most recently admitted request (largest id),
                    # the one that has made the least progress, so we lose the least work.
                    victim = max(running)
                    free += running.pop(victim)[1]   # remove it AND return all its blocks
                    waiting.insert(0, victim)        # front of the queue: it restarts from its prompt
                    preempt += 1
                    if victim == rid:                # we evicted ourselves: skip the rest
                        continue
                free -= 1                            # take one block for the new token
                held += 1

            running[rid] = [toks, held]

            # Step 3 · Finished? It has generated `out` tokens beyond its prompt: free its blocks.
            if toks >= p + out:
                free += held
                del running[rid]
                done += 1

    return dict(peak=peak, preemptions=preempt, steps=steps)


if __name__ == "__main__":
    reqs = [(100, 50 + (i * 37) % 350) for i in range(20)]   # prompt 100, outputs 50..399
    print("reserve", simulate(reqs, 64, policy="reserve"))   # peak 2, 0 preemptions, 2187 steps
    print("paged  ", simulate(reqs, 64, policy="paged"))     # peak 9, 50 preemptions, 1265 steps
