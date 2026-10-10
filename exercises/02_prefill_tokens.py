"""Exercise 02 · Round-robin vs prefix-aware router.

Setup
  - 12 requests arrive, one after another (request 0, 1, 2, ... 11).
  - Each request = a shared SYSTEM PROMPT + its own USER MESSAGE.
      * There are 3 different system prompts, A, B, C (ids 0, 1, 2), 1,000 tokens each.
        Requests cycle through them: A, B, C, A, B, C, ...  ->  prefix_id = i % 3
      * Every user message is 50 tokens and is always new.
  - There are 4 replicas (4 GPUs, each running the engine).
  - A replica keeps every system prompt it has prefilled in its cache. If a later request with
    the same prompt lands on the same replica, that prompt's 1,000 tokens are free (no prefill).

Question
  How many prompt tokens does the whole fleet prefill under each routing policy?
    'round_robin': request i goes to replica i % 4 (ignores the prompt)
    'prefix':      every request with the same prefix_id goes to the same replica (prefix_id % 4)

Worked example for round-robin (the two cycles, i % 4 and i % 3, don't line up):
    replica 0 gets requests 0, 4, 8   -> prompts A, B, C -> all new, 3 prefills of 1,000
    replica 1 gets requests 1, 5, 9   -> prompts B, C, A -> 3 prefills
    replica 2 gets requests 2, 6, 10  -> prompts C, A, B -> 3 prefills
    replica 3 gets requests 3, 7, 11  -> prompts A, B, C -> 3 prefills
    total = 12 x 1,000 (prompts) + 12 x 50 (user messages) = 12,600

How to think about it (same structure as exercise 01):
  0. Inputs: each request is (prefix_id, prefix_len, user_len).
  1. State: for each replica, the set of prefix_ids it has already cached (all start empty).
  2. For each request: pick a replica according to the policy.
  3. Cost of this request: user_len always, plus prefix_len only if that replica hasn't cached the prompt.
  4. Update the state: the replica now has that prompt cached.
  5. Output: the sum of all request costs.
"""


def total_prefill_tokens(requests, n_replicas, policy):
    """Return the total prompt tokens the fleet must prefill.

    requests:   list of (prefix_id, prefix_len, user_len), in arrival order
    n_replicas: number of replicas (4)
    policy:     'round_robin' or 'prefix'
    """
    # Step 1 · State: one empty set per replica (which prefix_ids it has cached)
    cached = [set() for _ in range(n_replicas)]

    total = 0
    for i, (prefix_id, prefix_len, user_len) in enumerate(requests):
        # Step 2 · Pick the replica
        if policy == "round_robin":
            replica = i % n_replicas
        else:
            replica = prefix_id % n_replicas

        # Step 3 · Cost of this request
        cost = user_len + (0 if prefix_id in cached[replica] else prefix_len)

        # Step 4 · Update the state
        cached[replica].add(prefix_id)

        total += cost

    # Step 5 · Output
    return total


if __name__ == "__main__":
    requests = [(i % 3, 1000, 50) for i in range(12)]   # A, B, C, A, B, C, ...
    print(total_prefill_tokens(requests, 4, "round_robin"))   # expect 12,600
    print(total_prefill_tokens(requests, 4, "prefix"))        # expect 3,600
