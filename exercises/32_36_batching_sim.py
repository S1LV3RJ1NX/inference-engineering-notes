"""Static vs continuous batching on the same queue of requests.

Steps:
1. All requests are queued at t = 0 in arrival order; one step = one token per running request.
2. Static: take `slots` requests at a time; the group runs for its longest member and
   everyone in it is handed back when the group ends.
3. Continuous: track when each slot frees up; each request takes the slot that frees first
   and finishes `n` steps later.
4. Useful share = tokens generated / (slots x total steps); average finish = mean finish step.

Run: uv run exercises/32_36_batching_sim.py
"""


def static(lengths: list[int], slots: int) -> tuple[int, list[int]]:
    """Groups of `slots` run together until their longest member is done."""
    t = 0
    finishes = []
    for g in range(0, len(lengths), slots):
        group = lengths[g:g + slots]
        group_steps = max(group)                       # padding: short rows ride along
        finishes += [t + group_steps] * len(group)     # everyone returns with the batch
        t += group_steps
    return t, finishes


def continuous(lengths: list[int], slots: int) -> tuple[int, list[int]]:
    """A freed slot goes to the next queued request right away."""
    free_at = [0] * slots          # step at which each slot becomes free
    finishes = []
    for n in lengths:
        i = free_at.index(min(free_at))   # FCFS: next request takes the slot that frees first
        free_at[i] = free_at[i] + n       # starts when the slot frees, runs n steps
        finishes.append(free_at[i])       # handed back the moment it's done
    total = max(free_at)                  # the batch drains when the busiest slot finishes
    return total, finishes


def simulate(lengths: list[int], slots: int) -> dict:
    """lengths: output tokens per request, in arrival order (all queued at t = 0).
    slots: how many requests can run at once.
    Returns {name: (total_steps, useful_share, avg_finish)} for static and continuous."""
    out = {}
    for name, run in (("static", static), ("continuous", continuous)):
        total, finishes = run(lengths, slots)
        useful = sum(lengths) / (slots * total)
        out[name] = (total, round(useful, 3), round(sum(finishes) / len(finishes), 1))
    return out


if __name__ == "__main__":
    print(simulate([100, 1000, 200, 300, 150, 400, 250, 700, 120, 500, 180, 350], 4))
    print("want static (2200, 0.483, 1633.3)  continuous (1150, 0.924, 635.0)")
    print(simulate([50, 150, 400, 800], 2))
    print("want static (950, 0.737, 550.0)  continuous (950, 0.737, 400.0)")
