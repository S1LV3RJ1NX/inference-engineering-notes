"""Exercise 03-04 · Amdahl calculator (time accounting, not adding speedups).

Setup
  A decode step takes 1.0 units of time today. Several optimizations each touch a different,
  non-overlapping part of it:
      "attention": 30% of the step, made 3x faster
      "mlp":       50% of the step, made 2x faster
  The remaining 20% is untouched.

Question
  What is the true end-to-end speedup? (Hint: it is NOT 3 + 2 = 5, and NOT 3 x 2 = 6.)

How to think about it (same structure as before):
  0. Inputs: each part is (fraction_of_original_step, speedup_of_that_part).
  1. Old time = 1.0 (work in fractions of the original step).
  2. Touched parts: each one now takes fraction / speedup.
  3. Untouched part: whatever fraction no optimization touched keeps its full time.
  4. New time = sum of steps 2 and 3.
  5. Output: old time / new time.
"""


def overall_speedup(parts):
    """parts: dict of name -> (fraction_of_step, speedup_of_that_part). Return old time / new time."""
    # Step 1 · Old time
    old_time = 1.0

    # Step 2 · Time of the touched parts after their speedups
    touched_new = sum(fraction / speedup for fraction, speedup in parts.values())

    # Step 3 · Time of the untouched part (unchanged)
    untouched = 1 - sum(fraction for fraction, _ in parts.values())

    # Step 4 · New time
    new_time = touched_new + untouched

    # Step 5 · Speedup
    return old_time / new_time


if __name__ == "__main__":
    print(overall_speedup({"attention": (0.3, 3)}))                  # expect 1.25
    print(overall_speedup({"attention": (0.3, 3), "mlp": (0.5, 2)})) # expect ~1.82
    print(overall_speedup({"attention": (0.3, 1e9)}))                # expect ~1.43 (the ceiling)
