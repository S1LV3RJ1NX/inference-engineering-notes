"""Exercise 37-41 · A KV block manager with reference counts and copy-on-write.

Fill in the four TODO methods. Run: uv run exercises/37_41_block_manager.py

Rules (lessons 38 and 40)
  - The pool has num_blocks blocks of block_size tokens. Free blocks wait in a FIFO free queue:
    take from the FRONT, return to the BACK.
  - allocate(seq, n): a new request gets just enough blocks for its n-token prompt.
  - fork(parent, child): child gets a copy of parent's block TABLE (same physical blocks),
    and every one of those blocks gains one reference.
  - append(seq): the sequence writes one new token.
      * last block full      -> take a fresh block (lazy allocation)
      * last block shared    -> copy-on-write: take a fresh block, point seq at it,
                                drop the old block's count by one, count a copy
      * otherwise            -> write in place, nothing to do
  - free(seq): walk its table TAIL FIRST, decrement each count, return blocks at 0 to the back.
  - No free block when one is needed -> raise MemoryError (the scheduler would preempt).

Expected output
  3 {'a': [0, 1, 2]}
  3 [4, 4, 4]
  6 3 {'a': [0, 1, 3], 'b': [0, 1, 4], 'c': [0, 1, 5], 'd': [0, 1, 2]}
  5 [6, 7, 8, 9, 3] [3, 3, 1, 0, 1, 1]
  4 0 {'x': [0, 1, 2], 'y': [0, 1, 3]}
  MemoryError {'x': 48, 'y': 33}
"""
from collections import deque


class BlockManager:
    def __init__(self, num_blocks, block_size=16):
        self.bs = block_size
        self.free_q = deque(range(num_blocks))   # free block ids, front = next to hand out
        self.ref = [0] * num_blocks               # ref[b] = how many tables point at block b
        self.tables = {}                          # seq -> list of physical block ids
        self.lens = {}                            # seq -> tokens it holds
        self.copies = 0                           # copy-on-write copies made

    def _take(self):
        """Pop a block from the front of the free queue and give it count 1."""
        if not self.free_q:
            raise MemoryError("no free block: preempt someone")
        b = self.free_q.popleft()
        self.ref[b] = 1
        return b

    def allocate(self, seq, n_tokens):
        needed_blocks = (n_tokens + self.bs - 1) // self.bs
        table = [self._take() for _ in range(needed_blocks)]
        self.tables[seq] = table
        self.lens[seq] = n_tokens
        return table

    def fork(self, parent, child):
        # Fixed: `= self.tables[parent]` made both names point at ONE list, so a later append or
        # copy-on-write in one sequence changed the other's table too. Copy the list; share blocks.
        self.tables[child] = list(self.tables[parent])
        self.lens[child] = self.lens[parent]
        for block in self.tables[child]:
            self.ref[block] += 1
        return child

    def append(self, seq):
        t, n = self.tables[seq], self.lens[seq]
        # Fixed: t[-1] is a physical block ID, not how full it is, so `t[-1] == self.bs` and
        # `t[-1] += 1` compared and changed the block number. Fullness comes from the length:
        # n tokens fill the last block exactly when n is a multiple of the block size.
        if n % self.bs == 0:
            t.append(self._take())                 # case 1: last block full -> fresh block
        elif self.ref[t[-1]] > 1:
            # Added: case 2, the last block is partly full AND shared -> copy-on-write.
            old = t[-1]
            t[-1] = self._take()                   # our own copy of that one block
            self.ref[old] -= 1                     # the others keep the original
            self.copies += 1
        # case 3: partly full and ours alone -> write in place, no block work
        self.lens[seq] = n + 1
        return seq

    def free(self, seq):
        t = self.tables.pop(seq)
        for block in reversed(t):                  # tail first: deep blocks are least reusable
            self.ref[block] -= 1
            if self.ref[block] == 0:
                self.free_q.append(block)
        del self.lens[seq]                         # Fixed: forget the length too
        return seq

    def used(self):
        return sum(r > 0 for r in self.ref)


if __name__ == "__main__":
    bm = BlockManager(10)
    bm.allocate("a", 40)                      # 16 + 16 + 8 -> 3 blocks
    print(bm.used(), bm.tables)
    for c in "bcd":
        bm.fork("a", c)                       # 4 answers share the prompt
    print(bm.used(), bm.ref[:3])
    for s in "abcd":
        bm.append(s)                          # first writes: 3 copies, the last writes in place
    print(bm.used(), bm.copies, bm.tables)
    bm.free("a")
    print(bm.used(), list(bm.free_q), bm.ref[:6])

    bm2 = BlockManager(4)
    bm2.allocate("x", 32)                     # prompt ends on a block boundary
    bm2.fork("x", "y")
    bm2.append("x")
    bm2.append("y")                           # fresh blocks, no copies
    print(bm2.used(), bm2.copies, bm2.tables)
    try:
        for _ in range(17):
            bm2.append("x")                   # fills block 2, then needs a 5th block
    except MemoryError:
        print("MemoryError", bm2.lens)
