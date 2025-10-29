from collections import deque
import xxhash
import numpy as np

from nano_pearl.pearl_engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    # PEARL KV cache Rollback
    def rollback(self, seq: Sequence, n: int) -> None:
        block_table = seq.block_table
        before_num_blocks = seq.num_blocks
        seq.rollback_tokens(n)
        after_num_blocks = seq.num_blocks
        if before_num_blocks == after_num_blocks:
            return
        for block_id in block_table[after_num_blocks:]:
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.block_table = seq.block_table[:after_num_blocks]

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """
        Decide whether to append a new block to the sequence when appending n tokens.
        When the sequence is already appended with n tokens, three situations may happen:
        1. required_blocks > current_blocks: we need to append a new block, and hash the last block
        2. required_blocks == current_blocks and the last block is full. we need to hash the last block
        3. required_blocks == current_blocks and the last block is not full. we do nothing.
        """
        block_table = seq.block_table
        required_blocks = seq.num_blocks
        current_blocks = len(block_table)
        if required_blocks > current_blocks:
            assert required_blocks == current_blocks + 1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)

            if self.blocks[block_table[-2]].hash == -1:
                # -1 is the new appended block, -2 is the last block, -3 is the prefix of the last block
                token_ids = seq.block(seq.num_blocks-2)
                prefix = self.blocks[block_table[-3]].hash if len(block_table) > 2 else -1
                h = self.compute_hash(token_ids, prefix)
                self.blocks[block_table[-2]].update(h, token_ids)
                self.hash_to_block_id[h] = block_table[-2]
        else:
            if seq.last_block_num_tokens == self.block_size:
                token_ids = seq.block(seq.num_blocks-1)
                prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
                h = self.compute_hash(token_ids, prefix)
                self.blocks[block_table[-1]].update(h, token_ids)
                self.hash_to_block_id[h] = block_table[-1]