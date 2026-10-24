"""Canonical state queue from dense integer marks, with bounded storage.

The optional mark_scan path replaces sort/unique on query-frontier rows with
mark -> scan of the key domain -> ordered scatter -> direct slot lookup.
One host count/error read remains for the existing variable-batch solver ABI.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _mark_keys(Keys, Marks, Metadata, ROWS: tl.constexpr,
               KEY_COUNT: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = row < ROWS
    key = tl.load(Keys + row, mask=active, other=0)
    valid = active & (key >= 0) & (key < KEY_COUNT)
    # Repeated query/node rows may name the same state.
    tl.atomic_or(Marks + key, 1, mask=valid)
    invalid = tl.sum((active & ~valid).to(tl.int32), axis=0)
    tl.atomic_add(Metadata + 1, invalid)


@triton.jit
def _scatter_keys(Marks, Prefix, Queue, SlotMap,
                  KEY_COUNT: tl.constexpr, CAPACITY: tl.constexpr,
                  BLOCK: tl.constexpr):
    key = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = key < KEY_COUNT
    mark = tl.load(Marks + key, mask=live, other=0)
    prefix = tl.load(Prefix + key, mask=live, other=0)
    slot = prefix - 1
    take = live & (mark != 0) & (slot >= 0) & (slot < CAPACITY)
    # Each marked key has a distinct prefix index: no atomic append and no
    # scheduling-dependent permutation of the state solver's initial vectors.
    tl.store(Queue + slot, key.to(tl.int32), mask=take)
    tl.store(SlotMap + key, tl.where(take, slot, -1), mask=live)


@triton.jit
def _lookup_slots(Keys, SlotMap, Slots, ROWS: tl.constexpr,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = row < ROWS
    key = tl.load(Keys + row, mask=active, other=0)
    # All active keys were bounds-checked by mark + boundary metadata.
    slot = tl.load(SlotMap + key, mask=active, other=-1)
    tl.store(Slots + row, slot, mask=active)


@torch.inference_mode()
def build_state_queue(row_keys, centroid_count, node_count, workspace):
    """Return canonical int32 keys and int32 row->state slots.

    In mark_scan mode both results alias workspace and last until the next
    queue build. The caller uses them on the workspace's execution stream.
    Default unique mode retains the established algorithm.
    """
    if workspace.plan.config.state_queue == "unique":
        keys, inverse = torch.unique(row_keys, sorted=True, return_inverse=True)
        return keys, inverse.to(torch.int32).contiguous()

    if row_keys.ndim != 1 or row_keys.dtype != torch.int32:
        raise TypeError("mark_scan requires the authority's int32 row keys")
    if row_keys.device != workspace.device or not row_keys.is_contiguous():
        raise ValueError("row keys must be contiguous on the workspace device")
    key_count = int(centroid_count) * int(node_count)
    if key_count <= 0:
        raise ValueError("state key domain must be nonempty")
    if (centroid_count > workspace.plan.centroid_capacity
            or node_count > workspace.plan.node_capacity):
        raise ValueError("state key domain exceeds the workspace plan")
    rows = int(row_keys.numel())
    workspace.require("frontier", rows, workspace.plan.config.frontier_capacity)
    marks = workspace.buffers["queue_marks"][:key_count]
    prefix = workspace.buffers["queue_prefix"][:key_count]
    slot_map = workspace.buffers["queue_slot_map"][:key_count]
    queue = workspace.buffers["queue_keys"]
    slots = workspace.buffers["queue_row_slots"][:rows]
    metadata = workspace.buffers["queue_metadata"]
    marks.zero_()
    metadata.zero_()
    if rows:
        _mark_keys[(triton.cdiv(rows, 256),)](
            row_keys, marks, metadata, ROWS=rows, KEY_COUNT=key_count,
            BLOCK=256, num_warps=4,
        )
    # The plan proves key_count < 2**31; this sums zero/one marks only.
    torch.cumsum(marks, dim=0, dtype=torch.int32, out=prefix)
    metadata[0].copy_(prefix[-1])
    required, invalid = (int(x) for x in metadata.cpu().tolist())
    if invalid:
        status = workspace.buffers["status"]
        status[0].fill_(3)
        status[1].zero_()
        status[2].fill_(invalid)
        status[3].fill_(key_count)
        raise RuntimeError(
            f"HTRC invalid state keys: {invalid} rows outside [0,{key_count})"
        )
    workspace.require("states", required, workspace.plan.state_capacity)
    _scatter_keys[(triton.cdiv(key_count, 256),)](
        marks, prefix, queue, slot_map, KEY_COUNT=key_count,
        CAPACITY=queue.numel(), BLOCK=256, num_warps=4,
    )
    if rows:
        _lookup_slots[(triton.cdiv(rows, 256),)](
            row_keys, slot_map, slots, ROWS=rows, BLOCK=256, num_warps=4,
        )
    return queue[:required], slots
