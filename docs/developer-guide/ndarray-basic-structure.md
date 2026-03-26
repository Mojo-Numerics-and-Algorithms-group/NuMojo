# Basic structure of NDArray

> Date: 2026-02-23
> This also applies to `Matrix` type.

The `NDArray` is wrapper around `DataContainer` and other metadata such as shape, strides, offset, and flags. The actual data is stored in the `DataContainer`. The `DataContainer` manages the memory allocation and reference counting for the data buffer.

```txt
NDArray[dtype]
  ├── shape, strides, offset, ndim, flags ...
  └── _buf: DataContainer[dtype]
              ├── ptr         → [data data data data ...]  (heap allocation A)
              ├── _refcount   → [Atomic(1)]                (heap allocation B)
              ├── ownership   = Managed
              └── size        = N
```

For each `DataContainer`, there are two heap allocations: one for the data buffer (`ptr`) and one for the reference count (`_refcount`).

The `NDArray` itself is a stack allocation that contains metadata and a pointer to the `DataContainer`.

When you create a **view** of an `NDArray`, it create a new `DataContainer` object that shares the same data buffer and the same reference count, and also increments the reference count by 1. This ensures that, when any `DataContainer` is destroyed, as long as the reference count is greater than 0, the data buffer will not be deallocated.

Below is an example of the reference counting mechanism when creating a view and destroying the original array:

```txt
Step 1: arr = NDArray(...), i.e., alloc()
  arr._buf.ptr       → [data...]      (alloc A)
  arr._buf._refcount → [Atomic(1)]    (alloc B)

Step 2: v = arr.view(), i.e., share()
  ┌─ arr._buf.ptr       ─→ [data...]      (alloc A, shared)
  │  arr._buf._refcount ─→ [Atomic(2)]    (alloc B, shared, refcount = 2)
  │
  └─ v._buf.ptr         ─╯
     v._buf._refcount   ─╯

Step 3: v is destroyed (__del__)
  fetch_sub → refcount = 1, not released

Step 4: arr is destroyed (__del__)
  fetch_sub → refcount = 0 → release alloc A (ptr.free()) + release alloc B (_refcount.free())
```
