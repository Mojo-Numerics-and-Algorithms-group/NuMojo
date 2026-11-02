# AcceleratorNDArray view and execution design

> Status: implemented owner/view trait model, with future TileTensor work.

This note records the intended owner/view boundary, the pointer-access rules,
and a future MAX `TileTensor` execution path. It is not a description of the
current stable `NDArray` implementation.

## Owner and view model

`AcceleratorNDArray` is the movable owner of its allocation. Calling `copy()`
creates a new owner with independent storage. Metadata-only transformations
produce `AcceleratorNDArrayView` values that borrow the owner's storage and own
a copied `NDLayout`.

View mutability is encoded by its origin rather than an independent writable
flag:

```mojo
struct AcceleratorNDArrayView[
    mut: Bool,
    //,
    dtype: DType,
    device: Device,
    origin: Origin[mut=mut],
](Movable):
    ...
```

- `array.view()` returns a view with `mut=False` and `ImmOrigin(o)`, even when
  `array` is currently mutable.
- `array.view_mut()` requires `o: MutOrigin` and returns a view with `mut=True`
  and origin `o`.
- Slicing, selection, transposition, and reshaping preserve the source view's
  origin and mutability.
- This phase tracks the whole owner origin. It deliberately does not depend on
  owned interior origins while that compiler feature remains unstable.

## Pointer-access invariant

Pointers stored inside owners and views are private implementation
details. Code must never obtain a pointer through direct field access such as
`array._ptr` or `view._ptr`.

Direct field access can lose the origin relationship expressed by the public
borrow and lets callers accidentally substitute the origin of a wrapper object
for the origin of the allocation. Every pointer must instead be obtained from
an `unsafe_ptr()` method whose return type states the correct data origin.

The one `NDArrayLike` trait exposes generic read and write pointer access:

```mojo
trait NDArrayLike:
    comptime ElementType: DType
    comptime DeviceType: Device
    comptime Writable: Bool

    def unsafe_ptr[o: Origin](
        ref[o] self,
    ) -> Pointer[Scalar[Self.ElementType], ImmOrigin(o)]:
        ...

    def unsafe_mut_ptr[o: MutOrigin](
        ref[o] self,
    ) -> Pointer[Scalar[Self.ElementType], o]:
        ...

```

`unsafe_ptr()` always returns `ImmOrigin(o)`, even when `o` is mutable. A
read-only view stored in a `var` therefore still produces only an immutable
pointer. Mojo 1.2 does not allow `where` clauses on trait methods, so each
concrete `unsafe_mut_ptr()` implementation begins with
`comptime assert Self.Writable`. The `o: MutOrigin` signature independently
requires a mutable receiver borrow. Owners set `Writable = True`; views set
`Writable = Self.mut`, so calling the generic mutable method on a view created
from an immutable owner fails during specialization.

For a view, the pointer is reborrowed through the current view origin. The view's
private stored pointer still carries the owner origin, so keeping the view alive
also keeps its owner alive. Copying or rebinding a read-only view preserves
`Writable == False` and cannot strengthen this capability.

Both methods return the full `Pointer` API for SIMD loads/stores, bitcasts,
offsets, and kernel arguments. Logical strided traversal remains the caller's
responsibility through `NDLayout`; a raw SIMD load is valid only after the
operation establishes the required physical contiguity.

## CPU and GPU boundaries

CPU SIMD implementations, including fully generic `NDArrayLike` code, may call
`unsafe_ptr()` or, for writable values, `unsafe_mut_ptr()`, after checking layout
requirements. Strided paths compute physical offsets through `NDLayout`.

GPU implementations also obtain pointers only through `unsafe_ptr()`. Origins
must be preserved through internal adapters for as long as the kernel API
allows. If an existing MAX kernel requires `ImmutAnyOrigin` or `MutAnyOrigin`,
origin erasure occurs only in the final launch adapter, after all validation.
The originating owner or view must remain alive until the queued operation no
longer needs its storage. The initial synchronous implementation
achieves this by synchronizing before tracked borrows are released. The planned
asynchronous implementation below instead makes queued work retain its storage
and context until completion.

## Planned accelerator execution architecture

The accelerator API will support ordinary eager operations and an explicit
fused elementwise path on top of a shared, asynchronous execution model. These
are three separate pieces of work and should be implemented in this order.

### 1. Shared device contexts and asynchronous eager operations

Use one lazily created default `DeviceContext` per physical device identity
(backend plus ordinal), rather than constructing a native context and command
queue for every allocation. A single universal context is not appropriate for
multiple GPUs or different backends. Advanced callers may eventually supply an
explicit context or stream, but ordinary allocation uses the shared default.

- `DeviceStorage` accepts or obtains the shared context, and output allocations
  inherit the context of their operands.
- Operations validate that all operands are on compatible devices and contexts.
- Kernels, fills, copies, and reductions returning device arrays enqueue work
  and return without synchronizing. Commands on the same queue provide ordering
  for dependent eager expressions.
- Synchronization happens only at host-observation boundaries: `to_host()`,
  printing, conversion to a host scalar or list, host file I/O, benchmark
  timing, and an explicit `synchronize()` call.
- A reduction returning a zero-dimensional device array remains asynchronous.
- Host-to-device initialization is asynchronous only when an owned or pinned
  staging buffer is retained until the copy completes. A temporary raw host
  pointer must either be retained through such a buffer or synchronized before
  it expires.
- Queued work must retain every input/output allocation it touches. Removing
  per-operation synchronization before this ownership rule and shared-context
  propagation are in place would introduce cross-context races and dangling
  borrows.
- Compiled kernel functions should be cached per device/context and kernel
  specialization instead of being recompiled for every operation.

An eager expression such as `exp(x * x)` therefore enqueues multiplication and
then exponentiation as two ordered kernels without an intermediate host wait.
It is asynchronous, but it is not fused and still uses an intermediate device
array.

### 2. `DevicePassable` projection

Make the owning `AcceleratorNDArray` `DevicePassable` by projecting it at the
launch boundary to a small device-side view. The owner, reference-counted
buffer, host context, and runtime-only metadata do not cross that boundary.

The first projection may be a contiguous data pointer with the element count
passed separately, matching the current kernels. The later projection should
contain device-compatible shape/stride metadata, or a rank-specialized
`TileTensor`, when non-contiguous device views are supported. In either case:

- `_to_device_type()` obtains the pointer through `unsafe_ptr()` or
  `unsafe_mut_ptr()`; it never reads a pointer field directly.
- Origin erasure, when required by the MAX launch ABI, occurs only inside this
  internal adapter.
- `DevicePassable` changes argument encoding and kernel-launch ergonomics. It
  does not transfer storage, synchronize work, select a context, or fuse eager
  operations.
- Read-only and writable public access remain distinct even if the internal GPU
  ABI ultimately requires an origin-erased pointer.

### 3. Eager and explicitly fused elementwise APIs

Keep NumPy-like eager operators as the default API. Add an explicit
`elementwise[step]` family for callers who want a scalar expression compiled as
one traversal/kernel:

```mojo
def square_exp(x):
    return exp(x * x)

var eager = exp(x * x)                  # two asynchronous kernels
var fused = elementwise[square_exp](x)  # one kernel, no intermediate array
```

The same `step` should drive a vectorized CPU implementation and a scalar-lane
GPU implementation. Start with contiguous unary and binary inputs, then add
broadcasting, multiple outputs, strided views, and rank-specialized
`TileTensor` adapters. Fusion is always explicit in this phase; ordinary Mojo
function composition over eager ndarray operators must not be presented as
automatic kernel fusion.

## NuMax owner/view reference audit

NuMax `v0.3.1` at commit `f0a963f0294922fdbf25f702115659c261798c8e`
was compiled and tested against the pinned Mojo 1.2/MAX 26.7 environment while
designing this model. NuMax does not provide an `arr.view()` method. Slicing,
`block[...]`, or an explicit `TensorView(arr.tile(), arr.context())` produces a
view, and those constructors require a mutable owner borrow.

Its `TensorLike.tile(ref self)` returns a `TileTensor` whose origin and
mutability are `origin_of(self)` for both the owner and view. `TensorView`
stores its owner's origin as `ImmOrigin(owner_origin)`, but internally performs
an unsafe mutability cast so that a mutable binding of the view yields a
writable tile. Consequently:

- an immutable owner can produce a read-only tile but cannot produce a
  `TensorView`;
- `var view = owner[...]` is writable and mutations reach the owner;
- a generic `def read[T: TensorLike](value: T)` can read owners and views;
- a generic function needs `mut value: T` to mutate either directly; and
- two views may alias because their stored owner origins are immutable.

The audit also found that binding-based mutability is not a complete read-only
guarantee. `TensorView` is `ImplicitlyCopyable`: an immutable view argument can
be copied into a local `var`, after which `tile()` or `copy_from_host()` becomes
writable and can mutate the original owner. The compiler correctly rejects a
direct mutating call on the immutable argument, but the copied-alias program
compiled and changed the parent tensor. NuMojo must therefore retain mutability
in the view type and must not recover write access merely because a
read-only view was copied into a mutable local binding.

## Future MAX TileTensor backend

`TileTensor` should be added as an execution and interoperability layer, not as
a replacement for `NDLayout`.

`TileTensor` is well suited to:

- flattened contiguous elementwise operations;
- fixed-rank kernels with runtime extents;
- shared CPU/GPU SIMD implementations;
- MAX `elementwise` launches; and
- zero-copy interoperation with APIs that already consume `TileTensor`.

Its layout rank is part of its compile-time type, while `AcceleratorNDArray`
has runtime rank. A single `TileTensor` type therefore cannot represent every
arbitrary-rank `NDLayout`. Arbitrary runtime-rank striding and broadcasting
must continue to use packed shape/stride metadata and custom offset kernels,
unless the input is materialized or dispatched to a rank-specialized path.

The planned hybrid is:

1. Convert contiguous array-like values to a flattened rank-one `TileTensor` and use it
   for elementwise CPU/GPU kernels (Needs testing).
2. Retain metadata-driven kernels for arbitrary runtime-rank strided and
   broadcast views.
3. Consider rank-specialized dispatch for common ranks only after measuring
   compile-time/code-size costs.
4. Add explicit zero-copy adapters under a future `numojo.max` interoperability
   module, for example contiguous flat and matrix tile adapters.

Tile adapters must follow the same pointer rule: obtain the source pointer by
calling `unsafe_ptr()`, construct the tile with that pointer's origin, and never
read an owner's or view's pointer field directly.
