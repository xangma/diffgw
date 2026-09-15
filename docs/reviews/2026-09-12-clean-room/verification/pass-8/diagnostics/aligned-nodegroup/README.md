# Charm++ over-aligned nodegroup allocator probe

This is a minimal before/after probe for the p8 failure mode. `AlignedNodeGroup`
contains an `alignas(64)` member, forcing `alignof(AlignedNodeGroup) >= 64`.
The constructor prints the address and asserts `this % 64 == 0`; a runtime that
uses plain `malloc` for nodegroups should fail the assertion on an
under-aligned allocation. The member write also gives the compiler an aligned
object to use when optimizing the constructor. The constructor queues a
completion entry on `Main`, and `Main` exits only from that callback, after
construction has completed.

Build with the same Charm++ compiler/runtime used by the target binary, for
example:

```sh
charmc aligned-nodegroup.ci
charmc -O2 -o aligned-nodegroup aligned-nodegroup.cpp
./aligned-nodegroup +p8
```

(The generated `.decl.h` and `.def.h` headers are build outputs and are not
checked in.) The probe checks 16 allocations and waits for their completion
callbacks before exiting. Run before and after the allocator change. The allocator fix must
preserve the runtime's existing destruction/free path: use an aligned allocation
API with its matching deallocation, or retain the original pointer if an
allocation wrapper is needed. Do not unconditionally call `free(this)` from the
chare.
