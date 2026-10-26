# Workspace API source provenance

This package is private-gemm-x86 0.1.22 from crates.io, upstream commit
`71872b9a98e6a64ef3c0c3b46da02cc2b19d2937` in
https://github.com/sarah-quinones/gemm-x64-v2. Its MIT license is retained.

It is included as a path dependency so the faer fork is independently usable
while the backend API can be proposed separately to its upstream repository.
No generated assembly kernels, packing routines or numerical algorithms change.

The changes add a fallibly allocated caller-owned `Workspace`, scope its use
through a non-dropping thread-local pointer, and factor the existing GEMM
driver to accept that buffer instead of initializing its owning TLS cache.
Scopes restore their predecessor on normal return or unwind. The legacy API
outside a scope retains its original allocation contract. Cache initialization
has an allocation-free CPUID/default path for workspace construction; ordinary
cache discovery remains available when no workspace has initialized it first.

Only sequential calls are covered by the allocation guarantee. The scope does
not initialize a thread pool or propagate to other threads. Kernel memory is
exclusively borrowed while the scope executes; native kernels call no user code.
