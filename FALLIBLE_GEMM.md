# Fallible GEMM workspace branch

Based on faer 0.24.4 at `0ecd846dc223752fd41e3ea95253b52532156c01`.
This branch adds an opt-in caller-owned packing workspace while preserving the
optimized native kernels and existing APIs outside the workspace scope.

`faer::linalg::matmul::GemmWorkspace::try_new()` returns an allocation error
instead of aborting. `workspace.with(|| operation())` supplies its storage to
sequential native GEMM calls, including the dense, triangular and indexed
dispatch sites used by decompositions. Reuse the workspace for subsequent
operations, then drop it normally. Outputs and decomposition scratch remain
the caller's responsibility and should also be allocated fallibly.

The backend's previous owning thread-local Vec is not initialized in this
scope. A non-dropping TLS pointer borrows the caller's workspace for the
duration of the closure. Nested scopes and unwinding restore the prior pointer.
Cache discovery uses an allocation-free CPUID/default path. On machines without
the optimized x86 backend, a scoped operation uses nano-gemm rather than the
legacy `gemm` dependency's hidden allocations. Outside the scope, dispatch is
unchanged. This guarantee covers native f32/f64/c32/c64 and `Par::Seq`, not
parallel executors, other threads, owned matrix constructors, or high-level
decompositions that allocate their own outputs/scratch.

The actual packing allocation belongs to private-gemm-x86. Its upstream
0.1.22 source is included as a path dependency so this branch is usable on its
own. The import is a separate commit; the workspace change can be reviewed
against it or proposed to that backend upstream separately. See
[source provenance](private-gemm-x86/FORK.md). The assembly kernels, packing
routines and numerical algorithms are unchanged. No upstream PR is created
by this branch.

`faer/tests/gemm_workspace.rs` denies the global allocator on a fresh thread,
checks recoverable cold allocation failure, then runs all four native scalar
types with allocation denied. It also exercises nested-scope unwinding. Plenty's
consumer tests additionally cover all three dispatch paths via blocked
decompositions, strided storage, cold and concurrent calls, and independent
residual calculations. A non-AVX QEMU run checks the portable scope path.

This branch contains allocation-contract changes, not a new kernel-tuning
algorithm. Workspace construction has a real cost; callers that perform many
small operations should reuse the workspace. Plenty implements bounded,
fallible reuse separately in its engine.
