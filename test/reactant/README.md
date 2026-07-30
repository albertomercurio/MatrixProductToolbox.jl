# Reactant / XLA tests

These tests live in their own environment on purpose: **Reactant must never become a
dependency of `MatrixProductToolbox`**, not even a weak one. The package is generic over
`AbstractArray` and needs no knowledge of Reactant — the `@tensor` calls work on traced arrays
because *Reactant itself* ships `ReactantTensorOperationsExt`.

Keeping Reactant here also keeps it out of the default CI matrix, where its large binary
artifacts would dominate the runtime.

## Running

```bash
GROUP=Reactant julia --project=. -e 'using Pkg; Pkg.test()'
```

or directly:

```bash
julia --project=test/reactant -e '
    using Pkg
    Pkg.develop(PackageSpec(path = pwd()))
    Pkg.instantiate()
    include("test/reactant/runtests.jl")'
```

## What is being verified

1. `@tensor` contractions run on Reactant arrays inside `@compile`, and the results match the
   CPU answer bit-for-bit-ish (to floating-point tolerance).
2. A gauge sweep with [`FixedTruncation`](../../src/factorizations.jl) traces successfully —
   this is the load-bearing claim of the design, since bond dimensions must be known at trace
   time.
3. `Truncation` (the cutoff policy) is *expected to fail* under tracing, because the rank it
   chooses is a runtime value. That failure is asserted, so the documented boundary between
   the two policies stays honest.

## Two rules that are easy to get wrong

- **Everything must run inside `@compile` / `@jit`.** Reactant's `select_backend` methods are
  `@reactant_overlay`'d, so they only fire under the overlay interpreter. Calling `@tensor` on
  a `ConcreteRArray` at top level falls back to the Strided scalar path and fails.
- **Compiled functions must return the mutated object.** Returning is what reliably propagates
  updated traced values out of a compiled region.
