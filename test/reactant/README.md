# Reactant / XLA tests

These tests live in their own environment on purpose: **Reactant must never become a
dependency of `MatrixProductToolbox`**, not even a weak one. The package is generic over
`AbstractArray` and needs no knowledge of Reactant — the `@tensor` calls work on traced arrays
because *Reactant itself* ships `ReactantTensorOperationsExt`.

Keeping Reactant here also keeps it out of the default CI matrix, where its large binary
artifacts would dominate the runtime.

## ⚠️ This environment currently pins a fork

`QRFactorization` calls `LinearAlgebra.qr`, and **no released Reactant can trace it**: upstream
wraps neither `lapack.geqrf` nor `lapack.orgqr`, and its `enzymexla.linalg.qr` op is not wired
up — its lowering forwards the QR result types straight into `geqrf`'s, so it only verifies when
`Q` and `R` have the same type, and the "R" it yields is the packed matrix rather than a
triangular one. The branch pinned in `Project.toml` wraps the two LAPACK ops directly instead:

<https://github.com/albertomercurio/Reactant.jl/pull/2>

**To undo once that is merged and released:** delete the `[sources]` section from
`Project.toml`, raise the `Reactant` compat bound to the first release carrying it, and
regenerate `Manifest.toml`. Nothing outside this directory refers to the fork — the package has
no Reactant dependency, weak or otherwise, so only this test group is affected.

Two limits of that branch worth knowing while it is pinned:

- Batched (`ndims > 2`) `geqrf`/`orgqr` work on CUDA and TPU but not on CPU. This package never
  hits it — every factorization is of a rank-2 reshape.
- There are no autodiff rules for either op, so a QR gauge sweep will not differentiate under
  Enzyme.

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
4. A gauge sweep with [`QRFactorization`](../../src/factorizations.jl) traces, agrees with both
   the CPU answer and the SVD-gauged one, and reaches `geqrf`/`orgqr` in the emitted HLO. That
   last check matters because `LinearAlgebra.qr` is intercepted by a `@reactant_overlay`: the
   failure to guard against is a silent fall-back that happens to produce the right numbers.
5. `QRFactorization` with a *binding* `FixedTruncation` is expected to fail — a QR keeps the
   full rank. Unlike (3) this is not an XLA limitation: the check compares shapes, so it fires
   the same way on every backend, and the assertion is that it still fires under tracing rather
   than dissolving into a traced boolean.
6. A **`ComplexF64`** gauge sweep with `QRFactorization` matches the CPU answer at `L = 10`,
   `χ = 8`. That size is chosen deliberately: it is past the point where Reactant's complex SVD
   segfaults, so the test would not pass if the QR shared that bug. It is what makes the QR the
   only way to gauge a complex state under Reactant today.

## Two rules that are easy to get wrong

- **Everything must run inside `@compile` / `@jit`.** Reactant's `select_backend` methods are
  `@reactant_overlay`'d, so they only fire under the overlay interpreter. Calling `@tensor` on
  a `ConcreteRArray` at top level falls back to the Strided scalar path and fails.
- **Compiled functions must return the mutated object.** Returning is what reliably propagates
  updated traced values out of a compiled region.
