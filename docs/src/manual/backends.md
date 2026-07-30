```@meta
CurrentModule = MatrixProductToolbox
```

# [Array backends](@id man-backends)

Site tensors are plain `AbstractArray`s, and the package contains **no backend-specific code at
all** — no `[weakdeps]`, no `[extensions]`, no configuration hooks. The dependencies are
`TensorOperations`, `LinearAlgebra`, `Random` and `Adapt`, and nothing else.

That works because every operation the package performs is either generic Julia (`reshape`,
`similar`, `*`, broadcast, `sum`), `LinearAlgebra.svd`, or a TensorOperations `@tensor` call — and
the ecosystem already supplies the per-backend pieces:

| Backend | Who provides the support |
|:---|:---|
| `Array` | Julia and TensorOperations' Strided/BLAS backends |
| `CuArray` | TensorOperations' CUDA extension, plus a cuTENSOR backend |
| Reactant | Reactant's own `ReactantTensorOperationsExt` |

## Moving between backends

One method, `Adapt.adapt_structure`, makes objects portable:

```julia
using Adapt

ψ_cpu = rand_mps(ComplexF64, fill(2, 10); maxdim = 32)

# CUDA
using CUDA
ψ_gpu = adapt(CuArray, ψ_cpu)          # or cu(ψ_cpu)
ψ_f32 = adapt(CuArray{ComplexF32}, ψ_cpu)

# Reactant
using Reactant
ψ_xla = Reactant.to_rarray(ψ_cpu)
```

Structural metadata — the orthogonality centre, bond dimensions, physical dimensions — is
host-side data and survives the move unchanged. Everything else behaves as on the CPU:

```julia
expect(adapt(CuArray, H), ψ_gpu)       # same call, same result
```

## Two idioms worth knowing

If you write algorithms on top of this package, follow these; they are why the source contains no
`CUDA.@allowscalar`:

- Reduce a `1×1` environment with `sum(E)`, never `E[1]`. Scalar `getindex` on a device array is
  either an error or a hidden host round-trip.
- Build boundary tensors with `fill!(similar(t, T, dims), one(T))`. All-ones is correct because
  boundary bonds have dimension 1.

## Reactant and XLA

The motivation is fusion: an MPS sweep is a long stream of small contractions, which is exactly
what XLA is good at collapsing into a few kernels.

```julia
using MatrixProductToolbox, Reactant, LinearAlgebra, Random

L, χ = 20, 16
H_cpu = tfim_mpo(ComplexF64, L)                                    # see the previous page
ψ_cpu = rand_mps(ComplexF64, fill(2, L); maxdim = χ)

H, ψ = Reactant.to_rarray(H_cpu), Reactant.to_rarray(ψ_cpu)

# Compile ONE whole sweep, not individual contractions -- fusion is the entire point.
sweep(H, ψ) = expect(H, canonicalize(ψ, L; trunc = FixedTruncation(χ)))

compiled = @compile sweep(H, ψ)
compiled(H, ψ)
```

Note the two deliberate choices: `FixedTruncation` rather than a cutoff, and one compiled
function covering the whole sweep.

### Seven rules

1. **Every `!` function returns its argument.** Returning is what reliably propagates updated
   traced values out of a compiled region; do not rely on argument mutation alone.
2. **Run inside `@compile` / `@jit`.** Reactant's TensorOperations `select_backend` methods are
   `@reactant_overlay`'d, so they only fire under the overlay interpreter. Calling `@tensor` on a
   `ConcreteRArray` at top level falls back to the Strided scalar path and fails.
3. **Compile one whole sweep**, not individual contractions.
4. **Use [`FixedTruncation`](@ref)** inside a compiled region. A cutoff policy picks its rank by
   comparing singular values, which makes the output shape a runtime value — something XLA cannot
   express. If you forget, [`Truncation`](@ref) detects it up front and raises an error that says
   so, rather than failing somewhere deep in the tracer. See
   [Canonical forms and truncation](@ref man-canonical).
5. **Plain `for` loops unroll at trace time.** Fine for modest `L`; reach for Reactant's
   `@trace for` when compile time becomes the bottleneck.
6. **Scalars are frozen** at trace time unless passed via `to_rarray(…; track_numbers = true)`. A
   bond dimension or tolerance given as a bare `Float64` becomes a compile-time constant.
7. **Compiled thunks do not validate shapes.** A Reactant array's shape is a struct field, not a
   type parameter, so two differently shaped arrays have the same Julia type. Cache thunks keyed
   on shapes yourself.

### Verifying that it fused

`@code_hlo` is the tool. Transposes should have been folded into `dot_general`, and there should
be no leftover `stablehlo.while` where you expected batching:

```julia
@code_hlo sweep(H, ψ)
```

!!! note "An honest caveat"
    Reactant's extension routes `@tensor` to TensorOperations' `BaseCopy` backend —
    `permutedims` into temporaries, then `reshape` and `mul!` — rather than emitting
    `Ops.dot_general` directly. XLA's algebraic simplifier should fold those transposes into the
    dot products, but that is an expectation, not a measurement. Benchmark before relying on it.
    If it underperforms, the fix belongs upstream in Reactant: this package's six contraction
    kernels would not change.

## Testing across backends

The test suite runs one shared list of probes — scalar-valued functions of `(H, ψ, φ)` — on every
backend. Adding a backend means supplying an adaptor and a runner, never rewriting a check:

```julia
backend_testset("Array",          identity,               (f, a, b, c) -> f(a, b, c))
backend_testset("CuArray",        ψ -> adapt(CuArray, ψ), (f, a, b, c) -> f(a, b, c))
backend_testset("ConcreteRArray", Reactant.to_rarray,     jit_run)
```

CUDA and Reactant live in sub-environments under `test/`, so neither enters this package's
dependency graph:

```bash
GROUP=Core      make test    # CPU
GROUP=CUDA      make test    # needs a GPU
GROUP=Reactant  make test    # runs on CPU, so it works in CI
```
