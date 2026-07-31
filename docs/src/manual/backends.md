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

| Backend | Who provides the support | You must load |
|:---|:---|:---|
| `Array` | Julia and TensorOperations' Strided/BLAS backends | — |
| `CuArray` | TensorOperations' cuTENSOR extension | `CUDA` **and `cuTENSOR`** |
| Reactant | Reactant's own `ReactantTensorOperationsExt` | `Reactant` |

!!! warning "`using cuTENSOR` is required on the GPU"
    TensorOperations' actual GPU contraction backend lives in its **cuTENSOR** extension; its CUDA
    extension only supplies device-aware allocation. With `CUDA` alone, `CuArray <: DenseArray`
    makes TensorOperations' `isstrided` check return `true`, `@tensor` selects the Strided *CPU*
    backend, and every contraction fails with `Scalar indexing is disallowed`.

    ```julia
    using CUDA, cuTENSOR    # both, always
    ```

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

## Current status of the XLA path

Everything below is measured, not assumed: Reactant v0.2.275, CPU backend, via
`benchmarks/reactant.jl` in the repository. Please re-measure on your own hardware and version
before drawing conclusions.

### Correctness

| Workload | Status |
|:---|:---|
| Contractions — `dot`, `expect`, `apply`, the transfer kernels | ✅ exact to machine precision at `L = 60`, `χ = 32` |
| Gauge and truncation in `Float64` | ✅ exact at `L = 40`, `χ = 32` |
| Gauge and truncation in `ComplexF64` | ❌ **broken upstream** |

The one broken cell is Reactant's **complex** SVD on the CPU backend. It is silently wrong at
small bond dimensions (relative error ≈ 0.26 at `L = 10`, `χ = 4`, with LAPACK `ZUNMBR` errors on
stderr) and segfaults from about `χ = 6` upwards. The real SVD is exact.

That is an upstream limitation, not a design problem here — nothing in the six contraction kernels
changes when it is fixed. But it does mean that today, **`canonicalize` and `truncate` under
Reactant are only usable in `Float64`.**

!!! warning "Check your results against the CPU"
    The complex failure mode is silent at small sizes. Always compare a newly compiled function
    against the plain-Julia answer before trusting it.

### Performance

Measured on an NVIDIA RTX 4090 (CUDA 13.0) against a single-threaded CPU baseline, via
`benchmarks/cuda.jl` and `benchmarks/reactant.jl`. Speedup is relative to CPU; **bold** is a win.

**Contractions** (`expect` — the transfer-kernel sweep, pure `@tensor`):

| Accelerator | `L` | `χ` | CPU | GPU | speedup |
|:---|---:|---:|---:|---:|---:|
| CUDA + cuTENSOR | 40 | 32 | 2.3 ms | 8.9 ms | 0.26× |
| CUDA + cuTENSOR | 60 | 64 | 15.9 ms | 13.6 ms | 1.17× |
| CUDA + cuTENSOR | 60 | 128 | 104 ms | 14.9 ms | **6.96×** |
| Reactant (XLA) | 60 | 64 | 12.1 ms | 4.4 ms | **2.72×** |
| Reactant (XLA) | 60 | 128 | 59.5 ms | 11.7 ms | **5.08×** |

**Everything is a function of bond dimension.** Below `χ ≈ 32` the tensors are too small for
either accelerator to beat BLAS, and the picture inverts. Reactant on the *CPU* backend, same
workload at `L = 60`:

| `χ` | plain Julia | XLA | speedup |
|---:|---:|---:|---:|
| 16 | 0.90 ms | 1.59 ms | 0.56× |
| 32 | 4.19 ms | 4.63 ms | 0.91× |
| 64 | 15.0 ms | 13.0 ms | **1.15×** |
| 128 | 50.5 ms | 49.6 ms | 1.02× |

So do not benchmark at toy sizes and conclude anything: XLA loses badly at `χ = 16` and wins at
`χ = 64`.

**Gauge sweeps** (`canonicalize` + `expect` — adds one SVD per bond):

| Accelerator | `L` | `χ` | CPU | GPU | speedup |
|:---|---:|---:|---:|---:|---:|
| CUDA + cuTENSOR | 40 | 32 | 8.6 ms | 349 ms | 0.02× |
| CUDA + cuTENSOR | 60 | 64 | 76 ms | 1341 ms | 0.06× |
| CUDA + cuTENSOR | 60 | 128 | 342 ms | 3298 ms | 0.10× |
| Reactant (XLA) | 60 | 64 | 32 ms | 413 ms | 0.08× |
| Reactant (XLA) | 60 | 128 | 206 ms | 990 ms | 0.21× |

Three conclusions:

1. **Contractions pay off at large bond dimension** — up to 7× on the GPU, and even ~1.15× on the
   CPU backend at `χ = 64`. Below `χ ≈ 32` neither accelerator is worth it.
2. **XLA fusion is real.** Reactant beats even cuTENSOR on the contraction workload at `χ = 64`
   (4.4 ms vs 13.6 ms), which is exactly the effect one hopes for from compiling a whole sweep
   into fused kernels. The cost is 13–20 s of compilation per shape signature.
3. **The SVD is the bottleneck, badly.** Both accelerators are *5–50× slower* than the CPU on a
   gauge sweep. A canonicalization is a long chain of small, sequential, data-dependent
   factorizations — close to the worst case for a GPU. This is the single strongest argument for
   adding a `QRFactorization` (far more GPU-friendly than an SVD) and for batching the
   factorizations across bonds.

### Is `BaseCopy` costing us anything?

Reactant's TensorOperations extension routes `@tensor` to TensorOperations' `BaseCopy` backend:
`permutedims` into materialized temporaries, then `reshape` and `mul!`. Reactant-aware tensor
libraries instead emit one `Ops.dot_general` per contraction, with explicit contracting and
batching dimensions and no input permutation at all.

`benchmarks/dot_general.jl` implements the MPO transfer kernel both ways and measures them
head to head. The answer is reassuring:

| | runtime | HLO transposes | compile |
|:---|---:|---:|---:|
| via `@tensor` (BaseCopy) | 4.44 ms | 237 | 13.0 s |
| via `Ops.dot_general` | 4.43 ms | 119 | 2.2 s |

(GPU, `L = 60`, `χ = 64`; the CPU backend shows at most a 1.2× runtime difference.)

**Runtime is a wash** — XLA's algebraic simplifier really does fold `BaseCopy`'s transposes into
the dot products, so going through TensorOperations costs essentially nothing at execution time.
What it does cost is **compile time: about 6×**, from twice as many transposes to simplify away.
That is invisible for a one-off calculation and will matter for DMRG, which compiles a new thunk
per shape signature. The fix belongs upstream in Reactant's extension — emitting `dot_general`
instead of selecting `BaseCopy` — which is why this package has no `dot_general` code path of its
own.

If you are reaching for an accelerator, the summary is: worth it for contraction-heavy work at
`χ ≳ 64`, not yet worth it for gauge sweeps.

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
