# MatrixProductToolbox.jl

Matrix-product states and operators for closed and open quantum systems. The design goals, in
order: **simplicity** (few concepts, a tiny kernel set), **flexibility** (every policy is a
type you can extend), and **platform agnosticism** (one code path for `Array`, `CuArray` and
Reactant/XLA).

Current status: the foundation layer. DMRG and TDVP are **not** implemented — see the roadmap
at the end. The conventions and kernels they need are fixed and tested.

## Standing instruction

**Design independently.** Do not consult the implementations of ITensors.jl, MPSKit.jl or
Tenet.jl when adding features here. The conventions below were derived from scratch and should
stay internally consistent rather than drift toward another package's choices.

## Index conventions — read this first

Getting these wrong is the single most expensive mistake available, so they are stated once and
never varied.

### Site tensors: `A[l, p₁, …, p_N, r]`

Left bond first, physical legs in the middle, right bond last.

| | Index order | Meaning |
|:---|:---|:---|
| `MPS` = `MatrixProduct{1}` | `A[a, s, a′]` | left bond, physical, right bond |
| `MPO` = `MatrixProduct{2}` | `W[w, s′, s, w′]` | left bond, **output** physical, **input** physical, right bond |

`W[w, s′, s, w′] = ⟨s′|Ŵ|s⟩`: the bra contracts with `s′`, the ket with `s`.

**Boundary bonds keep dimension 1** rather than being dropped. Every site tensor therefore has
exactly `N + 2` indices and all generic code is shape-agnostic — no rank-2 special cases.

This ordering is not arbitrary. It makes **both** canonicalization reshapes contiguous, so
left- and right-orthogonalization need *zero* `permutedims`:

- left-orthogonalize: `reshape(A, l * ∏p, r)`
- right-orthogonalize: `reshape(A, l, ∏p * r)`

It also lets an `N`-leg object reduce to the `N = 1` kernels for overlaps by fusing the
physical legs (`_fuse_physical`), which is why six kernels cover every `N`.

### Environments: `L[b, w, a]` / `R[b, w, a]`

`b` = bra bond, `w` = MPO bond, `a` = ket bond — top-to-bottom in the standard diagram.
Overlap environments (no MPO) are `L[b, a]`.

### `dense`: site 1 is the fastest-varying index

So `reshape(dense(ψ), physical_dims(ψ)...)` has dimension `i` equal to site `i`. This is the
permute-free choice, and it is the **opposite** of `kron`'s convention: a reference operator
must be built as `reduce(kron, reverse(ops))` (see `kron_ops` in `test/testutils.jl`).

## The gauge invariant: `center::UnitRange{Int}`

`center` is **the range of sites not known to be orthogonal**. Sites
`1:first(center)-1` are left-orthogonal; sites `last(center)+1:end` are right-orthogonal.

| Value | Meaning |
|:---|:---|
| `1:L` | nothing claimed — a freshly constructed object |
| `c:c` | mixed-canonical, centre at site `c` |
| `1:1` / `L:L` | right- / left-canonical |
| `i:i+1` | a two-site block |

**There is no sentinel.** Every value is meaningful, which is why `canonicalize!` has a single
code path with no "is it gauged yet?" branch:

```julia
lo, hi = first(center(ψ)), last(center(ψ))
for i in lo:(c - 1);    _orthogonalize_left!(ψ, i, …);  end
for i in hi:-1:(c + 1); _orthogonalize_right!(ψ, i, …); end
setcenter!(ψ, c:c)
```

For a fresh `1:L` this *is* the full sweep. Do not reintroduce a nullable centre.

Alternative gauges (Vidal/bond-canonical) carry extra data — the Λ tensors — so they belong in
their own `AbstractMatrixProduct` subtype, not as another flag here.

## Architecture

Ten source files, each mapping to one concept.

| File | Role |
|:---|:---|
| `src/abstract.jl` | `AbstractMatrixProduct{N}`, the five-method subtype interface, generic queries |
| `src/matrixproduct.jl` | the container, `MPS`/`MPO` aliases, `Base` and `Adapt` methods |
| `src/kernels.jl` | **the entire compute core** — six contractions |
| `src/factorizations.jl` | truncation policies and the SVD / QR factorizations |
| `src/canonical.jl` | `canonicalize!`, `truncate!` |
| `src/states.jl` | `rand_mps`, `product_mps`, `product_mpo`, `identity_mpo` |
| `src/observables.jl` | `dot`, `norm`, `expect`, entanglement |
| `src/apply.jl` | `apply(H, ψ)`, scalar arithmetic |
| `src/dense.jl` | conversion to the full dense array |

`export` statements live at the top of the file where each symbol is defined, grouped — never
one central list.

### `src/kernels.jl` is the seam

Six contractions are the whole compute core; everything else is built from them.

```julia
transfer_left(L, A, B)              transfer_right(R, A, B)
transfer_left(L, A, W, B)           transfer_right(R, A, W, B)
apply_effective(L, W, x, R)         apply_effective(L, W₁, W₂, x, R)
```

If the contraction strategy ever changes, this is the only file to touch.

Two rules for this file:

- **Hand-stage every contraction** into pairwise `@tensor` steps. `@tensoropt` without extent
  hints assumes all dimensions are equal and can pick a bad order; the optimal staging here is
  standard and known.
- **Never repeat an index within one tensor.** That would reach TensorOperations'
  `tensortrace!`, whose `BaseCopy` implementation — the one Reactant routes to — unrolls a
  Julia loop over diagonal slices at trace time.

The two `apply_effective` kernels are unused by the current public API. They exist so that the
effective-operator conventions DMRG and TDVP need are fixed and tested now; `test/kernels.jl`
validates them against `expect`.

### Two idioms that keep the package device-safe

Follow these in any new code — they are why there is no `@allowscalar` anywhere:

- Reduce a `1×1` environment with **`sum(E)`**, never `E[1]`.
- Build boundary tensors with **`_boundary(t, T, dims)`** = `fill!(similar(t, T, dims), one(T))`.
  All-ones is correct because boundary bonds have dimension 1.

Also: never form a full dimension product with `prod`. `dense`'s size guard uses
`_saturating_prod`, because `prod(fill(2, 40))^2` overflows `Int64` to exactly `0` and a naive
guard passes silently, then tries to allocate a 2⁴⁰ × 2⁴⁰ array.

## Truncation: two policies, one distinction

XLA requires array shapes to be known at trace time. It does **not** require uniform bond
dimensions — only that they be a function of *shapes alone*, never of numerical values.

| Policy | Rank rule | Depends on | `Array` / `CuArray` | Reactant |
|:---|:---|:---|:---:|:---:|
| `NoTruncation()` | `k = min(m, n)` | shape | ✅ | ✅ |
| `FixedTruncation(χ)` | `k = min(χ, m, n)` | shape | ✅ | ✅ |
| `Truncation(; maxdim, cutoff)` | smallest `k` meeting `cutoff` | singular values | ✅ | ❌ |

**`FixedTruncation` is not "every bond equals χ".** Near the chain ends `min(m, n) < χ`, so
bonds there are smaller — but deterministically, from the geometry. Two consequences: no
zero-padding is ever needed, and isometries stay exact (`Q'Q == I`, not a projector).

The cutoff policy's rank comes from `count(>(cutoff * total), reverse(cumsum(reverse(abs2.(s)))))`
— one device-side reduction, no scalar indexing, no host copy of the spectrum. That `count`
returning a host `Int` is the *only* place the two policies diverge, and the reason the cutoff
policy cannot be traced.

### Gauge and truncation are separate concerns

- `canonicalize!(ψ, c)` — pure gauge, exact, no truncation.
- `truncate!(ψ; …)` — sweeps to site `L`, back to site 1 applying the policy at every bond, then
  moves the centre to `to`. The two passes are what make the truncation optimal.

## Backends: no platform dependencies, no hooks

`Project.toml` has **no `[weakdeps]` and no `[extensions]`**. Deps are `TensorOperations`,
`LinearAlgebra`, `Random`, `Adapt`. Keep it that way.

Every backend works because the package only ever uses generic operations (`reshape`,
`similar`, `*`, broadcast, `sum`, `svd`) plus `@tensor`, and the ecosystem supplies the rest:

- **CUDA**: TensorOperations' GPU contraction backend is in its **cuTENSOR** extension, so users
  must load `CUDA` *and* `cuTENSOR`. Its CUDA extension alone only provides device-aware
  allocation; without cuTENSOR, `CuArray <: DenseArray` makes `isstrided` true, `@tensor` picks the
  Strided CPU backend, and everything fails on `Scalar indexing is disallowed`. Note the version
  coupling: TensorOperations 5.6 needs cuTENSOR **v6**, not v2.
- **Reactant/XLA**: Reactant ships `ReactantTensorOperationsExt` (PR #2944, 2026-06-09;
  Reactant ≥ 0.2.275, `TensorOperations = "5.6"`). It routes to TensorOperations' `BaseCopy`
  backend — `permutedims` into temporaries, `reshape`, `mul!` — rather than to
  `Ops.dot_general`. This was measured head to head (`benchmarks/dot_general.jl`): **runtime is a
  wash** — XLA's simplifier really does fold the transposes into the dots — but going through
  `BaseCopy` emits twice the HLO transposes and compiles **~6× slower** (13.0 s vs 2.2 s). That is
  irrelevant for one-off runs and will matter for DMRG, which compiles per shape signature. **The
  fix belongs upstream in Reactant's extension, not here**; do not add a `dot_general` path to this
  package, since it would mean a Reactant dependency for no runtime gain.

`Adapt.adapt_structure` is the single method that makes objects portable; it gives `cu(ψ)` and
`Reactant.to_rarray(ψ)` for free.

### Two factorizations, split on truncation

`SVDFactorization` is the default and the only one that can discard anything.
`QRFactorization` is for exact gauge sweeps and is measured at 2.0× (χ=16) to 3.0× (χ=128)
faster on a 20-site chain. Neither needs a hook: the `factorization` keyword carries the
choice, and both work on every backend.

**The GPU numbers are why this exists.** A gauge sweep runs 5–50× *slower* on a GPU than on the
CPU (`benchmarks/cuda.jl`, RTX 4090) because it is a chain of small sequential SVDs — while the
contractions in the same sweep run up to 7× *faster*. A QR is far more GPU-friendly; batching
factorizations across bonds is the other half of the fix and is still to do. Note also that
accelerator wins only appear for `χ ≳ 64`; benchmarking at `χ ≤ 32` shows XLA *losing*, so size
the benchmark before concluding anything.

**Under Reactant the QR is not merely faster — it is the only one that works in `ComplexF64`.**
Reactant's complex SVD is silently wrong at `χ = 4` and segfaults from about `χ = 6`; the QR is a
different lowering (`geqrf` + `ungqr`, never `gesdd`) and is exact there — measured at `6e-16`
(`L = 10`, `χ = 4`) and `1e-15` (`χ = 8`) against the CPU. `truncate!` still cannot run complex
under Reactant at all, since only the SVD can discard anything.

Whether a factorization may be handed a truncating policy is
`_supports_truncation(alg)` — internal, defaulting to `true`. `truncate!` consults it **before
the first sweep**, because only its middle sweep truncates; a late check would leave `ψ` regauged
by the first one and then throw.

Four things that are easy to get wrong here, all of them load-bearing:

- **`qr(M).Q` has a different type per backend.** It is a *square* `m × m` `AbstractQ` operator
  on `Array`/`CuArray` — Julia 1.10 decoupled `AbstractQ` from `AbstractMatrix` — but an
  already-materialized thin `m × k` matrix under Reactant, whose factorization has no
  `AbstractQ` wrapper. `_thin_q` dispatches on exactly that: two methods, not one shape test,
  because for a square bond matrix `m == k` the two cases are indistinguishable by shape.
- **`Qf * _eye(M, k)` rather than `Matrix(F.Q)`**, which would materialize the square factor and
  move a `CuArray` to the host. The multiply dispatches to LAPACK `ormqr` (cuSOLVER on a GPU).
- **Bind `F.Q` once.** Under Reactant every `getproperty(F, :Q)` re-emits the `orgqr` call.
- **`factorize_right` is a `qr` of the adjoint**, not an `lq` — CUDA's `lq` coverage is
  unreliable. `M' ≈ Q*R` gives `M ≈ R'*Q'`, and `Q'` has the orthonormal rows wanted. The
  adjoints are materialized with `copy`: the input because a lazy wrapper is a poor thing to
  hand a tracing backend, the outputs so both methods return dense factors.

Reactant's `qr` is **not in a released version**: no released Reactant wraps
`lapack.geqrf`/`lapack.orgqr`, and its `enzymexla.linalg.qr` op is unwired. `test/reactant/`
pins <https://github.com/albertomercurio/Reactant.jl/pull/2> through a `[sources]` block in its
`Project.toml`; delete that block and raise the compat bound once it lands upstream. There are
also **no AD rules** for `geqrf`/`orgqr`, so a QR gauge sweep will not differentiate under
Enzyme — worth knowing before DMRG/TDVP lean on it.

`CholeskyQRFactorization` remains viable on backends with `cholesky` but no `qr`. A
`QRSVDFactorization` (QR, then SVD the small `R`) would be able to truncate, but LAPACK's
`gesdd` already does that preprocessing for `m ≥ 5n/3` — and `m/n = d = 2` — so measure before
building it.

### Seven Reactant rules

1. Every `!` function **returns** the mutated object. Returning is what reliably propagates
   updated traced values out of a compiled region.
2. Reactant paths must run **inside** `@compile`/`@jit`. Its `select_backend` methods are
   `@reactant_overlay`'d and fire only under the overlay interpreter; outside, `@tensor` falls
   back to the Strided scalar path and errors.
3. Compile **one whole sweep**, not individual contractions. Fusion is the entire point.
4. Use `FixedTruncation` inside a compiled region.
5. Plain Julia `for` loops unroll at trace time. Fine for modest `L`; use `@trace for` when
   compile time becomes the bottleneck.
6. Runtime-varying scalars need `to_rarray(…; track_numbers = true)`, or they are frozen as
   compile-time constants.
7. Compiled thunks do **not** validate shapes (shape is a struct field, not a type parameter),
   so a shape-keyed cache is our responsibility when DMRG lands.

### Two traps that cost real debugging time

Both were found by running `GROUP=Reactant`, and both are invisible on the CPU:

- **A traced number is `<: Number` but *not* `<: Real`.** Never constrain a type parameter or an
  argument to `Real` on anything that can hold a *computed* scalar — it makes the method
  uncallable under tracing. (A `Real` bound on a policy *field* like `Truncation.cutoff` is fine:
  that is always a host value.) The same goes for `eltype`:
  `eltype(TracedRArray{Float64,2})` is `TracedRNumber{Float64}`, not `Float64`.
- **Build tensors outside the compiled region and pass them in.** An array constructed *inside* a
  traced function comes out as `Array{TracedRNumber{T},N}` — an array of traced scalars — not a
  `TracedRArray{T,N}`, and TensorOperations rejects the mix with `incompatible types for
  tensorcontract!`. So `Reactant.to_rarray(H)` on the outside, never `identity_mpo(...)` on the
  inside.
- **Never branch on a value, only on a shape.** `if total > 0` compiles to a traced boolean and
  cannot drive Julia control flow. `if k ≥ length(s)` is fine, because `k` and `length(s)` come
  from shapes. `_discarded_weight` is written this way deliberately; the comment there explains
  the trade-off it accepts.
- **`count` on a traced array can exhaust the compiler.** Not a clean error — LLVM ran out of
  memory even with a 24 GB cap. This is why `_truncation_rank(::Truncation, …)` checks that a
  comparison yields a host `Bool` *before* reaching its reduction, rather than validating the
  result afterwards: a late check never gets to run. When guarding an operation that a deferred
  backend cannot express, guard it **before** the expensive step, and test the guard by asserting
  on the error message rather than merely that something was thrown.

## Conventions for new code

- **Argument order is operator-first**: `expect(H, ψ)`, `apply(H, ψ)`, `H * ψ`. Matches
  QuTiP/QuantumToolbox. `dot(ψ, φ)` keeps LinearAlgebra's order, since it conjugates its first
  argument.
- **Complex conjugation is written `X^*`**, not with an overline, in both LaTeX and prose. The
  adjoint is `†`.
- **Keep docstrings short.** Signature, one-or-two-line summary, `# Arguments`, `# Returns`. Tables,
  derivations and long rationale belong in `docs/src/manual/`, not in a docstring — the docstring
  should link there instead. Abstract types get no `# Examples`: an example of an abstract type is
  necessarily an example of some concrete subtype, so it belongs on that subtype.
- **No speculative return values.** `factorize_left`/`factorize_right` return just `(Q, R)` /
  `(L, Q)`; the kept bond dimension is `size(Q, 2)`. An earlier version returned a
  `TruncationInfo` carrying the discarded weight, which nothing consumed while costing a reduction
  on every bond of every sweep. Per-sweep truncation-error reporting belongs with DMRG, which is
  the first thing that actually needs it.
- **Expectation values are unnormalised**: `expect` returns `⟨ψ|Ô|ψ⟩` with no division by
  `⟨ψ|ψ⟩`. Documented, and tested.
- **Mutation-first with copying wrappers**: `canonicalize!` is the primary implementation;
  `canonicalize(ψ, …) = canonicalize!(copy(ψ), …)`. Tensor-level work stays non-mutating —
  factorizations return new arrays — so mutation happens only at the container level.
- **`setindex!` resets the gauge** to `1:L`, because an arbitrary new tensor destroys
  orthogonality. Internal code that knows better uses `setsite!`.
- **Docstrings use `@doc raw"""`** (so LaTeX needs no escaping), with sections in the order
  `# Arguments` → `# Notes` → `# Returns` → `# Examples` → `# References`. Arguments are a
  bullet list with the name in backticks and no repeated type — types are already in the
  signature block. Structs use `# Fields (Attributes)`.
- **Explicit `return`** on the last line of every `function … end`.
- **Formatting is Runic's** (`make format`); CI enforces it.

## Working on this repo

```bash
make test                 # Core group: contractions, gauge, observables, CPU probes
GROUP=Core        make test
GROUP=Code-Quality make test   # Aqua + JET; memory hungry, keep it in its own process
GROUP=CUDA        make test    # test/cuda/,     own sub-environment, needs a GPU
GROUP=Reactant    make test    # test/reactant/, own sub-environment, runs on CPU
make format               # Runic
make docs                 # build documentation with doctests
```

**Run groups separately.** `Pkg.test()` with no `GROUP` runs everything including JET, which is
by far the heaviest part.

### Test structure

`test/testutils.jl` holds the shared fixtures. The important one is `BACKEND_PROBES`: a list of
**scalar-valued functions of `(H, ψ, φ)`**. Fixed arity and a scalar result are what let one
test body serve every backend — a backend supplies only an adaptor and a runner:

```julia
backend_testset("Array",          identity,                  (f, a, b, c) -> f(a, b, c))
backend_testset("CuArray",        ψ -> adapt(CuArray, ψ),    (f, a, b, c) -> f(a, b, c))
backend_testset("ConcreteRArray", Reactant.to_rarray,        jit_run)
```

Add a backend by adding those two functions. **Never** by rewriting a check.

`src/dense.jl` is the linchpin of correctness: with `dense(ψ)` and `dense(H)`, every observable
is checked against explicit dense linear algebra on a small chain, which leaves nowhere for an
index or conjugation error to hide.

CUDA and Reactant live in sub-environments under `test/` precisely so they never enter this
package's dependency graph. Reactant runs on CPU, so unlike CUDA it is usable in CI.

## Roadmap

Deliberately deferred, in rough order:

1. **DMRG** — needs an environment cache (`leftenv`/`rightenv` with validity bookkeeping) and an
   `EffectiveOperator` wrapper around the `apply_effective` kernels, exposed as a callable
   linear map so any Krylov solver consumes it. Reactant has no `eigen`, so the local
   eigensolver must be a hand-written Lanczos over matvecs.
2. **TDVP** — same environments, exponentiation instead of diagonalization.
3. **General Hamiltonian-MPO builders** from sums of local terms. Until then, build MPOs by hand
   — see `tfim_mpo` in `test/testutils.jl` and the documentation.
4. MPS addition / direct sum; needed by variational and Krylov methods.
5. **Clifford-Augmented MPS (CAMPS)**: `struct CliffordMPS <: AbstractMatrixProduct{1}` holding a
   tableau plus an `MPS`. This is why `src/abstract.jl` documents a five-method interface and
   why every algorithm is written against it rather than against the concrete type.
6. Infinite MPS; shape-keyed thunk caching for Reactant.
