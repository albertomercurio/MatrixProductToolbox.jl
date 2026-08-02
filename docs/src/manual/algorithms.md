```@meta
CurrentModule = MatrixProductToolbox
```

# [Algorithms](@id man-algorithms)

Two sweeping algorithms are implemented, both in their **two-site** form: [`dmrg`](@ref) for
ground states and [`tdvp`](@ref) for time evolution. They share every ingredient — the
[environment cache](@ref man-environments), the [`apply_effective`](@ref) kernels, the two-site
merge and split, the truncation policies — and differ only in what they do to the block once it
has been assembled: DMRG takes its lowest eigenvector, TDVP exponentiates it.

## Why two-site

A one-site sweep cannot change a bond dimension at all: it replaces one tensor by another of
exactly the same shape. Starting from a product state it would stay at ``\chi = 1`` forever, and
starting from a poor guess it would stay trapped in whatever manifold that guess spanned. The
two-site update merges neighbouring tensors into

```math
\Theta[a, s_1, s_2, a'] = \sum_m A[a, s_1, m] \, B[m, s_2, a']
```

acts on the whole block, and only then splits it again — so the bond between them is chosen
afresh, from the singular values of the *updated* block. The bond dimension grows on its own
until the truncation policy binds.

```@example alg
using MatrixProductToolbox
using LinearAlgebra
using Random

# -J Σ σᶻσᶻ - h Σ σˣ, the transverse-field Ising chain, as a bond-dimension-3 MPO.
function tfim(::Type{T}, L; J = 1.0, h = 0.5) where {T}
    σx = T[0 1; 1 0]; σz = T[1 0; 0 -1]; id = Matrix{T}(I, 2, 2)
    W = zeros(T, 3, 2, 2, 3)
    W[1, :, :, 1] = id;  W[1, :, :, 2] = σz;  W[1, :, :, 3] = -h * σx
    W[2, :, :, 3] = -J * σz;  W[3, :, :, 3] = id
    tensors = [copy(W) for _ in 1:L]
    tensors[1] = W[1:1, :, :, :]
    tensors[end] = W[:, :, :, 3:3]
    return MPO(tensors)
end

L = 10
H = tfim(ComplexF64, L; J = 1.0, h = 0.7)
ψ₀ = product_mps(fill(2, L), ones(Int, L))     # χ = 1 everywhere
maxbonddim(ψ₀)
```

```@example alg
sol = dmrg(H, ψ₀; nsweeps = 10, maxdim = 32, verbose = false)
sol.energy, maxbonddim(sol.state)
```

The bond dimension is an output, not an input.

## DMRG

Each sweep visits every bond left to right and then right to left. At each bond the two-site
block is replaced by the lowest eigenvector of the effective operator

```math
y[b, s_1', s_2', b'] = \sum L[b, w, a] \, W_1[w, s_1', s_1, w_m] \, W_2[w_m, s_2', s_2, w'] \,
                       \Theta[a, s_1, s_2, a'] \, R[b', w', a']
```

and split back with truncation, leaving the orthogonality centre on the site the sweep is moving
towards.

The state stays normalised without an explicit step, because [`krylov_eigen`](@ref) returns a
unit vector and splitting it leaves an isometry beside a centre tensor of unit norm. Only
truncation costs norm, and the sweep rescales the centre afterwards.

[`DMRGSolution`](@ref) reports the energy after every sweep and the largest relative discarded
weight of each. `energy` is the Rayleigh quotient of the *last local solve*, so it is variational
but is measured before that step's truncation; `expect(H, sol.state)` differs from it by the
discarded weight.

```@example alg
sol.energies
```

## TDVP

[`tdvp`](@ref) follows QuantumToolbox's `sesolve` in shape — `tdvp(H, ψ₀, tlist; e_ops, …)`
returning a [`TDVPSolution`](@ref) with `times`, `states` and `expect`.

!!! warning "`tlist` is the time step"
    One symmetric second-order TDVP step is taken per interval of `tlist`, so its spacing is the
    integration step and not merely where output is saved. Halving the spacing quarters the
    splitting error.

One step of `dt` is

```
left to right, over bonds 1 … L-1:
    evolve the two-site block forward by τ = -i·dt/2
    split; then evolve the one-site centre it leaves behind backward by -τ
right to left, over bonds L-1 … 1:
    the mirror image
```

Each half-sweep is first order on its own; running one and then its reverse makes the composition
symmetric, hence second order. The backward one-site step is not a correction bolted on — it
removes the double counting of the bond that two consecutive two-site blocks share, which is what
the projector-splitting derivation asks for.

```@example alg
Z = product_mpo([ComplexF64[1 0; 0 -1] for _ in 1:L])
ψᵢ = product_mps(fill(2, L), [isodd(i) ? 1 : 2 for i in 1:L])
evo = tdvp(H, ψᵢ, range(0, 2; length = 21); e_ops = (H, Z), maxdim = 64, progress = false)
round.(real.(evo.expect[2, 1:5]); digits = 6)
```

Energy is conserved to the truncation error:

```@example alg
E = real.(evo.expect[1, :]); maximum(abs, E .- E[1])
```

Singular values are **not** renormalised after a truncation: in real time the norm a truncation
costs measures how much of the state was thrown away, and hiding it would hide the error. Pass
`normalize = true` to rescale anyway.

### Imaginary time

`imaginary = true` evolves under ``e^{-\tau\hat H}`` instead. The step factor stays real, so a
`Float64` state stays `Float64`, and `normalize` defaults to `true`.

This is a second, independent route to a ground state, which is what makes it a useful check on
[`dmrg`](@ref) — the two share the environments, the split and the kernels, but not the local
solver.

!!! note "It converges at the rate of the gap"
    Imaginary time suppresses an excited state as ``e^{-2\tau\Delta}``. On a chain with a small
    gap the residual energy is set by ``\tau`` and cannot be removed by refining the time step or
    raising the bond dimension: the transverse-field Ising chain at ``L = 8``, ``h = 0.7`` has a
    gap of ``0.06``, and ``\tau = 20`` still leaves ``3\times10^{-3}``. DMRG has no such
    dependence. Use imaginary time to check a ground state, not to find one.

## [Environments](@id man-environments)

[`Environments`](@ref) caches the partial contractions of ``\langle\psi|\hat H|\psi\rangle``.
Entry `left[i]` covers sites `1:i-1` and `right[i]` covers `i:L`, so both vectors have length
`L+1` and the two boundary entries are the ``1\times1\times1`` environments. The pairing an
effective operator needs is

| block | left | right |
|:---|:---|:---|
| one site at `i` | `leftenv(env, H, ψ, i)` | `rightenv(env, H, ψ, i+1)` |
| two sites at `(i, i+1)` | `leftenv(env, H, ψ, i)` | `rightenv(env, H, ψ, i+2)` |

Validity is two integers rather than a flag per slot: `left[1:nleft]` and `right[nright:L+1]` are
current. Two suffice because the valid entries are always a prefix and a suffix — a left
environment can only be built from the one to its left, so invalidating one invalidates
everything after it. `MatrixProductToolbox.invalidate!(env, j)` declares that site `j` changed,
and the next [`leftenv`](@ref) or [`rightenv`](@ref) rebuilds only the step it is asked for.

```@example alg
env = Environments(H, sol.state)
sum(leftenv(env, H, sol.state, L + 1)) ≈ expect(H, sol.state)
```

## [Local solvers](@id man-local-solvers)

Both algorithms need a Krylov method: DMRG the lowest eigenpair of a Hermitian linear map, TDVP
the action of its exponential. Both are implemented here rather than taken from a Krylov package,
for two reasons — the dependency graph stays at four entries, and a package whose convergence
test is a branch on a value cannot be traced.

They are policies, so another solver can be plugged in from your own code by defining one method
on [`AbstractEigenSolver`](@ref) or [`AbstractExpSolver`](@ref) — a KrylovKit wrapper, say —
without this package depending on it.

```@example alg
sol_fast = dmrg(H, ψ₀; nsweeps = 6, maxdim = 32, verbose = false,
                solver = LanczosEigen(; krylovdim = 20, maxiter = 1))
sol_fast.energy
```

### Where tracing stops, and why

Lanczos has two halves.

1. Build an orthonormal Krylov basis ``V = [v_1 \ldots v_m]`` with ``m`` applications of the map,
   giving a small ``m \times m`` real symmetric tridiagonal ``T = V^\dagger A V``.
2. Solve the *projected* problem on ``T`` — lowest eigenpair, or ``e^{\tau T} e_1`` — and lift the
   answer back as ``x = V w``.

Half 1 is traceable as soon as ``m`` is fixed: it is dots, norms, broadcasts and calls to the
map. **Half 2 is the boundary**, and fixing the iteration count does not move it:

- ``\alpha_j = \mathrm{Re}\langle v_j, w\rangle`` is a traced scalar, so ``T`` is assembled from
  traced scalars;
- Reactant wraps no `eigen`/`eigh` at all, so there is nothing to call;
- `LinearAlgebra.exp` of a matrix picks its number of squarings from `opnorm(A)` — a branch on a
  value — and routes Hermitian input through `eigen`.

So the projected solve, and only the projected solve, is a policy:

| [`AbstractRitzSolver`](@ref) | lowest eigenpair | ``e^{\tau T} e_1`` | traceable |
|:---|:---|:---|:---:|
| [`DenseRitz`](@ref) (default) | `eigen(SymTridiagonal(α, β))` | via the same decomposition | ❌ |
| [`TracedRitz`](@ref) | a squared spectral filter, then Rayleigh-quotient iteration | scaling-and-squaring Padé | ✅ |

Selecting `TracedRitz` also turns the convergence test off, so the Lanczos runs its full
`krylovdim`. It is not a different *algorithm* — it computes the same two quantities by a
different numerical route, which is what lets a traced sweep be compared against the CPU answer
to floating-point tolerance rather than to a loose one.

```@example alg
a = dmrg(H, ψ₀; nsweeps = 6, trunc = FixedTruncation(32), verbose = false,
         solver = LanczosEigen(; krylovdim = 10, ritz = DenseRitz())).energy
b = dmrg(H, ψ₀; nsweeps = 6, trunc = FixedTruncation(32), verbose = false,
         solver = LanczosEigen(; krylovdim = 10, ritz = TracedRitz())).energy
a, b, abs(a - b)
```

### How the traceable policy works

Three ingredients, none of which branches on a value.

**Assembling the small matrix.** Every ``m \times m`` matrix is built as
`scalar .* hostpattern`, a broadcast of a Lanczos coefficient against a constant 0/1 array.
Broadcasting puts the result on whichever backend the *scalar* lives on: a host `Matrix` when the
coefficients are host `Real`s, as they are on `Array` and `CuArray`, and a traced array when they
are not. A `setindex!` would be a scalar write, which is illegal on a device array.

**The exponential.** Higham's scaling-and-squaring with a degree-13 Padé approximant, with the
one adaptive part — the number of squarings, normally chosen from `opnorm(A)` — frozen to a
compile-time constant. `s` squarings cover ``\|A\| \lesssim 5.4 \cdot 2^s``, so the default
``s = 12`` reaches ``2\times10^4`` and TDVP, where ``\|\tau H_{\text{eff}}\| = \|H\| \, dt / 2``,
never comes close. Over-squaring is the usual hazard of freezing `s`, but not here: ``A`` is
always ``\tau`` times a Hermitian tridiagonal, so ``e^A`` is normal — unitary in real time — and
repeated squaring of a normal matrix is well conditioned.

**The eigenpair.** A polynomial filter, built by repeated squaring. Gershgorin gives an interval
``[\ell, h]`` containing the spectrum, and

```math
B \;=\; I + \frac{h I - T}{h - \ell}
```

is positive definite with its *largest* eigenvalue at ``\lambda_1``. Squaring ``B`` ``k`` times
raises it to the power ``2^k``, so the lowest eigenvector wins by
``\big((h-\lambda_1)/(h-\lambda_2)\big)^{2^k}`` — a separation exponential in the number of
multiplies, where a plain power iteration would need ``2^k`` of them. Then a few steps of
Rayleigh-quotient iteration, which converges cubically once it is in the right basin.

Three details are load-bearing. **Renormalising after every squaring** is what makes a fixed
``k`` safe: the eigenvalues are raised to ``2^k \approx 10^7``, so without it the matrix would
overflow or underflow depending on where the spectrum happens to sit. Dividing by a norm changes
no ratio, and a norm is a reduction rather than a branch — so over-filtering costs nothing and
`filtersteps` may be set for the worst case. **Shifting by ``h + (h-\ell)``** rather than by
``h`` — the ``I +`` — keeps ``B \succ 0`` even when the Gershgorin interval degenerates to a
point, which is the case of a null effective operator; there ``B`` is the identity and the
eigenpair comes out as ``(0, e_1)`` instead of ``0/0``. And because ``h`` and ``\ell`` are
Gershgorin bounds rather than the true extremes, the filter's strength is set by the gap measured
against the *width of the discs*, which is the quantity ``k`` has to cover.

Two alternatives, both worse. An imaginary-time projection ``e^{-\beta T}`` is limited in
``\beta`` by the exponential's dynamic range, so where the gap is small next to the Gershgorin
width it under-separates and the polish then converges onto the *second* eigenvalue — measured on
7 of 4000 random Hermitian matrices, the worst 8.7 % above the true minimum, which is a
variational method silently returning an excited state. A shifted inverse iteration is worse
still: the only shift available a priori is ``\ell``, a full spectral radius below
``\lambda_{\min}``, and from there ``(T - \sigma I)^{-1}`` barely separates anything.

**Breakdown.** An eager Lanczos stops when the residual vanishes, because the Krylov space is
then invariant and there is no next direction. A traceable one cannot look, so it keeps going and
would divide by zero. Rather than floor the divisor — which puts a *zero* vector in the basis,
and with it a spurious eigenvalue at 0 whose eigenvector is zero — the recurrence repeats the
previous basis vector. That pads the projected matrix with ``\alpha_{j}I``, and ``\alpha_j`` is
the Rayleigh quotient of a genuine unit vector, so the padding can never sit below the true
projected minimum: it is unselectable, and harmless if it ties.

The cost of all this is honest: the projected solve emits a few tens of extra operations per
bond, and a fixed iteration count runs all `m` applications of the map where three would often
have converged. `DenseRitz` remains the better choice wherever it can run.

## Under Reactant

Everything the [Array backends](@ref man-backends) page says still applies, and two of its rules
become sharp here.

Use [`FixedTruncation`](@ref) and [`TracedRitz`](@ref) together: the first makes bond dimensions
a function of shapes alone, the second removes the last value-dependent branch. `dmrg` and `tdvp`
also switch their own sweep-to-sweep convergence tests off when handed a traceable solver, since
those are comparisons of energies and so branches of exactly the same kind.

```julia
sol = @jit dmrg(H, ψ₀; nsweeps = 4, trunc = FixedTruncation(16), verbose = false,
                solver = LanczosEigen(; krylovdim = 8, ritz = TracedRitz())).energy
```

!!! warning "Real-time TDVP needs a complex SVD"
    The two-site split has to truncate, and Reactant's complex SVD is broken on both backends —
    on CPU it fails to compile, on GPU it silently returns ``V^T`` where ``V^\dagger`` is wanted.
    Real-time TDVP under Reactant is therefore blocked upstream. DMRG on a real Hamiltonian in
    `Float64`, and imaginary-time TDVP, are unaffected.

`DenseRitz` refuses to run under tracing rather than falling back silently, and names the
alternative when it does:

```julia
julia> @jit dmrg(H, ψ₀; trunc = FixedTruncation(16))
ERROR: ArgumentError: `DenseRitz` solves the projected Lanczos problem with `LinearAlgebra.eigen`,
which needs the Krylov coefficients to be host numbers. […] Use `TracedRitz()` instead […]
```
