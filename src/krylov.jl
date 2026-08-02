#=
The local solvers: the lowest eigenpair of a Hermitian linear map (what DMRG needs) and the
action of its exponential (what TDVP needs). Both are Lanczos, and both are written here rather
than taken from a Krylov package so that the dependency graph stays at four entries and so that
the traced path below is possible at all.

Lanczos has two halves.

  1. Build an orthonormal Krylov basis V = [v₁ … v_m] with m applications of the map, giving a
     small m×m real symmetric tridiagonal T = VᴴAV.
  2. Solve the *projected* problem on T — lowest eigenpair, or exp(τT)e₁ — and lift the answer
     back as x = V·w.

Half 1 is traceable as soon as m is fixed: it is dots, norms, broadcasts and calls to the
caller's map. **Half 2 is the boundary**, and fixing the iteration count does not move it:

  * α[j] = real(dot(vⱼ, w)) is a traced scalar, so T is assembled from traced scalars;
  * Reactant wraps no `eigen`/`eigh` at all, so there is nothing to call;
  * `LinearAlgebra.exp(::Matrix)` picks its number of squarings from `opnorm(A)` — a branch on a
    value — and routes Hermitian input through `eigen`.

So the projected solve, and only the projected solve, is a policy: `DenseRitz` calls LAPACK,
`TracedRitz` computes the same two quantities with branch-free array operations. Running one
algorithm on every backend is what lets `test/backends.jl` compare a traced result against the
CPU answer to floating-point tolerance rather than to a loose one.

Two idioms hold this together, both of them consequences of where the coefficients live:

  * `scalar .* hostpattern` is how every small matrix is assembled. Broadcasting puts the result
    on whichever backend the *scalar* lives on — a host `Matrix` when the Lanczos coefficients
    are host `Real`s, as they are on `Array` and `CuArray`, and a traced array when they are not.
    A `setindex!` would be a scalar write and is illegal on a device array.
  * The basis size is `min(krylovdim, length(x))`. That is a **shape**, so it is legal under
    tracing, and it removes the one systematic cause of Lanczos breakdown here: near the ends of
    a chain a two-site tensor can be smaller than the requested Krylov dimension.
=#

export AbstractRitzSolver, DenseRitz, TracedRitz
export AbstractEigenSolver, AbstractExpSolver, LanczosEigen, LanczosExponential
export krylov_eigen, krylov_exp

# ---------------------------------------------------------------------------------------------
# Projected-solve policies
# ---------------------------------------------------------------------------------------------

@doc raw"""
    abstract type AbstractRitzSolver

Supertype for policies that solve the small projected problem at the end of a Lanczos run.

What distinguishes them is whether the `m × m` tridiagonal is handed to LAPACK or worked on with
branch-free array operations; only the latter can be traced. See the
[Algorithms](@ref man-algorithms) manual page.
"""
abstract type AbstractRitzSolver end

@doc raw"""
    struct DenseRitz <: AbstractRitzSolver

Solve the projected problem with `LinearAlgebra.eigen` on the small tridiagonal.

The default. Exact, adaptive, and the only policy that permits a value-dependent convergence
test — which also makes it unusable inside a Reactant `@compile` region, where
[`TracedRitz`](@ref) takes over.
"""
struct DenseRitz <: AbstractRitzSolver end

@doc raw"""
    struct TracedRitz <: AbstractRitzSolver
        iterations::Int
        squarings::Int
        filtersteps::Int
    end

    TracedRitz(; iterations = 3, squarings = 12, filtersteps = 24)

Solve the projected problem with branch-free array operations, so that a whole sweep traces.

The exponential is scaling-and-squaring Padé with a fixed number of squarings. The eigenpair uses
a spectral filter built by repeated squaring, then Rayleigh-quotient iteration to polish.
Selecting this policy also disables the convergence test, so the Lanczos runs its full
`krylovdim`.

# Fields (Attributes)
- `iterations::Int`: Rayleigh-quotient iterations after the filter.
- `squarings::Int`: squarings in the matrix exponential; `s` covers `‖A‖ ≲ 5.4 · 2^s`. Used by
  the exponential only.
- `filtersteps::Int`: squarings of the eigenvalue filter, whose degree is `2^filtersteps`. Larger
  resolves a smaller gap, at one small matrix multiply each.
"""
struct TracedRitz <: AbstractRitzSolver
    iterations::Int
    squarings::Int
    filtersteps::Int
end

TracedRitz(; iterations::Integer = 3, squarings::Integer = 12, filtersteps::Integer = 24) =
    TracedRitz(Int(iterations), Int(squarings), Int(filtersteps))

# Whether a policy forgoes every value-dependent branch. Internal: it is what turns the
# convergence test off, rather than the user having to set `tol = 0` to match.
_traceable(::AbstractRitzSolver) = false
_traceable(::TracedRitz) = true

# The exact point where a lazily-traced backend parts company with an eager one, checked on one
# cheap comparison before any work happens -- the same shape of guard as `_truncation_rank`'s,
# and for the same reason: a late check never gets the chance to fire.
function _require_host_ritz(::DenseRitz, n)
    ((n > zero(n)) isa Bool) || throw(
        ArgumentError(
            "`DenseRitz` solves the projected Lanczos problem with `LinearAlgebra.eigen`, which " *
                "needs the Krylov coefficients to be host numbers. On this array backend a " *
                "comparison yields $(typeof(n > zero(n))) rather than a `Bool`, so they are not: " *
                "there is no `eigen` to call, and the convergence test could not be evaluated " *
                "either. Use `TracedRitz()` instead — it computes the same quantities with " *
                "branch-free array operations."
        )
    )
    return nothing
end

_require_host_ritz(::TracedRitz, n) = nothing

# ---------------------------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------------------------

@doc raw"""
    abstract type AbstractEigenSolver

Supertype for solvers of the local eigenvalue problem that DMRG poses at every bond.

# Interface

    krylov_eigen(alg, f, x₀) -> (λ, v)

with `f` a callable Hermitian linear map. Define that one method to plug in another solver — a
KrylovKit wrapper, say — without this package depending on it.
"""
abstract type AbstractEigenSolver end

@doc raw"""
    abstract type AbstractExpSolver

Supertype for solvers of the local exponential that TDVP poses at every bond.

# Interface

    krylov_exp(alg, τ, f, x₀) -> y ≈ exp(τ f) x₀
"""
abstract type AbstractExpSolver end

@doc raw"""
    struct LanczosEigen{R<:AbstractRitzSolver} <: AbstractEigenSolver
        krylovdim::Int
        maxiter::Int
        tol::Float64
        ritz::R
    end

    LanczosEigen(; krylovdim = 12, maxiter = 2, tol = 1e-12, ritz = DenseRitz())

Lowest eigenpair by restarted Lanczos with full reorthogonalization.

# Fields (Attributes)
- `krylovdim::Int`: basis size per restart, capped at `length(x₀)`.
- `maxiter::Int`: number of restarts.
- `tol::Float64`: convergence tolerance, relative to the norm of the map — both for the Krylov
  breakdown test and for the eigenvalue between restarts. Ignored — and the
  Lanczos always runs its full `krylovdim` — when `ritz` is [`TracedRitz`](@ref).
- `ritz::R`: an [`AbstractRitzSolver`](@ref).
"""
struct LanczosEigen{R <: AbstractRitzSolver} <: AbstractEigenSolver
    krylovdim::Int
    maxiter::Int
    tol::Float64
    ritz::R
end

function LanczosEigen(;
        krylovdim::Integer = 12, maxiter::Integer = 2,
        tol::Real = 1.0e-12, ritz::AbstractRitzSolver = DenseRitz()
    )
    krylovdim ≥ 2 || throw(ArgumentError("`krylovdim` must be at least 2, got $krylovdim."))
    maxiter ≥ 1 || throw(ArgumentError("`maxiter` must be at least 1, got $maxiter."))
    return LanczosEigen(Int(krylovdim), Int(maxiter), Float64(tol), ritz)
end

@doc raw"""
    struct LanczosExponential{R<:AbstractRitzSolver} <: AbstractExpSolver
        krylovdim::Int
        tol::Float64
        ritz::R
    end

    LanczosExponential(; krylovdim = 16, tol = 1e-12, ritz = DenseRitz())

Action of a matrix exponential by Lanczos with full reorthogonalization.

# Fields (Attributes)
- `krylovdim::Int`: basis size, capped at `length(x₀)`.
- `tol::Float64`: breakdown tolerance, relative to the norm of the map. Ignored when `ritz` is
  [`TracedRitz`](@ref).
- `ritz::R`: an [`AbstractRitzSolver`](@ref).
"""
struct LanczosExponential{R <: AbstractRitzSolver} <: AbstractExpSolver
    krylovdim::Int
    tol::Float64
    ritz::R
end

function LanczosExponential(;
        krylovdim::Integer = 16, tol::Real = 1.0e-12, ritz::AbstractRitzSolver = DenseRitz()
    )
    krylovdim ≥ 1 || throw(ArgumentError("`krylovdim` must be at least 1, got $krylovdim."))
    return LanczosExponential(Int(krylovdim), Float64(tol), ritz)
end

# Whether a solver forgoes every value-dependent branch, and so can run inside a traced region.
# The fallback is `false`, which is the safe answer for a solver supplied from outside.
# `dmrg`/`tdvp` consult it for their own convergence tests, which are branches of the same kind.
_traceable(::AbstractEigenSolver) = false
_traceable(::AbstractExpSolver) = false
_traceable(alg::LanczosEigen) = _traceable(alg.ritz)
_traceable(alg::LanczosExponential) = _traceable(alg.ritz)

# `tol` is meaningless without a host comparison, so the policy decides it rather than the user
# having to keep the two fields consistent.
_tol(alg) = _traceable(alg) ? 0.0 : alg.tol

# ---------------------------------------------------------------------------------------------
# The Lanczos recurrence
# ---------------------------------------------------------------------------------------------

# `_floor_norm` (see `abstract.jl`) is what keeps a vanishing sub-diagonal from producing a `NaN`
# here. Flooring is safe rather than merely convenient: the floored direction is still
# orthonormalized against the whole basis, and a Rayleigh quotient of a unit vector can never
# fall below the smallest eigenvalue, so a noise direction cannot manufacture a Ritz value below
# the true one. It only wastes a basis vector.
#
# Build the Krylov basis. Returns `(V, α, β)` with `length(β) == length(V) - 1 == length(α) - 1`.
# `m` comes from a shape, so it is a compile-time quantity on every backend.
function _lanczos(f, v₁, m::Integer, tol::Real)
    w = f(v₁)
    V = [v₁]
    α = [real(dot(v₁, w))]
    β = similar(α, 0)
    # A running magnitude for the operator, against which a residual can be called negligible.
    # `|α₁|` is a Rayleigh quotient of a unit vector and the βs are residual norms, so the largest
    # of them is a lower bound on ‖f‖ — which is all a *relative* tolerance needs.
    scale = abs(α[1])
    for _ in 2:m
        # Modified Gram-Schmidt against the whole basis, run **twice**.
        #
        # Once is not enough, and the case that breaks it is the one DMRG is in constantly. Once
        # a sweep has half converged, the two-site block handed to this function is already
        # nearly the local eigenvector, so the first residual is tiny — β₁ ≈ 1e-5 against an
        # operator of norm ≈ 20 is typical. A single pass removes the components along the
        # existing basis only down to roundoff *of the vector before scaling*; dividing by that
        # tiny β then amplifies what is left by 1/β, and the error compounds along the
        # recurrence. Measured on a real sweep: ‖VᴴV - I‖ = 1.4, so T is no longer VᴴAV, and the
        # Ritz value comes out at -29.8 against a true minimum of -21.8 — below it, which a
        # variational method must never produce.
        #
        # "Twice is enough" (Kahan) is the standard remedy: the second pass works on a vector
        # that is already orthogonal to working precision, so what it removes is ~eps‖w‖ rather
        # than ~eps‖f(v)‖, and the amplification by 1/β lands back at roundoff. The cost is one
        # extra O(m²) set of dots per step, which is nothing beside the applications of `f`.
        for _ in 1:2, v in V
            w = w .- v .* dot(v, w)
        end
        b = _norm2(w)
        # Stop once the residual is negligible *relative to the operator*, which is the only
        # scale-free way to say it: an absolute threshold is meaningless because `f` carries the
        # units of an energy, and at ‖f‖ ≈ 20 a converged block leaves a residual around 2e-12 —
        # above a fixed 1e-12, so a fixed threshold would essentially never fire and every solve
        # would run the full `krylovdim` for nothing. Measured on a 20-site chain at χ = 128, that
        # was the whole of a 1.6× gap against an adaptively restarted KrylovKit.
        #
        # `tol` is a host field of the solver, so the outer test is decided at trace time; the
        # traceable policy sets it to zero and the loop always runs to `m`.
        tol > 0 && b ≤ tol * scale && break
        push!(β, b)
        scale = max(scale, b)
        # On an *exact* breakdown the next direction does not exist rather than merely being
        # small, and `w ./ b` is `0/0`. Flooring the divisor alone would push a **zero** vector
        # into the basis, which is worse than useless: it enters the projected matrix as a
        # spurious eigenvalue at 0 whose eigenvector is zero, so a positive-definite effective
        # operator reports 0 and hands back nothing. Repeating the previous basis vector instead
        # pads the projected matrix with `α_prev · I`, and `α_prev` is the Rayleigh quotient of a
        # genuine unit vector, so it can never sit *below* the true projected minimum — the
        # padding is unselectable, and harmless if it ties.
        #
        # A traceable solver cannot take the eager way out here, which is to notice `b == 0` and
        # stop. The blend is exact at both ends instead: `b / _floor_norm(b)` is `1` for every `b`
        # above the floor, so an ordinary step is untouched, and `0` at `b == 0`.
        d = _floor_norm(b)
        push!(V, w ./ d .+ (one(b) - b / d) .* V[end])
        w = f(V[end])
        push!(α, real(dot(V[end], w)))
    end
    return V, α, β
end

# x = Σⱼ w[j] · V[j], where `V` is a host `Vector` of tensors and `w` the coefficients from the
# projected solve.
#
# Two methods, dispatching on where the *coefficients* live. `DenseRitz` returns a host `Vector`
# whatever the array backend, because the Lanczos coefficients are host numbers there; `w[j]` is
# then free, and broadcasting a host scalar against a device tensor is fine. `TracedRitz` under
# tracing returns a traced array instead, and `w[j]` would be a scalar read of a device array —
# illegal. So that case stacks the basis and does one matrix-vector product, which is also fewer
# kernel launches than `m` separate broadcasts.
function _combine(V::AbstractVector, w::Vector)
    y = V[1] .* w[1]
    for j in 2:length(V)
        y = y .+ V[j] .* w[j]
    end
    return y
end

_combine(V::AbstractVector, w::AbstractVector) =
    reshape(reduce(hcat, map(vec, V)) * w, size(V[1]))

@doc raw"""
    krylov_eigen(alg, f, x₀)

Lowest algebraic eigenpair of a Hermitian linear map.

# Arguments
- `alg`: an [`AbstractEigenSolver`](@ref).
- `f`: the map, called as `f(x)`; it must be Hermitian and shape-preserving.
- `x₀`: the starting vector, of any array shape. A good guess matters — DMRG passes the current
  two-site tensor.

# Returns
- `λ`: the eigenvalue, real.
- `v`: the eigenvector, normalised and shaped like `x₀`.
"""
function krylov_eigen(alg::LanczosEigen, f, x₀)
    nrm = _norm2(x₀)
    _require_host_ritz(alg.ritz, nrm)
    tol = _tol(alg)
    m = min(alg.krylovdim, length(x₀))
    v = x₀ ./ _floor_norm(nrm)
    V, α, β = _lanczos(f, v, m, tol)
    λ, w = _ritz_eigen(alg.ritz, α, β)
    v = _combine(V, w)
    for _ in 2:(alg.maxiter)
        λ_prev = λ
        v = v ./ _floor_norm(_norm2(v))
        V, α, β = _lanczos(f, v, m, tol)
        λ, w = _ritz_eigen(alg.ritz, α, β)
        v = _combine(V, w)
        tol > 0 && abs(λ - λ_prev) ≤ tol * max(abs(λ), one(λ)) && break
    end
    return λ, v ./ _floor_norm(_norm2(v))
end

@doc raw"""
    krylov_exp(alg, τ, f, x₀)

The action ``e^{\tau f} x_0`` of the exponential of a Hermitian linear map.

# Arguments
- `alg`: an [`AbstractExpSolver`](@ref).
- `τ`: the step. `-im * dt` for real time, `-dt` for imaginary time.
- `f`: the map, called as `f(x)`; it must be Hermitian and shape-preserving.
- `x₀`: the vector to propagate, of any array shape.

# Returns
- `y`: the propagated vector, shaped like `x₀`.
"""
function krylov_exp(alg::LanczosExponential, τ::Number, f, x₀)
    nrm = _norm2(x₀)
    _require_host_ritz(alg.ritz, nrm)
    m = min(alg.krylovdim, length(x₀))
    V, α, β = _lanczos(f, x₀ ./ _floor_norm(nrm), m, _tol(alg))
    w = _ritz_exp(alg.ritz, τ, α, β)
    return _combine(V, w) .* nrm
end

# ---------------------------------------------------------------------------------------------
# The projected solve: LAPACK
# ---------------------------------------------------------------------------------------------

function _ritz_eigen(::DenseRitz, α::AbstractVector, β::AbstractVector)
    length(α) == 1 && return α[1], [one(α[1])]
    F = eigen(SymTridiagonal(collect(α), collect(β)))
    # `eigen` of a `SymTridiagonal` returns its values in ascending order.
    return F.values[1], F.vectors[:, 1]
end

function _ritz_exp(::DenseRitz, τ::Number, α::AbstractVector, β::AbstractVector)
    length(α) == 1 && return [exp(τ * α[1])]
    F = eigen(SymTridiagonal(collect(α), collect(β)))
    # exp(τT)e₁ = U exp(τΛ) Uᵀe₁, and Uᵀe₁ is the first *row* of U. Going through the
    # eigendecomposition rather than `exp` of the matrix keeps this exact for a Hermitian T and
    # costs one decomposition either way.
    return F.vectors * (exp.(τ .* F.values) .* view(F.vectors, 1, :))
end

# ---------------------------------------------------------------------------------------------
# The projected solve: branch-free
# ---------------------------------------------------------------------------------------------

# A constant 0/1 pattern, always a host array. It is only ever the *right* operand of a
# broadcast against a coefficient, which is what decides where the result lands.
function _pattern(m::Integer, entries)
    E = zeros(Float64, m, m)
    for (i, j) in entries
        E[i, j] = 1.0
    end
    return E
end

_eye_pattern(m::Integer) = _pattern(m, ((j, j) for j in 1:m))

# The symmetric tridiagonal `(α, β)` as a dense matrix.
#
# Assembled by broadcasting each coefficient against a constant pattern rather than by writing
# elements: `setindex!` is a scalar write and is illegal on a device array, whereas
# `scalar .* hostpattern` follows the scalar onto whichever backend it lives on.
function _tridiagonal(α::AbstractVector, β::AbstractVector)
    m = length(α)
    A = α[1] .* _pattern(m, ((1, 1),))
    for j in 2:m
        A = A .+ α[j] .* _pattern(m, ((j, j),))
    end
    for j in 1:(m - 1)
        A = A .+ β[j] .* _pattern(m, ((j, j + 1), (j + 1, j)))
    end
    return A
end

# The m×m identity beside the coefficients, by the same route.
_identity_like(α::AbstractVector) = one(α[1]) .* _eye_pattern(length(α))

# The first column of the identity, as a vector on the coefficients' backend.
function _first_unit(α::AbstractVector)
    m = length(α)
    e = zeros(Float64, m)
    e[1] = 1.0
    return one(α[1]) .* e
end

# Gershgorin bounds on the spectrum: `lo ≤ λ₁` and `λ_m ≤ hi`, from the discs
# `αⱼ ± (|βⱼ₋₁| + |βⱼ|)`. `reduce` runs over a host-length collection and the `j` comparisons are
# on the loop index, so nothing here branches on a value.
function _gershgorin_bounds(α::AbstractVector, β::AbstractVector)
    m = length(α)
    z = zero(α[1])
    radius(j) = (j > 1 ? abs(β[j - 1]) : z) + (j < m ? abs(β[j]) : z)
    return reduce(min, α[j] - radius(j) for j in 1:m),
        reduce(max, α[j] + radius(j) for j in 1:m)
end

# Hold a Rayleigh-quotient shift strictly below the eigenvalue it converges onto. Without it the
# shifted matrix goes exactly singular as the iteration converges and the solve returns `NaN`; a
# branch on the residual would be the eager way out, and is not available here.
#
# The offset is relative, so it survives any rescaling of H. Its size is a genuine trade-off and
# was measured: it caps how close two eigenvalues can be and still be resolved, so at 1e-8 a pair
# separated by 1e-9 stalls the residual at 1e-7, while at 1e-13 the same pair reaches 3e-11. On
# spectra that are *not* degenerate to within the offset the eigenvalue agrees with `DenseRitz`
# to 6e-15, i.e. exactly. Going below 1e-13 buys nothing and edges towards the singularity again.
_guard_shift(ρ) = ρ - oftype(ρ, 1.0e-13) * (one(ρ) + abs(ρ))

function _ritz_eigen(alg::TracedRitz, α::AbstractVector, β::AbstractVector)
    length(α) == 1 && return α[1], _first_unit(α)
    T = _tridiagonal(α, β)
    Id = _identity_like(α)
    lo, hi = _gershgorin_bounds(α, β)

    # Isolate the lowest eigenvector with a polynomial filter. `B = I + (hi·I - T)/(hi - lo)` is
    # positive definite with its *largest* eigenvalue at λ₁, so squaring it `k` times raises it to
    # the power 2ᵏ and the lowest eigenvector wins by `((hi-λ₁)/(hi-λ₂))^{2ᵏ}` — a separation
    # exponential in the number of multiplies, where a power iteration would need 2ᵏ of them.
    #
    # Three details are load-bearing:
    #
    # * **Renormalising after every squaring** is what makes a *fixed* number of steps safe. The
    #   eigenvalues are raised to 2ᵏ ≈ 10⁷, so without it the whole matrix would overflow or
    #   underflow depending on where the spectrum sits, and the answer would depend on a scale
    #   nobody controls. Dividing by a norm changes no ratio, and `_norm2` is a reduction rather
    #   than a branch. Over-filtering is then free: once the unwanted components fall below
    #   roundoff they stay there, so `filtersteps` may be set for the worst case.
    # * **Shifting by `hi + (hi - lo)` rather than by `hi`** — the `Id .+` — keeps `B ≻ 0` even
    #   when the Gershgorin interval degenerates to a point, which is exactly the case of a null
    #   effective operator. There `B` is the identity, the filter is a no-op, and the eigenpair
    #   comes out as `(0, e₁)` instead of `0/0`. It costs a factor of at most two in contrast,
    #   i.e. one extra squaring.
    # * `hi` and `lo` are Gershgorin bounds rather than the true extremes, so the filter's
    #   strength is set by `(λ₂ - λ₁)` measured against the *width of the discs*. That is the
    #   quantity `filtersteps` has to cover.
    #
    # An imaginary-time projection `exp(-βT)` was the previous route and is strictly worse: its
    # β is limited by the exponential's dynamic range, so on a spectrum whose gap is small next
    # to the Gershgorin width it under-separates and the polish below then converges onto the
    # *second* eigenvalue. Measured that way on 7 of 4000 random Hermitian matrices, the worst 8.7%
    # above the true minimum — a variational method silently returning an excited state. A shifted
    # inverse iteration is worse still: the only shift available a priori is `lo`, a full spectral
    # radius below λ_min, and from there `(T - σI)⁻¹` barely separates anything.
    B = Id .+ (hi .* Id .- T) ./ _floor_norm(hi - lo)
    for _ in 1:(alg.filtersteps)
        B = B * B
        B = B ./ _floor_norm(_norm2(B))
    end
    w = B * _first_unit(α)
    w = w ./ _floor_norm(_norm2(w))

    # Polish. Rayleigh-quotient iteration converges cubically once it is in the right basin, which
    # the filter has guaranteed.
    for _ in 1:(alg.iterations)
        y = (T .- _guard_shift(real(dot(w, T * w))) .* Id) \ w
        w = y ./ _floor_norm(_norm2(y))
    end
    return real(dot(w, T * w)), w
end

# exp(A) by scaling and squaring with a degree-13 Padé approximant: Higham's algorithm with the
# one adaptive part — the number of squarings, chosen there from `opnorm(A)` — frozen to a
# compile-time constant. `s` squarings cover ‖A‖ ≲ 5.4·2ˢ.
#
# Over-squaring is the usual hazard of freezing `s`, but not here: `A` is always `τ` times a
# Hermitian tridiagonal, so `exp(A)` is normal — unitary in real time — and repeated squaring of
# a normal matrix is well conditioned.
const _PADE13 = (
    64764752532480000.0, 32382376266240000.0, 7771770303897600.0, 1187353796428800.0,
    129060195264000.0, 10559470521600.0, 670442572800.0, 33522128640.0,
    1323241920.0, 40840800.0, 960960.0, 16380.0, 182.0, 1.0,
)

function _expm_pade(A::AbstractMatrix, Id::AbstractMatrix, squarings::Integer)
    c = _PADE13
    X = A ./ (2.0^squarings)
    X2 = X * X
    X4 = X2 * X2
    X6 = X2 * X4
    U = X * (
        X6 * (c[14] .* X6 .+ c[12] .* X4 .+ c[10] .* X2) .+
            c[8] .* X6 .+ c[6] .* X4 .+ c[4] .* X2 .+ c[2] .* Id
    )
    V = X6 * (c[13] .* X6 .+ c[11] .* X4 .+ c[9] .* X2) .+
        c[7] .* X6 .+ c[5] .* X4 .+ c[3] .* X2 .+ c[1] .* Id
    E = (V .- U) \ (V .+ U)
    for _ in 1:squarings
        E = E * E
    end
    return E
end

function _ritz_exp(alg::TracedRitz, τ::Number, α::AbstractVector, β::AbstractVector)
    A = τ .* _tridiagonal(α, β)
    # `one(τ)` carries the element type across, so the identity matches `A` without a `convert`
    # and without dividing by a `τ` that may be zero.
    Id = _identity_like(α) .* one(τ)
    return _expm_pade(A, Id, alg.squarings)[:, 1]
end
