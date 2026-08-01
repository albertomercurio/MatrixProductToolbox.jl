#=
Truncation policies and the matrix factorizations used to move the orthogonality centre.

Two factorizations, split on truncation: the SVD is the only one that can discard anything,
and the QR is faster on the sweeps that discard nothing. Both run on every backend.
=#

export AbstractTruncation, NoTruncation, Truncation, FixedTruncation
export AbstractFactorization, SVDFactorization, QRFactorization
export factorize_left, factorize_right

@doc raw"""
    abstract type AbstractTruncation

Supertype for bond-truncation policies: how many singular values a bond keeps.

What distinguishes the policies is whether that number depends on the array *shapes* alone or on
the singular *values*. Only the former can be used with a traced backend such as Reactant/XLA.
See the [Canonical forms and truncation](@ref man-canonical) manual page for the comparison.
"""
abstract type AbstractTruncation end

@doc raw"""
    struct NoTruncation <: AbstractTruncation

Keep the full rank `min(m, n)` of every bond matrix, so the factorization is exact.

The default for [`canonicalize!`](@ref), which is a pure gauge transformation.
"""
struct NoTruncation <: AbstractTruncation end

@doc raw"""
    struct Truncation{T<:Real} <: AbstractTruncation
        maxdim::Int
        cutoff::T
    end

    Truncation(; maxdim = typemax(Int), cutoff = 0.0)

Keep the smallest `k ≤ maxdim` whose relative discarded weight
``\varepsilon = \sum_{i>k}\sigma_i^2 / \sum_i \sigma_i^2`` does not exceed `cutoff`.

Bond dimensions adapt to the state, which is usually what you want on `Array` and `CuArray`.

!!! warning "Not traceable"
    `k` depends on the singular values, so the output shape is a run-time value. Use
    [`FixedTruncation`](@ref) inside a Reactant `@compile` region.

# Fields (Attributes)
- `maxdim::Int`: hard cap on the kept bond dimension.
- `cutoff::T`: maximum tolerated relative discarded weight.
"""
struct Truncation{T <: Real} <: AbstractTruncation
    maxdim::Int
    cutoff::T
end

Truncation(maxdim::Integer, cutoff::Real) = Truncation(Int(maxdim), cutoff)
Truncation(; maxdim::Integer = typemax(Int), cutoff::Real = 0.0) = Truncation(Int(maxdim), cutoff)

@doc raw"""
    struct FixedTruncation <: AbstractTruncation
        maxdim::Int
    end

Keep `k = min(maxdim, m, n)` singular values, decided by the bond matrix shape alone.

Bonds near the ends of a chain are therefore smaller than `maxdim`, but deterministically so.
That is what makes this policy traceable, and it means no zero-padding is needed, so the
resulting isometries are exact.

# Fields (Attributes)
- `maxdim::Int`: the target bond dimension, reached wherever the geometry allows.
"""
struct FixedTruncation <: AbstractTruncation
    maxdim::Int
end

FixedTruncation(; maxdim::Integer) = FixedTruncation(Int(maxdim))

# Number of singular values to keep. `s` is real, non-negative and sorted descending.
_truncation_rank(::NoTruncation, s::AbstractVector) = length(s)
_truncation_rank(trunc::FixedTruncation, s::AbstractVector) = min(trunc.maxdim, length(s))

function _truncation_rank(trunc::Truncation, s::AbstractVector)
    n = length(s)
    kmax = min(trunc.maxdim, n)
    # `cutoff` is a field of the policy, hence always a host value: safe to branch on.
    trunc.cutoff > 0 || return kmax
    w = abs2.(s)
    total = sum(w)
    # This is the exact point where a lazily-traced backend parts company with an eager one: the
    # rank below is chosen by *comparing values*, which only works if a comparison yields a host
    # `Bool`. Check that on one cheap comparison up front. Doing it here rather than after the
    # reduction is deliberate — `count` on a traced array can exhaust the compiler outright, so a
    # late check never gets the chance to fire.
    ((total > zero(total)) isa Bool) || throw(
        ArgumentError(
            "`Truncation` needs to compare singular values to pick a bond dimension, but on this " *
                "array backend a comparison yields $(typeof(total > zero(total))) rather than a " *
                "`Bool`. That means the resulting bond dimension — and hence the output shape — " *
                "would not be known until run time, which a deferred/traced backend such as " *
                "Reactant/XLA cannot express. Use `FixedTruncation` instead: its bond dimensions " *
                "follow from the array shapes alone."
        )
    )
    # Reverse-cumulative weights: tail[i] = Σ_{j ≥ i} σⱼ², decreasing in i. A single reduction
    # then counts how many leading values must be kept — no scalar indexing, and no host copy of
    # the spectrum, so this stays efficient on a GPU.
    tail = reverse(cumsum(reverse(w)))
    k = count(>(trunc.cutoff * total), tail)
    return clamp(k, 1, kmax)
end

@doc raw"""
    abstract type AbstractFactorization

Supertype for the matrix factorizations used to move the orthogonality centre.

A factorization splits a bond matrix into an isometry and a remainder.
[`SVDFactorization`](@ref) and [`QRFactorization`](@ref) are implemented; the abstract type is
here so that a Cholesky-QR, for backends with `cholesky` but no `qr`, can be added without
touching any algorithm.

# Interface

    factorize_left(alg, trunc, M)  -> (Q, R)   # Q'Q ≈ I, M ≈ Q * R
    factorize_right(alg, trunc, M) -> (L, Q)   # QQ' ≈ I, M ≈ L * Q

A factorization that cannot discard anything also defines
`MatrixProductToolbox._supports_truncation(alg) = false`, which is what lets [`truncate!`](@ref)
reject it before it mutates anything.
"""
abstract type AbstractFactorization end

# Whether `alg` may be handed a policy that actually discards something. Internal: it exists so
# `truncate!` can refuse up front rather than half-way through a sweep, not as a knob to set.
_supports_truncation(::AbstractFactorization) = true

@doc raw"""
    struct SVDFactorization <: AbstractFactorization

Factorize via the singular value decomposition.

Works on every backend, and is the only factorization that can truncate. The default everywhere.
"""
struct SVDFactorization <: AbstractFactorization end

@doc raw"""
    struct QRFactorization <: AbstractFactorization

Factorize via the unpivoted QR decomposition: exact, and 2–3× faster than
[`SVDFactorization`](@ref). Use it to *move* the orthogonality centre rather than to truncate.

It keeps the full rank `min(m, n)`, so it rejects [`Truncation`](@ref) outright and accepts a
[`FixedTruncation`](@ref) only where the cap does not bind. Being unpivoted it also cannot
*detect* rank deficiency: a null direction is kept rather than dropped, `Q` remaining an exact
isometry with the matching row of `R` near zero. See the
[Canonical forms and truncation](@ref man-canonical) manual page for the comparison.

Under Reactant it is also the *only* factorization that gauges a `ComplexF64` state correctly,
the complex SVD there being broken upstream — see [Array backends](@ref man-backends).
"""
struct QRFactorization <: AbstractFactorization end

_supports_truncation(::QRFactorization) = false

@doc raw"""
    factorize_left(alg, trunc, M)

Split a bond matrix as `M ≈ Q * R` with `Q` having orthonormal columns.

Used when sweeping left to right: `Q` becomes the left-orthogonal site tensor and `R` is absorbed
into the next site.

# Arguments
- `alg`: an [`AbstractFactorization`](@ref).
- `trunc`: an [`AbstractTruncation`](@ref) policy.
- `M`: the `m × n` bond matrix.

# Returns
- `Q`: `m × k` with `Q' * Q ≈ I`.
- `R`: `k × n`.
"""
function factorize_left(::SVDFactorization, trunc::AbstractTruncation, M::AbstractMatrix)
    F = svd(M)
    k = _truncation_rank(trunc, F.S)
    Q = F.U[:, 1:k]
    R = F.S[1:k] .* F.Vt[1:k, :]
    return Q, R
end

@doc raw"""
    factorize_right(alg, trunc, M)

Split a bond matrix as `M ≈ L * Q` with `Q` having orthonormal rows.

Used when sweeping right to left: `Q` becomes the right-orthogonal site tensor and `L` is absorbed
into the previous site.

# Arguments
- `alg`: an [`AbstractFactorization`](@ref).
- `trunc`: an [`AbstractTruncation`](@ref) policy.
- `M`: the `m × n` bond matrix.

# Returns
- `L`: `m × k`.
- `Q`: `k × n` with `Q * Q' ≈ I`.
"""
function factorize_right(::SVDFactorization, trunc::AbstractTruncation, M::AbstractMatrix)
    F = svd(M)
    k = _truncation_rank(trunc, F.S)
    L = F.U[:, 1:k] .* transpose(F.S[1:k])
    Q = F.Vt[1:k, :]
    return L, Q
end

# A QR keeps the full rank `k = min(m, n)`, so the only policies it can honour are the ones that
# ask for exactly that. Deciding this from `k` alone -- never from a value -- is what keeps the
# check traceable: it is a shape comparison, and so yields a host `Bool` on every backend.
_check_qr_rank(::NoTruncation, ::Integer) = nothing

function _check_qr_rank(trunc::FixedTruncation, k::Integer)
    trunc.maxdim ≥ k && return nothing
    return throw(
        ArgumentError(
            "`QRFactorization` cannot truncate, but `FixedTruncation($(trunc.maxdim))` asks it " *
                "to keep $(trunc.maxdim) of $k directions. Use `SVDFactorization()` for a " *
                "truncating sweep, or `QRFactorization()` only where the cap does not bind."
        )
    )
end

_check_qr_rank(::Truncation, ::Integer) = throw(
    ArgumentError(
        "`QRFactorization` cannot truncate: a QR produces no singular values, so there is " *
            "nothing for `Truncation` to rank a bond by. Use `SVDFactorization()` instead."
    )
)

# `qr(M).Q` is a *square* `m × m` `AbstractQ` operator on `Array` and `CuArray` -- Julia 1.10
# decoupled `AbstractQ` from `AbstractMatrix` -- but an already-materialized thin `m × k` matrix
# under Reactant, whose factorization has no `AbstractQ` wrapper. Two methods rather than one
# shape test, because for a square bond matrix `m == k` the two cases are indistinguishable by
# shape alone.
_thin_q(Qf::AbstractMatrix, ::AbstractMatrix, k::Integer) =
    size(Qf, 2) == k ? Qf : Qf[:, 1:k]

# Multiplying the operator by an `m × k` identity dispatches to LAPACK `ormqr` (cuSOLVER on a
# GPU), so the square `Q` is never materialized and nothing moves to the host.
_thin_q(Qf, M::AbstractMatrix, k::Integer) = Qf * _eye(M, k)

# An `m × k` identity beside `M`, on whichever device `M` lives. Only ever reached from the
# `AbstractQ` method above, i.e. never under tracing -- so `eltype(M)` is a genuine Julia type
# here and not a `TracedRNumber`. `diagind` is a function of the shape, so this stays free of
# scalar indexing.
function _eye(M::AbstractMatrix, k::Integer)
    T = eltype(M)
    E = fill!(similar(M, T, size(M, 1), k), zero(T))
    E[diagind(E)] .= one(T)
    return E
end

function factorize_left(::QRFactorization, trunc::AbstractTruncation, M::AbstractMatrix)
    k = minimum(size(M))
    _check_qr_rank(trunc, k)
    F = qr(M)
    # Bind `F.Q` once: under Reactant every `getproperty` re-emits the `orgqr` call.
    return _thin_q(F.Q, M, k), F.R
end

# Implemented as a QR of the adjoint rather than an `lq`, because CUDA's `lq` coverage is
# unreliable. `M' ≈ Q * R` gives `M ≈ R' * Q'`, and `Q'` has the orthonormal *rows* wanted here.
# The `copy`s materialize the adjoints: the inputs to `qr` because a lazy wrapper is a poor thing
# to hand a tracing backend, the outputs because `factorize_left` returns dense factors and the
# two methods must agree.
function factorize_right(alg::QRFactorization, trunc::AbstractTruncation, M::AbstractMatrix)
    Q, R = factorize_left(alg, trunc, copy(adjoint(M)))
    return copy(adjoint(R)), copy(adjoint(Q))
end
