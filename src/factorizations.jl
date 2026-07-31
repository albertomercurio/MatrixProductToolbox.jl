#=
Truncation policies and the matrix factorization used to move the orthogonality centre.

Only the SVD is implemented. It is the single code path on every backend: `Array`, `CuArray`
and Reactant arrays all support `LinearAlgebra.svd`, whereas `qr` is unavailable under
Reactant. See CLAUDE.md for why QR was deferred and how to add it.
=#

export AbstractTruncation, NoTruncation, Truncation, FixedTruncation
export AbstractFactorization, SVDFactorization
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

Supertype for the matrix factorization used to move the orthogonality centre.

A factorization splits a bond matrix into an isometry and a remainder. Only
[`SVDFactorization`](@ref) exists so far; the abstract type is here so that a QR for untruncated
sweeps, or a Cholesky-QR for backends with `cholesky` but no `qr`, can be added without touching
any algorithm.

# Interface

    factorize_left(alg, trunc, M)  -> (Q, R)   # Q'Q ≈ I, M ≈ Q * R
    factorize_right(alg, trunc, M) -> (L, Q)   # QQ' ≈ I, M ≈ L * Q
"""
abstract type AbstractFactorization end

@doc raw"""
    struct SVDFactorization <: AbstractFactorization

Factorize via the singular value decomposition.

Works on every backend, and is the only factorization that can truncate. The default everywhere.
"""
struct SVDFactorization <: AbstractFactorization end

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
