#=
Truncation policies and the matrix factorization used to move the orthogonality centre.

Only the SVD is implemented. It is the single code path on every backend: `Array`, `CuArray`
and Reactant arrays all support `LinearAlgebra.svd`, whereas `qr` is unavailable under
Reactant. See CLAUDE.md for why QR was deferred and how to add it.
=#

export AbstractTruncation, NoTruncation, Truncation, FixedTruncation, TruncationInfo
export AbstractFactorization, SVDFactorization
export factorize_left, factorize_right

@doc raw"""
    abstract type AbstractTruncation

Supertype for bond-truncation policies.

A policy answers one question: given the singular values of a bond matrix, how many of them
are kept? The distinction that matters is *what the answer depends on*:

| Policy | Rank rule | Depends on | `Array` / `CuArray` | Reactant / XLA |
|:---|:---|:---|:---:|:---:|
| [`NoTruncation`](@ref) | `k = min(m, n)` | shape | ✅ | ✅ |
| [`FixedTruncation`](@ref) | `k = min(maxdim, m, n)` | shape | ✅ | ✅ |
| [`Truncation`](@ref) | smallest `k` meeting `cutoff` | singular-value magnitudes | ✅ | ❌ |

XLA requires every array shape to be known at trace time. It does *not* require uniform
bond dimensions — only that they be a function of shapes alone. That is exactly the top two
rows, which is why they are traceable and [`Truncation`](@ref) is not.
"""
abstract type AbstractTruncation end

@doc raw"""
    struct NoTruncation <: AbstractTruncation

Keep the full rank `min(m, n)` of every bond matrix; the factorization is exact.

This is the default for [`canonicalize!`](@ref), which is a pure gauge transformation.
"""
struct NoTruncation <: AbstractTruncation end

@doc raw"""
    struct Truncation{T<:Real} <: AbstractTruncation
        maxdim::Int
        cutoff::T
    end

    Truncation(; maxdim = typemax(Int), cutoff = 0.0)

Discard singular values based on their magnitude.

Keeps the smallest `k ≤ maxdim` such that the relative discarded weight

```math
\varepsilon = \frac{\sum_{i > k} \sigma_i^2}{\sum_i \sigma_i^2}
```

does not exceed `cutoff`. Bond dimensions therefore adapt to the state, which is what you
usually want on `Array` and `CuArray`.

!!! warning "Not traceable"
    Because `k` depends on the singular values themselves, the output shape is a runtime
    value. This cannot be used inside a Reactant `@compile` region — use
    [`FixedTruncation`](@ref) there.

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

!!! note "Not a uniform bond dimension"
    Near the ends of the chain `min(m, n) < maxdim`, so bonds there are genuinely smaller —
    but *deterministically*, from the chain geometry. That is precisely what XLA needs, and
    it means no zero-padding is ever required, so the resulting isometries are exact.

# Fields (Attributes)
- `maxdim::Int`: the target bond dimension, reached wherever the geometry allows.
"""
struct FixedTruncation <: AbstractTruncation
    maxdim::Int
end

FixedTruncation(; maxdim::Integer) = FixedTruncation(Int(maxdim))

@doc raw"""
    struct TruncationInfo{T<:Number}
        dim::Int
        error::T
    end

Report of a single bond truncation.

# Fields (Attributes)
- `dim::Int`: the bond dimension that was kept.
- `error::T`: the relative discarded weight ``\sum_{i>k}\sigma_i^2 / \sum_i \sigma_i^2``.

# Notes
- `error` is a real, non-negative quantity, but the bound here is `Number` rather than `Real` on
  purpose: a traced number is `<: Number` and **not** `<: Real`, so a `Real` bound would make
  this struct unconstructible inside a Reactant `@compile` region.
"""
struct TruncationInfo{T <: Number}
    dim::Int
    error::T
end

Base.show(io::IO, info::TruncationInfo) = print(io, "TruncationInfo(dim=", info.dim, ", error=", info.error, ")")

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

# Relative weight discarded by keeping the leading `k` singular values.
#
# Deliberately free of any branch on a *value*: under tracing, a comparison such as `total > 0`
# yields a traced boolean, which cannot drive Julia control flow. Branching on `k` is fine, since
# it comes from shapes. The consequence is that an all-zero bond reports `NaN` rather than `0`,
# which is acceptable for a diagnostic field.
function _discarded_weight(s::AbstractVector, k::Integer)
    w = abs2.(s)
    total = sum(w)
    k ≥ length(s) && return zero(total)
    return sum(w[(k + 1):end]) / total
end

@doc raw"""
    abstract type AbstractFactorization

Supertype for the matrix factorization used to move the orthogonality centre.

A factorization splits a bond matrix into an isometry and a remainder. Only
[`SVDFactorization`](@ref) is currently implemented; the abstract type exists so that
alternatives (a QR for untruncated sweeps, or a Cholesky-QR on backends with `cholesky` but
no `qr`) can be added without touching any algorithm.

# Interface

    factorize_left(alg, trunc, M)  -> (Q, R, info)   # Q'Q ≈ I, M ≈ Q * R
    factorize_right(alg, trunc, M) -> (L, Q, info)   # QQ' ≈ I, M ≈ L * Q
"""
abstract type AbstractFactorization end

@doc raw"""
    struct SVDFactorization <: AbstractFactorization

Factorize via the singular value decomposition.

Works on every supported backend and is the only factorization that can truncate. This is
the default everywhere.
"""
struct SVDFactorization <: AbstractFactorization end

@doc raw"""
    factorize_left(alg, trunc, M)

Split a bond matrix as `M ≈ Q * R` with `Q` having orthonormal columns.

Used when sweeping left to right: `Q` becomes the left-orthogonal site tensor and `R` is
absorbed into the next site.

# Arguments
- `alg`: an [`AbstractFactorization`](@ref).
- `trunc`: an [`AbstractTruncation`](@ref) policy.
- `M`: the `m × n` bond matrix.

# Returns
- `Q`: `m × k` with `Q' * Q ≈ I`.
- `R`: `k × n`.
- `info::TruncationInfo`: the kept dimension and discarded weight.
"""
function factorize_left(::SVDFactorization, trunc::AbstractTruncation, M::AbstractMatrix)
    F = svd(M)
    k = _truncation_rank(trunc, F.S)
    info = TruncationInfo(k, _discarded_weight(F.S, k))
    Q = F.U[:, 1:k]
    R = F.S[1:k] .* F.Vt[1:k, :]
    return Q, R, info
end

@doc raw"""
    factorize_right(alg, trunc, M)

Split a bond matrix as `M ≈ L * Q` with `Q` having orthonormal rows.

Used when sweeping right to left: `Q` becomes the right-orthogonal site tensor and `L` is
absorbed into the previous site.

# Arguments
- `alg`: an [`AbstractFactorization`](@ref).
- `trunc`: an [`AbstractTruncation`](@ref) policy.
- `M`: the `m × n` bond matrix.

# Returns
- `L`: `m × k`.
- `Q`: `k × n` with `Q * Q' ≈ I`.
- `info::TruncationInfo`: the kept dimension and discarded weight.
"""
function factorize_right(::SVDFactorization, trunc::AbstractTruncation, M::AbstractMatrix)
    F = svd(M)
    k = _truncation_rank(trunc, F.S)
    info = TruncationInfo(k, _discarded_weight(F.S, k))
    L = F.U[:, 1:k] .* transpose(F.S[1:k])
    Q = F.Vt[1:k, :]
    return L, Q, info
end
