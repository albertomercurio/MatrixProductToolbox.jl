#=
Conversion to the full, exponentially large dense representation. Indispensable for testing
and for small examples; useless beyond ~20 sites, hence the size guard.

Index ordering: site 1 is the FASTEST-varying index, so that
`reshape(dense(ψ), physical_dims(ψ)...)` has dimension `i` equal to site `i`. This is the
choice that needs no `permutedims`. Note that it is the opposite of `kron`'s convention: a
reference operator must be assembled as `reduce(kron, reverse(ops))` to match.
=#

export dense

const _DENSE_MAXSIZE = 2^24

# Product of `dims`, saturating at `limit + 1` rather than overflowing.
#
# The full product must never be formed: for a 40-site MPO it is 2^80, which wraps Int64 around
# to exactly 0 and would let the guard below pass silently.
function _saturating_prod(dims, limit::Int)
    n = 1
    for d in dims
        d ≤ 0 && return 0
        n > limit ÷ d && return limit + 1
        n *= Int(d)
    end
    return n
end

function _check_dense_size(dims, maxsize::Integer)
    limit = Int(maxsize)
    _saturating_prod(dims, limit) ≤ limit || throw(
        ArgumentError(
            "Refusing to build a dense array with more than $limit entries. Pass `maxsize` " *
                "explicitly if this is intentional."
        )
    )
    return nothing
end

@doc raw"""
    dense(ψ; maxsize = 2^24)

Contract a matrix-product object into its full dense representation.

An [`MPS`](@ref) becomes a state vector of length `prod(physical_dims(ψ))`; an [`MPO`](@ref)
becomes a matrix. The result stays on the same array backend as the input — use
`Array(ψ)` to bring it to the host.

# Arguments
- `ψ`: an [`MPS`](@ref) or [`MPO`](@ref).
- `maxsize`: refuse to allocate more than this many entries, as a guard against accidentally
  contracting a large chain.

# Notes
- **Site 1 is the fastest-varying index**, so `reshape(dense(ψ), physical_dims(ψ)...)` has
  dimension `i` equal to site `i`. This is the opposite of `kron`'s ordering: build a
  reference operator as `reduce(kron, reverse(ops))` to compare.
- The cost is exponential in the number of sites. This is a testing and teaching tool.

# Returns
- `v`: a `Vector` for an [`MPS`](@ref), a `Matrix` for an [`MPO`](@ref).

# Examples
```jldoctest
julia> ψ = product_mps([2, 2], [1, 2]);

julia> round.(real(dense(ψ)); digits = 12)
4-element Vector{Float64}:
 0.0
 0.0
 1.0
 0.0
```
"""
function dense(ψ::AbstractMatrixProduct{1}; maxsize::Integer = _DENSE_MAXSIZE)
    _check_dense_size(physical_dims(ψ), maxsize)
    t = ψ[1]
    v = reshape(t, size(t, 2), size(t, 3))
    for i in 2:length(ψ)
        A = ψ[i]
        @tensor w[p, s, r] := v[p, a] * A[a, s, r]
        v = reshape(w, size(w, 1) * size(w, 2), size(w, 3))
    end
    return reshape(v, size(v, 1))
end

function dense(W::AbstractMatrixProduct{2}; maxsize::Integer = _DENSE_MAXSIZE)
    # The total entry count is prod(outputs) * prod(inputs), so flattening the per-site tuples
    # gives exactly the right product without ever multiplying the two halves together.
    _check_dense_size(Iterators.flatten(physical_dims(W)), maxsize)
    t = W[1]
    m = reshape(t, size(t, 2), size(t, 3), size(t, 4))
    for k in 2:length(W)
        A = W[k]
        # Output indices are kept adjacent, and likewise input indices, so that the fusing
        # reshape below stays contiguous.
        @tensor w[o, o2, i, i2, r] := m[o, i, a] * A[a, o2, i2, r]
        m = reshape(w, size(w, 1) * size(w, 2), size(w, 3) * size(w, 4), size(w, 5))
    end
    return reshape(m, size(m, 1), size(m, 2))
end

@doc raw"""
    Array(ψ; maxsize = 2^24)

Dense representation of `ψ` as a host `Array`.

Equivalent to `Array(dense(ψ))`: unlike [`dense`](@ref), this always moves the result to the
CPU.

# Arguments
- `ψ`: an [`MPS`](@ref) or [`MPO`](@ref).
- `maxsize`: forwarded to [`dense`](@ref).

# Returns
- `a::Array`: a `Vector` for an [`MPS`](@ref), a `Matrix` for an [`MPO`](@ref).
"""
Base.Array(ψ::AbstractMatrixProduct; kwargs...) = convert(Array, dense(ψ; kwargs...))
