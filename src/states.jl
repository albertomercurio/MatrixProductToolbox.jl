#=
Constructors for the matrix-product objects one actually starts from: random states, product
states, and product operators.

Hamiltonian MPO builders (from sums of local terms) are deliberately absent — building an
MPO by hand from a `Vector` of rank-4 arrays is a handful of lines, and a general builder is
a separate design problem. See the documentation for a worked transverse-field Ising example.
=#

export rand_mps, product_mps, product_mpo, identity_mpo

# Largest exactly representable bond dimensions for the given physical dimensions, capped at
# `maxdim`. Grown from the left, then clipped from the right, so the profile is symmetric and
# never claims more than the Hilbert space on either side supports.
function _mps_bond_dims(d::AbstractVector{<:Integer}, maxdim::Integer)
    L = length(d)
    maxdim ≥ 1 || throw(ArgumentError("`maxdim` must be at least 1, got $maxdim."))
    χ = ones(Int, L + 1)
    # Forward: no bond can exceed the Hilbert space to its left.
    for i in 1:L
        χ[i + 1] = _capped_product(χ[i], d[i], maxdim)
    end
    # The right boundary is trivial; restore it before clipping back, since the forward pass
    # has just overwritten it with the (irrelevant) left-hand bound.
    χ[L + 1] = 1
    # Backward: no bond can exceed the Hilbert space to its right either.
    for i in L:-1:2
        χ[i] = min(χ[i], _capped_product(χ[i + 1], d[i], maxdim))
    end
    return χ
end

# min(χ * d, cap) without overflowing when cap is typemax(Int).
_capped_product(χ::Int, d::Integer, cap::Integer) =
    χ ≥ cld(cap, d) ? Int(cap) : min(Int(cap), χ * Int(d))

@doc raw"""
    rand_mps([T = ComplexF64,] physical_dims; maxdim = typemax(Int), rng = default_rng())

A random, normalised [`MPS`](@ref) with the largest bond dimensions allowed by `maxdim`.

The tensors are filled with normally distributed entries, then right-canonicalised and
normalised, so the result satisfies `center(ψ) == 1:1` and `norm(ψ) ≈ 1`.

# Arguments
- `T`: element type of the site tensors.
- `physical_dims`: local dimension of each site, as a vector.
- `maxdim`: cap on the bond dimension.
- `rng`: random number generator.

# Returns
- `ψ::MPS`: a normalised random state.

# Examples
```jldoctest
julia> using Random

julia> ψ = rand_mps(ComplexF64, fill(2, 5); maxdim = 4, rng = MersenneTwister(0));

julia> bond_dims(ψ)
6-element Vector{Int64}:
 1
 2
 4
 4
 2
 1

julia> norm(ψ) ≈ 1
true
```
"""
function rand_mps(
        ::Type{T}, physical_dims::AbstractVector{<:Integer};
        maxdim::Integer = typemax(Int), rng::AbstractRNG = default_rng()
    ) where {T <: Number}
    isempty(physical_dims) && throw(ArgumentError("`physical_dims` must not be empty."))
    all(≥(1), physical_dims) || throw(ArgumentError("All physical dimensions must be at least 1."))
    χ = _mps_bond_dims(physical_dims, maxdim)
    tensors = [randn(rng, T, χ[i], physical_dims[i], χ[i + 1]) for i in eachindex(physical_dims)]
    ψ = MPS(tensors)
    canonicalize!(ψ, 1)
    return normalize!(ψ)
end

rand_mps(physical_dims::AbstractVector{<:Integer}; kwargs...) =
    rand_mps(ComplexF64, physical_dims; kwargs...)

@doc raw"""
    product_mps([T,] states)
    product_mps([T,] physical_dims, indices)

A [`MPS`](@ref) representing an unentangled product state, with all bonds of dimension 1.

The first form takes the local state vectors directly. The second builds the computational
basis state ``|i_1 i_2 \ldots i_L\rangle`` from a list of local dimensions and one-based
indices.

The norms of sites `2:L` are moved onto site 1, which makes the result exactly
right-canonical, so `center(ψ) == 1:1` without any factorization.

# Arguments
- `T`: element type of the site tensors; inferred from `states` when omitted.
- `states`: a vector of local state vectors.
- `physical_dims`: local dimension of each site.
- `indices`: one-based basis index selected at each site.

# Returns
- `ψ::MPS`: the product state.

# Examples
```jldoctest
julia> ψ = product_mps([2, 2, 2], [1, 2, 1]);

julia> bond_dims(ψ)
4-element Vector{Int64}:
 1
 1
 1
 1

julia> dense(ψ)[1 + 2 * (2 - 1) + 4 * (1 - 1)] ≈ 1
true
```
"""
function product_mps(::Type{T}, states::AbstractVector{<:AbstractVector}) where {T <: Number}
    isempty(states) && throw(ArgumentError("`states` must not be empty."))
    tensors = [reshape(convert(Vector{T}, collect(v)), 1, length(v), 1) for v in states]
    # Shift every local norm onto site 1: the product is unchanged, but sites 2:L become
    # exactly orthogonal, so the state is right-canonical with no work.
    for i in 2:length(tensors)
        n = norm(tensors[i])
        iszero(n) && throw(ArgumentError("Local state $i has zero norm."))
        tensors[i] = tensors[i] ./ n
        tensors[1] = tensors[1] .* n
    end
    return MPS(tensors, 1)
end

function product_mps(states::AbstractVector{<:AbstractVector})
    T = mapreduce(eltype, promote_type, states)
    return product_mps(T, states)
end

function product_mps(
        ::Type{T}, physical_dims::AbstractVector{<:Integer},
        indices::AbstractVector{<:Integer}
    ) where {T <: Number}
    length(physical_dims) == length(indices) || throw(
        DimensionMismatch(
            "Got $(length(physical_dims)) physical dimensions but $(length(indices)) indices."
        )
    )
    states = map(zip(physical_dims, indices)) do (d, i)
        (1 ≤ i ≤ d) || throw(ArgumentError("Basis index $i is out of range for dimension $d."))
        v = zeros(T, d)
        v[i] = one(T)
        return v
    end
    return product_mps(T, states)
end

product_mps(physical_dims::AbstractVector{<:Integer}, indices::AbstractVector{<:Integer}) =
    product_mps(ComplexF64, physical_dims, indices)

@doc raw"""
    product_mpo([T,] operators)

A [`MPO`](@ref) representing a product of single-site operators, with all bonds of
dimension 1.

# Arguments
- `T`: element type of the site tensors; inferred from `operators` when omitted.
- `operators`: a vector of matrices, one per site, each indexed `[output, input]`.

# Returns
- `W::MPO`: the product operator.

# Examples
```jldoctest
julia> σz = [1.0 0.0; 0.0 -1.0];

julia> W = product_mpo([σz, σz]);

julia> bond_dims(W)
3-element Vector{Int64}:
 1
 1
 1
```
"""
function product_mpo(::Type{T}, operators::AbstractVector{<:AbstractMatrix}) where {T <: Number}
    isempty(operators) && throw(ArgumentError("`operators` must not be empty."))
    tensors = [reshape(convert(Matrix{T}, collect(o)), 1, size(o, 1), size(o, 2), 1) for o in operators]
    return MPO(tensors)
end

function product_mpo(operators::AbstractVector{<:AbstractMatrix})
    T = mapreduce(eltype, promote_type, operators)
    return product_mpo(T, operators)
end

@doc raw"""
    identity_mpo([T = ComplexF64,] physical_dims)

The identity [`MPO`](@ref) on the given local dimensions.

# Arguments
- `T`: element type of the site tensors.
- `physical_dims`: local dimension of each site.

# Returns
- `W::MPO`: the identity operator, with all bonds of dimension 1.

# Examples
```jldoctest
julia> ψ = rand_mps(ComplexF64, fill(2, 4); maxdim = 4);

julia> expect(identity_mpo([2, 2, 2, 2]), ψ) ≈ 1
true
```
"""
function identity_mpo(::Type{T}, physical_dims::AbstractVector{<:Integer}) where {T <: Number}
    return product_mpo(T, [Matrix{T}(I, d, d) for d in physical_dims])
end

identity_mpo(physical_dims::AbstractVector{<:Integer}) = identity_mpo(ComplexF64, physical_dims)
