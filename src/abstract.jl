#=
Abstract supertype for one-dimensional matrix-product objects, together with the
interface that every concrete subtype must implement and the generic queries that
are derived from it.
=#

export AbstractMatrixProduct
export nphysical, physical_dims, bond_dims, maxbonddim

@doc raw"""
    abstract type AbstractMatrixProduct{N}

Supertype for one-dimensional matrix-product objects, where `N` is the number of physical
legs carried by every site.

A matrix-product object is a chain of `L` site tensors, each with the index order

```
A[l, p₁, …, p_N, r]
```

where `l` is the left bond, `p₁ … p_N` are the physical legs and `r` is the right bond.
Boundary bonds have dimension `1` rather than being omitted, so every site tensor has
exactly `N + 2` indices and all generic code is shape-agnostic.

`N = 1` is a matrix-product state ([`MPS`](@ref)) and `N = 2` is a matrix-product operator
([`MPO`](@ref)). Larger `N` is allowed but has no dedicated support.

# Notes
- This is a strictly one-dimensional container: every site has exactly one left and one
  right bond. Generic tensor networks with arbitrary connectivity are out of scope.
- `N` varies the number of physical legs per site, never the connectivity.

# Interface

A concrete subtype `T <: AbstractMatrixProduct{N}` must implement

- `Base.length(::T)`: the number of sites.
- `Base.getindex(::T, i::Integer)`: the site tensor at site `i`.
- [`center`](@ref)`(::T)`: the orthogonality centre as a `UnitRange{Int}`.
- `MatrixProductToolbox.setsite!(::T, i::Integer, t)`: replace a site tensor *without*
  touching the gauge label.
- `MatrixProductToolbox.setcenter!(::T, r::UnitRange{Int})`: set the gauge label.

Everything else ([`physical_dims`](@ref), [`bond_dims`](@ref), [`canonicalize!`](@ref),
[`expect`](@ref), …) is generic and follows from these five.
"""
abstract type AbstractMatrixProduct{N} end

@doc raw"""
    nphysical(ψ)

Number of physical legs carried by every site.

# Arguments
- `ψ`: an [`AbstractMatrixProduct`](@ref), or its type.

# Returns
- `N::Int`: `1` for an [`MPS`](@ref), `2` for an [`MPO`](@ref).

# Examples
```jldoctest
julia> nphysical(product_mps([2, 2, 2], [1, 1, 1]))
1
```
"""
nphysical(::AbstractMatrixProduct{N}) where {N} = N
nphysical(::Type{<:AbstractMatrixProduct{N}}) where {N} = N

# Generic fallback for subtypes that do not specialise it; `MatrixProduct` answers from its
# type parameter instead.
scalartype(ψ::AbstractMatrixProduct) = eltype(ψ[1])

@doc raw"""
    setsite!(ψ, i, t)

Replace the site tensor at site `i` *without* invalidating the gauge label.

This is part of the [`AbstractMatrixProduct`](@ref) interface and is intended for
algorithms that know the gauge is preserved. Use `ψ[i] = t` instead when the gauge
should be invalidated.

# Arguments
- `ψ`: the matrix-product object to mutate.
- `i`: the site index.
- `t`: the new site tensor.

# Returns
- `ψ`: the mutated object.
"""
function setsite! end

@doc raw"""
    setcenter!(ψ, r)

Set the gauge label of `ψ` to the site range `r`.

This is part of the [`AbstractMatrixProduct`](@ref) interface. See [`center`](@ref) for the
meaning of the range.

# Arguments
- `ψ`: the matrix-product object to mutate.
- `r`: the new orthogonality centre, a `UnitRange{Int}`.

# Returns
- `ψ`: the mutated object.
"""
function setcenter! end

@doc raw"""
    physical_dims(ψ)

Physical dimensions of every site.

# Arguments
- `ψ`: an [`AbstractMatrixProduct`](@ref).

# Returns
- `dims`: for `N = 1`, a `Vector{Int}` of local dimensions; for general `N`, a
  `Vector{NTuple{N,Int}}`.

# Examples
```jldoctest
julia> physical_dims(product_mps([2, 3, 2], [1, 1, 1]))
3-element Vector{Int64}:
 2
 3
 2
```
"""
function physical_dims(ψ::AbstractMatrixProduct{N}) where {N}
    return [ntuple(k -> size(ψ[i], k + 1), Val(N)) for i in 1:length(ψ)]
end

physical_dims(ψ::AbstractMatrixProduct{1}) = [size(ψ[i], 2) for i in 1:length(ψ)]

@doc raw"""
    bond_dims(ψ)

Dimensions of all `L + 1` bonds, including the two trivial boundary bonds.

Entry `i` is the dimension of the bond to the left of site `i`, so `bond_dims(ψ)[1]` and
`bond_dims(ψ)[end]` are both `1`.

# Arguments
- `ψ`: an [`AbstractMatrixProduct`](@ref).

# Returns
- `dims::Vector{Int}`: a vector of length `length(ψ) + 1`.

# Examples
```jldoctest
julia> bond_dims(product_mps([2, 2, 2], [1, 1, 1]))
4-element Vector{Int64}:
 1
 1
 1
 1
```
"""
function bond_dims(ψ::AbstractMatrixProduct)
    L = length(ψ)
    dims = Vector{Int}(undef, L + 1)
    dims[1] = size(ψ[1], 1)
    for i in 1:L
        t = ψ[i]
        dims[i + 1] = size(t, ndims(t))
    end
    return dims
end

@doc raw"""
    maxbonddim(ψ)

Largest bond dimension of `ψ`.

# Arguments
- `ψ`: an [`AbstractMatrixProduct`](@ref).

# Returns
- `χ::Int`: `maximum(bond_dims(ψ))`.
"""
maxbonddim(ψ::AbstractMatrixProduct) = maximum(bond_dims(ψ))

# Fuse the physical legs of a site tensor into a single index, turning a rank-(N+2) tensor
# into the rank-3 form that the N = 1 kernels expect. The (l, p…, r) index order makes this
# reshape contiguous, hence free.
_fuse_physical(t::AbstractArray) = reshape(t, size(t, 1), :, size(t, ndims(t)))

# Physical dimensions of a single site tensor, as a statically sized tuple.
_physsize(t::AbstractArray, ::Val{N}) where {N} = ntuple(k -> size(t, k + 1), Val(N))

# A boundary environment: all entries equal to one. Correct because boundary bonds have
# dimension 1, and free of scalar indexing so it works on any backend.
_boundary(t::AbstractArray, ::Type{T}, dims::Tuple) where {T} = fill!(similar(t, T, dims), one(T))
