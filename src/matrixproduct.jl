#=
The concrete one-dimensional matrix-product container, its `MPS` / `MPO` aliases, and the
`Base` / `Adapt` methods that make it behave like a well-mannered Julia object.
=#

export MatrixProduct, MPS, MPO
export center, iscanonical, isleftcanonical, isrightcanonical

@doc raw"""
    mutable struct MatrixProduct{N,T,A<:AbstractArray{T}} <: AbstractMatrixProduct{N}
        tensors::Vector{A}
        center::UnitRange{Int}
    end

A one-dimensional chain of site tensors with `N` physical legs each.

Each site tensor has the index order `A[l, p₁, …, p_N, r]`; see
[`AbstractMatrixProduct`](@ref) for the full convention. The two aliases

```julia
const MPS = MatrixProduct{1}    # A[a, s, a′]
const MPO = MatrixProduct{2}    # W[w, s′, s, w′]
```

are distinct types.

# Fields (Attributes)
- `tensors::Vector{A}`: the `L` site tensors, each with `N + 2` indices.
- `center::UnitRange{Int}`: the orthogonality centre; see [`center`](@ref).

# Constructors

    MatrixProduct(tensors, center = 1:length(tensors))
    MatrixProduct{N}(tensors, center = 1:length(tensors))

`N` is inferred from `ndims` of the site tensors when omitted. `center` may also be given as
an `Integer`, which is interpreted as the single-site range `c:c`. The constructor validates
that every tensor has `N + 2` indices, that neighbouring bonds match, that both boundary
bonds have dimension `1`, and that `center` lies within `1:L`.

!!! note "No implicit canonicalization"
    Construction never gauges or otherwise modifies the tensors you pass in. A fresh object
    therefore has `center == 1:L`, which claims nothing. Call [`canonicalize!`](@ref) to
    establish a gauge.

# Examples
```jldoctest
julia> ψ = MPS([reshape([1.0, 0.0], 1, 2, 1), reshape([0.0, 1.0], 1, 2, 1)])
MPS{Float64}(L=2, d=[2, 2], χ=[1, 1, 1], center=1:2)

julia> bond_dims(ψ)
3-element Vector{Int64}:
 1
 1
 1
```
"""
mutable struct MatrixProduct{N, T, A <: AbstractArray{T}} <: AbstractMatrixProduct{N}
    tensors::Vector{A}
    center::UnitRange{Int}

    function MatrixProduct{N, T, A}(
            tensors::Vector{A}, center::UnitRange{Int}
        ) where {N, T, A <: AbstractArray{T}}
        _check_sites(Val(N), tensors)
        _check_center(center, length(tensors))
        return new{N, T, A}(tensors, center)
    end
end

function MatrixProduct{N}(
        tensors::Vector{A}, center::UnitRange{Int} = 1:length(tensors)
    ) where {N, T, A <: AbstractArray{T}}
    return MatrixProduct{N, T, A}(tensors, center)
end

MatrixProduct{N}(tensors::Vector{<:AbstractArray}, c::Integer) where {N} =
    MatrixProduct{N}(tensors, Int(c):Int(c))

# Forgiving fallback: narrow the element type of e.g. a `Vector{Any}` before dispatching.
function MatrixProduct{N}(tensors::AbstractVector, center = 1:length(tensors)) where {N}
    narrowed = map(identity, tensors)
    narrowed isa Vector{<:AbstractArray} || throw(
        ArgumentError(
            "Expected a vector of arrays, got a $(typeof(tensors)) whose elements do not " *
                "share a common array type."
        )
    )
    return MatrixProduct{N}(narrowed, center)
end

function MatrixProduct(
        tensors::Vector{<:AbstractArray{<:Any, M}}, center = 1:length(tensors)
    ) where {M}
    M ≥ 3 || throw(
        ArgumentError(
            "Site tensors need at least 3 indices (left bond, ≥1 physical, right bond), got $M."
        )
    )
    return MatrixProduct{M - 2}(tensors, center)
end

@doc raw"""
    const MPS = MatrixProduct{1}

A matrix-product **state**: one physical leg per site, with site tensors indexed `A[a, s, a′]`
(left bond, physical, right bond).

This is an alias, not a wrapper, so `MPS` and [`MPO`](@ref) are still distinct types for dispatch
while sharing every gauge, truncation and backend method.

# Examples
```jldoctest
julia> ψ = MPS([reshape(ComplexF64[1, 0], 1, 2, 1)])
MPS{ComplexF64}(L=1, d=[2], χ=[1, 1], center=1:1)

julia> ψ isa MatrixProduct{1}
true
```
"""
const MPS = MatrixProduct{1}

@doc raw"""
    const MPO = MatrixProduct{2}

A matrix-product **operator**: two physical legs per site, with site tensors indexed
`W[w, s′, s, w′]` (left bond, output physical, input physical, right bond), so that
``W[w, s', s, w'] = \langle s' | \hat{W} | s \rangle``.

The bra contracts with the output leg `s′` and the ket with the input leg `s`.

An `MPO` also serves as a density matrix in matrix-product form, which is why open quantum
systems need no additional types.

# Examples
```jldoctest
julia> W = MPO([reshape(ComplexF64[1 0; 0 -1], 1, 2, 2, 1)])
MPO{ComplexF64}(L=1, d=[(2, 2)], χ=[1, 1], center=1:1)

julia> physical_dims(W)
1-element Vector{Tuple{Int64, Int64}}:
 (2, 2)
```
"""
const MPO = MatrixProduct{2}

function _check_sites(::Val{N}, tensors::AbstractVector) where {N}
    L = length(tensors)
    L ≥ 1 || throw(ArgumentError("A matrix-product object needs at least one site tensor."))
    for i in 1:L
        t = tensors[i]
        ndims(t) == N + 2 || throw(
            ArgumentError(
                "Site tensor $i has $(ndims(t)) indices, but a MatrixProduct{$N} requires " *
                    "$(N + 2) (left bond, $N physical, right bond)."
            )
        )
    end
    size(tensors[1], 1) == 1 || throw(
        ArgumentError("The left boundary bond must have dimension 1, got $(size(tensors[1], 1)).")
    )
    size(tensors[L], N + 2) == 1 || throw(
        ArgumentError(
            "The right boundary bond must have dimension 1, got $(size(tensors[L], N + 2))."
        )
    )
    for i in 1:(L - 1)
        r = size(tensors[i], N + 2)
        l = size(tensors[i + 1], 1)
        r == l || throw(
            DimensionMismatch(
                "Bond mismatch between sites $i and $(i + 1): right bond $r ≠ left bond $l."
            )
        )
    end
    return nothing
end

function _check_center(center::UnitRange{Int}, L::Int)
    (!isempty(center) && first(center) ≥ 1 && last(center) ≤ L) || throw(
        ArgumentError(
            "The orthogonality centre must be a non-empty range within 1:$L, got $center."
        )
    )
    return nothing
end

Base.length(ψ::MatrixProduct) = length(ψ.tensors)
Base.getindex(ψ::MatrixProduct, i::Integer) = ψ.tensors[i]
Base.firstindex(::MatrixProduct) = firstindex(ψ.tensors)
Base.lastindex(ψ::MatrixProduct) = lastindex(ψ.tensors)
Base.eachindex(ψ::MatrixProduct) = eachindex(ψ.tensors)

Base.eltype(::Type{<:MatrixProduct{N, T}}) where {N, T} = T
Base.eltype(ψ::MatrixProduct) = eltype(typeof(ψ))

scalartype(::Type{<:MatrixProduct{N, T}}) where {N, T} = T
scalartype(ψ::MatrixProduct) = scalartype(typeof(ψ))

@doc raw"""
    setindex!(ψ, t, i)

Replace the site tensor at site `i`, **invalidating the gauge**.

Because an arbitrary new tensor destroys any orthogonality relations, this resets
[`center`](@ref) to `1:length(ψ)`. Algorithms that know the gauge survives should use the
internal `MatrixProductToolbox.setsite!` instead.

# Arguments
- `ψ`: the matrix-product object to mutate.
- `t`: the new site tensor, with `nphysical(ψ) + 2` indices.
- `i`: the site index.

# Returns
- `ψ`: the mutated object.
"""
function Base.setindex!(ψ::MatrixProduct{N}, t::AbstractArray, i::Integer) where {N}
    ndims(t) == N + 2 || throw(
        ArgumentError(
            "Expected a tensor with $(N + 2) indices for a MatrixProduct{$N}, got $(ndims(t))."
        )
    )
    ψ.tensors[i] = t
    ψ.center = 1:length(ψ)
    return ψ
end

setsite!(ψ::MatrixProduct, i::Integer, t::AbstractArray) = (ψ.tensors[i] = t; ψ)
setcenter!(ψ::MatrixProduct, r::UnitRange{Int}) = (ψ.center = r; ψ)
setcenter!(ψ::MatrixProduct, c::Integer) = setcenter!(ψ, Int(c):Int(c))

@doc raw"""
    center(ψ)

The orthogonality centre of `ψ`: the range of sites **not known** to be orthogonal.

The invariant is that sites `1:first(center(ψ))-1` are left-orthogonal and sites
`last(center(ψ))+1:end` are right-orthogonal. Every value is meaningful, so there is no
"unknown" sentinel:

| Value | Meaning |
|:---|:---|
| `1:L` | nothing claimed — e.g. a freshly constructed object |
| `c:c` | mixed-canonical with the centre at site `c` |
| `1:1` | right-canonical |
| `L:L` | left-canonical |
| `i:i+1` | a two-site block |

# Arguments
- `ψ`: an [`AbstractMatrixProduct`](@ref).

# Returns
- `r::UnitRange{Int}`: the orthogonality centre.

# Examples
```jldoctest
julia> ψ = rand_mps(ComplexF64, [2, 2, 2, 2]; maxdim = 4);

julia> center(ψ)
1:1

julia> center(canonicalize!(ψ, 3))
3:3
```
"""
center(ψ::MatrixProduct) = ψ.center

@doc raw"""
    iscanonical(ψ)

Whether `ψ` is in a single-site canonical form, i.e. [`center`](@ref) spans exactly one site.

# Arguments
- `ψ`: an [`AbstractMatrixProduct`](@ref).

# Returns
- `::Bool`
"""
iscanonical(ψ::AbstractMatrixProduct) = length(center(ψ)) == 1

@doc raw"""
    isleftcanonical(ψ)

Whether the orthogonality centre sits on the last site, i.e. every site but the last is
left-orthogonal.

# Arguments
- `ψ`: an [`AbstractMatrixProduct`](@ref).

# Returns
- `::Bool`
"""
isleftcanonical(ψ::AbstractMatrixProduct) = center(ψ) == length(ψ):length(ψ)

@doc raw"""
    isrightcanonical(ψ)

Whether the orthogonality centre sits on the first site, i.e. every site but the first is
right-orthogonal.

# Arguments
- `ψ`: an [`AbstractMatrixProduct`](@ref).

# Returns
- `::Bool`
"""
isrightcanonical(ψ::AbstractMatrixProduct) = center(ψ) == 1:1

Base.copy(ψ::MatrixProduct{N}) where {N} = MatrixProduct{N}(map(copy, ψ.tensors), ψ.center)

_mpname(::Val{1}) = "MPS"
_mpname(::Val{2}) = "MPO"
_mpname(::Val{N}) where {N} = "MatrixProduct{$N}"

function Base.show(io::IO, ψ::MatrixProduct{N, T}) where {N, T}
    print(
        io, _mpname(Val(N)), "{", T, "}(L=", length(ψ),
        ", d=", physical_dims(ψ), ", χ=", bond_dims(ψ), ", center=", center(ψ), ")"
    )
    return nothing
end

Base.show(io::IO, ::MIME"text/plain", ψ::MatrixProduct) = show(io, ψ)

@doc raw"""
    Adapt.adapt_structure(to, ψ)

Move every site tensor of `ψ` to another array backend.

This is the single method that makes `ψ` portable: it gives `CUDA.cu(ψ)`,
`Adapt.adapt(Array, ψ)` and `Reactant.to_rarray(ψ)` for free, without this package knowing
anything about those libraries.

# Arguments
- `to`: the target passed on to `Adapt.adapt`, e.g. an array type.
- `ψ`: an [`AbstractMatrixProduct`](@ref).

# Returns
- `φ`: a new object of the same kind whose tensors live on the target backend.
"""
function adapt_structure(to, ψ::MatrixProduct{N}) where {N}
    return MatrixProduct{N}(map(t -> adapt(to, t), ψ.tensors), ψ.center)
end
