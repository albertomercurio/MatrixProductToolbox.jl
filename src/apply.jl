#=
Applying an MPO to an MPS, and scalar arithmetic.
=#

export apply

@doc raw"""
    apply(H, ψ; trunc = NoTruncation(), to = 1, factorization = SVDFactorization())

Apply the [`MPO`](@ref) `H` to the [`MPS`](@ref) `ψ`.

Each pair of site tensors is contracted over the physical index and the MPO and MPS bonds are
fused, so the bond dimensions **multiply**: a state of bond dimension `χ` acted on by an
operator of bond dimension `D` comes back with bond dimension `χD`. Pass `trunc` to compress
the result in the same call.

# Arguments
- `H`: an [`MPO`](@ref) with the same number of sites as `ψ`.
- `ψ`: an [`MPS`](@ref).
- `trunc`: an [`AbstractTruncation`](@ref). The default leaves the result exact; pass e.g.
  `Truncation(; maxdim = 32)` to compress, or [`FixedTruncation`](@ref) for a
  Reactant-compatible sweep.
- `to`: where to leave the orthogonality centre, when truncating.
- `factorization`: an [`AbstractFactorization`](@ref), when truncating.

# Notes
- Without truncation the result is exact but its bond dimension grows, so applying an operator
  repeatedly needs a `trunc`.

# Returns
- `φ::MPS`: the state `H|ψ⟩`.

# Examples
```jldoctest
julia> ψ = rand_mps(ComplexF64, fill(2, 4); maxdim = 4);

julia> σx = ComplexF64[0 1; 1 0]; H = product_mpo([σx, σx, σx, σx]);

julia> dense(apply(H, ψ)) ≈ dense(H) * dense(ψ)
true
```
"""
function apply(
        H::AbstractMatrixProduct{2}, ψ::AbstractMatrixProduct{1};
        trunc::AbstractTruncation = NoTruncation(),
        to::Integer = 1,
        factorization::AbstractFactorization = SVDFactorization()
    )
    L = length(ψ)
    L == length(H) || throw(
        DimensionMismatch("Operator has $(length(H)) sites but the state has $L.")
    )
    tensors = map(1:L) do i
        W, t = H[i], ψ[i]
        size(W, 3) == size(t, 2) || throw(
            DimensionMismatch(
                "At site $i the operator acts on dimension $(size(W, 3)) but the state has " *
                    "local dimension $(size(t, 2))."
            )
        )
        # Output indices are grouped so that both fusing reshapes stay contiguous.
        @tensor y[wl, al, s′, wr, ar] := W[wl, s′, s, wr] * t[al, s, ar]
        reshape(
            y, size(y, 1) * size(y, 2), size(y, 3), size(y, 4) * size(y, 5)
        )
    end
    φ = MPS(tensors)
    trunc isa NoTruncation && return φ
    return truncate!(φ; trunc, to, factorization)
end

@doc raw"""
    *(H, ψ)

Shorthand for `apply(H, ψ)`, i.e. an exact application with no truncation.

# Arguments
- `H`: an [`MPO`](@ref).
- `ψ`: an [`MPS`](@ref).

# Returns
- `φ::MPS`: the state `H|ψ⟩`.
"""
Base.:*(H::AbstractMatrixProduct{2}, ψ::AbstractMatrixProduct{1}) = apply(H, ψ)

@doc raw"""
    *(α, ψ)
    *(ψ, α)
    /(ψ, α)
    -(ψ)

Scale a matrix-product object by a number.

Only one site tensor is touched — the orthogonality centre when there is one — so the gauge
label survives.

# Arguments
- `α`: the scalar.
- `ψ`: an [`AbstractMatrixProduct`](@ref).

# Returns
- `φ`: a scaled copy of `ψ`, with the element type promoted as needed.
"""
function Base.:*(α::Number, ψ::AbstractMatrixProduct{N}) where {N}
    S = promote_type(typeof(α), scalartype(ψ))
    c = first(center(ψ))
    # Multiplying by `one(S)` promotes the element type through plain broadcasting, so no
    # `convert` is needed and the whole thing stays traceable on any backend.
    tensors = [ψ[i] .* (i == c ? α * one(S) : one(S)) for i in 1:length(ψ)]
    return MatrixProduct{N}(tensors, center(ψ))
end

Base.:*(ψ::AbstractMatrixProduct, α::Number) = α * ψ
Base.:/(ψ::AbstractMatrixProduct, α::Number) = inv(α) * ψ
Base.:-(ψ::AbstractMatrixProduct) = -one(scalartype(ψ)) * ψ
