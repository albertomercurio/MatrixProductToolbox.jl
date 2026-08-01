#=
Scalar quantities extracted from a matrix-product object: overlaps, norms, expectation
values and entanglement.

Everything here is a sweep of the transfer kernels from `kernels.jl`, plus two idioms that
keep the package free of scalar indexing on device arrays:

  * a 1×1 environment is reduced with `sum(E)`, never `E[1]`;
  * boundary environments are built with `_boundary`, i.e. `fill!(similar(...), one(T))`.
=#

export expect, entanglement_entropy, entanglement_spectrum

function _check_same_sites(ψ::AbstractMatrixProduct, φ::AbstractMatrixProduct)
    length(ψ) == length(φ) || throw(
        DimensionMismatch("Got $(length(ψ)) sites and $(length(φ)) sites.")
    )
    return nothing
end

@doc raw"""
    dot(ψ, φ)

The overlap ``\langle \psi | \phi \rangle``, conjugating `ψ` as `LinearAlgebra.dot` does.

Contracts the full network from left to right; neither argument needs to be canonical.

# Arguments
- `ψ`: the bra, an [`AbstractMatrixProduct`](@ref); it is conjugated.
- `φ`: the ket, with the same number of sites and the same physical dimensions.

# Returns
- `z`: the overlap, of the promoted element type.

# Examples
```jldoctest
julia> ψ = product_mps([2, 2], [1, 1]); φ = product_mps([2, 2], [1, 2]);

julia> dot(ψ, ψ) ≈ 1 && abs(dot(ψ, φ)) < 1e-14
true
```
"""
function dot(ψ::AbstractMatrixProduct{N}, φ::AbstractMatrixProduct{N}) where {N}
    _check_same_sites(ψ, φ)
    T = promote_type(scalartype(ψ), scalartype(φ))
    E = _boundary(φ[1], T, (1, 1))
    for i in 1:length(ψ)
        E = transfer_left(E, _fuse_physical(φ[i]), _fuse_physical(ψ[i]))
    end
    return sum(E)
end

@doc raw"""
    norm(ψ)

The Euclidean norm ``\sqrt{\langle \psi | \psi \rangle}``.

# Arguments
- `ψ`: an [`AbstractMatrixProduct`](@ref).

# Notes
- When `ψ` is canonical the norm is just the norm of the centre tensor, so this is `O(χ²d)`.
  Otherwise the full network is contracted.

# Returns
- `n::Real`: the norm.
"""
function norm(ψ::AbstractMatrixProduct)
    iscanonical(ψ) && return norm(ψ[only(center(ψ))])
    return sqrt(abs(real(dot(ψ, ψ))))
end

@doc raw"""
    normalize!(ψ)

Rescale `ψ` to unit norm, in place.

Canonicalises first if needed, then divides the centre tensor by its norm — so only one site
tensor is touched.

# Arguments
- `ψ`: the matrix-product object to normalise.

# Returns
- `ψ`: the same object, mutated.
"""
function normalize!(ψ::AbstractMatrixProduct)
    iscanonical(ψ) || canonicalize!(ψ, 1)
    c = only(center(ψ))
    n = norm(ψ[c])
    iszero(n) && throw(ArgumentError("Cannot normalise a matrix-product object of zero norm."))
    setsite!(ψ, c, ψ[c] ./ n)
    return ψ
end

@doc raw"""
    normalize(ψ)

Copying version of [`normalize!`](@ref).

# Arguments
- `ψ`: an [`AbstractMatrixProduct`](@ref).

# Returns
- `φ`: a normalised copy of `ψ`.
"""
normalize(ψ::AbstractMatrixProduct) = normalize!(copy(ψ))

@doc raw"""
    expect(H, ψ)
    expect(op, ψ, site)
    expect(op, ψ)

Expectation value of an operator in the state `ψ`.

The three forms take, respectively, an [`MPO`](@ref) acting on the whole chain, a single-site
matrix acting on `site`, and a single-site matrix evaluated at every site.

!!! note "Not normalised"
    The result is ``\langle \psi | \hat{O} | \psi \rangle`` without dividing by
    ``\langle \psi | \psi \rangle``. Call [`normalize!`](@ref) first, or divide by
    `norm(ψ)^2`.

# Arguments
- `H`: an [`MPO`](@ref) with the same number of sites as `ψ`.
- `op`: a single-site operator, indexed `[output, input]`.
- `ψ`: an [`MPS`](@ref).
- `site`: the site at which `op` acts.

# Returns
- `z`: the expectation value; a `Vector` of them for the all-sites form.

# Examples
```jldoctest
julia> σz = ComplexF64[1 0; 0 -1];

julia> ψ = product_mps([2, 2, 2], [1, 2, 1]);

julia> real(expect(σz, ψ))
3-element Vector{Float64}:
  1.0
 -1.0
  1.0
```
"""
function expect(H::AbstractMatrixProduct{2}, ψ::AbstractMatrixProduct{1})
    _check_same_sites(H, ψ)
    T = promote_type(scalartype(H), scalartype(ψ))
    E = _boundary(ψ[1], T, (1, 1, 1))
    for i in 1:length(ψ)
        E = transfer_left(E, ψ[i], H[i], ψ[i])
    end
    return sum(E)
end

function expect(op::AbstractMatrix, ψ::AbstractMatrixProduct{1}, site::Integer)
    L = length(ψ)
    (1 ≤ site ≤ L) || throw(ArgumentError("`site` must lie in 1:$L, got $site."))
    d = size(ψ[site], 2)
    size(op) == (d, d) || throw(
        DimensionMismatch("Operator of size $(size(op)) does not match local dimension $d.")
    )
    # Gauging to `site` keeps everything on the caller's backend, unlike padding the operator
    # out to a full-chain MPO with identity matrices.
    if !(iscanonical(ψ) && only(center(ψ)) == site)
        return expect(op, canonicalize(ψ, site), site)
    end
    t = ψ[site]
    @tensor y[a, s′, b] := op[s′, s] * t[a, s, b]
    return dot(t, y)
end

function expect(op::AbstractMatrix, ψ::AbstractMatrixProduct{1})
    L = length(ψ)
    φ = canonicalize(ψ, 1)
    first_val = expect(op, φ, 1)
    vals = Vector{typeof(first_val)}(undef, L)
    vals[1] = first_val
    for i in 2:L
        canonicalize!(φ, i)
        vals[i] = expect(op, φ, i)
    end
    return vals
end

@doc raw"""
    entanglement_spectrum(ψ, bond; factorization = SVDFactorization())

Schmidt coefficients across `bond`, normalised so that ``\sum_i \lambda_i^2 = 1``.

`bond` is the cut between sites `bond` and `bond + 1`, so it ranges over `1:length(ψ)-1`.

# Arguments
- `ψ`: an [`AbstractMatrixProduct`](@ref).
- `bond`: the bond to cut at.
- `factorization`: used to *gauge* `ψ` first. The spectrum always comes from a separate
  `svdvals`, so [`QRFactorization`](@ref) is a free speed-up here.

# Returns
- `λ`: the Schmidt coefficients in descending order.
"""
function entanglement_spectrum(
        ψ::AbstractMatrixProduct, bond::Integer;
        factorization::AbstractFactorization = SVDFactorization()
    )
    L = length(ψ)
    (1 ≤ bond ≤ L - 1) || throw(ArgumentError("`bond` must lie in 1:$(L - 1), got $bond."))
    φ = canonicalize(ψ, bond; factorization)
    t = φ[bond]
    s = svdvals(reshape(t, :, size(t, ndims(t))))
    n = norm(s)
    return iszero(n) ? s : s ./ n
end

@doc raw"""
    entanglement_entropy(ψ, bond; factorization = SVDFactorization())

Von Neumann entanglement entropy across `bond`, in nats.

Computed as ``S = -\sum_i \lambda_i^2 \log \lambda_i^2`` from the normalised
[`entanglement_spectrum`](@ref), so it is insensitive to the norm of `ψ`.

# Arguments
- `ψ`: an [`AbstractMatrixProduct`](@ref).
- `bond`: the bond to cut at, in `1:length(ψ)-1`.
- `factorization`: forwarded to [`entanglement_spectrum`](@ref).

# Returns
- `S::Real`: the entropy in nats; divide by `log(2)` for bits.

# Examples
```jldoctest
julia> bell = MPS([reshape(ComplexF64[1, 0, 0, 1] ./ sqrt(2), 1, 2, 2), reshape(ComplexF64[1, 0, 0, 1], 2, 2, 1)]);

julia> entanglement_entropy(bell, 1) ≈ log(2)
true
```
"""
function entanglement_entropy(
        ψ::AbstractMatrixProduct, bond::Integer;
        factorization::AbstractFactorization = SVDFactorization()
    )
    p = abs2.(entanglement_spectrum(ψ, bond; factorization))
    return -sum(x -> x > zero(x) ? x * log(x) : zero(x), p)
end
