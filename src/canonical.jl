#=
Gauge transformations: moving the orthogonality centre, and truncating every bond.

The whole file rests on the `center::UnitRange` invariant. Because a freshly built object
carries `1:L` — "nothing claimed" — rather than a sentinel, moving the centre needs no
special case for an ungauged object: the two loops below simply run over the full chain.
=#

export canonicalize, canonicalize!, truncate!
export isleftorthogonal, isrightorthogonal

# Push the right factor of site `i` into site `i+1`, leaving site `i` left-orthogonal.
function _orthogonalize_left!(
        ψ::AbstractMatrixProduct{N}, i::Integer,
        trunc::AbstractTruncation, alg::AbstractFactorization
    ) where {N}
    t = ψ[i]
    χl = size(t, 1)
    dims = _physsize(t, Val(N))
    Q, R = factorize_left(alg, trunc, reshape(t, :, size(t, N + 2)))
    setsite!(ψ, i, reshape(Q, χl, dims..., size(Q, 2)))
    setsite!(ψ, i + 1, _absorb_left(R, ψ[i + 1]))
    return ψ
end

# Push the left factor of site `i` into site `i-1`, leaving site `i` right-orthogonal.
function _orthogonalize_right!(
        ψ::AbstractMatrixProduct{N}, i::Integer,
        trunc::AbstractTruncation, alg::AbstractFactorization
    ) where {N}
    t = ψ[i]
    χr = size(t, N + 2)
    dims = _physsize(t, Val(N))
    L, Q = factorize_right(alg, trunc, reshape(t, size(t, 1), :))
    setsite!(ψ, i, reshape(Q, size(Q, 1), dims..., χr))
    setsite!(ψ, i - 1, _absorb_right(ψ[i - 1], L))
    return ψ
end

# t′[r, p…, a] = Σ_l R[r, l] * t[l, p…, a]  — reshape and one matrix multiply, nothing else.
function _absorb_left(R::AbstractMatrix, t::AbstractArray)
    rest = size(t)[2:end]
    return reshape(R * reshape(t, size(t, 1), :), size(R, 1), rest...)
end

# t′[l, p…, k] = Σ_r t[l, p…, r] * L[r, k]
function _absorb_right(t::AbstractArray, L::AbstractMatrix)
    n = ndims(t)
    front = size(t)[1:(n - 1)]
    return reshape(reshape(t, :, size(t, n)) * L, front..., size(L, 2))
end

#=
Two-site blocks, for DMRG and TDVP.

Merging and splitting are the same shape of operation as the two absorptions above -- a reshape
and one matrix multiply -- which is why they live here rather than in `kernels.jl`. The index
order `[l, p…, r]` makes every reshape below contiguous, so no `permutedims` appears here either,
and `kernels.jl` stays at the six contractions it claims.
=#

# Θ[l, p…, q…, r] = Σ_m A[l, p…, m] * B[m, q…, r].
function _merge_sites(A::AbstractArray, B::AbstractArray)
    front = size(A)[1:(ndims(A) - 1)]
    back = size(B)[2:end]
    M = reshape(A, :, size(A, ndims(A))) * reshape(B, size(B, 1), :)
    return reshape(M, front..., back...)
end

# The matricization a two-site block is split across: left bond and the first site's physical
# legs against the second site's physical legs and the right bond.
function _twosite_matrix(Θ::AbstractArray, ::Val{N}) where {N}
    dims1 = ntuple(k -> size(Θ, k + 1), Val(N))
    return reshape(Θ, size(Θ, 1) * prod(dims1), :)
end

# Split a two-site block back onto sites `i` and `i+1`, in place, returning the relative discarded
# weight. `direction` says which way the sweep is moving: the orthogonality centre is left on site
# `i+1` for `Val(:right)` and on site `i` for `Val(:left)`, so the site the sweep is leaving behind
# is the one that becomes orthogonal.
function _split_twosite!(
        ψ::AbstractMatrixProduct{N}, i::Integer, Θ::AbstractArray,
        trunc::AbstractTruncation, alg::AbstractFactorization, ::Val{:right}
    ) where {N}
    χl, χr = size(Θ, 1), size(Θ, ndims(Θ))
    dims1 = ntuple(k -> size(Θ, k + 1), Val(N))
    dims2 = ntuple(k -> size(Θ, k + 1 + N), Val(N))
    Q, R, ε = _factorize_left(alg, trunc, _twosite_matrix(Θ, Val(N)))
    setsite!(ψ, i, reshape(Q, χl, dims1..., size(Q, 2)))
    setsite!(ψ, i + 1, reshape(R, size(R, 1), dims2..., χr))
    setcenter!(ψ, Int(i) + 1)
    return ε
end

function _split_twosite!(
        ψ::AbstractMatrixProduct{N}, i::Integer, Θ::AbstractArray,
        trunc::AbstractTruncation, alg::AbstractFactorization, ::Val{:left}
    ) where {N}
    χl, χr = size(Θ, 1), size(Θ, ndims(Θ))
    dims1 = ntuple(k -> size(Θ, k + 1), Val(N))
    dims2 = ntuple(k -> size(Θ, k + 1 + N), Val(N))
    L, Q, ε = _factorize_right(alg, trunc, _twosite_matrix(Θ, Val(N)))
    setsite!(ψ, i, reshape(L, χl, dims1..., size(L, 2)))
    setsite!(ψ, i + 1, reshape(Q, size(Q, 1), dims2..., χr))
    setcenter!(ψ, Int(i))
    return ε
end

# Scale the orthogonality centre to unit norm, without the zero-norm check `normalize!` makes.
# That check is `iszero(n) && throw(…)` — a branch on a value, which a traced backend cannot
# express — so the sweeping algorithms, which renormalise after every truncation, divide by a
# floored norm instead. Requires a single-site centre, which is what a finished sweep leaves.
function _normalize_center!(ψ::AbstractMatrixProduct)
    c = only(center(ψ))
    setsite!(ψ, c, ψ[c] ./ _floor_norm(_norm2(ψ[c])))
    return ψ
end

@doc raw"""
    canonicalize!(ψ, c = 1; trunc = NoTruncation(), factorization = SVDFactorization())

Move the orthogonality centre of `ψ` to site `c`, in place.

Afterwards sites `1:c-1` are left-orthogonal, sites `c+1:end` are right-orthogonal, site `c`
carries the norm, and [`center`](@ref)`(ψ) == c:c`. With the default `trunc` this is an exact
gauge transformation: it leaves the represented state untouched.

# Arguments
- `ψ`: the matrix-product object to gauge.
- `c`: the target site for the orthogonality centre.
- `trunc`: an [`AbstractTruncation`](@ref) policy. Only bonds the sweep crosses are affected,
  so use [`truncate!`](@ref) to truncate every bond.
- `factorization`: an [`AbstractFactorization`](@ref). The default `trunc` discards nothing, so
  [`QRFactorization`](@ref) applies here and is 2–3× faster.

# Notes
- Only the bonds between the current centre and `c` are touched, so repeatedly moving the
  centre one site at a time costs one factorization per step.
- A freshly constructed object has `center == 1:L`, so the first call sweeps the whole chain.

# Returns
- `ψ`: the same object, mutated.

# Examples
```jldoctest
julia> ψ = rand_mps(ComplexF64, [2, 2, 2, 2]; maxdim = 4);

julia> canonicalize!(ψ, 3);

julia> center(ψ)
3:3

julia> isleftorthogonal(ψ[1]) && isrightorthogonal(ψ[4])
true
```
"""
function canonicalize!(
        ψ::AbstractMatrixProduct, c::Integer = 1;
        trunc::AbstractTruncation = NoTruncation(),
        factorization::AbstractFactorization = SVDFactorization()
    )
    L = length(ψ)
    (1 ≤ c ≤ L) || throw(ArgumentError("The orthogonality centre must lie in 1:$L, got $c."))
    lo, hi = first(center(ψ)), last(center(ψ))
    for i in lo:(c - 1)
        _orthogonalize_left!(ψ, i, trunc, factorization)
    end
    for i in hi:-1:(c + 1)
        _orthogonalize_right!(ψ, i, trunc, factorization)
    end
    setcenter!(ψ, Int(c):Int(c))
    return ψ
end

@doc raw"""
    canonicalize(ψ, c = 1; kwargs...)

Copying version of [`canonicalize!`](@ref).

# Arguments
- `ψ`: the matrix-product object to gauge.
- `c`: the target site for the orthogonality centre.
- `kwargs`: forwarded to [`canonicalize!`](@ref).

# Returns
- `φ`: a gauged copy of `ψ`.
"""
canonicalize(ψ::AbstractMatrixProduct, c::Integer = 1; kwargs...) =
    canonicalize!(copy(ψ), c; kwargs...)

@doc raw"""
    truncate!(ψ; maxdim, cutoff, trunc, to = 1, factorization = SVDFactorization())

Truncate **every** bond of `ψ`, in place.

Sweeps first to the last site to establish a gauge, then back to the first site applying
`trunc` at each bond, and finally moves the centre to `to`. The two-pass structure is what
makes the truncation optimal: each discarded singular value is the true Schmidt coefficient
of the bond.

# Arguments
- `ψ`: the matrix-product object to truncate.
- `maxdim`: convenience shorthand for `Truncation(; maxdim)`.
- `cutoff`: convenience shorthand for `Truncation(; cutoff)`.
- `trunc`: an [`AbstractTruncation`](@ref). Defaults to `Truncation(maxdim, cutoff)`; pass
  [`FixedTruncation`](@ref) explicitly for a Reactant-compatible sweep.
- `to`: the site where the orthogonality centre is left.
- `factorization`: an [`AbstractFactorization`](@ref).

# Notes
- Truncation is not a gauge transformation: it changes the represented state unless every
  discarded singular value is zero.
- `factorization` must be able to truncate unless `trunc` is [`NoTruncation`](@ref), which rules
  out [`QRFactorization`](@ref). That is checked before anything is mutated, so a rejected
  combination leaves `ψ` untouched rather than half-swept.

# Returns
- `ψ`: the same object, mutated.

# Examples
```jldoctest
julia> ψ = rand_mps(ComplexF64, [2, 2, 2, 2, 2, 2]; maxdim = 8);

julia> bond_dims(ψ)
7-element Vector{Int64}:
 1
 2
 4
 8
 4
 2
 1

julia> bond_dims(truncate!(ψ; maxdim = 3))
7-element Vector{Int64}:
 1
 2
 3
 3
 3
 2
 1
```
"""
function truncate!(
        ψ::AbstractMatrixProduct;
        maxdim::Integer = typemax(Int), cutoff::Real = 0.0,
        trunc::AbstractTruncation = Truncation(maxdim, cutoff),
        to::Integer = 1,
        factorization::AbstractFactorization = SVDFactorization()
    )
    L = length(ψ)
    (1 ≤ to ≤ L) || throw(ArgumentError("The orthogonality centre must lie in 1:$L, got $to."))
    # Ask before mutating anything. Of the three sweeps below only the middle one truncates, so a
    # factorization that cannot truncate would otherwise fail after the first sweep had already
    # regauged `ψ` -- an error and a half-transformed object.
    (_supports_truncation(factorization) || trunc isa NoTruncation) || throw(
        ArgumentError(
            "`truncate!` was given $(typeof(factorization)), which cannot truncate, together " *
                "with $(typeof(trunc)). Use `SVDFactorization()` here; a non-truncating " *
                "factorization belongs in `canonicalize!`, or in `truncate!` with " *
                "`trunc = NoTruncation()`."
        )
    )
    canonicalize!(ψ, L; factorization)
    canonicalize!(ψ, 1; trunc, factorization)
    return canonicalize!(ψ, to; factorization)
end

@doc raw"""
    truncate(ψ; kwargs...)

Copying version of [`truncate!`](@ref).

# Arguments
- `ψ`: the matrix-product object to truncate.
- `kwargs`: forwarded to [`truncate!`](@ref).

# Returns
- `φ`: a truncated copy of `ψ`.
"""
Base.truncate(ψ::AbstractMatrixProduct; kwargs...) = truncate!(copy(ψ); kwargs...)

@doc raw"""
    isleftorthogonal(t; atol = √eps)

Whether a site tensor is left-orthogonal, i.e. `Q' * Q ≈ I` after fusing its left bond with
its physical legs.

# Arguments
- `t`: a site tensor with index order `[l, p…, r]`.
- `atol`: absolute tolerance passed to `isapprox`.

# Returns
- `::Bool`
"""
function isleftorthogonal(t::AbstractArray; atol::Real = sqrt(eps(real(float(eltype(t))))))
    M = reshape(t, :, size(t, ndims(t)))
    return isapprox(M' * M, I; atol = atol)
end

@doc raw"""
    isrightorthogonal(t; atol = √eps)

Whether a site tensor is right-orthogonal, i.e. `Q * Q' ≈ I` after fusing its physical legs
with its right bond.

# Arguments
- `t`: a site tensor with index order `[l, p…, r]`.
- `atol`: absolute tolerance passed to `isapprox`.

# Returns
- `::Bool`
"""
function isrightorthogonal(t::AbstractArray; atol::Real = sqrt(eps(real(float(eltype(t))))))
    M = reshape(t, size(t, 1), :)
    return isapprox(M * M', I; atol = atol)
end
