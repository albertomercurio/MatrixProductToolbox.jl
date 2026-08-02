#=
The environment cache: the partial contractions of ⟨ψ|Ĥ|ψ⟩ that a sweeping algorithm would
otherwise rebuild from scratch at every site.

`left[i]` covers sites `1 … i-1` and `right[i]` covers sites `i … L`, which is exactly the
convention the reference sweeps in `test/testutils.jl` use, and exactly the pairing
`test/kernels.jl` asserts for `apply_effective`:

    one site  at i        ->  leftenv(…, i),  rightenv(…, i+1)
    two sites at (i, i+1) ->  leftenv(…, i),  rightenv(…, i+2)

Validity is two integers rather than a flag per slot. `left[1:nleft]` and `right[nright:L+1]`
are the entries known to be current; everything outside is garbage and is recomputed on demand.
Two integers suffice because the valid entries are always a prefix and a suffix: a left
environment can only be built from the one to its left, so invalidating `left[k]` invalidates
everything after it too.
=#

export Environments, leftenv, rightenv

@doc raw"""
    mutable struct Environments{A<:AbstractArray}
        left::Vector{A}
        right::Vector{A}
        nleft::Int
        nright::Int
    end

    Environments(H, ψ)

Cache of the partial contractions of ``\langle \psi | \hat{H} | \psi \rangle``.

Entry `left[i]` covers sites `1:i-1` and entry `right[i]` covers sites `i:L`, so both vectors
have length `L + 1` and the two boundary entries are the `1×1×1` environments. They are indexed
`[bra bond, MPO bond, ket bond]` like every other environment in the package.

Use [`leftenv`](@ref) and [`rightenv`](@ref) to read an entry — they extend the cache as far as
needed and no further — and `MatrixProductToolbox.invalidate!` to declare a site changed.

# Fields (Attributes)
- `left::Vector{A}`: `left[i]` contracts sites `1:i-1`.
- `right::Vector{A}`: `right[i]` contracts sites `i:L`.
- `nleft::Int`: `left[1:nleft]` are current.
- `nright::Int`: `right[nright:L+1]` are current.

# Examples
```jldoctest
julia> ψ = rand_mps(ComplexF64, fill(2, 4); maxdim = 4);

julia> H = product_mpo([ComplexF64[1 0; 0 -1] for _ in 1:4]);

julia> env = Environments(H, ψ);

julia> size(leftenv(env, H, ψ, 1))
(1, 1, 1)

julia> sum(leftenv(env, H, ψ, 5)) ≈ expect(H, ψ)
true
```
"""
mutable struct Environments{A <: AbstractArray}
    left::Vector{A}
    right::Vector{A}
    nleft::Int
    nright::Int
end

function Environments(H::AbstractMatrixProduct{2}, ψ::AbstractMatrixProduct{1})
    L = length(ψ)
    L == length(H) || throw(
        DimensionMismatch("Operator has $(length(H)) sites but the state has $L.")
    )
    T = promote_type(scalartype(H), scalartype(ψ))
    boundary = _boundary(ψ[1], T, (1, 1, 1))
    left = Vector{typeof(boundary)}(undef, L + 1)
    right = Vector{typeof(boundary)}(undef, L + 1)
    left[1] = boundary
    right[L + 1] = boundary
    return Environments(left, right, 1, L + 1)
end

Base.length(env::Environments) = length(env.left) - 1

function Base.show(io::IO, env::Environments)
    print(
        io, "Environments(L=", length(env),
        ", left valid through ", env.nleft, ", right valid from ", env.nright, ")"
    )
    return nothing
end

Base.show(io::IO, ::MIME"text/plain", env::Environments) = show(io, env)

function _check_env_site(env::Environments, i::Integer)
    L = length(env)
    return (1 ≤ i ≤ L + 1) || throw(
        ArgumentError("An environment index must lie in 1:$(L + 1), got $i.")
    )
end

@doc raw"""
    leftenv(env, H, ψ, i)

The environment covering sites `1:i-1`, extending the cache only as far as needed.

# Arguments
- `env`: an [`Environments`](@ref) cache.
- `H`: the [`MPO`](@ref).
- `ψ`: the [`MPS`](@ref).
- `i`: the site whose left environment is wanted, in `1:length(ψ)+1`.

# Returns
- `L`: the environment, indexed `[bra bond, MPO bond, ket bond]`.
"""
function leftenv(
        env::Environments, H::AbstractMatrixProduct{2}, ψ::AbstractMatrixProduct{1}, i::Integer
    )
    _check_env_site(env, i)
    for k in env.nleft:(i - 1)
        env.left[k + 1] = transfer_left(env.left[k], ψ[k], H[k], ψ[k])
    end
    env.nleft = max(env.nleft, Int(i))
    return env.left[i]
end

@doc raw"""
    rightenv(env, H, ψ, i)

The environment covering sites `i:length(ψ)`, extending the cache only as far as needed.

# Arguments
- `env`: an [`Environments`](@ref) cache.
- `H`: the [`MPO`](@ref).
- `ψ`: the [`MPS`](@ref).
- `i`: the first site the environment covers, in `1:length(ψ)+1`.

# Returns
- `R`: the environment, indexed `[bra bond, MPO bond, ket bond]`.
"""
function rightenv(
        env::Environments, H::AbstractMatrixProduct{2}, ψ::AbstractMatrixProduct{1}, i::Integer
    )
    _check_env_site(env, i)
    for k in env.nright:-1:(i + 1)
        env.right[k - 1] = transfer_right(env.right[k], ψ[k - 1], H[k - 1], ψ[k - 1])
    end
    env.nright = min(env.nright, Int(i))
    return env.right[i]
end

@doc raw"""
    invalidate!(env, j)

Declare that site `j` of the state has changed.

Every left environment that contracts site `j` — `left[j+1]` and beyond — and every right
environment that contracts it — `right[j]` and leftwards — is dropped. Nothing is recomputed
here; the next [`leftenv`](@ref) or [`rightenv`](@ref) rebuilds only what it is asked for.

# Arguments
- `env`: an [`Environments`](@ref) cache.
- `j`: the site that changed.

# Returns
- `env`: the same cache, mutated.
"""
function invalidate!(env::Environments, j::Integer)
    env.nleft = min(env.nleft, Int(j))
    env.nright = max(env.nright, Int(j) + 1)
    return env
end
