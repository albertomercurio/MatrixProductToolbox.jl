#=
Two-site TDVP, and the real-time driver built on it.

One step of `dt` is the symmetric second-order composition:

    left to right, over bonds 1 … L-1:
        evolve the two-site block forward by τ = -i·dt/2
        split; then evolve the *one-site* centre it leaves behind backward by -τ
    right to left, over bonds L-1 … 1:
        the mirror image

Each half-sweep is first order on its own; running one and then its reverse makes the composition
symmetric, hence second order in `dt`. The backward one-site step is not a correction bolted on:
it removes the double counting of the bond that two consecutive two-site blocks share, which is
what the projector-splitting derivation of TDVP asks for.

Imaginary time is the same code with a real `τ`, which is why `dmrg` and `tdvp` can be used to
cross-check each other on the same Hamiltonian.

Unlike DMRG the singular values are *not* renormalised after a truncation: for real time the norm
that a truncation costs is a genuine measure of how much of the state was thrown away, and hiding
it would hide the error. `normalize = true` opts into rescaling, and is the default in imaginary
time, where the norm decays by construction.
=#

export tdvp, tdvp!, TDVPSolution

@doc raw"""
    struct TDVPSolution{Tt,S,E,T}

What [`tdvp`](@ref) returns. The field names follow QuantumToolbox's `TimeEvolutionSol`.

# Fields (Attributes)
- `times::Vector{Tt}`: the time points, i.e. the `tlist` that was asked for.
- `states::Vector{S}`: the state at each time when `save_states` is set, otherwise just the last.
- `expect::Matrix{E}`: `expect[k, n]` is the `k`-th observable at `times[n]`.
- `truncation_errors::Vector{T}`: the largest relative discarded weight of each step.
"""
struct TDVPSolution{Tt, S, E, T}
    times::Vector{Tt}
    states::Vector{S}
    expect::Matrix{E}
    truncation_errors::Vector{T}
end

function Base.show(io::IO, sol::TDVPSolution)
    print(
        io, "TDVPSolution(times=", length(sol.times),
        ", states=", length(sol.states),
        ", e_ops=", size(sol.expect, 1),
        ", truncation error=",
        isempty(sol.truncation_errors) ? "-" : maximum(sol.truncation_errors), ")"
    )
    return nothing
end

Base.show(io::IO, ::MIME"text/plain", sol::TDVPSolution) = show(io, sol)

# The half-step generator. Real for imaginary time, so a real state stays real.
_tdvp_tau(dt::Real, imaginary::Bool) = imaginary ? -dt / 2 : -im * dt / 2

# A step of `τ` must not widen the state's element type, or `setsite!` would be handed a tensor
# the container cannot hold. Comparing promoted *types* rather than values keeps this a host
# question on every backend, including one where `scalartype` is a traced number.
function _check_step_type(ψ::AbstractMatrixProduct, τ::Number)
    T = scalartype(ψ)
    promote_type(T, typeof(τ)) === T && return nothing
    return throw(
        ArgumentError(
            "Evolving a state of element type $T by a step of type $(typeof(τ)) would widen it " *
                "to $(promote_type(T, typeof(τ))). Real-time TDVP needs a complex state — build " *
                "it with e.g. `rand_mps(ComplexF64, …)` — while imaginary time (`imaginary = " *
                "true`) works in either."
        )
    )
end

# Evolve the one-site orthogonality centre at site `j` by `τ`, in place.
function _tdvp_backward!(
        ψ::AbstractMatrixProduct{1}, H::AbstractMatrixProduct{2}, env::Environments,
        solver::AbstractExpSolver, τ::Number, j::Integer
    )
    Lenv = leftenv(env, H, ψ, j)
    Renv = rightenv(env, H, ψ, j + 1)
    W = H[j]
    C = krylov_exp(solver, τ, x -> apply_effective(Lenv, W, x, Renv), ψ[j])
    setsite!(ψ, j, C)
    invalidate!(env, j)
    return ψ
end

# One bond of a half-sweep: evolve the two-site block forward, split, and evolve the one-site
# centre back again — except at the far end of the sweep, where there is no next block to hand
# the bond on to.
function _tdvp_update!(
        ψ::AbstractMatrixProduct{1}, H::AbstractMatrixProduct{2}, env::Environments,
        solver::AbstractExpSolver, τ::Number, trunc::AbstractTruncation,
        alg::AbstractFactorization, i::Integer, direction::Union{Val{:left}, Val{:right}}
    )
    Lenv = leftenv(env, H, ψ, i)
    Renv = rightenv(env, H, ψ, i + 2)
    Wl, Wr = H[i], H[i + 1]
    Θ = _merge_sites(ψ[i], ψ[i + 1])
    Θ = krylov_exp(solver, τ, x -> apply_effective(Lenv, Wl, Wr, x, Renv), Θ)
    ε = _split_twosite!(ψ, i, Θ, trunc, alg, direction)
    invalidate!(env, i)
    invalidate!(env, i + 1)

    L = length(ψ)
    if direction isa Val{:right}
        i < L - 1 && _tdvp_backward!(ψ, H, env, solver, -τ, i + 1)
    else
        i > 1 && _tdvp_backward!(ψ, H, env, solver, -τ, i)
    end
    return ε
end

function _tdvp_halfsweep!(
        ψ::AbstractMatrixProduct{1}, H::AbstractMatrixProduct{2}, env::Environments,
        solver::AbstractExpSolver, τ::Number, trunc::AbstractTruncation,
        alg::AbstractFactorization, direction::Union{Val{:left}, Val{:right}}
    )
    L = length(ψ)
    bonds = direction isa Val{:right} ? (1:(L - 1)) : ((L - 1):-1:1)
    ε = _tdvp_update!(ψ, H, env, solver, τ, trunc, alg, first(bonds), direction)
    for i in Iterators.drop(bonds, 1)
        ε = max(ε, _tdvp_update!(ψ, H, env, solver, τ, trunc, alg, i, direction))
    end
    return ε
end

# One full step of `dt`.
function _tdvp_step!(ψ, H, env, solver, dt::Real, trunc, alg, imaginary::Bool)
    τ = _tdvp_tau(dt, imaginary)
    ε₁ = _tdvp_halfsweep!(ψ, H, env, solver, τ, trunc, alg, Val(:right))
    ε₂ = _tdvp_halfsweep!(ψ, H, env, solver, τ, trunc, alg, Val(:left))
    return max(ε₁, ε₂)
end

function _tdvp_log(step::Integer, nsteps::Integer, t)
    print(stderr, "\r[TDVP] step ", lpad(step, 5), "/", nsteps, "   t = ", t, "     ")
    step == nsteps && print(stderr, "\n")
    return nothing
end

# One column of the observable table. The element type is named rather than inferred so that an
# empty `e_ops` still produces a typed (zero-row) column, and so that a traced backend produces
# traced entries rather than a `Vector{Any}`.
_expect_column(e_ops, ψ, ::Type{E}) where {E} = E[expect(op, ψ) for op in e_ops]

@doc raw"""
    tdvp(H, ψ₀, tlist; e_ops, imaginary, maxdim, cutoff, trunc, solver, factorization,
         normalize, save_states, progress)
    tdvp!(H, ψ₀, tlist; kwargs...)

Time evolution of `ψ₀` under `H` by two-site TDVP.

`tdvp!` evolves `ψ₀` in place; `tdvp` works on a copy. **`tlist` sets the time step**: one
symmetric second-order TDVP step is taken per interval, so the spacing of `tlist` is the
integration step, not merely where output is saved.

# Arguments
- `H`: the generator, an [`MPO`](@ref). It must be Hermitian — the local solver is Lanczos.
- `ψ₀`: the initial [`MPS`](@ref). Real time needs a complex element type.
- `tlist`: increasing times, starting at the initial time.
- `e_ops`: observables, each an [`MPO`](@ref), evaluated at every point of `tlist`.
- `imaginary`: evolve under ``e^{-\tau \hat{H}}`` rather than ``e^{-i t \hat{H}}``. Combined with
  `normalize` this converges to the ground state, which is how it cross-checks [`dmrg`](@ref).
- `maxdim`, `cutoff`: shorthand for `Truncation(; maxdim, cutoff)`.
- `trunc`: an [`AbstractTruncation`](@ref). Pass [`FixedTruncation`](@ref) inside a Reactant
  `@compile` region.
- `solver`: an [`AbstractExpSolver`](@ref) for the local exponentials.
- `factorization`: an [`AbstractFactorization`](@ref) for the splits; it must be able to truncate.
- `normalize`: rescale to unit norm after every step. Defaults to `imaginary`.
- `save_states`: keep the state at every time rather than only the last. Defaults to `true` when
  no `e_ops` were given.
- `progress`: print progress to `stderr`.

# Notes
- Expectation values follow this package's convention and are **not** divided by
  ``\langle\psi|\psi\rangle``. With `normalize = true`, or in real time where the norm is
  conserved up to truncation, the distinction does not arise.
- Two-site TDVP grows bond dimensions on its own until `trunc` binds, so it does not suffer the
  projection error that traps one-site TDVP in the starting manifold.
- Imaginary time suppresses an excited state only as ``e^{-2\tau\Delta}``, so on a chain with a
  small gap it converges slowly and the residual energy is set by ``\tau``, not by the time step
  or the bond dimension. [`dmrg`](@ref) has no such dependence and is the better ground-state
  method; imaginary time earns its place as an independent check on it.

# Returns
- `sol::TDVPSolution`.

# Examples
```jldoctest
julia> H = product_mpo([ComplexF64[1 0; 0 -1] for _ in 1:4]);   # σᶻ₁ ⊗ σᶻ₂ ⊗ σᶻ₃ ⊗ σᶻ₄

julia> ψ₀ = product_mps([2, 2, 2, 2], [1, 2, 1, 2]);

julia> sol = tdvp(H, ψ₀, 0.0:0.05:0.2; e_ops = (H,), progress = false);

julia> all(z -> isapprox(z, 0; atol = 1e-10), sol.expect .- sol.expect[1, 1])
true
```
"""
function tdvp(
        H::AbstractMatrixProduct{2}, ψ₀::AbstractMatrixProduct{1}, tlist::AbstractVector; kwargs...
    )
    return tdvp!(H, copy(ψ₀), tlist; kwargs...)
end

@doc (@doc tdvp) function tdvp!(
        H::AbstractMatrixProduct{2}, ψ::AbstractMatrixProduct{1}, tlist::AbstractVector;
        e_ops = (),
        imaginary::Bool = false,
        maxdim::Integer = typemax(Int), cutoff::Real = 1.0e-12,
        trunc::AbstractTruncation = Truncation(maxdim, cutoff),
        solver::AbstractExpSolver = LanczosExponential(),
        factorization::AbstractFactorization = SVDFactorization(),
        normalize::Bool = imaginary,
        save_states::Bool = isempty(e_ops),
        progress::Bool = true
    )
    L = length(ψ)
    L ≥ 2 || throw(ArgumentError("Two-site TDVP needs at least 2 sites, got $L."))
    L == length(H) || throw(
        DimensionMismatch("Generator has $(length(H)) sites but the state has $L.")
    )
    length(tlist) ≥ 2 || throw(
        ArgumentError("`tlist` needs at least two entries, got $(length(tlist)).")
    )
    issorted(tlist) || throw(ArgumentError("`tlist` must be increasing."))
    all(op -> length(op) == L, e_ops) || throw(
        DimensionMismatch("Every entry of `e_ops` must have the same number of sites as the state.")
    )
    (_supports_truncation(factorization) || trunc isa NoTruncation) || throw(
        ArgumentError(
            "`tdvp` was given $(typeof(factorization)), which cannot truncate, together with " *
                "$(typeof(trunc)). The two-site update grows every bond, so the split has to be " *
                "able to discard: use `SVDFactorization()`."
        )
    )
    times = collect(float(eltype(tlist)), tlist)
    _check_step_type(ψ, _tdvp_tau(one(eltype(times)), imaginary))

    canonicalize!(ψ, 1; factorization)
    normalize && _normalize_center!(ψ)
    env = Environments(H, ψ)

    E = promote_type(scalartype(H), scalartype(ψ))
    nsteps = length(times) - 1
    every = max(1, nsteps ÷ 100)
    columns = [_expect_column(e_ops, ψ, E)]
    states = [copy(ψ)]

    # One step, including everything recorded afterwards. Closed over rather than inlined so that
    # the first step can be peeled off the loop below: `truncation_errors` takes its element type
    # from a computed value, which under tracing is a traced number and not a `Float64`.
    function step!(step::Int)
        ε = _tdvp_step!(ψ, H, env, solver, times[step + 1] - times[step], trunc, factorization, imaginary)
        canonicalize!(ψ, 1; factorization)
        normalize && _normalize_center!(ψ)
        push!(columns, _expect_column(e_ops, ψ, E))
        save_states ? push!(states, copy(ψ)) : (states[1] = copy(ψ))
        progress && (step % every == 0 || step == nsteps) && _tdvp_log(step, nsteps, times[step + 1])
        return ε
    end

    errors = [step!(1)]
    for step in 2:nsteps
        push!(errors, step!(step))
    end

    return TDVPSolution(times, states, reduce(hcat, columns), errors)
end
