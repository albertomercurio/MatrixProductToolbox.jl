#=
Two-site DMRG.

The whole algorithm is: merge two neighbouring site tensors, find the lowest eigenvector of the
effective operator that `apply_effective` builds from the cached environments, split the result
back with truncation, and step. Every ingredient already exists — `_merge_sites` and
`_split_twosite!` in `canonical.jl`, `apply_effective` in `kernels.jl`, `Environments` in
`environments.jl`, `krylov_eigen` in `krylov.jl` — so this file is the sweep bookkeeping and
nothing else.

Two-site rather than one-site because the two-site update grows the bond dimension by itself: the
truncation of the merged block is free to keep directions that were not in either site's original
column space. A one-site sweep cannot change a bond dimension at all and needs a subspace
expansion bolted on to escape a bad initial state.

The state stays normalised without any explicit step: `krylov_eigen` returns a unit vector, and
splitting it leaves an isometry beside a centre tensor of unit norm. Only truncation costs norm,
which is why the sweep renormalises the centre afterwards rather than trusting it.
=#

export dmrg, dmrg!, DMRGSolution

@doc raw"""
    struct DMRGSolution{S,E,T}

What [`dmrg`](@ref) returns.

# Fields (Attributes)
- `state::S`: the optimised [`MPS`](@ref), normalised, with its orthogonality centre at site 1.
- `energy::E`: the variational energy of the final local solve.
- `energies::Vector{E}`: the energy after each sweep.
- `truncation_errors::Vector{T}`: the largest relative discarded weight of each sweep.
- `sweeps::Int`: how many sweeps ran, which is less than `nsweeps` when the run converged.
- `converged::Bool`: whether the energy met `tol` for `patience` consecutive sweeps.
"""
struct DMRGSolution{S, E, T}
    state::S
    energy::E
    energies::Vector{E}
    truncation_errors::Vector{T}
    sweeps::Int
    converged::Bool
end

function Base.show(io::IO, sol::DMRGSolution)
    print(
        io, "DMRGSolution(sweeps=", sol.sweeps,
        ", energy=", sol.energy,
        ", truncation error=", isempty(sol.truncation_errors) ? "-" : last(sol.truncation_errors),
        ", converged=", sol.converged, ")"
    )
    return nothing
end

Base.show(io::IO, ::MIME"text/plain", sol::DMRGSolution) = show(io, sol)

# One bond. Returns the local eigenvalue and the discarded weight.
function _dmrg_update!(
        ψ::AbstractMatrixProduct{1}, H::AbstractMatrixProduct{2}, env::Environments,
        solver::AbstractEigenSolver, trunc::AbstractTruncation, alg::AbstractFactorization,
        i::Integer, direction::Union{Val{:left}, Val{:right}}
    )
    Lenv = leftenv(env, H, ψ, i)
    Renv = rightenv(env, H, ψ, i + 2)
    Wl, Wr = H[i], H[i + 1]
    Θ = _merge_sites(ψ[i], ψ[i + 1])
    λ, Θ = krylov_eigen(solver, x -> apply_effective(Lenv, Wl, Wr, x, Renv), Θ)
    ε = _split_twosite!(ψ, i, Θ, trunc, alg, direction)
    # Both sites were rewritten, so every environment that contracts either of them is stale. The
    # next `leftenv`/`rightenv` rebuilds exactly the one step the sweep advanced by.
    invalidate!(env, i)
    invalidate!(env, i + 1)
    return λ, ε
end

# One half-sweep. `direction` is `Val(:right)` (bonds 1 … L-1) or `Val(:left)` (bonds L-1 … 1).
# Returns the last local eigenvalue and the largest discarded weight seen.
#
# The first bond is peeled off the loop so that `λ` and `ε` take their types from a value rather
# than from an initialiser: `zero(real(scalartype(ψ)))` would be wrong under tracing, where the
# scalar type is a traced number rather than a `Float64`.
function _dmrg_halfsweep!(
        ψ::AbstractMatrixProduct{1}, H::AbstractMatrixProduct{2}, env::Environments,
        solver::AbstractEigenSolver, trunc::AbstractTruncation, alg::AbstractFactorization,
        direction::Union{Val{:left}, Val{:right}}
    )
    L = length(ψ)
    bonds = direction isa Val{:right} ? (1:(L - 1)) : ((L - 1):-1:1)
    λ, ε = _dmrg_update!(ψ, H, env, solver, trunc, alg, first(bonds), direction)
    for i in Iterators.drop(bonds, 1)
        λ, εᵢ = _dmrg_update!(ψ, H, env, solver, trunc, alg, i, direction)
        ε = max(ε, εᵢ)
    end
    return λ, ε
end

# A full sweep is right then left, so that it ends where it began.
function _dmrg_sweep!(ψ, H, env, solver, trunc, alg)
    _, ε₁ = _dmrg_halfsweep!(ψ, H, env, solver, trunc, alg, Val(:right))
    λ, ε₂ = _dmrg_halfsweep!(ψ, H, env, solver, trunc, alg, Val(:left))
    _normalize_center!(ψ)
    return λ, max(ε₁, ε₂)
end

function _dmrg_log(sweep::Integer, λ, ε, χ::Integer)
    print(
        stderr, "[DMRG] sweep ", lpad(sweep, 3),
        "   E = ", λ, "   ε = ", ε, "   χ = ", χ, "\n"
    )
    return nothing
end

@doc raw"""
    dmrg(H, ψ₀; nsweeps, maxdim, cutoff, trunc, solver, factorization, tol, patience, verbose)
    dmrg!(H, ψ₀; kwargs...)

Ground state of `H` by two-site DMRG.

`dmrg!` optimises `ψ₀` in place; `dmrg` works on a copy. Each sweep visits every bond left to
right and then right to left, replacing each two-site block by the lowest eigenvector of the
effective operator and truncating on the way out.

# Arguments
- `H`: the Hamiltonian, an [`MPO`](@ref). It must be Hermitian — the local solver is Lanczos.
- `ψ₀`: the starting [`MPS`](@ref). A random state of the target bond dimension is the usual
  choice; see [`rand_mps`](@ref).
- `nsweeps`: maximum number of sweeps.
- `maxdim`, `cutoff`: shorthand for `Truncation(; maxdim, cutoff)`.
- `trunc`: an [`AbstractTruncation`](@ref). Pass [`FixedTruncation`](@ref) inside a Reactant
  `@compile` region.
- `solver`: an [`AbstractEigenSolver`](@ref) for the local eigenproblem.
- `factorization`: an [`AbstractFactorization`](@ref) for the splits. It must be able to
  truncate, so [`QRFactorization`](@ref) is rejected unless `trunc` is
  [`NoTruncation`](@ref).
- `tol`: relative energy change below which a sweep counts as converged. Ignored when `solver`
  is traceable, since the test is a branch on a value.
- `patience`: consecutive converged sweeps required before stopping early.
- `verbose`: print one line per sweep to `stderr`.

# Notes
- `energy` is the Rayleigh quotient of the last local solve, so it is variational but is measured
  *before* that step's truncation. `expect(H, sol.state)` differs from it by the discarded weight.
- The two-site update grows bond dimensions on its own, so `ψ₀` may start far below `maxdim`.

# Returns
- `sol::DMRGSolution`.

# Examples

``-\sum_i \hat{\sigma}^x_i`` as a bond-dimension-2 MPO, whose ground energy is exactly `-L`:

```jldoctest
julia> using Random

julia> W = zeros(ComplexF64, 2, 2, 2, 2);

julia> W[1, :, :, 1] = W[2, :, :, 2] = [1 0; 0 1]; W[1, :, :, 2] = -[0 1; 1 0];

julia> H = MPO([W[1:1, :, :, :], W, W, W, W[:, :, :, 2:2]]);

julia> ψ₀ = rand_mps(ComplexF64, fill(2, 5); maxdim = 4, rng = MersenneTwister(1));

julia> sol = dmrg(H, ψ₀; nsweeps = 5, verbose = false);

julia> isapprox(sol.energy, -5; atol = 1e-8)
true
```
"""
function dmrg(H::AbstractMatrixProduct{2}, ψ₀::AbstractMatrixProduct{1}; kwargs...)
    return dmrg!(H, copy(ψ₀); kwargs...)
end

@doc (@doc dmrg) function dmrg!(
        H::AbstractMatrixProduct{2}, ψ::AbstractMatrixProduct{1};
        nsweeps::Integer = 10,
        maxdim::Integer = typemax(Int), cutoff::Real = 1.0e-12,
        trunc::AbstractTruncation = Truncation(maxdim, cutoff),
        solver::AbstractEigenSolver = LanczosEigen(),
        factorization::AbstractFactorization = SVDFactorization(),
        tol::Real = 1.0e-10,
        patience::Integer = 2,
        verbose::Bool = true
    )
    L = length(ψ)
    L ≥ 2 || throw(ArgumentError("Two-site DMRG needs at least 2 sites, got $L."))
    L == length(H) || throw(
        DimensionMismatch("Operator has $(length(H)) sites but the state has $L.")
    )
    nsweeps ≥ 1 || throw(ArgumentError("`nsweeps` must be at least 1, got $nsweeps."))
    # Ask before mutating anything, exactly as `truncate!` does: a factorization that cannot
    # truncate would otherwise fail part-way through the first sweep, leaving `ψ` half-optimised.
    (_supports_truncation(factorization) || trunc isa NoTruncation) || throw(
        ArgumentError(
            "`dmrg` was given $(typeof(factorization)), which cannot truncate, together with " *
                "$(typeof(trunc)). The two-site update grows every bond, so the split has to be " *
                "able to discard: use `SVDFactorization()`."
        )
    )
    # The convergence test is a comparison of energies, so it goes the way of every other
    # value-dependent branch in this package -- off, when the solver is one that can be traced.
    conv_tol = _traceable(solver) ? 0.0 : Float64(tol)

    canonicalize!(ψ, 1; factorization)
    env = Environments(H, ψ)

    λ, ε = _dmrg_sweep!(ψ, H, env, solver, trunc, factorization)
    energies = [λ]
    errors = [ε]
    verbose && _dmrg_log(1, λ, ε, maxbonddim(ψ))

    converged = false
    streak = 0
    sweeps = 1
    for sweep in 2:nsweeps
        λ, ε = _dmrg_sweep!(ψ, H, env, solver, trunc, factorization)
        push!(energies, λ)
        push!(errors, ε)
        sweeps = sweep
        verbose && _dmrg_log(sweep, λ, ε, maxbonddim(ψ))
        # `conv_tol > 0` is a host comparison of a host field, so the traced path never reaches
        # the energy comparison behind it.
        if conv_tol > 0
            streak = abs(λ - energies[sweep - 1]) ≤ conv_tol * max(abs(λ), one(λ)) ? streak + 1 : 0
            if streak ≥ patience
                converged = true
                break
            end
        end
    end

    canonicalize!(ψ, 1; factorization)
    _normalize_center!(ψ)
    return DMRGSolution(ψ, λ, energies, errors, sweeps, converged)
end
