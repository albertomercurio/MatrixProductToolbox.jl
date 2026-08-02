#=
DMRG is checked against exact diagonalization of `dense(H)`, which is the same trick the
observables use: on a chain short enough to diagonalise, there is nowhere for an index, a
conjugation or a gauge error to hide.

Both the energy *and* the state are compared -- the energy alone would pass for a state that is
merely close in energy, which is exactly what a subtly wrong sweep produces.
=#

# The ground state of `H` by brute force, for chains short enough to afford it.
function _exact_ground(H)
    F = eigen(Hermitian(dense(H)))
    return F.values[1], F.vectors[:, 1]
end

@testset "DMRG finds the ground state ($(T))" for T in (Float64, ComplexF64)
    L = 8
    H = tfim_mpo(T, L; J = 1.0, h = 0.7)
    E₀, gs = _exact_ground(H)
    ψ₀ = rand_mps(T, fill(2, L); maxdim = 4, rng = MersenneTwister(51))
    sol = dmrg(H, ψ₀; nsweeps = 12, maxdim = 32, verbose = false)

    @test sol.energy ≈ E₀ atol = 1.0e-8
    @test abs(dot(gs, dense(sol.state))) ≈ 1 atol = 1.0e-6
    @test norm(sol.state) ≈ 1
    @test center(sol.state) == 1:1
    # `energy` is the last local Rayleigh quotient; with an unbinding truncation it must agree
    # with a full contraction of the state it returns.
    @test real(expect(H, sol.state)) ≈ sol.energy atol = 1.0e-8
    @test sol.converged
    @test sol.sweeps ≤ 12
    @test length(sol.energies) == sol.sweeps
    @test length(sol.truncation_errors) == sol.sweeps
end

@testset "DMRG on a Hamiltonian with a known spectrum" begin
    # -Σᵢ σˣᵢ is a sum of commuting single-site terms: ground energy exactly -L, ground state the
    # product of |+⟩, and ⟨σˣ⟩ = 1 at every site. No reference diagonalization needed at all, so
    # a sign or conjugation error has nothing to hide behind.
    L = 6
    H = tfim_mpo(ComplexF64, L; J = 0.0, h = 1.0)
    ψ₀ = rand_mps(ComplexF64, fill(2, L); maxdim = 4, rng = MersenneTwister(52))
    sol = dmrg(H, ψ₀; nsweeps = 6, maxdim = 8, verbose = false)
    @test sol.energy ≈ -L atol = 1.0e-10
    @test real(expect(σx, sol.state)) ≈ fill(1.0, L) atol = 1.0e-6
end

@testset "the energy is variational and decreases" begin
    L = 8
    H = tfim_mpo(ComplexF64, L; J = 1.0, h = 0.9)
    E₀, _ = _exact_ground(H)
    ψ₀ = rand_mps(ComplexF64, fill(2, L); maxdim = 2, rng = MersenneTwister(53))
    sol = dmrg(H, ψ₀; nsweeps = 8, maxdim = 8, tol = 0.0, verbose = false)

    @test all(≥(E₀ - 1.0e-9), sol.energies)
    # Monotone up to the truncation, which is the only thing that can raise the energy again.
    @test all(sol.energies[i + 1] ≤ sol.energies[i] + 1.0e-8 for i in 1:(length(sol.energies) - 1))
    @test length(sol.energies) == 8            # `tol = 0` disables the early exit
    @test !sol.converged
end

@testset "the two-site update grows the bond dimension" begin
    # A one-site sweep could never do this: starting at χ = 1 it would stay there. This is the
    # whole reason the two-site variant is the one implemented.
    L = 8
    H = tfim_mpo(ComplexF64, L; J = 1.0, h = 0.5)
    ψ₀ = product_mps(fill(2, L), ones(Int, L))
    @test maxbonddim(ψ₀) == 1
    sol = dmrg(H, ψ₀; nsweeps = 10, maxdim = 16, verbose = false)
    @test maxbonddim(sol.state) > 1
    @test sol.energy ≈ _exact_ground(H)[1] atol = 1.0e-8
end

@testset "truncation is respected and reported" begin
    L = 10
    H = tfim_mpo(ComplexF64, L; J = 1.0, h = 1.0)
    ψ₀ = rand_mps(ComplexF64, fill(2, L); maxdim = 4, rng = MersenneTwister(54))

    sol = dmrg(H, ψ₀; nsweeps = 4, trunc = FixedTruncation(4), verbose = false)
    @test maxbonddim(sol.state) ≤ 4
    @test all(≥(0), sol.truncation_errors)

    # A binding truncation discards something; an unbinding one discards nothing.
    tight = dmrg(H, ψ₀; nsweeps = 4, trunc = FixedTruncation(2), verbose = false)
    loose = dmrg(H, ψ₀; nsweeps = 4, maxdim = 64, cutoff = 0.0, verbose = false)
    @test maximum(tight.truncation_errors) > maximum(loose.truncation_errors)
    @test tight.energy > loose.energy           # a smaller manifold cannot do better
end

@testset "`dmrg` copies and `dmrg!` mutates" begin
    L = 6
    H = tfim_mpo(ComplexF64, L)
    ψ₀ = rand_mps(ComplexF64, fill(2, L); maxdim = 4, rng = MersenneTwister(55))
    reference = dense(ψ₀)

    sol = dmrg(H, ψ₀; nsweeps = 3, maxdim = 8, verbose = false)
    @test dense(ψ₀) ≈ reference                 # untouched
    @test !(sol.state === ψ₀)

    sol! = dmrg!(H, ψ₀; nsweeps = 3, maxdim = 8, verbose = false)
    @test sol!.state === ψ₀
    @test sol!.energy ≈ sol.energy
end

@testset "argument checking" begin
    H = tfim_mpo(ComplexF64, 4)
    ψ = rand_mps(ComplexF64, fill(2, 4); maxdim = 4, rng = MersenneTwister(56))
    single = MPS([reshape(ComplexF64[1, 0], 1, 2, 1)])

    @test_throws ArgumentError dmrg(product_mpo([ComplexF64[1 0; 0 -1]]), single; verbose = false)
    @test_throws DimensionMismatch dmrg(tfim_mpo(ComplexF64, 5), ψ; verbose = false)
    @test_throws ArgumentError dmrg(H, ψ; nsweeps = 0, verbose = false)

    # A factorization that cannot truncate is refused *before* anything is mutated, exactly as
    # `truncate!` refuses it, so a rejected combination leaves the state alone.
    reference = dense(ψ)
    err = try
        dmrg!(H, ψ; factorization = QRFactorization(), verbose = false)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("SVDFactorization", err.msg)
    @test dense(ψ) ≈ reference
end

@testset "the solver is a policy" begin
    L = 8
    H = tfim_mpo(Float64, L; J = 1.0, h = 0.6)
    E₀, _ = _exact_ground(H)
    ψ₀ = rand_mps(Float64, fill(2, L); maxdim = 4, rng = MersenneTwister(57))

    # The traceable Ritz policy differs only in how the projected problem is solved, so at a
    # fixed `krylovdim` it must reach the same answer. This is the CPU-side half of the claim the
    # Reactant tests lean on.
    dense_sol = dmrg(
        H, ψ₀; nsweeps = 8, trunc = FixedTruncation(16), verbose = false,
        solver = LanczosEigen(; krylovdim = 10, ritz = DenseRitz())
    )
    traced_sol = dmrg(
        H, ψ₀; nsweeps = 8, trunc = FixedTruncation(16), verbose = false,
        solver = LanczosEigen(; krylovdim = 10, ritz = TracedRitz())
    )

    @test dense_sol.energy ≈ E₀ atol = 1.0e-8
    @test traced_sol.energy ≈ dense_sol.energy atol = 1.0e-8
    # `TracedRitz` also turns the sweep-to-sweep convergence test off, since that is a comparison
    # of energies and so a branch on a value.
    @test length(traced_sol.energies) == 8
end

@testset "show" begin
    H = tfim_mpo(ComplexF64, 4)
    ψ₀ = rand_mps(ComplexF64, fill(2, 4); maxdim = 4, rng = MersenneTwister(58))
    s = sprint(show, dmrg(H, ψ₀; nsweeps = 2, maxdim = 8, verbose = false))
    @test occursin("DMRGSolution", s)
    @test occursin("energy=", s)
end
