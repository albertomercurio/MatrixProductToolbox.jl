#=
TDVP is checked against `exp(-i t H)` applied to the dense state vector. On a chain short enough
to exponentiate, and with a bond dimension large enough that nothing is discarded, two-site TDVP
is *exact*, so the comparison is to machine precision rather than to an integration tolerance.

Where truncation binds the reference is no longer available, and the checks become the physical
invariants instead: norm and energy conservation in real time, and convergence to the DMRG ground
state in imaginary time.
=#

@testset "TDVP reproduces the exact propagator" begin
    L = 6
    H = tfim_mpo(ComplexF64, L; J = 1.0, h = 0.7)
    Hd = dense(H)
    ψ₀ = product_mps(fill(2, L), [1, 2, 1, 2, 1, 2])
    v₀ = dense(ψ₀)

    t = 0.4
    tlist = range(0, t; length = 9)
    # χ = 2^(L÷2) is enough to represent any state of this chain, so nothing is discarded and the
    # only error left is the second-order splitting -- which is also exact once the manifold is
    # the whole Hilbert space.
    sol = tdvp(H, ψ₀, tlist; maxdim = 2^(L ÷ 2), cutoff = 0.0, progress = false)

    @test dense(sol.states[end]) ≈ exp(-im * t * Hd) * v₀ rtol = 1.0e-9
    @test sol.times == collect(tlist)
    @test length(sol.states) == length(tlist)
    @test maximum(sol.truncation_errors) < 1.0e-10

    # Every intermediate time, not only the last.
    for (n, tn) in enumerate(tlist)
        @test dense(sol.states[n]) ≈ exp(-im * tn * Hd) * v₀ rtol = 1.0e-9
    end
end

@testset "real time conserves norm and energy" begin
    L = 8
    H = tfim_mpo(ComplexF64, L; J = 1.0, h = 0.5)
    ψ₀ = product_mps(fill(2, L), [1, 1, 2, 1, 2, 2, 1, 2])
    sol = tdvp(H, ψ₀, range(0, 1.0; length = 21); e_ops = (H,), maxdim = 32, progress = false)

    E = real.(vec(sol.expect))
    @test maximum(abs, E .- E[1]) < 1.0e-8
    @test norm(sol.states[end]) ≈ 1 rtol = 1.0e-8
    # No `e_ops`-free run was asked for, so only the final state is kept.
    @test length(sol.states) == 1
    @test size(sol.expect) == (1, 21)
end

@testset "imaginary time converges to the DMRG ground state" begin
    # The two algorithms share the environments, the split and the kernels but not the local
    # solver, so agreeing on the ground energy is a genuine cross-check rather than a tautology.
    #
    # h = 2 puts the chain deep in the disordered phase, where the gap is ≈ 2.2. That matters:
    # imaginary time suppresses an excited state only as exp(-2τΔ), so at h = 0.7 — gap 0.06 —
    # the same τ = 10 would leave a residual of ~3e-3 that no amount of bond dimension or
    # time-step refinement removes. DMRG has no such dependence, which is the honest reason it is
    # the ground-state method and this is the cross-check.
    L = 8
    H = tfim_mpo(Float64, L; J = 1.0, h = 2.0)
    E₀ = eigvals(Symmetric(real(dense(H))))[1]
    ψ₀ = rand_mps(Float64, fill(2, L); maxdim = 8, rng = MersenneTwister(61))

    sol = tdvp(H, ψ₀, range(0, 10; length = 101); imaginary = true, maxdim = 16, progress = false)
    ψ = sol.states[end]
    @test norm(ψ) ≈ 1 rtol = 1.0e-8            # `normalize` defaults to `imaginary`
    @test real(expect(H, ψ)) ≈ E₀ atol = 1.0e-9

    dmrg_energy = dmrg(H, ψ₀; nsweeps = 10, maxdim = 16, verbose = false).energy
    @test real(expect(H, ψ)) ≈ dmrg_energy atol = 1.0e-9
end

@testset "imaginary time keeps a real state real" begin
    # `_tdvp_tau` is real for imaginary time precisely so that a `Float64` state does not widen
    # to complex and become unassignable to its own container.
    L = 6
    H = tfim_mpo(Float64, L)
    ψ₀ = rand_mps(Float64, fill(2, L); maxdim = 4, rng = MersenneTwister(62))
    sol = tdvp(H, ψ₀, range(0, 0.5; length = 6); imaginary = true, maxdim = 8, progress = false)
    @test scalartype(sol.states[end]) === Float64

    # Real time on a real state would widen it, and is refused with a message that says so.
    err = try
        tdvp(H, ψ₀, range(0, 0.5; length = 6); maxdim = 8, progress = false)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("complex state", err.msg)
end

@testset "the time list is the time step" begin
    L = 6
    H = tfim_mpo(ComplexF64, L; J = 1.0, h = 0.8)
    Hd = dense(H)
    ψ₀ = product_mps(fill(2, L), [2, 1, 2, 1, 2, 1])
    t = 1.0
    reference = exp(-im * t * Hd) * dense(ψ₀)

    # Truncation is switched off, so the only error is the second-order splitting: halving the
    # step must cut it by about four.
    coarse = tdvp(H, ψ₀, range(0, t; length = 5); maxdim = 2^(L ÷ 2), cutoff = 0.0, progress = false)
    fine = tdvp(H, ψ₀, range(0, t; length = 9); maxdim = 2^(L ÷ 2), cutoff = 0.0, progress = false)
    e_coarse = norm(dense(coarse.states[end]) - reference)
    e_fine = norm(dense(fine.states[end]) - reference)
    @test e_fine ≤ e_coarse

    # A non-uniform list is integrated interval by interval.
    uneven = tdvp(H, ψ₀, [0.0, 0.1, 0.35, 0.6, 1.0]; maxdim = 2^(L ÷ 2), cutoff = 0.0, progress = false)
    @test dense(uneven.states[end]) ≈ reference rtol = 1.0e-6
end

@testset "observables and saving" begin
    L = 6
    H = tfim_mpo(ComplexF64, L; J = 1.0, h = 0.6)
    Z = product_mpo([ComplexF64[1 0; 0 -1] for _ in 1:L])
    ψ₀ = product_mps(fill(2, L), ones(Int, L))
    tlist = range(0, 0.5; length = 6)

    sol = tdvp(H, ψ₀, tlist; e_ops = (H, Z), maxdim = 16, progress = false)
    @test size(sol.expect) == (2, 6)
    # The values are the observables of the saved trajectory, so recomputing them must agree.
    full = tdvp(H, ψ₀, tlist; maxdim = 16, save_states = true, progress = false)
    @test length(full.states) == 6
    for n in 1:6
        @test sol.expect[1, n] ≈ expect(H, full.states[n]) rtol = 1.0e-8
        @test sol.expect[2, n] ≈ expect(Z, full.states[n]) rtol = 1.0e-8
    end

    # No observables at all still produces a well-formed, empty table.
    none = tdvp(H, ψ₀, tlist; maxdim = 16, progress = false)
    @test size(none.expect) == (0, 6)
end

@testset "the solver is a policy" begin
    L = 6
    H = tfim_mpo(ComplexF64, L; J = 1.0, h = 0.7)
    ψ₀ = product_mps(fill(2, L), [1, 2, 1, 2, 1, 2])
    tlist = range(0, 0.3; length = 4)

    dense_sol = tdvp(
        H, ψ₀, tlist; trunc = FixedTruncation(16), progress = false,
        solver = LanczosExponential(; krylovdim = 12, ritz = DenseRitz())
    )
    traced_sol = tdvp(
        H, ψ₀, tlist; trunc = FixedTruncation(16), progress = false,
        solver = LanczosExponential(; krylovdim = 12, ritz = TracedRitz())
    )
    @test dense(traced_sol.states[end]) ≈ dense(dense_sol.states[end]) rtol = 1.0e-8
end

@testset "`tdvp` copies and `tdvp!` mutates" begin
    L = 6
    H = tfim_mpo(ComplexF64, L)
    ψ₀ = product_mps(fill(2, L), [1, 2, 1, 2, 1, 2])
    reference = dense(ψ₀)
    tlist = range(0, 0.2; length = 3)

    sol = tdvp(H, ψ₀, tlist; maxdim = 8, progress = false)
    @test dense(ψ₀) ≈ reference
    sol! = tdvp!(H, ψ₀, tlist; maxdim = 8, progress = false)
    @test dense(ψ₀) ≈ dense(sol!.states[end])
    @test dense(sol!.states[end]) ≈ dense(sol.states[end])
end

@testset "argument checking" begin
    L = 4
    H = tfim_mpo(ComplexF64, L)
    ψ = product_mps(fill(2, L), ones(Int, L))

    @test_throws DimensionMismatch tdvp(tfim_mpo(ComplexF64, 5), ψ, [0.0, 0.1]; progress = false)
    @test_throws ArgumentError tdvp(H, ψ, [0.0]; progress = false)
    @test_throws ArgumentError tdvp(H, ψ, [0.0, 0.2, 0.1]; progress = false)
    @test_throws DimensionMismatch tdvp(
        H, ψ, [0.0, 0.1]; e_ops = (tfim_mpo(ComplexF64, 5),), progress = false
    )
    err = try
        tdvp(H, ψ, [0.0, 0.1]; factorization = QRFactorization(), progress = false)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("SVDFactorization", err.msg)
end

@testset "show" begin
    H = tfim_mpo(ComplexF64, 4)
    ψ₀ = product_mps(fill(2, 4), ones(Int, 4))
    s = sprint(show, tdvp(H, ψ₀, [0.0, 0.1]; maxdim = 8, progress = false))
    @test occursin("TDVPSolution", s)
    @test occursin("times=2", s)
end
