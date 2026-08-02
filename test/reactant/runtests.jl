#=
The shared backend probes under Reactant, plus the XLA-specific claims.

Run with:

    GROUP=Reactant julia --project=. -e 'using Pkg; Pkg.test()'

Reactant lives in this sub-environment so that it never becomes a dependency of the package —
not even a weak one. The `@tensor` calls work on traced arrays because *Reactant itself* ships
`ReactantTensorOperationsExt`; nothing in this package knows Reactant exists.

Two rules that are easy to get wrong, and that shape this file:

  * Everything must run inside `@compile` / `@jit`. Reactant's TensorOperations `select_backend`
    methods are `@reactant_overlay`'d, so they only fire under the overlay interpreter. Calling
    `@tensor` on a `ConcreteRArray` at top level falls back to the Strided scalar path.
  * Bond dimensions must be trace-time constants, which is exactly what `FixedTruncation`
    guarantees and what `Truncation` (the cutoff policy) cannot.
=#

using MatrixProductToolbox
using Test

using LinearAlgebra
using Random
using Reactant
using TensorOperations: @tensor

include(joinpath(@__DIR__, "..", "testutils.jl"))

# The Reactant runner: compile the probe, run it, and bring the scalar back to the host. Fixed
# arity, because `@jit` needs to see the call expression rather than a splat.
_host(x::Reactant.ConcreteRNumber) = Reactant.to_number(x)
_host(x::AbstractArray) = only(Array(x))
_host(x::Number) = x

jit_run(f, a, b, c) = _host(@jit f(a, b, c))

# Relative reconstruction error of an SVD. Written with the package's own broadcast idiom rather
# than `Diagonal`, so that it traces the same way the factorizations do.
function _svd_residual(A)
    F = svd(A)
    return norm(F.U .* transpose(F.S) * F.Vt .- A) / norm(A)
end

# Whether Reactant's *complex* `svd` reconstructs its input on this backend.
#
# This is deliberately a runtime capability probe and **not** a test of which device is in use.
# The two backends fail differently: the CPU one cannot compile `zgesdd_` at all (`'llvm.call' op
# operand type mismatch for operand 13`, `LWORK` passed by value instead of by reference), while
# the GPU one throws nothing and silently returns `Vᵀ` where `Vᴴ` is wanted — `svdvals` is exact
# and both factors are valid isometries, so only the reconstruction identity catches it. Gating on
# "is this a GPU?" would run the complex tests on precisely the backend that lies.
#
# It also turns green by itself once the upstream bug is fixed, at which point the `@test_skip`
# below becomes a real test with no edit here.
function complex_svd_reconstructs()
    A = ComplexF64[2 1 + im 0 0; 1 - im 3 im 0; 0 -im 2 1; 0 0 1 4]
    return try
        _host(@jit _svd_residual(Reactant.to_rarray(A))) < 1.0e-10
    catch
        false
    end
end

@testset "Reactant / XLA" verbose = true begin
    Reactant.set_default_backend("cpu")

    # The same probe lists the CPU and CUDA groups run; only the adaptor and the runner differ.
    # They are split by element type on purpose — see the note below.
    @testset "contractions (ComplexF64)" begin
        backend_testset(
            "ConcreteRArray", Reactant.to_rarray, jit_run;
            probes = CONTRACTION_PROBES, T = ComplexF64
        )
    end

    # Reactant's *complex* SVD on the CPU backend is currently broken: measured against the CPU
    # answer it is silently wrong at small bond dimensions (relative error ~0.26 at L = 10, χ = 4)
    # and segfaults from χ ≈ 6 upwards. The real SVD is exact. So the gauge probes — the only ones
    # that call `svd` — run in Float64, where they genuinely verify this package's canonicalization
    # under XLA rather than passing by luck at one small size.
    @testset "gauge and truncation (Float64)" begin
        backend_testset(
            "ConcreteRArray", Reactant.to_rarray, jit_run;
            probes = GAUGE_PROBES, T = Float64
        )
    end

    # Float64 throughout below, since everything here goes through `svd`.
    H_cpu, ψ_cpu, _ = backend_inputs(Float64)
    H, ψ = Reactant.to_rarray(H_cpu), Reactant.to_rarray(ψ_cpu)

    @testset "compile once, run many" begin
        # Compile one whole sweep rather than individual contractions. Note this is about
        # *correctness of the workflow*, not speed: on the CPU backend XLA measures at roughly
        # parity with plain Julia for this workload (see benchmarks/reactant.jl).
        sweep(H, ψ) = expect(H, canonicalize(ψ, BACKEND_SITES; trunc = FixedTruncation(2)))

        compiled = @compile sweep(H, ψ)
        reference = sweep(H_cpu, ψ_cpu)
        @test _host(compiled(H, ψ)) ≈ reference
        # A compiled thunk is reusable, which is the whole reason to hold on to one.
        @test _host(compiled(H, ψ)) ≈ reference
    end

    @testset "QR gauge sweep" begin
        # `QRFactorization` reaches `LinearAlgebra.qr`, whose Reactant path is quite unlike the
        # `svd` one: `geqrf` + `orgqr` rather than a single op, and a factorization object whose
        # `Q` is the materialized thin factor instead of the square `AbstractQ` operator that
        # `Array` and `CuArray` hand back. Hence a sweep of its own.
        qr_sweep(H, ψ) =
            expect(H, canonicalize(ψ, BACKEND_SITES; factorization = QRFactorization()))

        compiled = @compile qr_sweep(H, ψ)
        reference = qr_sweep(H_cpu, ψ_cpu)
        @test _host(compiled(H, ψ)) ≈ reference
        @test _host(compiled(H, ψ)) ≈ reference

        # The gauge is exact, so the QR and the SVD must agree on the value even though they
        # disagree on the tensors that produce it.
        svd_sweep(H, ψ) = expect(H, canonicalize(ψ, BACKEND_SITES))
        @test _host(@jit qr_sweep(H, ψ)) ≈ _host(@jit svd_sweep(H, ψ))
    end

    @testset "QR gauges in ComplexF64, where the SVD cannot" begin
        # The reason `QRFactorization` matters here, over and above being faster: Reactant's
        # complex *SVD* is broken on the CPU backend — silently wrong at χ = 4 and segfaulting
        # from about χ = 6 — which is why the gauge probes above run in Float64. A QR is a
        # different lowering (`geqrf` + `ungqr`, no `gesdd`) and is unaffected.
        #
        # χ = 8 is chosen deliberately: it is past the point where the SVD segfaults, so this
        # would not pass if the QR shared the bug. Keep it if the upstream SVD is ever fixed —
        # it then simply stops being the only way to do this.
        L, χ = 10, 8
        Hc = tfim_mpo(ComplexF64, L)
        ψc = rand_mps(ComplexF64, fill(2, L); maxdim = χ, rng = MersenneTwister(14))
        sweep(H, ψ) = expect(H, canonicalize(ψ, L; factorization = QRFactorization()))

        got = _host(@jit sweep(Reactant.to_rarray(Hc), Reactant.to_rarray(ψc)))
        reference = sweep(Hc, ψc)
        @test got ≈ reference rtol = 1.0e-10
    end

    @testset "QR cannot truncate" begin
        # A QR keeps the full rank, so a *binding* `FixedTruncation` is rejected. Unlike the
        # cutoff policy below this is not an XLA limitation at all -- the check is a comparison
        # of shapes, so it fires identically on every backend. What matters here is that it
        # still fires under tracing, rather than being silently ignored or dissolving into a
        # traced boolean.
        qr_trunc(H, ψ, φ) = expect(
            H, canonicalize(
                ψ, BACKEND_SITES;
                trunc = FixedTruncation(2), factorization = QRFactorization()
            )
        )
        err = try
            jit_run(qr_trunc, H, ψ, ψ)
            nothing
        catch e
            e
        end
        @test err !== nothing
        @test occursin("SVDFactorization", sprint(showerror, err))
    end

    @testset "cutoff truncation is not traceable" begin
        # `Truncation` picks its rank by comparing singular values, so the output shape becomes a
        # runtime value and XLA cannot express it. Asserting the failure keeps the documented
        # boundary between the two policies honest rather than aspirational.
        # Three arguments, to match `jit_run`'s arity -- otherwise this "fails" with a MethodError
        # and passes for entirely the wrong reason.
        cutoff_sweep(H, ψ, φ) = expect(
            H, canonicalize(ψ, BACKEND_SITES; trunc = Truncation(; maxdim = 2, cutoff = 1.0e-8))
        )
        err = try
            jit_run(cutoff_sweep, H, ψ, ψ)
            nothing
        catch e
            e
        end
        @test err !== nothing
        # And it should fail with the guidance, not with something cryptic from deep inside the
        # tracer.
        @test occursin("FixedTruncation", sprint(showerror, err))
    end

    @testset "contractions lower to dot_general" begin
        # Not a performance assertion — just a check that the sweep lowers to HLO at all, and
        # that the contractions became matrix products rather than something pathological.
        # Inspect this module by hand when tuning: transposes should be folded into dot_general.
        sweep(H, ψ) = expect(H, canonicalize(ψ, BACKEND_SITES; trunc = FixedTruncation(2)))
        hlo = sprint(show, @code_hlo sweep(H, ψ))
        @test occursin("dot_general", hlo)
    end

    @testset "the QR sweep reaches the LAPACK lowering" begin
        # `LinearAlgebra.qr` is intercepted by a `@reactant_overlay`, so the failure mode to
        # guard against is a silent fall-back to some generic path that happens to produce the
        # right numbers. Seeing the LAPACK calls in the HLO is what says the overlay actually
        # fired.
        qr_sweep(H, ψ) =
            expect(H, canonicalize(ψ, BACKEND_SITES; factorization = QRFactorization()))
        svd_sweep(H, ψ) = expect(H, canonicalize(ψ, BACKEND_SITES))

        hlo = sprint(show, @code_hlo qr_sweep(H, ψ))
        @test occursin("geqrf", hlo)
        # `Q` is formed by `orgqr` for a real element type and `ungqr` for a complex one; match
        # either, so the check does not hinge on which one these inputs happen to use.
        @test occursin("orgqr", hlo) || occursin("ungqr", hlo)
        @test occursin("dot_general", hlo)

        # Differential rather than a guess at what the SVD lowers to: whatever the QR path emits
        # for its factorization, the SVD path must not, and vice versa. This is the assertion
        # that the `factorization` keyword actually reaches the backend.
        @test !occursin("geqrf", sprint(show, @code_hlo svd_sweep(H, ψ)))
    end

    #=
    The sweeping algorithms.

    `ALGORITHM_PROBES` uses `FixedTruncation` and `TracedRitz` together, which is exactly what
    makes a whole sweep traceable: bond dimensions follow from shapes alone, the Lanczos runs a
    fixed number of steps, and the projected solve never branches on a value. `dmrg` and `tdvp`
    additionally switch off their own sweep-to-sweep convergence tests when handed a traceable
    solver, since those are comparisons of energies and so branches of the same kind.

    Because `TracedRitz` solves the *same* projected problem as the default rather than running a
    different algorithm, these compare against the CPU answer at the ordinary tolerance — which
    is the whole reason the policy was written that way.
    =#
    @testset "algorithms (Float64)" begin
        backend_testset(
            "ConcreteRArray", Reactant.to_rarray, jit_run;
            probes = ALGORITHM_PROBES, T = Float64
        )
    end

    @testset "`DenseRitz` refuses to trace" begin
        # The failure to guard against is a silent fall-back, so this asserts on the message: the
        # guard exists to name the way out, not merely to stop.
        dense_ritz(H, ψ, φ) = dmrg(
            H, ψ; nsweeps = 1, trunc = FixedTruncation(ALGORITHM_CHI), verbose = false,
            solver = LanczosEigen(; krylovdim = ALGORITHM_KRYLOVDIM, ritz = DenseRitz())
        ).energy
        err = try
            jit_run(dense_ritz, H, ψ, ψ)
            nothing
        catch e
            e
        end
        @test err !== nothing
        @test occursin("TracedRitz", sprint(showerror, err))
    end

    @testset "a DMRG sweep lowers to HLO" begin
        # The two things a sweep must reach: `dot_general` for the effective-operator
        # applications, and the LAPACK SVD for the two-site splits.
        sweep(H, ψ) = dmrg(
            H, ψ; nsweeps = 1, trunc = FixedTruncation(ALGORITHM_CHI), verbose = false,
            solver = LanczosEigen(; krylovdim = ALGORITHM_KRYLOVDIM, ritz = TracedRitz())
        ).energy
        hlo = sprint(show, @code_hlo sweep(H, ψ))
        @test occursin("dot_general", hlo)
        @test occursin("gesdd", hlo) || occursin("gesvd", hlo) || occursin("svd", hlo)
    end

    @testset "real-time TDVP needs a complex SVD" begin
        # Two-site TDVP has to truncate, and truncation means an SVD, so real time in `ComplexF64`
        # is only possible where Reactant's complex SVD works. It currently does not — on either
        # backend — so this is probed rather than assumed, and skips with a message instead of
        # failing. See `complex_svd_reconstructs` above for why the probe is a reconstruction
        # identity and not a device-type test.
        if complex_svd_reconstructs()
            backend_testset(
                "ConcreteRArray", Reactant.to_rarray, jit_run;
                probes = COMPLEX_ALGORITHM_PROBES, T = ComplexF64
            )
        else
            @info """
            Skipping complex-time TDVP: Reactant's complex `svd` does not reconstruct its input on
            this backend. On CPU it fails to compile `zgesdd_`; on GPU it returns `Vᵀ` in place of
            `Vᴴ` and throws nothing. Imaginary-time TDVP and `Float64` DMRG are unaffected, and
            this test starts running by itself once the upstream bug is fixed.
            """
            @test_skip complex_svd_reconstructs()
        end
    end
end
