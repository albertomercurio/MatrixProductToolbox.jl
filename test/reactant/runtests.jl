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
end
