#=
The shared backend probes on `CuArray`.

Run with:

    GROUP=CUDA julia --project=. -e 'using Pkg; Pkg.test()'

CUDA lives in this sub-environment so that it never becomes a dependency of the package. GitHub
runners have no GPU, so this group is not part of the default CI matrix — it is meant for local
runs and self-hosted GPU CI.

Nothing here is CUDA-specific beyond the adaptor: `CuArray` works because every operation the
package performs is generic (`reshape`, `similar`, `*`, broadcast, `sum`, `svd`).

**`using cuTENSOR` is required, not optional.** TensorOperations' actual GPU contraction backend
lives in its cuTENSOR extension; its CUDA extension only supplies device-aware allocation. Without
cuTENSOR loaded, `CuArray <: DenseArray` makes `isstrided` return `true`, `@tensor` selects the
Strided CPU backend, and every contraction dies on `Scalar indexing is disallowed`.
=#

using MatrixProductToolbox
using Test

using Adapt
using CUDA
using LinearAlgebra
using Random
using TensorOperations: @tensor
using cuTENSOR

include(joinpath(@__DIR__, "..", "testutils.jl"))

@testset "CUDA" verbose = true begin
    if !CUDA.functional()
        @warn "CUDA is not functional on this machine; skipping the CuArray backend probes."
        @test_skip false
    else
        CUDA.versioninfo()

        # Eager backend: the runner is just application, exactly as for `Array`.
        backend_testset("CuArray", ψ -> adapt(CuArray, ψ), (f, a, b, c) -> f(a, b, c))

        @testset "no scalar indexing" begin
            # `allowscalar(false)` turns any accidental host round-trip into an error, which is
            # the property the `sum(E)` and `fill!(similar(...))` idioms exist to guarantee.
            CUDA.allowscalar(false)
            H_cpu, ψ_cpu, φ_cpu = backend_inputs()
            H, ψ, φ = adapt(CuArray, H_cpu), adapt(CuArray, ψ_cpu), adapt(CuArray, φ_cpu)
            for (label, probe) in BACKEND_PROBES
                @testset "$label" begin
                    @test probe(H, ψ, φ) ≈ probe(H_cpu, ψ_cpu, φ_cpu)
                end
            end
        end

        @testset "single precision" begin
            H_cpu, ψ_cpu, _ = backend_inputs()
            ψ = adapt(CuArray{ComplexF32}, ψ_cpu)
            H = adapt(CuArray{ComplexF32}, H_cpu)
            @test scalartype(ψ) === ComplexF32
            @test expect(H, ψ) ≈ expect(H_cpu, ψ_cpu) rtol = 1.0e-4
        end
    end
end
