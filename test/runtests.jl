using MatrixProductToolbox
using Test

using Adapt
using LinearAlgebra
using Pkg
using Random
using TensorOperations: @tensor

# Test groups, selected with the GROUP environment variable:
#
#   Core          contractions, gauge, observables, CPU backend probes   (default, no extra deps)
#   Code-Quality  Aqua + JET (memory hungry; keep it in its own process)
#   CUDA          the shared backend probes on CuArray
#   Reactant      the shared backend probes under @jit, plus XLA specifics
#
# CUDA and Reactant live in their own sub-environments under test/, so that neither ever becomes
# a dependency of this package. Reactant runs on CPU, so unlike CUDA it is usable in CI.
const GROUP = get(ENV, "GROUP", "All")
const GROUP_LIST = ["All", "Core", "Code-Quality", "CUDA", "Reactant"]
GROUP in GROUP_LIST || throw(
    ArgumentError("Unknown GROUP = $GROUP.\nThe allowed groups are: $GROUP_LIST\n")
)

# Activate one of the sub-environments under test/ and hand over to its own runtests.jl.
function run_subenv(name::AbstractString)
    subenv = joinpath(@__DIR__, name)
    Pkg.activate(subenv)
    Pkg.develop(Pkg.PackageSpec(path = dirname(@__DIR__)))
    Pkg.instantiate()
    return include(joinpath(subenv, "runtests.jl"))
end

if GROUP in ("All", "Core")
    include("testutils.jl")

    @testset "MatrixProductToolbox.jl" verbose = true begin
        @testset "MatrixProduct" begin
            include("matrixproduct.jl")
        end
        @testset "Kernels" begin
            include("kernels.jl")
        end
        @testset "Canonical forms and truncation" begin
            include("canonical.jl")
        end
        @testset "Observables" begin
            include("observables.jl")
        end
        @testset "Local solvers" begin
            include("krylov.jl")
        end
        @testset "Environments" begin
            include("environments.jl")
        end
        @testset "DMRG" begin
            include("dmrg.jl")
        end
        @testset "TDVP" begin
            include("tdvp.jl")
        end
        @testset "Backends" verbose = true begin
            include("backends.jl")
        end
    end
end

if GROUP in ("All", "Code-Quality")
    using Aqua
    using JET

    include("code_quality.jl")
end

GROUP == "CUDA" && run_subenv("cuda")
GROUP == "Reactant" && run_subenv("reactant")
