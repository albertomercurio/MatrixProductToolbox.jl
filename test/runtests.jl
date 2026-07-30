using MatrixProductToolbox
using Test
using Aqua
using JET

@testset "MatrixProductToolbox.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(MatrixProductToolbox)
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(MatrixProductToolbox; target_defined_modules = true)
    end
    # Write your tests here.
end
