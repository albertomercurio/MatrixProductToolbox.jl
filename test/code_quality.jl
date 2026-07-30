@testset "Code quality" verbose = true begin
    @testset "Aqua.jl" begin
        Aqua.test_all(MatrixProductToolbox; ambiguities = false)
        @testset "Ambiguities" begin
            Aqua.test_ambiguities(MatrixProductToolbox)
        end
    end
    @testset "JET.jl" begin
        JET.test_package(MatrixProductToolbox; target_modules = (MatrixProductToolbox,))
    end
end
