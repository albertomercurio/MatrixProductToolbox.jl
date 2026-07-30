@testset "construction and queries" begin
    tensors = [randn(ComplexF64, 1, 2, 3), randn(ComplexF64, 3, 4, 2), randn(ComplexF64, 2, 2, 1)]
    ψ = MPS(tensors)

    @test ψ isa MPS
    @test ψ isa MatrixProduct{1}
    @test ψ isa AbstractMatrixProduct{1}
    @test !(ψ isa MPO)

    @test length(ψ) == 3
    @test nphysical(ψ) == 1
    @test physical_dims(ψ) == [2, 4, 2]
    @test bond_dims(ψ) == [1, 3, 2, 1]
    @test maxbonddim(ψ) == 3
    @test scalartype(ψ) === ComplexF64
    @test eltype(ψ) === ComplexF64
    @test ψ[2] === tensors[2]
    @test firstindex(ψ) == 1 && lastindex(ψ) == 3

    # `N` is inferred from the rank of the site tensors.
    @test MatrixProduct(tensors) isa MPS
    @test MatrixProduct([randn(1, 2, 2, 1)]) isa MPO

    # A fresh object claims nothing about its gauge.
    @test center(ψ) == 1:3
    @test !iscanonical(ψ)
end

@testset "MPO queries" begin
    W = MPO([randn(ComplexF64, 1, 2, 3, 4), randn(ComplexF64, 4, 3, 2, 1)])
    @test nphysical(W) == 2
    @test physical_dims(W) == [(2, 3), (3, 2)]
    @test bond_dims(W) == [1, 4, 1]
end

@testset "constructor validation" begin
    # Wrong number of indices for the declared N.
    @test_throws ArgumentError MPS([randn(1, 2, 2, 1)])
    @test_throws ArgumentError MPO([randn(1, 2, 1)])

    # Boundary bonds must be trivial.
    @test_throws ArgumentError MPS([randn(2, 2, 1)])
    @test_throws ArgumentError MPS([randn(1, 2, 2)])

    # Neighbouring bonds must agree.
    @test_throws DimensionMismatch MPS([randn(1, 2, 3), randn(2, 2, 1)])

    # The centre must be a non-empty range inside 1:L.
    @test_throws ArgumentError MPS([randn(1, 2, 1)], 0:0)
    @test_throws ArgumentError MPS([randn(1, 2, 1)], 1:2)
    @test_throws ArgumentError MPS(Vector{Array{Float64, 3}}())

    # Site tensors need a left bond, at least one physical leg and a right bond.
    @test_throws ArgumentError MatrixProduct([randn(1, 2)])
end

@testset "gauge bookkeeping" begin
    ψ = rand_mps(ComplexF64, fill(2, 4); maxdim = 4)
    canonicalize!(ψ, 2)
    @test center(ψ) == 2:2
    @test iscanonical(ψ)

    # `setindex!` invalidates the gauge, because an arbitrary tensor breaks orthogonality.
    ψ[2] = randn(ComplexF64, size(ψ[2])...)
    @test center(ψ) == 1:4
    @test !iscanonical(ψ)

    # `setsite!` is the internal escape hatch that preserves the label.
    canonicalize!(ψ, 3)
    MatrixProductToolbox.setsite!(ψ, 3, ψ[3])
    @test center(ψ) == 3:3

    @test_throws ArgumentError ψ[1] = randn(ComplexF64, 1, 2, 2, 1)

    @test isrightcanonical(canonicalize!(ψ, 1))
    @test isleftcanonical(canonicalize!(ψ, 4))
end

@testset "copy is independent" begin
    ψ = rand_mps(ComplexF64, fill(2, 3); maxdim = 2)
    φ = copy(ψ)
    fill!(φ[1], 0)
    @test !iszero(ψ[1])
    @test center(φ) == center(ψ)
end

@testset "show" begin
    ψ = product_mps([2, 2], [1, 2])
    @test sprint(show, ψ) == "MPS{ComplexF64}(L=2, d=[2, 2], χ=[1, 1, 1], center=1:1)"
    W = product_mpo([σz])
    @test sprint(show, W) == "MPO{ComplexF64}(L=1, d=[(2, 2)], χ=[1, 1], center=1:1)"
end

@testset "states" begin
    ψ = product_mps([2, 3, 2], [1, 2, 1])
    @test bond_dims(ψ) == [1, 1, 1, 1]
    @test center(ψ) == 1:1              # product states are canonical for free
    @test norm(ψ) ≈ 1
    v = dense(ψ)
    @test v[1 + 2 * (2 - 1) + 6 * (1 - 1)] ≈ 1     # site 1 varies fastest
    @test sum(abs2, v) ≈ 1

    # Local norms are shifted onto site 1, so an unnormalised input stays right-canonical.
    φ = product_mps([[3.0, 0.0], [0.0, 2.0]])
    @test norm(φ) ≈ 6
    @test center(φ) == 1:1

    @test_throws ArgumentError product_mps([2, 2], [1, 3])
    @test_throws DimensionMismatch product_mps([2, 2], [1])
    @test_throws ArgumentError product_mps([[1.0, 0.0], [0.0, 0.0]])

    # `rand_mps` bond profile is capped from both ends by the local Hilbert spaces.
    @test bond_dims(rand_mps(ComplexF64, fill(2, 6); maxdim = 8)) == [1, 2, 4, 8, 4, 2, 1]
    @test bond_dims(rand_mps(ComplexF64, fill(2, 5); maxdim = 4)) == [1, 2, 4, 4, 2, 1]
    @test bond_dims(rand_mps(ComplexF64, fill(2, 6))) == [1, 2, 4, 8, 4, 2, 1]
    @test scalartype(rand_mps(Float64, fill(2, 3))) === Float64

    @test bond_dims(identity_mpo([2, 3])) == [1, 1, 1]
    @test dense(identity_mpo([2, 2])) ≈ I
end

@testset "Adapt" begin
    ψ = rand_mps(ComplexF64, fill(2, 4); maxdim = 4)
    φ = adapt(Array, ψ)
    @test φ isa MPS
    @test center(φ) == center(ψ)
    @test dense(φ) ≈ dense(ψ)
end

@testset "scalar arithmetic" begin
    ψ = rand_mps(ComplexF64, fill(2, 4); maxdim = 4)
    @test dense(2 * ψ) ≈ 2 * dense(ψ)
    @test dense(ψ * 2) ≈ 2 * dense(ψ)
    @test dense(ψ / 2) ≈ dense(ψ) / 2
    @test dense(-ψ) ≈ -dense(ψ)
    @test center(2 * ψ) == center(ψ)          # scaling the centre preserves the gauge

    # Element type is promoted rather than silently truncated.
    real_ψ = rand_mps(Float64, fill(2, 3); maxdim = 2)
    @test scalartype(im * real_ψ) === ComplexF64
end
