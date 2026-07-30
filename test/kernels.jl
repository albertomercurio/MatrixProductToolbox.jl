#=
The six contraction kernels are the entire compute core, so they get pinned down directly
rather than only through the functions built on them. Every check is against an independent
reference: either a sweep in the opposite direction, or the dense contraction.
=#

@testset "overlap transfer kernels" begin
    L = 5
    ψ = rand_mps(ComplexF64, fill(2, L); maxdim = 4, rng = MersenneTwister(3))
    φ = rand_mps(ComplexF64, fill(2, L); maxdim = 4, rng = MersenneTwister(4))
    T = ComplexF64
    reference = dot(dense(ψ), dense(φ))

    # Sweeping left to right, ψ is the bra (conjugated) and φ the ket.
    E = fill!(similar(ψ[1], T, (1, 1)), one(T))
    for i in 1:L
        E = transfer_left(E, φ[i], ψ[i])
    end
    @test sum(E) ≈ reference

    # Sweeping right to left must give the same answer.
    E = fill!(similar(ψ[1], T, (1, 1)), one(T))
    for i in L:-1:1
        E = transfer_right(E, φ[i], ψ[i])
    end
    @test sum(E) ≈ reference
end

@testset "operator transfer kernels" begin
    L = 5
    ψ = rand_mps(ComplexF64, fill(2, L); maxdim = 4, rng = MersenneTwister(5))
    H = tfim_mpo(ComplexF64, L)
    T = ComplexF64
    v = dense(ψ)
    reference = dot(v, dense(H) * v)

    E = fill!(similar(ψ[1], T, (1, 1, 1)), one(T))
    for i in 1:L
        E = transfer_left(E, ψ[i], H[i], ψ[i])
    end
    @test sum(E) ≈ reference

    E = fill!(similar(ψ[1], T, (1, 1, 1)), one(T))
    for i in L:-1:1
        E = transfer_right(E, ψ[i], H[i], ψ[i])
    end
    @test sum(E) ≈ reference

    # The bra pairs with the *output* leg of the MPO, so a non-Hermitian operator must come
    # back as ⟨ψ|W|φ⟩ rather than its adjoint.
    W = product_mpo([ComplexF64[0 1; 0 0] for _ in 1:L])       # strictly lower/upper triangular
    φ = rand_mps(ComplexF64, fill(2, L); maxdim = 4, rng = MersenneTwister(6))
    E = fill!(similar(ψ[1], T, (1, 1, 1)), one(T))
    for i in 1:L
        E = transfer_left(E, φ[i], W[i], ψ[i])
    end
    @test sum(E) ≈ dot(dense(ψ), dense(W) * dense(φ))
end

@testset "one-site effective operator" begin
    L = 6
    ψ = rand_mps(ComplexF64, fill(2, L); maxdim = 4, rng = MersenneTwister(7))
    H = tfim_mpo(ComplexF64, L)
    reference = expect(H, ψ)

    for c in 1:L
        canonicalize!(ψ, c)
        Lenv = left_env(ψ, H, c)
        Renv = right_env(ψ, H, c + 1)
        x = ψ[c]
        y = apply_effective(Lenv, H[c], x, Renv)
        @test size(y) == size(x)
        @test dot(x, y) ≈ reference
    end
end

@testset "two-site effective operator" begin
    L = 6
    ψ = rand_mps(ComplexF64, fill(2, L); maxdim = 4, rng = MersenneTwister(8))
    H = tfim_mpo(ComplexF64, L)
    reference = expect(H, ψ)

    for c in 1:(L - 1)
        canonicalize!(ψ, c)
        Lenv = left_env(ψ, H, c)
        Renv = right_env(ψ, H, c + 2)
        a, b = ψ[c], ψ[c + 1]
        @tensor x[l, s1, s2, r] := a[l, s1, m] * b[m, s2, r]
        y = apply_effective(Lenv, H[c], H[c + 1], x, Renv)
        @test size(y) == size(x)
        @test dot(x, y) ≈ reference
    end
end

@testset "type stability" begin
    L = 4
    ψ = rand_mps(ComplexF64, fill(2, L); maxdim = 4)
    H = tfim_mpo(ComplexF64, L)
    canonicalize!(ψ, 2)
    T = ComplexF64

    # A left-moving environment is indexed by the left bonds, so it starts at site 1; a
    # right-moving one is indexed by the right bonds, so it starts at site L.
    E2 = fill!(similar(ψ[1], T, (1, 1)), one(T))
    @inferred transfer_left(E2, ψ[1], ψ[1])
    @inferred transfer_right(E2, ψ[L], ψ[L])

    E3 = fill!(similar(ψ[1], T, (1, 1, 1)), one(T))
    @inferred transfer_left(E3, ψ[1], H[1], ψ[1])
    @inferred transfer_right(E3, ψ[L], H[L], ψ[L])

    Lenv = left_env(ψ, H, 2)
    Renv = right_env(ψ, H, 3)
    @inferred apply_effective(Lenv, H[2], ψ[2], Renv)

    a, b = ψ[2], ψ[3]
    @tensor x[l, s1, s2, r] := a[l, s1, m] * b[m, s2, r]
    @inferred apply_effective(left_env(ψ, H, 2), H[2], H[3], x, right_env(ψ, H, 4))
end
