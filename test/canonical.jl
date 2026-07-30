@testset "canonicalization is an exact gauge transformation" begin
    L = 6
    ψ = rand_mps(ComplexF64, fill(2, L); maxdim = 8, rng = MersenneTwister(1))
    reference = dense(ψ)

    for c in 1:L
        canonicalize!(ψ, c)
        @test center(ψ) == c:c
        # Sites left of the centre are left-orthogonal, sites right of it right-orthogonal.
        @test all(isleftorthogonal(ψ[i]) for i in 1:(c - 1))
        @test all(isrightorthogonal(ψ[i]) for i in (c + 1):L)
        # And the represented state is untouched.
        @test dense(ψ) ≈ reference
        @test norm(ψ) ≈ 1
    end

    # Moving back and forth repeatedly must not drift.
    for c in vcat(1:L, L:-1:1, [3, 1, L, 4])
        canonicalize!(ψ, c)
    end
    @test dense(ψ) ≈ reference

    @test_throws ArgumentError canonicalize!(ψ, 0)
    @test_throws ArgumentError canonicalize!(ψ, L + 1)
end

@testset "canonicalize does not mutate" begin
    ψ = rand_mps(ComplexF64, fill(2, 4); maxdim = 4)
    before = center(ψ)
    φ = canonicalize(ψ, 3)
    @test center(ψ) == before
    @test center(φ) == 3:3
    @test dense(φ) ≈ dense(ψ)
end

@testset "canonicalization works for any N" begin
    # The gauge code is written against the fused (l, p…, r) reshape, so an MPO gauges too.
    W = tfim_mpo(ComplexF64, 4)
    reference = dense(W)
    canonicalize!(W, 3)
    @test center(W) == 3:3
    @test all(isleftorthogonal(W[i]) for i in 1:2)
    @test isrightorthogonal(W[4])
    @test dense(W) ≈ reference
end

@testset "truncation policies" begin
    M = diagm([1.0, 0.1, 0.01, 0.001])
    total = sum(abs2, [1.0, 0.1, 0.01, 0.001])

    # NoTruncation keeps the full rank and reports no error.
    _, _, info = factorize_left(SVDFactorization(), NoTruncation(), M)
    @test info.dim == 4
    @test info.error == 0

    # FixedTruncation keeps min(maxdim, m, n) -- decided by shape alone.
    for k in 1:4
        Q, R, info = factorize_left(SVDFactorization(), FixedTruncation(k), M)
        @test info.dim == k
        @test size(Q) == (4, k)
        @test Q' * Q ≈ I                                   # exact isometry, no padding
        @test info.error ≈ sum(abs2, [1.0, 0.1, 0.01, 0.001][(k + 1):end]) / total
    end
    # Asking for more than the shape allows is clamped, not padded with zero columns.
    _, _, info = factorize_left(SVDFactorization(), FixedTruncation(10), M)
    @test info.dim == 4

    # The documented promise: because the rank is capped at min(m, n) rather than padded up to
    # `maxdim`, the isometry stays exact even when the bond matrix is rank-deficient. Padding
    # would have produced zero columns and only a projector.
    rank_deficient = [1.0 0.0; 0.0 0.0; 0.0 0.0; 0.0 0.0]     # 4×2, rank 1
    Q, R, info = factorize_left(SVDFactorization(), FixedTruncation(4), rank_deficient)
    @test info.dim == 2
    @test size(Q) == (4, 2)
    @test Q' * Q ≈ I
    @test Q * R ≈ rank_deficient

    # Cutoff keeps the smallest k whose discarded weight fits the tolerance.
    _, _, info = factorize_left(SVDFactorization(), Truncation(; cutoff = 1.0e-3), M)
    @test info.dim == 2
    @test info.error ≤ 1.0e-3
    # maxdim wins when it is the tighter constraint.
    _, _, info = factorize_left(SVDFactorization(), Truncation(; maxdim = 1, cutoff = 1.0e-3), M)
    @test info.dim == 1

    # factorize_right mirrors it, with orthonormal rows instead of columns.
    Lf, Q, info = factorize_right(SVDFactorization(), FixedTruncation(2), M)
    @test size(Lf) == (4, 2) && size(Q) == (2, 4)
    @test Q * Q' ≈ I
    F = svd(M)
    @test Lf * Q ≈ F.U[:, 1:2] * diagm(F.S[1:2]) * F.Vt[1:2, :]   # the rank-2 truncation of M
end

@testset "truncate!" begin
    L = 6
    ψ = rand_mps(ComplexF64, fill(2, L); maxdim = 8, rng = MersenneTwister(2))
    exact = dense(ψ)
    @test bond_dims(ψ) == [1, 2, 4, 8, 4, 2, 1]

    # Truncating at or above the exact rank changes nothing.
    @test dense(truncate(ψ; maxdim = 8)) ≈ exact
    @test dense(truncate(ψ; trunc = NoTruncation())) ≈ exact

    # Truncating below it shrinks every bond it can and loses a little fidelity.
    φ = truncate(ψ; maxdim = 3)
    @test bond_dims(φ) == [1, 2, 3, 3, 3, 2, 1]
    @test center(φ) == 1:1
    @test abs(dot(φ, ψ))^2 < 1
    @test abs(dot(φ, ψ))^2 > 0.9
    @test norm(φ) ≤ 1 + 1.0e-10

    # The two policies agree here, because the geometric cap binds before the cutoff does.
    @test bond_dims(truncate(ψ; trunc = FixedTruncation(3))) == bond_dims(φ)
    @test dense(truncate(ψ; trunc = FixedTruncation(3))) ≈ dense(φ)

    # `to` controls where the centre is left.
    @test center(truncate(ψ; maxdim = 3, to = 4)) == 4:4
    @test_throws ArgumentError truncate(ψ; maxdim = 3, to = 0)

    # In-place version mutates and returns the same object.
    χ = bond_dims(ψ)
    out = truncate!(ψ; maxdim = 2)
    @test out === ψ
    @test bond_dims(ψ) != χ
end

@testset "type stability" begin
    ψ = rand_mps(ComplexF64, fill(2, 4); maxdim = 4)
    @test (@inferred canonicalize!(ψ, 3)) === ψ
    @test (@inferred truncate!(ψ; maxdim = 2)) === ψ
    M = randn(ComplexF64, 8, 4)
    @inferred factorize_left(SVDFactorization(), FixedTruncation(2), M)
    @inferred factorize_right(SVDFactorization(), FixedTruncation(2), M)
end
