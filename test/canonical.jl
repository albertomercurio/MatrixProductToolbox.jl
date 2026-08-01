@testset "canonicalization is an exact gauge transformation ($(nameof(typeof(alg))))" for
    alg in (SVDFactorization(), QRFactorization())
    L = 6
    ψ = rand_mps(ComplexF64, fill(2, L); maxdim = 8, rng = MersenneTwister(1))
    reference = dense(ψ)

    for c in 1:L
        canonicalize!(ψ, c; factorization = alg)
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
        canonicalize!(ψ, c; factorization = alg)
    end
    @test dense(ψ) ≈ reference

    @test_throws ArgumentError canonicalize!(ψ, 0; factorization = alg)
    @test_throws ArgumentError canonicalize!(ψ, L + 1; factorization = alg)
end

@testset "the two factorizations agree on the state, not on the tensors" begin
    # A gauge is not unique: the SVD and the QR pick different isometries. What must agree is
    # the object they represent -- which is the whole content of "gauge transformation".
    ψ = rand_mps(ComplexF64, fill(2, 6); maxdim = 8, rng = MersenneTwister(7))
    svd_gauged = canonicalize(ψ, 3; factorization = SVDFactorization())
    qr_gauged = canonicalize(ψ, 3; factorization = QRFactorization())

    @test dense(svd_gauged) ≈ dense(qr_gauged) ≈ dense(ψ)
    @test bond_dims(svd_gauged) == bond_dims(qr_gauged)
    @test !all(svd_gauged[i] ≈ qr_gauged[i] for i in 1:6)
end

@testset "canonicalize does not mutate" begin
    ψ = rand_mps(ComplexF64, fill(2, 4); maxdim = 4)
    before = center(ψ)
    φ = canonicalize(ψ, 3)
    @test center(ψ) == before
    @test center(φ) == 3:3
    @test dense(φ) ≈ dense(ψ)
end

@testset "canonicalization works for any N ($(nameof(typeof(alg))))" for
    alg in (SVDFactorization(), QRFactorization())
    # The gauge code is written against the fused (l, p…, r) reshape, so an MPO gauges too.
    W = tfim_mpo(ComplexF64, 4)
    reference = dense(W)
    canonicalize!(W, 3; factorization = alg)
    @test center(W) == 3:3
    @test all(isleftorthogonal(W[i]) for i in 1:2)
    @test isrightorthogonal(W[4])
    @test dense(W) ≈ reference
end

@testset "truncation policies" begin
    σ = [1.0, 0.1, 0.01, 0.001]
    M = diagm(σ)
    # The kept rank is read off the returned shapes, which is the only thing that matters:
    # `size(Q, 2)` *is* the new bond dimension.
    kept(trunc) = size(first(factorize_left(SVDFactorization(), trunc, M)), 2)

    # NoTruncation keeps the full rank.
    @test kept(NoTruncation()) == 4

    # FixedTruncation keeps min(maxdim, m, n) -- decided by shape alone.
    for k in 1:4
        Q, R = factorize_left(SVDFactorization(), FixedTruncation(k), M)
        @test size(Q) == (4, k)
        @test Q' * Q ≈ I                                   # exact isometry, no padding
        # Exactly the leading k singular values survive.
        @test sort(svdvals(Q * R); rev = true)[1:k] ≈ σ[1:k]
    end
    # Asking for more than the shape allows is clamped, not padded with zero columns.
    @test kept(FixedTruncation(10)) == 4

    # The documented promise: because the rank is capped at min(m, n) rather than padded up to
    # `maxdim`, the isometry stays exact even when the bond matrix is rank-deficient. Padding
    # would have produced zero columns and only a projector.
    rank_deficient = [1.0 0.0; 0.0 0.0; 0.0 0.0; 0.0 0.0]     # 4×2, rank 1
    Q, R = factorize_left(SVDFactorization(), FixedTruncation(4), rank_deficient)
    @test size(Q) == (4, 2)
    @test Q' * Q ≈ I
    @test Q * R ≈ rank_deficient

    # Cutoff keeps the smallest k whose discarded weight fits the tolerance: dropping σ₃ and σ₄
    # costs (0.01² + 0.001²)/Σσ² ≈ 1e-4 ≤ 1e-3, whereas also dropping σ₂ would cost ≈ 1e-2.
    @test kept(Truncation(; cutoff = 1.0e-3)) == 2
    @test kept(Truncation(; cutoff = 1.0e-6)) == 3
    # maxdim wins when it is the tighter constraint.
    @test kept(Truncation(; maxdim = 1, cutoff = 1.0e-3)) == 1

    # factorize_right mirrors it, with orthonormal rows instead of columns.
    Lf, Q = factorize_right(SVDFactorization(), FixedTruncation(2), M)
    @test size(Lf) == (4, 2) && size(Q) == (2, 4)
    @test Q * Q' ≈ I
    F = svd(M)
    @test Lf * Q ≈ F.U[:, 1:2] * diagm(F.S[1:2]) * F.Vt[1:2, :]   # the rank-2 truncation of M
end

@testset "QR factorization" begin
    # Tall, square and wide: the sweep only ever hands `factorize_left` a tall matrix, but the
    # contract is stated for any shape and `_thin_q` takes a different path for each.
    @testset "$(m)x$(n)" for (m, n) in ((8, 4), (4, 4), (4, 8))
        M = randn(MersenneTwister(1), ComplexF64, m, n)
        k = min(m, n)

        Q, R = factorize_left(QRFactorization(), NoTruncation(), M)
        @test size(Q) == (m, k) && size(R) == (k, n)   # full rank kept, hence exact
        @test Q' * Q ≈ I
        @test Q * R ≈ M
        @test R ≈ triu(R)

        Lf, Qr = factorize_right(QRFactorization(), NoTruncation(), M)
        @test size(Lf) == (m, k) && size(Qr) == (k, n)
        @test Qr * Qr' ≈ I
        @test Lf * Qr ≈ M
    end

    M = randn(MersenneTwister(1), ComplexF64, 8, 4)
    kept(trunc) = size(first(factorize_left(QRFactorization(), trunc, M)), 2)

    # A QR keeps the full rank, so the only policies it accepts are those that ask for it.
    for trunc in (Truncation(; cutoff = 1.0e-8), Truncation(; maxdim = 2), FixedTruncation(2))
        err = try
            factorize_left(QRFactorization(), trunc, M)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        # Point at the fix, not just at the failure.
        @test occursin("SVDFactorization", sprint(showerror, err))
        @test_throws ArgumentError factorize_right(QRFactorization(), trunc, M)
    end

    # A cap that does not bind is fine, and so is one clamped by the shape.
    @test kept(NoTruncation()) == 4
    @test kept(FixedTruncation(4)) == 4
    @test kept(FixedTruncation(10)) == 4

    # Unpivoted QR cannot see rank deficiency: the factorization is still exact and `Q` still an
    # exact isometry, but the null direction is kept rather than dropped -- the one place the two
    # factorizations genuinely disagree, so it is pinned down here.
    rank_deficient = [1.0 0.0; 0.0 0.0; 0.0 0.0; 0.0 0.0]     # 4×2, rank 1
    Q, R = factorize_left(QRFactorization(), NoTruncation(), rank_deficient)
    @test size(Q) == (4, 2)                                   # not (4, 1): the rank is not detected
    @test Q' * Q ≈ I
    @test Q * R ≈ rank_deficient
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

@testset "truncate! rejects a factorization that cannot truncate" begin
    ψ = rand_mps(ComplexF64, fill(2, 6); maxdim = 8, rng = MersenneTwister(3))
    dims, ctr, state = bond_dims(ψ), center(ψ), dense(ψ)

    err = try
        truncate!(ψ; maxdim = 3, factorization = QRFactorization())
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("SVDFactorization", sprint(showerror, err))

    # The point of checking up front: `truncate!` sweeps three times and only the middle sweep
    # truncates, so a late check would leave `ψ` regauged by the first one.
    @test bond_dims(ψ) == dims
    @test center(ψ) == ctr
    @test dense(ψ) ≈ state

    # With nothing to discard there is nothing to object to, and the sweep is exact.
    out = truncate!(ψ; trunc = NoTruncation(), factorization = QRFactorization())
    @test out === ψ
    @test dense(ψ) ≈ state
    @test bond_dims(ψ) == dims
end

@testset "type stability" begin
    ψ = rand_mps(ComplexF64, fill(2, 4); maxdim = 4)
    @test (@inferred canonicalize!(ψ, 3)) === ψ
    @test (@inferred truncate!(ψ; maxdim = 2)) === ψ
    @test (@inferred canonicalize!(ψ, 3; factorization = QRFactorization())) === ψ
    M = randn(ComplexF64, 8, 4)
    @inferred factorize_left(SVDFactorization(), FixedTruncation(2), M)
    @inferred factorize_right(SVDFactorization(), FixedTruncation(2), M)
    @inferred factorize_left(QRFactorization(), NoTruncation(), M)
    @inferred factorize_right(QRFactorization(), NoTruncation(), M)
end
