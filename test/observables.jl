#=
Everything here is checked against explicit dense linear algebra on a small chain. That is
the linchpin of the suite: if `dense` is right, these comparisons leave nowhere for an index
or conjugation mistake to hide.
=#

@testset "overlaps and norms" begin
    L = 5
    ψ = rand_mps(ComplexF64, fill(2, L); maxdim = 4, rng = MersenneTwister(9))
    φ = rand_mps(ComplexF64, fill(2, L); maxdim = 4, rng = MersenneTwister(10))
    u, v = dense(ψ), dense(φ)

    @test dot(ψ, φ) ≈ dot(u, v)
    @test dot(φ, ψ) ≈ dot(v, u)
    @test dot(ψ, φ) ≈ conj(dot(φ, ψ))          # conjugate-linear in the first argument
    @test dot(ψ, ψ) ≈ dot(u, u)
    @test norm(ψ) ≈ norm(u)

    # `norm` uses the centre tensor when there is one, and the full contraction otherwise;
    # both routes must agree.
    ungauged = MPS(map(copy, ψ.tensors))
    @test center(ungauged) == 1:L
    @test norm(ungauged) ≈ norm(ψ)

    @test_throws DimensionMismatch dot(ψ, rand_mps(ComplexF64, fill(2, L + 1); maxdim = 2))

    # Frobenius overlap of two MPOs works too: any N reduces to the N = 1 kernels by fusing
    # the physical legs.
    W = tfim_mpo(ComplexF64, L)
    @test dot(W, W) ≈ sum(abs2, dense(W))
end

@testset "normalize" begin
    ψ = 3 * rand_mps(ComplexF64, fill(2, 4); maxdim = 4)
    @test norm(ψ) ≈ 3
    direction = dense(ψ) / norm(ψ)

    φ = normalize(ψ)
    @test norm(φ) ≈ 1
    @test dense(φ) ≈ direction
    @test norm(ψ) ≈ 3                          # the copying version leaves ψ alone

    out = normalize!(ψ)
    @test out === ψ
    @test norm(ψ) ≈ 1

    zero_state = MPS([zeros(ComplexF64, 1, 2, 1)])
    @test_throws ArgumentError normalize!(zero_state)
end

@testset "expectation values" begin
    L = 5
    ψ = rand_mps(ComplexF64, fill(2, L); maxdim = 4, rng = MersenneTwister(11))
    H = tfim_mpo(ComplexF64, L)
    v = dense(ψ)

    @test expect(H, ψ) ≈ dot(v, dense(H) * v)
    @test imag(expect(H, ψ)) ≈ 0 atol = 1.0e-12          # Hermitian operator, normalised state
    @test expect(identity_mpo(fill(2, L)), ψ) ≈ 1

    # Single-site operators, at every site and for every gauge position.
    for op in (σx, σy, σz)
        reference = [dot(v, site_op(op, L, i) * v) for i in 1:L]
        @test expect(op, ψ) ≈ reference
        for i in 1:L
            @test expect(op, ψ, i) ≈ reference[i]
        end
        # The fast path (centre already at the site) and the slow path must agree.
        canonicalize!(ψ, 3)
        @test expect(op, ψ, 3) ≈ reference[3]
        @test expect(op, ψ, 1) ≈ reference[1]
    end

    @test_throws ArgumentError expect(σz, ψ, 0)
    @test_throws ArgumentError expect(σz, ψ, L + 1)
    @test_throws DimensionMismatch expect(ComplexF64[1 0 0; 0 1 0; 0 0 1], ψ, 1)
    @test_throws DimensionMismatch expect(tfim_mpo(ComplexF64, L + 1), ψ)

    # Expectation values are unnormalised, as documented.
    scaled = 2 * ψ
    @test expect(H, scaled) ≈ 4 * expect(H, ψ)
end

@testset "apply" begin
    L = 5
    ψ = rand_mps(ComplexF64, fill(2, L); maxdim = 4, rng = MersenneTwister(12))
    H = tfim_mpo(ComplexF64, L)
    v = dense(ψ)

    φ = apply(H, ψ)
    @test dense(φ) ≈ dense(H) * v
    @test dense(H * ψ) ≈ dense(H) * v
    # Bond dimensions multiply exactly, since `apply` without `trunc` is lossless.
    @test bond_dims(φ) == bond_dims(ψ) .* bond_dims(H)
    @test bond_dims(φ) == [1, 6, 12, 12, 6, 1]

    # Truncating in the same call compresses the result.
    compressed = apply(H, ψ; trunc = Truncation(; maxdim = 4))
    @test maxbonddim(compressed) ≤ 4
    @test abs(dot(compressed, φ)) / (norm(compressed) * norm(φ)) > 0.9

    @test_throws DimensionMismatch apply(tfim_mpo(ComplexF64, L + 1), ψ)
    @test_throws DimensionMismatch apply(identity_mpo(fill(3, L)), ψ)
end

@testset "entanglement" begin
    # A product state has no entanglement across any cut.
    p = product_mps([2, 2, 2, 2], [1, 2, 1, 2])
    for b in 1:3
        @test entanglement_entropy(p, b) ≈ 0 atol = 1.0e-12
        @test entanglement_spectrum(p, b) ≈ [1.0]
    end

    # A Bell pair carries exactly log 2 nats.
    bell = bell_mps()
    @test entanglement_entropy(bell, 1) ≈ log(2)
    @test entanglement_spectrum(bell, 1) ≈ [1, 1] ./ sqrt(2)

    # The spectrum is normalised, so it ignores the norm of the state.
    ψ = rand_mps(ComplexF64, fill(2, 6); maxdim = 8, rng = MersenneTwister(13))
    for b in 1:5
        @test sum(abs2, entanglement_spectrum(ψ, b)) ≈ 1
        @test entanglement_entropy(5 * ψ, b) ≈ entanglement_entropy(ψ, b)
        @test entanglement_entropy(ψ, b) ≥ 0
    end
    # The middle cut of a random state saturates near the maximum for its bond dimension.
    @test entanglement_entropy(ψ, 3) ≤ log(8) + 1.0e-10

    @test_throws ArgumentError entanglement_entropy(ψ, 0)
    @test_throws ArgumentError entanglement_entropy(ψ, 6)

    # The gauge sweep and the spectrum are separate steps: the sweep only has to produce *a*
    # canonical form, and the `svdvals` that follows is what fixes the Schmidt coefficients. So
    # gauging with a QR -- which has no singular values of its own -- must change nothing.
    for b in 1:5
        @test entanglement_spectrum(ψ, b; factorization = QRFactorization()) ≈
            entanglement_spectrum(ψ, b)
        @test entanglement_entropy(ψ, b; factorization = QRFactorization()) ≈
            entanglement_entropy(ψ, b)
    end
end

@testset "dense" begin
    ψ = product_mps([2, 3], [2, 3])
    # Site 1 varies fastest, so reshaping recovers one dimension per site.
    a = reshape(dense(ψ), 2, 3)
    @test a[2, 3] ≈ 1
    @test sum(abs2, a) ≈ 1

    @test Array(ψ) isa Vector{ComplexF64}
    @test Array(ψ) ≈ dense(ψ)
    @test Array(tfim_mpo(ComplexF64, 3)) isa Matrix{ComplexF64}

    # The size guard fires before allocating something enormous.
    big = rand_mps(ComplexF64, fill(2, 40); maxdim = 2)
    @test_throws ArgumentError dense(big)
    @test_throws ArgumentError dense(tfim_mpo(ComplexF64, 40))
end
