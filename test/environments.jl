#=
`left_env` and `right_env` in `test/testutils.jl` are the specification the cache has to meet:
they sweep the transfer kernels with no caching at all, so agreeing with them is exactly the
statement that memoization changed nothing.

The second half is the harder claim: that the invalidation bookkeeping is right. A cache that is
correct only when nothing has been mutated is useless to a sweeping algorithm, so the tests below
rewrite site tensors and re-ask.
=#

@testset "the cache agrees with an uncached sweep" begin
    L = 6
    H = tfim_mpo(ComplexF64, L; J = 1.0, h = 0.7)
    ψ = rand_mps(ComplexF64, fill(2, L); maxdim = 6, rng = MersenneTwister(41))
    env = Environments(H, ψ)

    @test length(env) == L
    # Asked out of order, so the cache has to extend in both directions rather than only forward.
    for i in vcat(1:(L + 1), (L + 1):-1:1, [3, L, 1, 4])
        @test leftenv(env, H, ψ, i) ≈ left_env(ψ, H, i)
        @test rightenv(env, H, ψ, i) ≈ right_env(ψ, H, i)
    end

    # The two extremes are the whole expectation value, from either side.
    @test sum(leftenv(env, H, ψ, L + 1)) ≈ expect(H, ψ)
    @test sum(rightenv(env, H, ψ, 1)) ≈ expect(H, ψ)

    # Boundaries live on the state's backend and have dimension 1 on every leg.
    @test size(leftenv(env, H, ψ, 1)) == (1, 1, 1)
    @test size(rightenv(env, H, ψ, L + 1)) == (1, 1, 1)

    @test_throws ArgumentError leftenv(env, H, ψ, 0)
    @test_throws ArgumentError rightenv(env, H, ψ, L + 2)
end

@testset "invalidation after a site changes" begin
    L = 5
    H = tfim_mpo(ComplexF64, L; J = 1.0, h = 0.4)
    ψ = rand_mps(ComplexF64, fill(2, L); maxdim = 4, rng = MersenneTwister(42))
    env = Environments(H, ψ)

    # Warm every entry, so that a stale one would be *returned* rather than rebuilt by accident.
    for i in 1:(L + 1)
        leftenv(env, H, ψ, i)
        rightenv(env, H, ψ, i)
    end

    for j in 1:L
        t = copy(ψ[j])
        MatrixProductToolbox.setsite!(ψ, j, ψ[j] .+ 0.3 .* randn(MersenneTwister(j), size(t)...))
        MatrixProductToolbox.invalidate!(env, j)
        for i in 1:(L + 1)
            @test leftenv(env, H, ψ, i) ≈ left_env(ψ, H, i)
            @test rightenv(env, H, ψ, i) ≈ right_env(ψ, H, i)
        end
    end
end

@testset "the cache is not consulted beyond what it claims" begin
    # `left[i]` covers sites `1:i-1` and `right[i]` covers `i:L`, so changing site `j` must leave
    # `left[1:j]` and `right[j+1:L+1]` untouched. Checked by mutating a site and confirming that
    # the entries that do not contract it come back bit-identical.
    L = 5
    H = tfim_mpo(ComplexF64, L)
    ψ = rand_mps(ComplexF64, fill(2, L); maxdim = 4, rng = MersenneTwister(43))
    env = Environments(H, ψ)
    before_left = [copy(leftenv(env, H, ψ, i)) for i in 1:(L + 1)]
    before_right = [copy(rightenv(env, H, ψ, i)) for i in 1:(L + 1)]

    j = 3
    MatrixProductToolbox.setsite!(ψ, j, 2 .* ψ[j])
    MatrixProductToolbox.invalidate!(env, j)
    @test all(leftenv(env, H, ψ, i) == before_left[i] for i in 1:j)
    @test all(rightenv(env, H, ψ, i) == before_right[i] for i in (j + 1):(L + 1))
end

@testset "show" begin
    H = tfim_mpo(ComplexF64, 4)
    ψ = rand_mps(ComplexF64, fill(2, 4); maxdim = 2, rng = MersenneTwister(44))
    @test occursin("Environments(L=4", sprint(show, Environments(H, ψ)))
end
