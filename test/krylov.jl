#=
The local solvers are checked against dense linear algebra on an explicit matrix, so that the
reference is `eigen` and `exp` themselves rather than another Krylov method.

The load-bearing check is the last one: `DenseRitz` and `TracedRitz` differ only in how they
solve the *projected* problem, so at a fixed `krylovdim` they must agree to floating point. That
equivalence is what lets the Reactant tests compare a traced sweep against the CPU answer to a
tight tolerance instead of a loose one, so it is asserted here on the CPU first.
=#

# A Hermitian linear map from an explicit matrix, applied so that the solvers see an array of the
# same shape they were handed -- which is how DMRG and TDVP call them.
_map_of(A) = x -> reshape(A * vec(x), size(x))

function _random_hermitian(::Type{T}, n, rng) where {T}
    A = randn(rng, T, n, n)
    return A + A'
end

# Stands in for a traced scalar: `<: Number`, but a comparison of two of them does not yield a
# host `Bool`. That is the one property `_require_host_ritz` tests for, so it is the one property
# this needs, and faking it here keeps the guard's test out of the Reactant sub-environment.
struct NotHostReal <: Number end
Base.:>(::NotHostReal, ::NotHostReal) = NotHostReal()
Base.zero(::NotHostReal) = NotHostReal()

@testset "Lanczos eigensolver against `eigen` ($(nameof(typeof(ritz))))" for
    ritz in (DenseRitz(), TracedRitz())
    rng = MersenneTwister(31)
    for (n, T) in ((24, Float64), (24, ComplexF64), (40, ComplexF64))
        A = _random_hermitian(T, n, rng)
        exact = eigvals(Hermitian(A))
        x₀ = randn(rng, T, n)

        # A Krylov space as large as the problem is exact.
        λ, v = krylov_eigen(LanczosEigen(; krylovdim = n, maxiter = 2, ritz), _map_of(A), x₀)
        @test λ ≈ exact[1]
        @test norm(v) ≈ 1
        @test norm(A * v - λ * v) < 1.0e-8 * max(1, abs(λ))

        # A smaller one is variational: never below the true ground state, and converging to it.
        λ_small, _ = krylov_eigen(LanczosEigen(; krylovdim = 4, maxiter = 1, ritz), _map_of(A), x₀)
        λ_big, _ = krylov_eigen(LanczosEigen(; krylovdim = 12, maxiter = 1, ritz), _map_of(A), x₀)
        @test λ_small ≥ exact[1] - 1.0e-10
        @test λ_big ≥ exact[1] - 1.0e-10
        @test λ_big ≤ λ_small + 1.0e-10
    end
end

@testset "Lanczos exponential against `exp` ($(nameof(typeof(ritz))))" for
    ritz in (DenseRitz(), TracedRitz())
    rng = MersenneTwister(32)
    for (n, T) in ((20, Float64), (20, ComplexF64))
        A = _random_hermitian(T, n, rng)
        x₀ = randn(rng, T, n)
        for τ in (-0.05im, -0.3im, -0.05, 0.02)
            # A real step must not widen a real problem, which is what makes imaginary-time TDVP
            # usable on a `Float64` state.
            (T <: Real && τ isa Complex) && continue
            y = krylov_exp(LanczosExponential(; krylovdim = n, ritz), τ, _map_of(A), x₀)
            @test y ≈ exp(τ * A) * x₀
            @test eltype(y) === promote_type(T, typeof(τ))
        end
    end
end

@testset "the solvers preserve the shape they are given" begin
    # DMRG hands them a rank-4 two-site block and TDVP a rank-3 site tensor; neither reshapes
    # first, so the recurrences have to be shape-agnostic.
    rng = MersenneTwister(33)
    for shape in ((2, 3, 4), (3, 2, 2, 3))
        n = prod(shape)
        A = _random_hermitian(ComplexF64, n, rng)
        x₀ = randn(rng, ComplexF64, shape...)
        λ, v = krylov_eigen(LanczosEigen(; krylovdim = n), _map_of(A), x₀)
        @test size(v) == shape
        @test λ ≈ eigvals(Hermitian(A))[1]
        y = krylov_exp(LanczosExponential(; krylovdim = n), -0.1im, _map_of(A), x₀)
        @test size(y) == shape
        @test vec(y) ≈ exp(-0.1im * A) * vec(x₀)
    end
end

@testset "a Krylov space larger than the problem" begin
    # Near the ends of a chain a two-site block can be smaller than `krylovdim`, so the basis is
    # capped at `length(x₀)`. That cap is a *shape*, which is what keeps it legal under tracing;
    # without it the recurrence would break down and divide by a vanishing sub-diagonal.
    rng = MersenneTwister(34)
    for n in 1:5
        A = _random_hermitian(Float64, n, rng)
        x₀ = randn(rng, n)
        for ritz in (DenseRitz(), TracedRitz())
            λ, v = krylov_eigen(LanczosEigen(; krylovdim = 16, maxiter = 3, ritz), _map_of(A), x₀)
            @test λ ≈ eigvals(Symmetric(A))[1]
            @test norm(A * v - λ * v) < 1.0e-8 * max(1, abs(λ))
            y = krylov_exp(LanczosExponential(; krylovdim = 16, ritz), -0.1, _map_of(A), x₀)
            @test y ≈ exp(-0.1 * A) * x₀
        end
    end
end

@testset "a start vector that is already almost an eigenvector" begin
    #=
    The case a naive Lanczos gets wrong, and the one DMRG is in constantly: once a sweep has half
    converged, the two-site block handed to the solver is already nearly the local eigenvector, so
    the very first residual is tiny — β₁ ≈ 1e-5 against an operator of norm ≈ 20 — and dividing by
    it amplifies whatever a single Gram-Schmidt pass left behind. The basis stops being
    orthonormal, T is no longer VᴴAV, and the Ritz value drops *below* the true minimum, which a
    variational method must never produce. Measured before the fix: -29.8 against a true -21.8.

    Two assertions, because either alone would have missed it: the Ritz value must be a genuine
    variational bound, and it must agree with the Rayleigh quotient of the vector returned beside
    it. The second is what pins the basis down.
    =#
    rng = MersenneTwister(36)
    n = 80
    A = _random_hermitian(Float64, n, rng)
    F = eigen(Symmetric(A))
    exact, v₁ = F.values[1], F.vectors[:, 1]
    f = _map_of(A)

    for gap in (1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8, 1.0e-10, 0.0)
        # A start vector a controlled distance from the exact eigenvector, including *at* it.
        x₀ = v₁ .+ gap .* randn(rng, n)
        for kd in (4, 8, 12, 20, 30), ritz in (DenseRitz(), TracedRitz())
            λ, v = krylov_eigen(LanczosEigen(; krylovdim = kd, maxiter = 1, ritz), f, x₀)
            @test λ ≥ exact - 1.0e-9
            @test λ ≈ dot(v, f(v)) / dot(v, v) atol = 1.0e-8
        end
    end
end

@testset "the two Ritz policies solve the same problem" begin
    # Same recurrence, same fixed `krylovdim`, different projected solve. Compared over many
    # random problems and scales, because this is the assumption every cross-backend comparison
    # of `dmrg`/`tdvp` rests on.
    rng = MersenneTwister(35)
    for trial in 1:60
        n = rand(rng, 2:24)
        T = isodd(trial) ? Float64 : ComplexF64
        scale = 10.0^rand(rng, -2:2)
        A = scale .* _random_hermitian(T, n, rng)
        x₀ = randn(rng, T, n)
        kd = min(rand(rng, 2:16), n)

        λd, vd = krylov_eigen(LanczosEigen(; krylovdim = kd, maxiter = 2, ritz = DenseRitz()), _map_of(A), x₀)
        λt, vt = krylov_eigen(LanczosEigen(; krylovdim = kd, maxiter = 2, ritz = TracedRitz()), _map_of(A), x₀)
        @test λt ≈ λd rtol = 1.0e-10
        # Residuals rather than an overlap: where two eigenvalues are degenerate the eigenvector
        # is not unique, so two correct solvers may return different vectors from one eigenspace.
        # What must match is how well each solves the problem.
        @test norm(A * vt - λt * vt) ≤ 1.0e-8 * scale + 10 * norm(A * vd - λd * vd)

        τ = isodd(trial) ? -0.2 / scale : -0.2im / scale
        yd = krylov_exp(LanczosExponential(; krylovdim = kd, ritz = DenseRitz()), τ, _map_of(A), x₀)
        yt = krylov_exp(LanczosExponential(; krylovdim = kd, ritz = TracedRitz()), τ, _map_of(A), x₀)
        @test yt ≈ yd rtol = 1.0e-9
    end
end

@testset "`DenseRitz` refuses a backend it cannot branch on" begin
    # `_require_host_ritz` is the same shape of guard as `_truncation_rank`'s and fires for the
    # same reason. It is asserted on the *message* rather than merely on something being thrown,
    # because the point of the guard is that it names the way out.
    err = try
        MatrixProductToolbox._require_host_ritz(DenseRitz(), NotHostReal())
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("TracedRitz", err.msg)
    @test MatrixProductToolbox._require_host_ritz(TracedRitz(), NotHostReal()) === nothing
end

@testset "solver construction" begin
    @test_throws ArgumentError LanczosEigen(; krylovdim = 1)
    @test_throws ArgumentError LanczosEigen(; maxiter = 0)
    @test_throws ArgumentError LanczosExponential(; krylovdim = 0)

    # Only the traceable policy switches the convergence test off, and it does so on its own so
    # that `tol` and `ritz` cannot be left inconsistent.
    @test MatrixProductToolbox._tol(LanczosEigen(; tol = 1.0e-9)) == 1.0e-9
    @test MatrixProductToolbox._tol(LanczosEigen(; tol = 1.0e-9, ritz = TracedRitz())) == 0.0
    @test !MatrixProductToolbox._traceable(LanczosEigen())
    @test MatrixProductToolbox._traceable(LanczosEigen(; ritz = TracedRitz()))
    @test MatrixProductToolbox._traceable(LanczosExponential(; ritz = TracedRitz()))
end
