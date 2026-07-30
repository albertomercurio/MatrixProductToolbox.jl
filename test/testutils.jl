#=
Shared fixtures for the test suite.
=#

const σx = ComplexF64[0 1; 1 0]
const σy = ComplexF64[0 -im; im 0]
const σz = ComplexF64[1 0; 0 -1]

@doc raw"""
    tfim_mpo(T, L; J = 1.0, h = 0.5)

Transverse-field Ising Hamiltonian ``-J\sum \hat{\sigma}^z_i \hat{\sigma}^z_{i+1} - h\sum
\hat{\sigma}^x_i`` as an `MPO` of bond dimension 3, assembled by hand.

Doubles as the worked example in the documentation: building an MPO amounts to filling a
rank-4 array and trimming the two boundary bonds to dimension 1.
"""
function tfim_mpo(::Type{T}, L::Integer; J::Real = 1.0, h::Real = 0.5) where {T <: Number}
    L ≥ 2 || throw(ArgumentError("Need at least 2 sites."))
    id = Matrix{T}(I, 2, 2)
    W = zeros(T, 3, 2, 2, 3)
    W[1, :, :, 1] = id           # still to the left of every term
    W[1, :, :, 2] = σz           # open a σᶻσᶻ bond
    W[1, :, :, 3] = -h * σx      # place the field
    W[2, :, :, 3] = -J * σz      # close the σᶻσᶻ bond
    W[3, :, :, 3] = id           # already past every term
    tensors = [copy(W) for _ in 1:L]
    tensors[1] = W[1:1, :, :, :]         # start in the "nothing yet" state
    tensors[end] = W[:, :, :, 3:3]       # end in the "all done" state
    return MPO(tensors)
end

# Dense operator matching `dense`'s index ordering, in which site 1 varies fastest. That is
# the reverse of `kron`'s convention, hence the `reverse`.
kron_ops(ops) = reduce(kron, reverse(ops))

# A single-site operator padded out to the full chain.
function site_op(op::AbstractMatrix, L::Integer, site::Integer)
    d = size(op, 1)
    return kron_ops([i == site ? op : Matrix{eltype(op)}(I, d, d) for i in 1:L])
end

# Reference environments, built by explicit transfer sweeps with no caching. This is exactly
# what the future environment cache will memoize, so it doubles as the specification.
function left_env(ψ, H, i)                      # covers sites 1 … i-1
    T = promote_type(scalartype(H), scalartype(ψ))
    E = fill!(similar(ψ[1], T, (1, 1, 1)), one(T))
    for k in 1:(i - 1)
        E = transfer_left(E, ψ[k], H[k], ψ[k])
    end
    return E
end

function right_env(ψ, H, i)                     # covers sites i … L
    T = promote_type(scalartype(H), scalartype(ψ))
    E = fill!(similar(ψ[1], T, (1, 1, 1)), one(T))
    for k in length(ψ):-1:i
        E = transfer_right(E, ψ[k], H[k], ψ[k])
    end
    return E
end

# The maximally entangled two-site state, as an MPS with bond dimension 2.
bell_mps() = MPS(
    [
        reshape(ComplexF64[1, 0, 0, 1] ./ sqrt(2), 1, 2, 2),
        reshape(ComplexF64[1, 0, 0, 1], 2, 2, 1),
    ]
)

#=
Backend genericity, in one place.

Every probe below is a **scalar-valued function of `(H, ψ, φ)`**, which is what makes a single
test body work across every backend:

  * the arity is fixed, so a backend only has to supply a three-argument runner;
  * the result is a number, so there is nothing backend-specific to unwrap;
  * `FixedTruncation` is used for the gauge probes, since its bond dimensions follow from the
    shapes alone and are therefore legal under XLA tracing as well.

`test/backends.jl` runs these eagerly on `Array`, `test/cuda/` on `CuArray`, and
`test/reactant/` through `@jit`. Adding a backend means supplying an adaptor and a runner —
never rewriting a check.
=#
const BACKEND_SITES = 5
const BACKEND_CHI = 4

const BACKEND_PROBES = [
    "dot" => (H, ψ, φ) -> dot(ψ, φ),
    "self-overlap" => (H, ψ, φ) -> dot(ψ, ψ),
    "expect" => (H, ψ, φ) -> expect(H, ψ),
    "apply then overlap" => (H, ψ, φ) -> dot(φ, apply(H, ψ)),
    "scalar multiply" => (H, ψ, φ) -> dot(2 * ψ, ψ),
    "canonicalize" => (H, ψ, φ) -> expect(H, canonicalize(ψ, BACKEND_SITES)),
    "canonicalize mid-chain" => (H, ψ, φ) -> expect(H, canonicalize(ψ, 3)),
    "fixed truncation" =>
        (H, ψ, φ) -> expect(H, canonicalize(ψ, BACKEND_SITES; trunc = FixedTruncation(2))),
]

# The reference inputs, on the CPU. A backend adapts these and compares against the values the
# probes produce here.
function backend_inputs()
    L = BACKEND_SITES
    ψ = rand_mps(ComplexF64, fill(2, L); maxdim = BACKEND_CHI, rng = MersenneTwister(14))
    φ = rand_mps(ComplexF64, fill(2, L); maxdim = BACKEND_CHI, rng = MersenneTwister(15))
    return tfim_mpo(ComplexF64, L), ψ, φ
end

@doc raw"""
    backend_testset(name, adaptor, run)

Run every probe in `BACKEND_PROBES` on one array backend and compare against the CPU answer.

# Arguments
- `name`: label for the testset.
- `adaptor`: moves a matrix-product object to the backend, e.g. `ψ -> adapt(CuArray, ψ)`.
- `run`: invokes a three-argument probe and returns a host `Number`. Eager backends pass
  `(f, a, b, c) -> f(a, b, c)`; Reactant wraps the call in `@jit`.
"""
function backend_testset(name::AbstractString, adaptor, run)
    return @testset "$name" begin
        H_cpu, ψ_cpu, φ_cpu = backend_inputs()
        H, ψ, φ = adaptor(H_cpu), adaptor(ψ_cpu), adaptor(φ_cpu)

        # Structural metadata must survive the move: it is host-side data, not array data.
        @test ψ isa MPS
        @test H isa MPO
        @test center(ψ) == center(ψ_cpu)
        @test bond_dims(ψ) == bond_dims(ψ_cpu)
        @test physical_dims(ψ) == physical_dims(ψ_cpu)

        for (label, probe) in BACKEND_PROBES
            @testset "$label" begin
                @test run(probe, H, ψ, φ) ≈ probe(H_cpu, ψ_cpu, φ_cpu)
            end
        end
    end
end
