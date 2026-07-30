```@meta
CurrentModule = MatrixProductToolbox
```

# [Matrix products](@id man-matrix-products)

## One container, `N` physical legs

There is a single concrete type, [`MatrixProduct{N,T,A}`](@ref), where `N` is the number of
physical legs each site carries. The familiar names are aliases:

```julia
const MPS = MatrixProduct{1}    # A[a, s, a′]
const MPO = MatrixProduct{2}    # W[w, s′, s, w′]
```

They are still *distinct types*, so `dot(::MPS, ::MPS)` and `apply(::MPO, ::MPS)` dispatch
separately — while gauge, truncation and backend code is written once.

```@example mp
using MatrixProductToolbox
using LinearAlgebra

ψ = product_mps([2, 2, 2], [1, 2, 1])
ψ isa MPS, ψ isa MatrixProduct{1}, ψ isa MPO
```

`N` is inferred from the rank of the site tensors, so you rarely name it:

```@example mp
MatrixProduct([reshape(ComplexF64[1, 0], 1, 2, 1)]) isa MPS
```

!!! note "Strictly one-dimensional"
    Every site has exactly one left bond and one right bond. Generic tensor networks with
    arbitrary connectivity (PEPS, MERA, tree networks) need a graph-structured container and are
    deliberately out of scope. `N` varies the number of physical legs, never the connectivity.

### Open quantum systems come for free

Because `N` is a parameter rather than a hard-coded 1 or 2, open-system representations need no
new types:

- a **density matrix** in matrix-product form is a `MatrixProduct{2}`;
- a **vectorised** state evolving under a Liouvillian is a `MatrixProduct{1}` with local
  dimension ``d^2``, acted on by a `MatrixProduct{2}` Liouvillian.

Gauging and truncation already work for any `N`, because they operate on the fused
`(l, p…, r)` reshape rather than on a fixed rank.

## Queries

```@example mp
ψ = rand_mps(ComplexF64, [2, 3, 2, 3]; maxdim = 4)
length(ψ), nphysical(ψ), scalartype(ψ)
```

```@example mp
physical_dims(ψ), bond_dims(ψ), maxbonddim(ψ)
```

[`bond_dims`](@ref) has `length(ψ) + 1` entries and includes the two trivial boundary bonds, so
`bond_dims(ψ)[i]` is the bond to the *left* of site `i`.

For `N ≥ 2`, [`physical_dims`](@ref) returns one tuple per site:

```@example mp
physical_dims(identity_mpo([2, 3]))
```

## Building states

```@example mp
# A computational basis state |1 2 1⟩ (one-based indices).
product_mps([2, 2, 2], [1, 2, 1])
```

```@example mp
# Or from explicit local vectors -- they need not be normalised.
product_mps([[1.0, 1.0] ./ sqrt(2), [3.0, 0.0]])
```

```@example mp
# A random state saturating the given bond dimension.
rand_mps(ComplexF64, fill(2, 5); maxdim = 4)
```

## Building an operator by hand

There is no general Hamiltonian-MPO builder yet, and for many models one is not needed: an MPO
is a `Vector` of rank-4 arrays, and writing one down directly is short and transparent.

The standard construction reads the operator off a finite-state machine that scans the chain
left to right. For the transverse-field Ising model

```math
\hat{H} = -J \sum_i \hat{\sigma}^z_i \hat{\sigma}^z_{i+1} - h \sum_i \hat{\sigma}^x_i
```

three internal states suffice: *nothing placed yet*, *a ``\hat\sigma^z`` is open*, and *all terms
placed*.

```@example mp
function tfim_mpo(::Type{T}, L::Integer; J = 1.0, h = 0.5) where {T}
    id = Matrix{T}(I, 2, 2)
    σx = T[0 1; 1 0]
    σz = T[1 0; 0 -1]

    W = zeros(T, 3, 2, 2, 3)
    W[1, :, :, 1] = id          # still to the left of every term
    W[1, :, :, 2] = σz          # open a σᶻσᶻ bond
    W[1, :, :, 3] = -h * σx     # place the field
    W[2, :, :, 3] = -J * σz     # close the σᶻσᶻ bond
    W[3, :, :, 3] = id          # already past every term

    tensors = [copy(W) for _ in 1:L]
    tensors[1] = W[1:1, :, :, :]        # start in the "nothing yet" state
    tensors[end] = W[:, :, :, 3:3]      # end in the "all done" state
    return MPO(tensors)
end

H = tfim_mpo(ComplexF64, 6)
```

Trimming the first and last tensors is what keeps the boundary bonds at dimension 1, as the
constructor requires.

Check it against exact diagonalization on a small chain:

```@example mp
H4 = tfim_mpo(ComplexF64, 4)
ground = minimum(real(eigvals(Hermitian(dense(H4)))))
round(ground; digits = 8)
```

## Applying an operator

[`apply`](@ref) contracts the physical indices and fuses the bonds, so bond dimensions
**multiply**:

```@example mp
ψ = rand_mps(ComplexF64, fill(2, 6); maxdim = 4)
bond_dims(ψ), bond_dims(H), bond_dims(apply(H, ψ))
```

That growth is why repeated application needs a truncation:

```@example mp
bond_dims(apply(H, ψ; trunc = Truncation(; maxdim = 8)))
```

`H * ψ` is shorthand for the exact, untruncated form.

## Observables

```@example mp
normalize!(ψ)
round(real(expect(H, ψ)); digits = 6)
```

!!! warning "Expectation values are not normalised"
    [`expect`](@ref) returns ``\langle \psi | \hat{O} | \psi \rangle`` without dividing by
    ``\langle \psi | \psi \rangle``. Call [`normalize!`](@ref) first, or divide by `norm(ψ)^2`.

Single-site operators can be measured at one site or at every site:

```@example mp
σz = ComplexF64[1 0; 0 -1]
round.(real(expect(σz, ψ)); digits = 4)
```

## The dense representation

[`dense`](@ref) contracts everything into a state vector or matrix. It costs exponential memory,
so it is a testing and teaching tool — and it refuses to allocate beyond `maxsize`.

```@example mp
small = product_mps([2, 3], [2, 3])
reshape(dense(small), 2, 3)
```

!!! note "Site 1 varies fastest"
    `reshape(dense(ψ), physical_dims(ψ)...)` has dimension `i` equal to site `i`. This is the
    ordering that needs no `permutedims`, and it is the **reverse** of `kron`'s convention: to
    compare against a `kron`-built reference, assemble it as `reduce(kron, reverse(ops))`.

## Extending the package

Algorithms are written against [`AbstractMatrixProduct`](@ref), not the concrete type. A subtype
supplies five methods — `length`, `getindex`, [`center`](@ref), `setsite!` and `setcenter!` —
and everything generic follows.

This is what a Clifford-augmented MPS will look like:

```julia
struct CliffordMPS <: AbstractMatrixProduct{1}
    tableau::CliffordTableau
    mps::MPS
end
```
