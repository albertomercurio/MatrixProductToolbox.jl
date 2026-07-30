```@meta
CurrentModule = MatrixProductToolbox
```

# MatrixProductToolbox.jl

Matrix-product states and operators for closed and open quantum systems, built around three
goals:

- **Simplicity.** Six tensor contractions are the entire compute core. Everything else — norms,
  overlaps, expectation values, operator application — is built from them.
- **Flexibility.** One generic container, [`MatrixProduct`](@ref), parametrised by the number `N`
  of physical legs per site. `N = 1` is a state, `N = 2` an operator, and open-system objects
  need no new types.
- **Platform agnosticism.** Site tensors are plain `AbstractArray`s, so `Array`, `CuArray` and
  Reactant/XLA arrays all work through the *same* code path. The package has no
  platform-specific dependencies and no backend hooks.

## Installation

```julia
using Pkg
Pkg.add("MatrixProductToolbox")
```

## A first look

```@example intro
using MatrixProductToolbox
using LinearAlgebra
using Random

ψ = rand_mps(ComplexF64, fill(2, 6); maxdim = 8, rng = MersenneTwister(0))
```

Bond dimensions are capped from both ends by the local Hilbert spaces, so `maxdim = 8` is
reached only in the middle of the chain:

```@example intro
bond_dims(ψ)
```

`rand_mps` returns a normalised, right-canonical state. Moving the orthogonality centre is an
exact gauge transformation — it changes the tensors but not the state:

```@example intro
canonicalize!(ψ, 3)
center(ψ), norm(ψ)
```

```@example intro
isleftorthogonal(ψ[1]), isrightorthogonal(ψ[6])
```

Measure a local operator on every site:

```@example intro
σz = ComplexF64[1 0; 0 -1]
round.(real(expect(σz, ψ)); digits = 4)
```

Compress the state, and see what it cost:

```@example intro
φ = truncate(ψ; maxdim = 3)
bond_dims(φ), round(abs(dot(φ, ψ))^2; digits = 6)
```

## Index conventions

Every site tensor has the index order

```
A[l, p₁, …, p_N, r]
```

— left bond first, physical legs in the middle, right bond last. Boundary bonds keep dimension
`1` rather than being dropped, so every tensor has exactly `N + 2` indices and no routine needs a
special case for the ends of the chain.

| | Index order | Meaning |
|:---|:---|:---|
| [`MPS`](@ref) | `A[a, s, a′]` | left bond, physical, right bond |
| [`MPO`](@ref) | `W[w, s′, s, w′]` | left bond, **output** physical, **input** physical, right bond |

For an [`MPO`](@ref), `W[w, s′, s, w′] = ⟨s′|Ŵ|s⟩`: the bra contracts with `s′` and the ket
with `s`.

This ordering is chosen so that **both** canonicalization reshapes are contiguous —
`reshape(A, l * ∏p, r)` going left to right and `reshape(A, l, ∏p * r)` going right to left — so
gauging a matrix-product object needs no `permutedims` at all.

## Where to go next

- [Matrix products](@ref man-matrix-products) — the container, the `MPS`/`MPO` aliases, and how
  to build an operator by hand.
- [Canonical forms and truncation](@ref man-canonical) — the gauge invariant and the two
  truncation policies.
- [Array backends](@ref man-backends) — CPU, CUDA and Reactant/XLA.
- [API](@ref api) — the full reference.

## What is not here yet

DMRG and TDVP are not implemented. The pieces they need — the effective-operator kernels and
their index conventions — are in place and tested, so those algorithms will be additive rather
than a redesign. See [`apply_effective`](@ref).
