# MatrixProductToolbox

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://albertomercurio.github.io/MatrixProductToolbox.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://albertomercurio.github.io/MatrixProductToolbox.jl/dev/)
[![Build Status](https://github.com/albertomercurio/MatrixProductToolbox.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/albertomercurio/MatrixProductToolbox.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

Matrix-product states and operators for closed and open quantum systems.

- **Simple.** Six tensor contractions are the entire compute core. Norms, overlaps, expectation
  values and operator application are all built from them.
- **Flexible.** One generic container, `MatrixProduct{N}`, where `N` counts the physical legs per
  site. `MPS = MatrixProduct{1}`, `MPO = MatrixProduct{2}`, and open-system objects need no new
  types.
- **Platform agnostic.** Site tensors are plain `AbstractArray`s, so `Array`, `CuArray` and
  Reactant/XLA arrays all run through the *same* code path. There are no platform-specific
  dependencies and no backend hooks — just `TensorOperations`, `LinearAlgebra`, `Random` and
  `Adapt`.

## Quick start

```julia
using MatrixProductToolbox, LinearAlgebra

ψ = rand_mps(ComplexF64, fill(2, 6); maxdim = 8)
bond_dims(ψ)                      # [1, 2, 4, 8, 4, 2, 1]

canonicalize!(ψ, 3)               # exact gauge transformation
center(ψ)                         # 3:3
norm(ψ) ≈ norm(ψ[3])              # the centre tensor carries the norm

σz = ComplexF64[1 0; 0 -1]
expect(σz, ψ)                     # one value per site

φ = truncate(ψ; maxdim = 3)       # compress
abs(dot(φ, ψ))^2                  # ... and check the fidelity
```

Operators are ordinary vectors of rank-4 arrays, so writing one by hand is short — see the
[transverse-field Ising example](https://albertomercurio.github.io/MatrixProductToolbox.jl/dev/manual/matrix_products/)
in the documentation.

## GPU and XLA

```julia
using Adapt, CUDA
ψ_gpu = adapt(CuArray, ψ)         # everything else is unchanged

using Reactant
ψ_xla = Reactant.to_rarray(ψ)
sweep(H, ψ) = expect(H, canonicalize(ψ, length(ψ); trunc = FixedTruncation(8)))
compiled = @compile sweep(H_xla, ψ_xla)
```

Two notes on the XLA path: compile *one whole sweep* rather than individual contractions, since
fusion is the point; and use `FixedTruncation`, whose bond dimensions follow from the chain
geometry rather than from singular-value magnitudes, because XLA needs shapes at trace time. See
[Array backends](https://albertomercurio.github.io/MatrixProductToolbox.jl/dev/manual/backends/).

## Status

This is the foundation layer. **DMRG and TDVP are not implemented yet** — the effective-operator
kernels they build on (`apply_effective`) are in place and tested, so those algorithms are
additive rather than a redesign. Clifford-augmented MPS is the longer-term target, which is why
algorithms are written against an `AbstractMatrixProduct` interface rather than the concrete type.

## Development

```bash
make test                      # Core group
GROUP=Code-Quality make test   # Aqua + JET
GROUP=Reactant     make test   # XLA path, runs on CPU
GROUP=CUDA         make test   # needs a GPU
make format                    # Runic
make docs
```

Run groups separately: JET is by far the heaviest part of the suite.
