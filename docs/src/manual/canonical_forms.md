```@meta
CurrentModule = MatrixProductToolbox
```

# [Canonical forms and truncation](@id man-canonical)

## The gauge invariant

A matrix-product representation is not unique: you can insert ``G G^{-1}`` on any bond without
changing the state. Fixing that freedom — putting the object in a *canonical form* — is what makes
norms cheap, truncation optimal, and (later) DMRG stable.

This package tracks the gauge with a single field, [`center`](@ref), which is **the range of
sites not known to be orthogonal**:

- sites `1:first(center)-1` are left-orthogonal, ``\sum_{l,p} \overline{A[l,p,r']} A[l,p,r] =
  \delta_{r'r}``;
- sites `last(center)+1:end` are right-orthogonal, ``\sum_{p,r} A[l,p,r]
  \overline{A[l',p,r]} = \delta_{ll'}``.

| Value | Meaning |
|:---|:---|
| `1:L` | nothing claimed — a freshly constructed object |
| `c:c` | mixed-canonical, centre at site `c` |
| `1:1` / `L:L` | right- / left-canonical |
| `i:i+1` | a two-site block |

Every value carries information, so there is no "unknown" sentinel to test for.

```@example gauge
using MatrixProductToolbox
using LinearAlgebra
using Random

ψ = rand_mps(ComplexF64, fill(2, 5); maxdim = 4, rng = MersenneTwister(0))
center(ψ), iscanonical(ψ), isrightcanonical(ψ)
```

A hand-built object claims nothing until you gauge it:

```@example gauge
raw = MPS([randn(ComplexF64, 1, 2, 2), randn(ComplexF64, 2, 2, 2), randn(ComplexF64, 2, 2, 1)])
center(raw), iscanonical(raw)
```

```@example gauge
center(canonicalize!(raw, 2))
```

That this needs no special case is the practical payoff of the convention: `canonicalize!` walks
from `first(center)` up to `c` and from `last(center)` down to `c`, which for a fresh `1:L` *is*
the full sweep.

## Moving the centre

[`canonicalize!`](@ref) is a pure gauge transformation — exact, and it leaves the state alone:

```@example gauge
before = dense(ψ)
canonicalize!(ψ, 4)
dense(ψ) ≈ before, center(ψ)
```

```@example gauge
all(isleftorthogonal(ψ[i]) for i in 1:3), isrightorthogonal(ψ[5])
```

Only the bonds between the current centre and the target are touched, so sweeping the centre one
site at a time costs one factorization per step — the access pattern DMRG and TDVP want.

With a canonical form, the norm is just the norm of the centre tensor:

```@example gauge
norm(ψ) ≈ norm(ψ[4])
```

[`canonicalize`](@ref) is the non-mutating variant. Throughout the package, `!` functions mutate
and return their argument, and the copying version is defined as
`f(ψ, …) = f!(copy(ψ), …)`.

## Truncation: two policies

Truncation is *not* a gauge transformation — it changes the state. Use [`truncate!`](@ref), which
sweeps to one end, then back applying the policy at **every** bond, so each discarded value is a
true Schmidt coefficient of that cut.

The choice of policy is the one place where the target hardware matters.

| Policy | Rank rule | Depends on | `Array` / `CuArray` | Reactant / XLA |
|:---|:---|:---|:---:|:---:|
| [`NoTruncation`](@ref) | `k = min(m, n)` | shape | ✅ | ✅ |
| [`FixedTruncation`](@ref) | `k = min(maxdim, m, n)` | shape | ✅ | ✅ |
| [`Truncation`](@ref) | smallest `k` meeting `cutoff` | singular-value magnitudes | ✅ | ❌ |

XLA requires every array shape to be known at trace time. It does **not** require uniform bond
dimensions — only that they follow from *shapes alone*. That is the top two rows.

### Cutoff truncation

Adaptive, and what you normally want on CPU or GPU. Bond dimensions follow the state:

```@example gauge
ψ = rand_mps(ComplexF64, fill(2, 8); maxdim = 16, rng = MersenneTwister(1))
bond_dims(ψ)
```

```@example gauge
bond_dims(truncate(ψ; maxdim = 4))
```

```@example gauge
bond_dims(truncate(ψ; cutoff = 1e-2))
```

`cutoff` bounds the *relative discarded weight*
``\varepsilon = \sum_{i>k}\sigma_i^2 / \sum_i \sigma_i^2`` on each bond.

### Fixed truncation

```@example gauge
bond_dims(truncate(ψ; trunc = FixedTruncation(4)))
```

!!! note "`FixedTruncation` is not a uniform bond dimension"
    Near the ends of the chain `min(m, n) < maxdim`, so those bonds are genuinely smaller — but
    *deterministically*, from the chain geometry. That is exactly what XLA needs. Two happy
    consequences: no zero-padding is ever required, and the isometries stay exact (`Q'Q == I`
    rather than a projector).

Where the geometric cap binds before the cutoff does, the two policies agree exactly — which is
worth knowing when moving code between backends.

## Inspecting a factorization

The factorization layer is available directly, and reports what it discarded:

```@example gauge
M = diagm([1.0, 0.1, 0.01, 0.001])
Q, R, info = factorize_left(SVDFactorization(), Truncation(; cutoff = 1e-3), M)
info
```

```@example gauge
size(Q), Q' * Q ≈ I, Q * R ≈ M
```

[`factorize_left`](@ref) produces orthonormal *columns* for left-to-right sweeps;
[`factorize_right`](@ref) produces orthonormal *rows* for the other direction.

### Why the SVD is the only factorization

A QR would be roughly 2–3× faster on untruncated sweeps, but the SVD is needed for truncation
regardless, and it is the one factorization available on *every* backend — Reactant has no `qr`.
Using it everywhere means a single code path and no per-backend configuration.

[`AbstractFactorization`](@ref) and the `factorization` keyword exist so that adding
`QRFactorization` later is purely additive:

```julia
canonicalize!(ψ, 3; factorization = SVDFactorization())
```

## Entanglement

With a canonical form, the Schmidt spectrum across a bond is one SVD away:

```@example gauge
round.(entanglement_spectrum(ψ, 4); digits = 4)
```

```@example gauge
round.([entanglement_entropy(ψ, b) for b in 1:7]; digits = 4)
```

The spectrum is normalised, so the entropy ignores the norm of the state. A product state gives
zero and a Bell pair gives exactly ``\log 2``:

```@example gauge
p = product_mps([2, 2], [1, 2])
bell = MPS([reshape(ComplexF64[1, 0, 0, 1] ./ sqrt(2), 1, 2, 2), reshape(ComplexF64[1, 0, 0, 1], 2, 2, 1)])
entanglement_entropy(p, 1), entanglement_entropy(bell, 1) ≈ log(2)
```
