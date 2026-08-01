```@meta
CurrentModule = MatrixProductToolbox
```

# [API](@id api)

```@contents
Pages = ["api.md"]
Depth = 2
```

## Types

```@docs
AbstractMatrixProduct
MatrixProduct
MPS
MPO
```

## Construction

```@docs
rand_mps
product_mps
product_mpo
identity_mpo
```

## Queries

```@docs
nphysical
physical_dims
bond_dims
maxbonddim
center
iscanonical
isleftcanonical
isrightcanonical
isleftorthogonal
isrightorthogonal
Base.setindex!
```

## Canonical forms and truncation

```@docs
canonicalize!
canonicalize
truncate!
Base.truncate
```

### Truncation policies

```@docs
AbstractTruncation
NoTruncation
Truncation
FixedTruncation
```

### Factorizations

```@docs
AbstractFactorization
SVDFactorization
QRFactorization
factorize_left
factorize_right
```

## Contraction kernels

These six functions are the entire compute core of the package.

```@docs
transfer_left
transfer_right
apply_effective
```

## Observables

```@docs
dot
norm
normalize!
normalize
expect
entanglement_spectrum
entanglement_entropy
```

## Operators

```@docs
apply
Base.:*
```

## Dense representation

```@docs
dense
Base.Array
```

## Backend support

```@docs
adapt_structure
```

## Interface for new subtypes

A subtype of [`AbstractMatrixProduct`](@ref) implements `Base.length`, `Base.getindex`,
[`center`](@ref), and these two internal setters. Everything else in the package is generic.

```@docs
setsite!
setcenter!
```

## Index

```@index
Pages = ["api.md"]
```
