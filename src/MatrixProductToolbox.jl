module MatrixProductToolbox

using LinearAlgebra

## Standard Julia libraries
import LinearAlgebra: dot, norm, normalize, normalize!, qr, svd, svdvals, I
import Random: AbstractRNG, default_rng

## other dependencies (in alphabetical order)
import Adapt: adapt, adapt_structure
import TensorOperations: @tensor, scalartype

# Export functions from the other modules

## LinearAlgebra
export dot, norm, normalize, normalize!

## TensorOperations
export scalartype

## Abstract supertype and interface
include("abstract.jl")

## Core container
include("matrixproduct.jl")

## Contraction kernels: the entire compute core
include("kernels.jl")

## Truncation policies and factorizations
include("factorizations.jl")

## Gauge transformations
include("canonical.jl")

## Construction
include("states.jl")

## Observables and operations
include("observables.jl")
include("apply.jl")
include("dense.jl")

## Local solvers and the environment cache the sweeping algorithms are built from
include("krylov.jl")
include("environments.jl")

## Algorithms
include("dmrg.jl")
include("tdvp.jl")

end
