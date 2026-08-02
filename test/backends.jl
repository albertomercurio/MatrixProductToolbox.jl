#=
The CPU baseline for the shared backend probes defined in `testutils.jl`.

This is not a redundant test: it establishes that the probe list itself is self-consistent, so
that a failure on `CuArray` or under Reactant points at the backend rather than at the probes.
The identity adaptor also exercises the `Adapt` path.

The other backends run the very same list:

    GROUP=CUDA      julia --project=. test/runtests.jl     # test/cuda/
    GROUP=Reactant  julia --project=. test/runtests.jl     # test/reactant/
=#

backend_testset("Array (identity adaptor)", identity, (f, a, b, c) -> f(a, b, c))
backend_testset(
    "Array (algorithms)", identity, (f, a, b, c) -> f(a, b, c); probes = ALGORITHM_PROBES
)
backend_testset(
    "Array (complex algorithms)", identity, (f, a, b, c) -> f(a, b, c);
    probes = COMPLEX_ALGORITHM_PROBES
)

backend_testset("Array (via Adapt)", ψ -> adapt(Array, ψ), (f, a, b, c) -> f(a, b, c))
