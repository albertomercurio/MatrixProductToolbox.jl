#=
The complete compute core of the package: six tensor contractions. Norms, overlaps,
expectation values, MPO application and (later) DMRG / TDVP are all built from these.

Index conventions
-----------------
site tensor   A[a, s, a′]        left bond, physical, right bond
MPO tensor    W[w, s′, s, w′]    left bond, output physical, input physical, right bond
environment   L[b, w, a]         bra bond, MPO bond, ket bond   (L[b, a] without an MPO)

Every contraction is staged explicitly into pairwise `@tensor` steps rather than written as
one four-tensor expression: `@tensoropt` without extent hints assumes all dimensions are
equal and can choose a poor order, whereas the optimal staging here is standard and known.

No kernel repeats an index within a single tensor, so TensorOperations' `tensortrace!` is
never reached. That matters because its `BaseCopy` implementation — the one Reactant routes
to — unrolls a Julia loop over diagonal slices at trace time.
=#

export transfer_left, transfer_right, apply_effective

@doc raw"""
    transfer_left(L, A, B)

Grow an overlap environment to the right by one site.

Contracts

```math
L'[b', a'] = \sum_{b, s, a} L[b, a] \, B[b, s, b']^* \, A[a, s, a']
```

so `A` is the ket tensor and `B` the bra tensor (the conjugated one).

# Arguments
- `L`: the current environment, indexed `[bra bond, ket bond]`.
- `A`: the ket site tensor, indexed `[a, s, a′]`.
- `B`: the bra site tensor, indexed `[b, s, b′]`; it is conjugated.

# Returns
- `L′`: the environment one site further right, indexed `[bra bond, ket bond]`.
"""
function transfer_left(L::AbstractMatrix, A::AbstractArray{<:Any, 3}, B::AbstractArray{<:Any, 3})
    @tensor t[a, s, b′] := L[b, a] * conj(B[b, s, b′])
    @tensor L′[b′, a′] := t[a, s, b′] * A[a, s, a′]
    return L′
end

@doc raw"""
    transfer_left(L, A, W, B)

Grow an operator environment to the right by one site.

Contracts

```math
L'[b', w', a'] = \sum_{b, w, a, s, s'} L[b, w, a] \, B[b, s', b']^* \,
                 W[w, s', s, w'] \, A[a, s, a']
```

The bra pairs with the *output* physical index `s′` of `W` and the ket with the *input*
index `s`, matching ``W[w, s', s, w'] = \langle s' | \hat{W} | s \rangle``.

# Arguments
- `L`: the current environment, indexed `[bra bond, MPO bond, ket bond]`.
- `A`: the ket site tensor, indexed `[a, s, a′]`.
- `W`: the MPO site tensor, indexed `[w, s′, s, w′]`.
- `B`: the bra site tensor, indexed `[b, s′, b′]`; it is conjugated.

# Returns
- `L′`: the environment one site further right, indexed `[bra bond, MPO bond, ket bond]`.
"""
function transfer_left(
        L::AbstractArray{<:Any, 3}, A::AbstractArray{<:Any, 3},
        W::AbstractArray{<:Any, 4}, B::AbstractArray{<:Any, 3}
    )
    @tensor t1[w, a, s′, b′] := L[b, w, a] * conj(B[b, s′, b′])
    @tensor t2[a, s, w′, b′] := t1[w, a, s′, b′] * W[w, s′, s, w′]
    @tensor L′[b′, w′, a′] := t2[a, s, w′, b′] * A[a, s, a′]
    return L′
end

@doc raw"""
    transfer_right(R, A, B)

Grow an overlap environment to the left by one site.

Contracts

```math
R'[b, a] = \sum_{b', s, a'} B[b, s, b']^* \, R[b', a'] \, A[a, s, a']
```

# Arguments
- `R`: the current environment, indexed `[bra bond, ket bond]`.
- `A`: the ket site tensor, indexed `[a, s, a′]`.
- `B`: the bra site tensor, indexed `[b, s, b′]`; it is conjugated.

# Returns
- `R′`: the environment one site further left, indexed `[bra bond, ket bond]`.
"""
function transfer_right(R::AbstractMatrix, A::AbstractArray{<:Any, 3}, B::AbstractArray{<:Any, 3})
    @tensor t[b, s, a′] := conj(B[b, s, b′]) * R[b′, a′]
    @tensor R′[b, a] := t[b, s, a′] * A[a, s, a′]
    return R′
end

@doc raw"""
    transfer_right(R, A, W, B)

Grow an operator environment to the left by one site.

Contracts

```math
R'[b, w, a] = \sum_{b', w', a', s, s'} B[b, s', b']^* \, W[w, s', s, w'] \,
              A[a, s, a'] \, R[b', w', a']
```

# Arguments
- `R`: the current environment, indexed `[bra bond, MPO bond, ket bond]`.
- `A`: the ket site tensor, indexed `[a, s, a′]`.
- `W`: the MPO site tensor, indexed `[w, s′, s, w′]`.
- `B`: the bra site tensor, indexed `[b, s′, b′]`; it is conjugated.

# Returns
- `R′`: the environment one site further left, indexed `[bra bond, MPO bond, ket bond]`.
"""
function transfer_right(
        R::AbstractArray{<:Any, 3}, A::AbstractArray{<:Any, 3},
        W::AbstractArray{<:Any, 4}, B::AbstractArray{<:Any, 3}
    )
    @tensor t1[b, s′, w′, a′] := conj(B[b, s′, b′]) * R[b′, w′, a′]
    @tensor t2[b, w, s, a′] := t1[b, s′, w′, a′] * W[w, s′, s, w′]
    @tensor R′[b, w, a] := t2[b, w, s, a′] * A[a, s, a′]
    return R′
end

@doc raw"""
    apply_effective(L, W, x, R)

Apply the one-site effective operator to a site tensor.

Contracts

```math
y[b, s', b'] = \sum_{a, w, s, w', a'} L[b, w, a] \, W[w, s', s, w'] \, x[a, s, a'] \,
               R[b', w', a']
```

with `L` the environment left of the site and `R` the environment right of it. The result
carries the same index order as a site tensor, so it can be fed straight back into a
matrix-product object.

This is the linear map that DMRG diagonalises and that TDVP exponentiates. It is provided
now so that the effective-operator index conventions are fixed and tested; the environment
cache that feeds it arrives with those algorithms.

# Arguments
- `L`: the left environment, indexed `[b, w, a]`.
- `W`: the MPO site tensor, indexed `[w, s′, s, w′]`.
- `x`: the site tensor to act on, indexed `[a, s, a′]`.
- `R`: the right environment, indexed `[b′, w′, a′]`.

# Returns
- `y`: the result, indexed `[b, s′, b′]`.
"""
function apply_effective(
        L::AbstractArray{<:Any, 3}, W::AbstractArray{<:Any, 4},
        x::AbstractArray{<:Any, 3}, R::AbstractArray{<:Any, 3}
    )
    @tensor t1[b, w, s, a′] := L[b, w, a] * x[a, s, a′]
    @tensor t2[b, s′, w′, a′] := t1[b, w, s, a′] * W[w, s′, s, w′]
    @tensor y[b, s′, b′] := t2[b, s′, w′, a′] * R[b′, w′, a′]
    return y
end

@doc raw"""
    apply_effective(L, W₁, W₂, x, R)

Apply the two-site effective operator to a two-site tensor.

Contracts

```math
y[b, s_a', s_b', b'] = \sum L[b, w, a] \, W_1[w, s_a', s_a, w_m] \,
                       W_2[w_m, s_b', s_b, w'] \, x[a, s_a, s_b, a'] \, R[b', w', a']
```

# Arguments
- `L`: the left environment, indexed `[b, w, a]`.
- `W₁`: the MPO tensor of the left site of the pair.
- `W₂`: the MPO tensor of the right site of the pair.
- `x`: the two-site tensor, indexed `[a, sₐ, s_b, a′]`.
- `R`: the right environment, indexed `[b′, w′, a′]`.

# Returns
- `y`: the result, indexed `[b, sₐ′, s_b′, b′]`.
"""
function apply_effective(
        L::AbstractArray{<:Any, 3}, W1::AbstractArray{<:Any, 4}, W2::AbstractArray{<:Any, 4},
        x::AbstractArray{<:Any, 4}, R::AbstractArray{<:Any, 3}
    )
    @tensor t1[b, w, sa, sb, a′] := L[b, w, a] * x[a, sa, sb, a′]
    @tensor t2[b, sa′, wm, sb, a′] := t1[b, w, sa, sb, a′] * W1[w, sa′, sa, wm]
    @tensor t3[b, sa′, sb′, w′, a′] := t2[b, sa′, wm, sb, a′] * W2[wm, sb′, sb, w′]
    @tensor y[b, sa′, sb′, b′] := t3[b, sa′, sb′, w′, a′] * R[b′, w′, a′]
    return y
end
