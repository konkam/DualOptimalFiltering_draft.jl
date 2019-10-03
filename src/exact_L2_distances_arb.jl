# using SpecialFunctions, StatsFuns, Base.Iterators
# using Nemo

"""
    prod_2_gammas_arb(α1::arb, β1::arb, α2::arb, β2::arb)

Compute ``\\int_{\\mathbb{R}^+} Gamma(α1, β1)Gammas(α2, β2)`` exactly using arbitrary precision computation. This formula is only valid for ``α1 + α2 > 1``.

# Examples
```julia-repl
julia> DualOptimalFiltering.prod_2_gammas_arb(1, 1, 1, 1) #== 0.5
0.500000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
julia> DualOptimalFiltering.prod_2_gammas_arb(2, 0.5, 1, 0.5) #==  log(0.5^2*0.5 )
0.12500000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
```
"""
function prod_2_gammas_arb(α1::arb, β1::arb, α2::arb, β2::arb)
    @assert α1 + α2 > 1 "The formula for the L2 distance is only valid for α1 + α2 > 1"
    return β1^α1 * β2^α2 * Nemo.gamma(α1 + α2 - 1) / (Nemo.gamma(α1)*Nemo.gamma(α2)*(β1 + β2)^(α1 + α2 - 1))
end

prod_2_gammas_arb(α1::Real, β1::Real, α2::Real, β2::Real) = prod_2_gammas_arb(RR(α1), RR(β1), RR(α2), RR(β2))

CC = ComplexField(nbits)

function L2_dist_Gamma_mixtures_arb(wlist_1::Array{T, 1}, αlist_1::Array{U1, 1}, βlist_1::Array{V1, 1}, wlist_2::Array{T, 1}, αlist_2::Array{U2, 1}, βlist_2::Array{V2, 1}) where {T <: Real, U1 <: Real, V1 <: Real, U2 <: Real, V2 <: Real}
    I = length(wlist_1)
    J = length(wlist_2)
    wlist_1_arb = RR.(wlist_1)
    wlist_2_arb = RR.(wlist_2)
    αlist_1_arb = RR.(αlist_1)
    βlist_1_arb = RR.(βlist_1)
    αlist_2_arb = RR.(αlist_2)
    βlist_2_arb = RR.(βlist_2)

    first_term = sum(wlist_1_arb[i] * wlist_1_arb[j] * prod_2_gammas_arb(αlist_1_arb[i], βlist_1_arb[i], αlist_1_arb[j], βlist_1_arb[j]) for (i,j) in Base.Iterators.product(1:I, 1:I))

    second_term = sum(wlist_2_arb[i] * wlist_2_arb[j] * prod_2_gammas_arb(αlist_2_arb[i], βlist_2_arb[i], αlist_2_arb[j], βlist_2_arb[j]) for (i,j) in Base.Iterators.product(1:J, 1:J))

    third_term = - 2 * sum(wlist_1_arb[i] * wlist_2_arb[j] * prod_2_gammas_arb(αlist_1_arb[i], βlist_1_arb[i], αlist_2_arb[j], βlist_2_arb[j]) for (i,j) in Base.Iterators.product(1:I, 1:J))

    # println(first_term + second_term)
    # println(third_term)
    # println(first_term + second_term + third_term)

    return sqrt(CC(first_term + second_term + third_term)) |> real
    # return first_term + second_term + third_term
end

"""
    int_prod_2_Dir_arb(α1::Array{T, 1}, α2::Array{T, 1}) where T <: arb
    int_prod_2_Dir_arb(α1::Array{T, 1}, α2::Array{T, 1}) where T <: Real


Compute ``\\int_{\\nabla_K} Dir(α1)Dir(α2)`` exactly. This formula is only valid for ``\\forall j \\in \\{1,\\ldots, K\\}, \\alpha_{1,j} + \\alpha_{2,j} > 1``.

# Examples
```julia-repl
julia> DualOptimalFiltering.int_prod_2_Dir_arb([1,1], [1,1])
1.
julia> DualOptimalFiltering.int_prod_2_Dir_arb(2, 0.5, 1, 0.5) #==  0.5^2*0.5
1.125
```
"""
function int_prod_2_Dir_arb(α1::Array{T, 1}, α2::Array{T, 1}) where T <: arb
    @assert all(α1 .+ α2 .> 1) "The formula for the L2 distance is only valid for α1 + α2 > 1 for each component of α"
    return mvbeta_arb(α1 .+ α2 .- 1)/(mvbeta_arb(α1)*mvbeta_arb(α2))
end

int_prod_2_Dir_arb(α1::Array{T, 1}, α2::Array{T, 1}) where T <: Real = int_prod_2_Dir_arb(RR.(α1), RR.(α2))

"""
    mvbeta_arb(α::AbstractArray{T,1}) where T <: arb
    mvbeta_arb(α::AbstractArray{T,1}) where T <: Real

Compute the log of the [multivariate beta function](https://en.wikipedia.org/wiki/Beta_function#Multivariate_beta_function): ``\\Beta(\\alpha_1,\\alpha_2,\\ldots\\alpha_n) = \\frac{\\Gamma(\\alpha_1)\\Gamma(\\alpha_2) \\cdots \\Gamma(\\alpha_n)}{\\Gamma(\\alpha_1 + \\alpha_2 + \\cdots + \\alpha_n)}``.

# Examples
```julia-repl
julia> mvbeta_arb([1,1])
1.0
```
"""
function mvbeta_arb(α::AbstractArray{T,1}) where T <: arb
    return prod(Nemo.gamma.(α))/Nemo.gamma(sum(α))
end

mvbeta_arb(α::AbstractArray{T,1}) where T <: Real = mvbeta_arb(RR.(α))

function L2_dist_Dirichlet_mixtures_arb(wlist_1, αlist_1, wlist_2, αlist_2)
    wlist_1_arb = RR.(wlist_1)
    wlist_2_arb = RR.(wlist_2)
    αlist_1_arb = R.(αlist_1)
    αlist_2_arb = R.(αlist_2)

    I = length(wlist_1)
    J = length(wlist_2)

    first_term = sum(wlist_1_arb[i] * wlist_1_arb[j] * int_prod_2_Dir_arb(αlist_1_arb[i], αlist_1_arb[j]) for (i,j) in Base.Iterators.product(1:I, 1:I))

    second_term = sum(wlist_2_arb[i] * wlist_2_arb[j] * int_prod_2_Dir_arb(αlist_2_arb[i], αlist_2_arb[j]) for (i,j) in Base.Iterators.product(1:J, 1:J))

    third_term = - 2 * sum(wlist_1_arb[i] * wlist_2_arb[j] * int_prod_2_Dir_arb(αlist_1_arb[i], αlist_2_arb[j]) for (i,j) in Base.Iterators.product(1:I, 1:J))

    return sqrt(CC(first_term + second_term + third_term)) |> real

end
