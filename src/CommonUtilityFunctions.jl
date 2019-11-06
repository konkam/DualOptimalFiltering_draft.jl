using DataFrames, IterTools, DataStructures, Nemo

function bind_rows(dflist)
    vcat(dflist...)
end

"Normalises a vector"
function normalise(x::AbstractArray)
    if length(x) == 0
        error("cannot normalise a vector of length 0")
    end
    return x/sum(x)
end

"Normalises a vector of logweights"
function lognormalise(logx::AbstractArray)
    if length(logx) == 0
        error("cannot lognormalise a vector of length 0")
    end
    return logx .- logsumexp(logx)
end

"Normalises an Accumulator"
function normalise(x::Accumulator)
    normalisation_constant = sum(values(x))
    for k in keys(x)
        x[k] = x[k]/normalisation_constant
    end
    return x
end



"Normalises an Accumulator with logweights"
function normalise_logAccumulator(x::Accumulator)
    log_normalisation_constant = logsumexp(values(x))
    for k in keys(x)
        x[k] = x[k] - log_normalisation_constant
    end
    return x
end

function get_quantiles_from_mass(mass)
    qinf = 0.5*(1-mass)
    return (qinf, 1-qinf)
end

function log_binomial_safe_but_slow(n::Int64, k::Int64)
    @assert n >= 0
    @assert k >= 0
    @assert k <= n
    if k == 0 || k == n
        return 0
    elseif k == 1 || k == n-1
        return log(n)
    else
        return sum(log(i) for i in (n-k+1):n) - sum(log(i) for i in 2:k)
    end
end

"Converts a dictionary to a new dictionary with log values"
function convert_weights_to_logweights(weights_dict)
    return Dict(k => log.(v) for (k,v) in weights_dict)
end

"Converts a dictionary to a new dictionary with exponential values"
function convert_logweights_to_weights(weights_dict)
    return Dict(k => exp.(v) for (k,v) in weights_dict)
end

# @memoize function log_binomial_safe_but_slow_mem(n::Int64, k::Int64)
#     log_binomial_safe_but_slow(n::Int64, k::Int64)
# end

# function loghypergeom_pdf(i::Array{Int64,1}, m::Array{Int64,1}, si::Int64, sm::Int64)
#     return sum(log_binomial_safe_but_slow.(m,i)) - log_binomial_safe_but_slow(sm, si)
# end
function loghypergeom_pdf(i::Array{Int64,1}, m::Array{Int64,1}, si::Int64, sm::Int64)
    return sum(log_binomial_safe_but_slow(m[k],i[k]) for k in eachindex(m)) - log_binomial_safe_but_slow(sm, si)
end

# function loghypergeom_pdf_inner_mem(i::Array{Int64,1}, m::Array{Int64,1}, si::Int64, sm::Int64)
#     return sum(log_binomial_safe_but_slow_mem.(m,i)) - log_binomial_safe_but_slow_mem(sm, si)
# end
#
# function loghypergeom_pdf_mem(i::Array{Int64,1}, m::Array{Int64,1}, si::Int64, sm::Int64)
#     return loghypergeom_pdf_inner_mem(i::Array{Int64,1}, m::Array{Int64,1}, si::Int64, sm::Int64)
# end

function descending_fact_no0(x::Real, n::Int64)
    return prod(x-i for i in 0:(n-1))
end

function descending_fact0(x::Real, n::Int64)
    if(n==0)
        return 1
    else
        return descending_fact_no0(x, n)
    end
end

ascending_fact0(x::Real, n::Int64) = descending_fact0(x + n - 1, n)


function log_descending_fact_no0(x::Real, n::Int64)
    return sum(log(x-i) for i in 0:(n-1))
end

function log_descending_fact(x::Real, n::Int64)
    if(n==0)
        return 0
    else
        return log_descending_fact_no0(x, n)
    end
end

log_ascending_fact(x::Real, n::Int64) = log_descending_fact(x + n - 1, n)

function truncate_float(x, digits_after_comma)
    floor(x*10^digits_after_comma)/10^digits_after_comma
end

function keep_last_k(x, k)
    if length(x) <= k
        return x
    else
        return x[(end-k+1):end]
    end
end

function kmax(x::AbstractArray{T, 1}, k::Integer) where T <: Number
    if length(x) < k
        error("length(x) < $k, cannot take the $k largest elements of a vector of size $(length(x))")
    else
        res = x[(end-k+1):end]
        return kmax_rec(x, k, findmin(res), res)
    end
end

import Base.length

function length(x::Union{IterTools.Distinct})
    l = 0
    for k in x
        l +=1
    end
    return l
end

function kmax_rec(x::AbstractArray{T, 1}, k::Integer, smallest::Tuple{T,U}, res::AbstractArray{T, 1}) where {T <: Number, U <: Integer}
    # println("x = $x")
    # println("smallest = $smallest")
    # println("res = $res")
    if length(x) == k
        return res
    else
        if x[1] > smallest[1]
            res[smallest[2]] = x[1]
        end
        return kmax_rec(x[2:end], k, findmin(res), res)
    end
end


function kmax_safe_but_slow(x::AbstractArray{T, 1}, k::Integer) where T <: Number
    if length(x) < k
        error("length(x) < $k, cannot take the $k largest elements of a vector of size $(length(x))")
    elseif length(x) == k
        return sort(x)
    else
        res = Array{T}(undef, k)
        xtmp = x
        for i in 1:k
            max_found = findmax(xtmp)
            res[k-i+1] = max_found[1]
            xtmp = vcat(xtmp[1:(max_found[2]-1)], xtmp[(max_found[2]+1):end])
        end
        return res
    end
end




function flat2(arr::Array{Array{T,1},1}) where T<:Number
    rst = T[]
    grep(v) = for x in v
        if isa(x, Array) grep(x) else push!(rst, x) end
    end
    grep(arr)
    rst
end

function create_gamma_mixture_pdf(δ, θ, Λ, wms)
    #use 1/θ because of the way the Gamma distribution is parameterised in Julia Distributions.jl
    return x -> sum(wms.*Float64[pdf(Gamma(δ/2 + m, 1/θ),x) for m in Λ])
end

function create_gamma_mixture_cdf(δ, θ, Λ, wms)
    function gamma_mixture_cdf(x::T) where T<:Real
        return sum(wms.*Float64[cdf(Gamma(δ/2 + m, 1 ./ θ), x) for m in Λ])
    end
    return gamma_mixture_cdf
end

function test_equal_spacing_of_observations(data; override = false, digits_after_comma_for_time_precision = 4)
    if !override&&(data |> keys |> collect |> sort |> diff |> x -> round.(x; digits = digits_after_comma_for_time_precision) |> unique |> length > 1)
        println(data |> keys |> collect |> sort |> diff |> x -> truncate_float.(x,digits_after_comma_for_time_precision) |> unique)
        error("Think twice about precomputing all terms, as the time intervals are not equal. You can go ahead using the option 'override = true.'")
    end
end

function log_pochammer(x::Real, n::Integer)
    if n==0
        return 0
    elseif n==1
        return log(x)
    else
        return sum(log(x + i) for i in 0:(n-1))
    end
end

function assert_constant_time_step_and_compute_it(data)
    Δts = keys(data) |> collect |> sort |> diff |> unique
    if length(Δts) > 1
        test_equal_spacing_of_observations(data; override = false)
    end
    Δt = mean(Δts)
    return Δt
end

import Base.max
function max(x::Nemo.arb, y::Nemo.arb)
    if x == y
        return x
    elseif x < y
        return y
    else
        return x
    end
end
import StatsFuns.logaddexp
function logaddexp(x::Nemo.arb, y::Nemo.arb)
    # x or y is  NaN  =>  NaN
    # x or y is +Inf  => +Inf
    # x or y is -Inf  => other value
    isfinite(x) && isfinite(y) || return max(x,y)
    x > y ? x + log1p(exp(y - x)) : y + log1p(exp(x - y))
end
