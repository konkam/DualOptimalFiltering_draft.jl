using DataFrames

function bind_rows(dflist)
    vcat(dflist...)
end

"Normalises a vector"
function normalise(x)
    return x/sum(x)
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

@memoize function log_binomial_safe_but_slow_mem(n::Int64, k::Int64)
    log_binomial_safe_but_slow(n::Int64, k::Int64)
end

# function loghypergeom_pdf(i::Array{Int64,1}, m::Array{Int64,1}, si::Int64, sm::Int64)
#     return sum(log_binomial_safe_but_slow.(m,i)) - log_binomial_safe_but_slow(sm, si)
# end
function loghypergeom_pdf(i::Array{Int64,1}, m::Array{Int64,1}, si::Int64, sm::Int64)
    return sum(log_binomial_safe_but_slow(m[k],i[k]) for k in 1:length(m)) - log_binomial_safe_but_slow(sm, si)
end

function loghypergeom_pdf_inner_mem(i::Array{Int64,1}, m::Array{Int64,1}, si::Int64, sm::Int64)
    return sum(log_binomial_safe_but_slow_mem.(m,i)) - log_binomial_safe_but_slow_mem(sm, si)
end

function loghypergeom_pdf_mem(i::Array{Int64,1}, m::Array{Int64,1}, si::Int64, sm::Int64)
    return loghypergeom_pdf_inner_mem(i::Array{Int64,1}, m::Array{Int64,1}, si::Int64, sm::Int64)
end

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
    return sum(log(x + i) for i in 0:(n-1))
end
