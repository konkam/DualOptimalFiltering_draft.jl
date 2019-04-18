using StatsFuns, SpecialFunctions

function logμπh_NegBin(m::Integer, θ, α, y::Integer)
    # Be careful that the NegativeBinomial is not parameterised as in Wikipedia, p replaced by 1-p
    return logpdf(NegativeBinomial(α + m ,θ/(1+θ)), y)
end
function logμπh_NegBin(m::Integer, θ, α, y::AbstractArray{T, 1}; λ = 1) where T <: Integer
    s = sum(y)
    n = length(y)
    p = θ/(θ+n*λ)
    # Be careful that the NegativeBinomial is not parameterised as in Wikipedia, p replaced by 1-p
    return logpdf(NegativeBinomial(α + m, p), s) + lfactorial(s)-s*log(n) - sum(lfactorial.(y))
end

function logμπh(m::Integer, θ, α, y::Integer; λ = 1)
    return y*log(λ) + (α+m)*log(θ) + SpecialFunctions.lgamma(m+y+α) - lfactorial(y) - SpecialFunctions.lgamma(α+m) - (y+α+m) * log(θ + λ)
end
function logμπh(m::Integer, θ, α, y::AbstractArray{T, 1}; λ = 1) where T <: Integer
    s = sum(y)
    n = length(y)
    return s*log(λ) + (α+m)*log(θ) + SpecialFunctions.lgamma(m+s+α) - sum(lfactorial.(y)) - SpecialFunctions.lgamma(α+m) - (s+α+m) * log(θ + n*λ)
end

function logμπh_arb(m::Integer, θ, α, y::AbstractArray{T, 1}; λ = 1) where T <: Integer
    α_arb = RR(α)
    θ_arb = RR(θ)
    s = sum(y)
    n = length(y)
    return s*log(λ) + (α_arb+m)*log(θ_arb) + Nemo.lgamma(m+s+α_arb) - sum(Nemo.log.(RR.(Nemo.fac.(y)))) - Nemo.lgamma(α_arb+m) - (s+α_arb+m) * log(θ_arb + n*λ)

end

function logμπh_precomputed(m::Integer, θ, α, y::AbstractArray{T, 1}, precomputed_lgamma_α, precomputed_lfactorial; λ = 1) where T <: Integer
    s = sum(y)
    n = length(y)
    #@show m+s+1
    return @inbounds s*log(λ) + (α+m)*log(θ) + precomputed_lgamma_α[m+s+1] - precomputed_lgamma_α[m+1] - sum(view(precomputed_lfactorial, y)) - (s+α+m) * log(θ + n*λ)
end

function precompute_lgamma_α(α, data)
    return SpecialFunctions.lgamma.(i + α for i in 0:sum(sum.(values(data))))
end

function precompute_lfactorial(data)
    return  SpecialFunctions.lfactorial.(1:maximum(maximum.(values(data))))
end

# logμπh = logμπh_another_param

function logμπh_inside(m, θ, α, y; λ = 1)
    return map(mm -> logμπh(mm::Integer, θ, α, y; λ = λ), m)
end

function logμπh(m::AbstractArray{U, 1}, θ, α, y; λ = 1)  where U<:Integer
    return logμπh_inside(m, θ, α, y; λ = λ)
end
# function logμπh(m::UnitRange{Int64}, θ, α, y)
#     return logμπh_inside(m, θ, α, y)
# end

μπh(m, θ, α, y; λ = 1) = exp.(logμπh(m, θ, α, y; λ = λ))

logμπh_param_δγ(m, θ, δ, y; λ = 1) = logμπh(m, θ, δ/2, y; λ = λ)
logμmθ_param_δγ(m, θ, δ, y; λ = 1) = logμπh(m, θ, δ/2, y; λ = λ)

# function logpmmi(m::Integer, i::Integer, t, γ, σ, θ)
#     return logpdf(Binomial(m, γ/σ^2*1/(θ*exp(2*γ*t)+γ/σ^2-θ)), m-i)
# end

function t_CIR(y::Number, m::Integer)
    return m + y
end
function t_CIR(y::Number, m::Array{U, 1}) where U<:Integer
    return m .+ y
end
function t_CIR(y::Array{T, 1}, m::Integer) where T<:Number
    return m + sum(y)
end
function t_CIR_inside(y::Array{T, 1}, m) where T<:Number
    return m .+ sum(y)
end
function t_CIR(y::Array{T, 1}, m::Array{U, 1}) where {T<:Number, U<:Integer}
    return t_CIR_inside(y::Array{T, 1}, m)
end
function t_CIR(y::Array{T, 1}, m::UnitRange{Int64}) where T<:Number
    return t_CIR_inside(y::Array{T, 1}, m)
end

function Λ_prime_1D(Λ::Union{Array{T,1}, UnitRange{Int64}}) where T<:Integer
    return 0:maximum(Λ)
end

function next_Λ_from_Λ_prime(Λ_prime, y, t)
    return t(y, Λ_prime)
end

function θ_prime_from_θ_CIR(θ, Δt, γ, σ)
    return γ/σ^2* θ*exp(2*γ*Δt)/(θ*exp(2*γ*Δt)+γ/σ^2-θ)
end

function θ_from_θ_prime(y, θ_prime, T)
    return T(y, θ_prime)
end

T_CIR(y, θ; λ = 1) = θ + λ*length(y)

function next_wms_from_wms_prime(wms_prime, Λ_prime, y, θ_prime, α)#update
    #Make sure we deal correctly with weights equal to 0
    #Probably harmonise what can be improved in the filtering algorithm
    unnormalised_wms = wms_prime .* DualOptimalFiltering.μπh(Λ_prime, θ_prime, α, y)
    return unnormalised_wms |> DualOptimalFiltering.normalise
end

function next_logwms_from_log_wms_prime1D(log_wms_prime, Λ_prime, y, θ_prime, α)#update
    #Make sure we deal correctly with weights equal to 0
    #Probably harmonise what can be improved in the filtering algorithm
    unnormalised_log_wms = log_wms_prime .+ logμπh(Λ_prime, θ_prime, α, y)
    return unnormalised_log_wms .- StatsFuns.logsumexp(unnormalised_log_wms)
end

function next_wms_prime_from_wms(wms, Λ, Δt, θ, γ, σ)
    nΛ = length(Λ)
    wms_prime = zeros(maximum(Λ)+1)
    p = γ/σ^2*1/(θ*exp(2*γ*Δt) + γ/σ^2 - θ)
    for k in 1:nΛ
        m = Λ[k]
        for n in 0:m
            idx = n+1
            wms_prime[idx] += wms[k]*pdf(Binomial(m, p), n)
        end
    end
    return wms_prime
end

# function next_log_wms_prime_from_log_wms1D4(log_wms, Λ, Δt, θ, γ, σ)
function next_log_wms_prime_from_log_wms1D(log_wms, Λ, Δt, θ, γ, σ)
    #The speed gain seems like a modest 20%, but much less allocations though.
    nΛ = length(Λ)
    maxΛ = maximum(Λ)
    wms_prime = zeros(maxΛ+1)
    all_u = zeros(maxΛ+1)
    all_u .= -Inf
    p = γ/σ^2*1/(θ*exp(2*γ*Δt) + γ/σ^2 - θ)
    @inbounds for k in eachindex(Λ)
        m = Λ[k]
        for n in 0:m
            idx = n+1

            x = log_wms[k] + logpdf(Binomial(m, p), n)
            if x <= all_u[idx]
                wms_prime[idx] += exp(x - all_u[idx])
            else
                wms_prime[idx] *= exp(all_u[idx] - x)
                wms_prime[idx] += 1.0
                all_u[idx] = x
            end
        end
    end
    # return @. log(wms_prime) + all_u
    return log.(wms_prime) .+ all_u
end

function logμν_i_minus_1(Λ_prime_i_minus_1, log_wms_prime_i_minus_1, θ_prime_i_minus_1, yi, δ)
    StatsFuns.logsumexp(log_wms_prime_i_minus_1 .+ logμπh_param_δγ(Λ_prime_i_minus_1, θ_prime_i_minus_1, δ, yi))
end

function log_likelihood_CIR(δ, γ, σ, λ, data)
    times = data |> keys |> collect |> sort

    α = δ/2#Alternative parametrisation

    res = Dict()

    res[times[1]] = logμπh_param_δγ(0, γ/σ^2, δ, data[times[1]])

    #Prior
    Λ_prime = [0]
    θ_prime = γ/σ^2

    #1st update
    Λ = next_Λ_from_Λ_prime(Λ_prime, data[times[1]], t_CIR)
    θ = θ_from_θ_prime(data[times[1]], θ_prime, T_CIR)
    log_wms = [0]

    #1st prediction
    Λ_prime = Λ_prime_1D(Λ)
    log_wms_prime = next_log_wms_prime_from_log_wms1D(log_wms, Λ, times[2]-times[1], θ, γ, σ)
    θ_prime = θ_prime_from_θ_CIR(θ, times[2]-times[1], γ, σ)

    res[times[2]] = res[times[1]] + logμν_i_minus_1(Λ_prime, log_wms_prime, θ_prime, data[times[2]], δ)

    for i in 2:(length(times)-1)
        t = times[i]
        next_t = times[i+1]
        Δt = next_t-t

        #Update
        Λ = next_Λ_from_Λ_prime(Λ_prime, data[t], t_CIR)
        θ = θ_from_θ_prime(data[t], θ_prime, T_CIR)
        log_wms = next_logwms_from_log_wms_prime1D(log_wms_prime, Λ_prime, data[t], θ_prime, α)

         #Prediction
        Λ_prime = Λ_prime_1D(Λ)
        θ_prime = θ_prime_from_θ_CIR(θ, Δt, γ, σ)
        log_wms_prime = next_log_wms_prime_from_log_wms1D(log_wms, Λ, Δt, θ, γ, σ)

        res[next_t] = res[t] + logμν_i_minus_1(Λ_prime, log_wms_prime, θ_prime, data[next_t], δ)

    end

    return res
end


function log_likelihood_CIR_keep_fixed_number(δ, γ, σ, λ, data, fixed_number::Int64; silence = false)
    # println("filter_WF_mem2")

    prune_keeping_fixed_number_log_wms_(Λ_of_t, log_wms_of_t) =
        prune_keeping_fixed_number_log_wms(Λ_of_t, log_wms_of_t, fixed_number)

    log_likelihood_pruning(δ, γ, σ, λ, data, prune_keeping_fixed_number_log_wms_; silence = silence)

end

function log_likelihood_CIR_keep_above_threshold(δ, γ, σ, λ, data, ε::Float64; silence = false)
    # println("filter_WF_mem2")

    prune_keeping_above_threshold_log_wms_(Λ_of_t, log_wms_of_t) = prune_keeping_above_threshold_log_wms(Λ_of_t, log_wms_of_t, ε)

    log_likelihood_pruning(δ, γ, σ, λ, data, prune_keeping_above_threshold_log_wms_; silence = silence)

end

function log_likelihood_CIR_fixed_fraction(δ, γ, σ, λ, data, fraction::Float64; silence = false)
    # println("filter_WF_mem2")
    prune_keeping_fixed_fraction_log_wms_(Λ_of_t, log_wms_of_t) = prune_keeping_fixed_fraction_log_wms(Λ_of_t, log_wms_of_t, fraction)

    log_likelihood_pruning(δ, γ, σ, λ, data, prune_keeping_fixed_fraction_log_wms_; silence = silence)

end

function log_likelihood_pruning(δ, γ, σ, λ, data, do_the_pruning_log_wms::Function; silence = false)
    times = data |> keys |> collect |> sort

    α = δ/2#Alternative parametrisation

    res = Dict()

    res[times[1]] = logμπh_param_δγ(0, γ/σ^2, δ, data[times[1]])

    #Prior
    Λ_prime = [0]
    θ_prime = γ/σ^2

    #1st update
    Λ = next_Λ_from_Λ_prime(Λ_prime, data[times[1]], t_CIR)
    θ = θ_from_θ_prime(data[times[1]], θ_prime, T_CIR)
    log_wms = [0]

    pruned_Λ, pruned_log_wms = do_the_pruning_log_wms(Λ, log_wms)

    #1st prediction
    Λ_prime = Λ_prime_1D(pruned_Λ)
    log_wms_prime = next_log_wms_prime_from_log_wms1D(pruned_log_wms, pruned_Λ, times[2]-times[1], θ, γ, σ)
    θ_prime = θ_prime_from_θ_CIR(θ, times[2]-times[1], γ, σ)

    res[times[2]] = res[times[1]] + logμν_i_minus_1(Λ_prime, log_wms_prime, θ_prime, data[times[2]], δ)

    for i in 2:(length(times)-1)
        t = times[i]
        next_t = times[i+1]
        Δt = next_t-t

        #Update
        Λ = next_Λ_from_Λ_prime(Λ_prime, data[t], t_CIR)
        θ = θ_from_θ_prime(data[t], θ_prime, T_CIR)
        log_wms = next_logwms_from_log_wms_prime1D(log_wms_prime, Λ_prime, data[t], θ_prime, α)

        #Pruning
        pruned_Λ, pruned_log_wms = do_the_pruning_log_wms(Λ, log_wms)


         #Prediction
        Λ_prime = Λ_prime_1D(pruned_Λ)
        θ_prime = θ_prime_from_θ_CIR(θ, Δt, γ, σ)
        log_wms_prime = next_log_wms_prime_from_log_wms1D(pruned_log_wms, pruned_Λ, Δt, θ, γ, σ)

        res[next_t] = res[t] + logμν_i_minus_1(Λ_prime, log_wms_prime, θ_prime, data[next_t], δ)

    end

    return res
end
