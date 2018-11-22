using Roots
using Distributions

function create_mixture_density(δ, θ, Λ, wms)
    #use 1/θ because of the way the Gamma distribution is parameterised in Julia Distributions.jl
    return x -> sum(wms.*Float64[pdf(Gamma(δ/2 + m, 1/θ),x) for m in Λ])
end

function create_gamma_mixture_parameters(δ, θ, Λ)
    α = [δ/2 + m for m in Λ]
    β = [θ for m in Λ]
    return α, β
end

function create_mixture_components(δ, θ, Λ)
    #use 1/θ because of the way the Gamma distribution is parameterised in Julia Distributions.jl
    return [Gamma(δ/2 + m, 1/θ) for m in Λ]
end

function compute_quantile_mixture_hpi(δ, θ, Λ, wms, q::Float64)
    #use 1/θ because of the way the Gamma distribution is parameterised in Julia Distributions.jl
    f = x -> sum(wms.*Float64[cdf(Gamma(δ/2 + m, 1/θ),x) for m in Λ])
    return fzero(x -> f(x)-q, 0, 10^9)
end

# function update_CIR_params_debug(wms::Array{Ty,1}, δ::Ty, θ::Ty, λ::Ty, Λ, y::Array{Tz,1}; debug = false) where {Ty<:Number,Tz<:Integer}
#     ny = sum(y)
#     J = length(y)
#     function lpga(m)
# #         θ^(δ/2+m) / (θ+λ)^(δ/2+m+ny) * gamma(δ/2+m+ny)/gamma(δ/2+m)/prod([factorial(yi) for yi in y])
#         lres = (δ/2+m)*log(θ) - (δ/2+m+ny)*log(θ+J*λ) + lgamma(δ/2+m+ny) - lgamma(δ/2+m) - sum([lfactorial(yi) for yi in y])
#         return lres
#     end
#
#     pga(m) = exp(lpga(m))
#
#     filter_ = wms .== 0
#
#     lwms_hat = log.(wms) .+ map(lpga, Λ)
#
#     wms_hat = exp.(lwms_hat)
#     wms_hat[filter_] = 0
#     wms_hat = wms_hat |> normalise
#
#     if debug&&any(isnan, wms_hat)
#         println("NAs in update step")
#         println("wms")
#         println(wms)
#         println("lpga")
#         println(map(lpga, Λ))
#         println("pga")
#         println(map(pga, Λ))
#         println("wms_hat")
#         println(wms_hat)
#     end
#
#     return θ + J*λ, Λ + ny, wms_hat
# end

function update_CIR_params(wms::Array{Ty,1}, δ::Real, θ::Real, λ::Real, Λ, y::Array{Tz,1}) where {Ty<:Number,Tz<:Integer}
    α = δ/2#Alternative parametrisation

    ny = sum(y)
    J = length(y)

    return θ + J*λ, Λ .+ ny, next_wms_from_wms_prime(wms, Λ, y, θ, α)
end

function update_CIR_params_logweights(logweights::Array{Ty,1}, δ::Ty, θ::Ty, λ::Ty, Λ, y::Array{Tz,1}) where {Ty<:Number,Tz<:Integer}
    α = δ/2#Alternative parametrisation

    return T_CIR(y, θ), t_CIR(y, Λ), next_log_wms_from_log_wms_prime(logweights, Λ, y, θ, α)
end

function update_CIR_params_debug(wms::Array{Ty,1}, δ::Ty, θ::Ty, λ::Ty, Λ, y::Array{Tz,1}; debug = false) where {Ty<:Number,Tz<:Integer}
    update_CIR_params(wms, δ, θ, λ, Λ, y)
end
# function update_CIR_params(wms::Array{Ty,1}, δ::Ty, θ::Ty, λ::Ty, Λ, y::Array{Tz,1}) where {Ty<:Number,Tz<:Integer}
#     update_CIR_params_debug(wms::Array{Ty,1}, δ::Ty, θ::Ty, λ::Ty, Λ, y::Array{Tz,1}; debug = false)
# end

# function predict_CIR_params_debug(wms::Array{Ty,1}, δ::Ty, θ::Ty, γ::Ty, σ::Ty, Λ, t::Ty; debug = false) where Ty<:Number
    # M = maximum(Λ)
    # minm = minimum(Λ)
    #
    # w_dict = Dict(zip(Λ, wms))
    # p = γ/σ^2*1/(θ*exp(2*γ*t) + γ/σ^2 - θ)
    #
    # function wmpm_mmi(m, mmi)
    #     w_dict[m]*pdf(Binomial(m, p), mmi)
    # end
    #
    # θ_new = p * θ * exp(2*γ*t)
    # Λ_new = 0:M
    # wms_new = map(mmi -> sum(Float64[wmpm_mmi(m, mmi) for m in M:-1:max(mmi,minm)]), 0:M)
    #
    # if debug&&any(isnan, wms_new)
    #     println("NAs in predict step")
    #     println(wms_new)
    # end

#     return θ_new, Λ_new, wms_new
#
# end

function predict_CIR_params_debug(wms::Array{Ty,1}, δ::Ty, θ::Ty, γ::Ty, σ::Ty, Λ, t::Ty; debug = false) where Ty<:Number

    p = γ/σ^2*1/(θ*exp(2*γ*t) + γ/σ^2 - θ)
    θ_new = p * θ * exp(2*γ*t)
    Λ_new = 0:maximum(Λ)

    return θ_new, Λ_new, next_wms_prime_from_wms(wms, Λ, t, θ, γ, σ)

end

function predict_CIR_params_logweights(logweights::Array{Ty,1}, δ::Ty, θ::Ty, γ::Ty, σ::Ty, Λ, Δt::Ty; debug = false) where Ty<:Number

    return θ_prime_from_θ_CIR(θ, Δt, γ, σ), Λ_prime_1D(Λ), next_log_wms_prime_from_log_wms(logweights, Λ, Δt, θ, γ, σ)

end

function predict_CIR_params(wms::Array{Ty,1}, δ::Ty, θ::Ty, γ::Ty, σ::Ty, Λ, t::Ty) where Ty<:Number
    predict_CIR_params_debug(wms, δ, θ, γ, σ, Λ, t; debug = false)
end

function get_next_filtering_distribution_debug(current_Λ, current_wms, current_θ, current_time, next_time, δ, γ, σ, λ, next_y)
    predicted_θ, predicted_Λ, predicted_wms = predict_CIR_params_debug(current_wms, δ, current_θ, γ, σ, current_Λ, next_time-current_time; debug = true)
    filtered_θ, filtered_Λ, filtered_wms = update_CIR_params_debug(predicted_wms, δ, predicted_θ, λ, predicted_Λ, next_y; debug = true)

    return filtered_θ, filtered_Λ, filtered_wms
end

function get_next_filtering_distribution(current_Λ, current_wms, current_θ, current_time, next_time, δ, γ, σ, λ, next_y)
    predicted_θ, predicted_Λ, predicted_wms = predict_CIR_params(current_wms, δ, current_θ, γ, σ, current_Λ, next_time-current_time)
    filtered_θ, filtered_Λ, filtered_wms = update_CIR_params(predicted_wms, δ, predicted_θ, λ, predicted_Λ, next_y)

    return filtered_θ, filtered_Λ, filtered_wms
end

function get_next_filtering_distribution_logweights(current_Λ, current_logweights, current_θ, current_time, next_time, δ, γ, σ, λ, next_y)
    predicted_θ, predicted_Λ, predicted_logweights = predict_CIR_params_logweights(current_logweights, δ, current_θ, γ, σ, current_Λ, next_time-current_time)
    filtered_θ, filtered_Λ, filtered_logweights = update_CIR_params_logweights(predicted_logweights, δ, predicted_θ, λ, predicted_Λ, next_y)

    return filtered_θ, filtered_Λ, filtered_logweights
end

function filter_CIR_debug(δ, γ, σ, λ, data)

    times = keys(data) |> collect |> sort
    Λ_of_t = Dict()
    wms_of_t = Dict()
    θ_of_t = Dict()

    filtered_θ, filtered_Λ, filtered_wms = update_CIR_params([1.], δ, γ/σ^2, λ, [0], data[times[1]])

    Λ_of_t[times[1]] = filtered_Λ
    wms_of_t[times[1]] = filtered_wms # = 1.
    θ_of_t[times[1]] = filtered_θ

    for k in 1:(length(times)-1)
        filtered_θ, filtered_Λ, filtered_wms = get_next_filtering_distribution_debug(filtered_Λ, filtered_wms, filtered_θ, times[k], times[k+1], δ, γ, σ, λ, data[times[k+1]])
        Λ_of_t[times[k+1]] = filtered_Λ
        wms_of_t[times[k+1]] = filtered_wms
        θ_of_t[times[k+1]] = filtered_θ
    end

    return Λ_of_t, wms_of_t, θ_of_t

end

function filter_CIR(δ, γ, σ, λ, data; silence = false)

    times = keys(data) |> collect |> sort
    Λ_of_t = Dict()
    wms_of_t = Dict()
    θ_of_t = Dict()

    filtered_θ, filtered_Λ, filtered_wms = update_CIR_params([1.], δ, γ/σ^2, λ, [0], data[0])

    Λ_of_t[0] = filtered_Λ
    wms_of_t[0] = filtered_wms # = 1.
    θ_of_t[0] = filtered_θ

    for k in 1:(length(times)-1)
        if (!silence)
            println("Step index: $k")
            println("Number of components: $(length(filtered_Λ))")
        end
        filtered_θ, filtered_Λ, filtered_wms = get_next_filtering_distribution(filtered_Λ, filtered_wms, filtered_θ, times[k], times[k+1], δ, γ, σ, λ, data[times[k+1]])
        Λ_of_t[times[k+1]] = filtered_Λ
        wms_of_t[times[k+1]] = filtered_wms
        θ_of_t[times[k+1]] = filtered_θ
    end

    return Λ_of_t, wms_of_t, θ_of_t

end

function filter_CIR_logweights(δ, γ, σ, λ, data; silence = false)

    times = keys(data) |> collect |> sort
    Λ_of_t = Dict()
    wms_of_t = Dict()
    θ_of_t = Dict()

    filtered_θ, filtered_Λ, filtered_logweights = update_CIR_params_logweights([0.], δ, γ/σ^2, λ, [0], data[0])

    Λ_of_t[0] = filtered_Λ
    wms_of_t[0] = exp.(filtered_logweights) # = 1.
    θ_of_t[0] = filtered_θ

    for k in 1:(length(times)-1)
        if (!silence)
            println("Step index: $k")
            println("Number of components: $(length(filtered_Λ))")
        end
        filtered_θ, filtered_Λ, filtered_logweights = get_next_filtering_distribution_logweights(filtered_Λ, filtered_logweights, filtered_θ, times[k], times[k+1], δ, γ, σ, λ, data[times[k+1]])
        Λ_of_t[times[k+1]] = filtered_Λ
        wms_of_t[times[k+1]] = exp.(filtered_logweights)
        θ_of_t[times[k+1]] = filtered_θ
    end

    return Λ_of_t, wms_of_t, θ_of_t

end

# function transition_CIR(n::Integer, Dt::Real, x0::Real, δ, γ, σ)
#     ks = rand(Poisson(γ/σ^2*x0/(exp(2*γ*Dt)-1)), n)
#     rate = γ/σ^2*exp(2*γ*Dt)/(exp(2*γ*Dt)-1)
#     xs = Float64[rand(Gamma(k+δ/2, 1/rate)) for k in ks]
# end

function rCIR(n::Integer, Dt::Real, x0::Real, δ, γ, σ)
    β = γ/σ^2*exp(2*γ*Dt)/(exp(2*γ*Dt)-1)
    if n == 1
        ks = rand(Poisson(γ/σ^2*x0/(exp(2*γ*Dt)-1)))
        return rand(Gamma(ks+δ/2, 1/β))
    else
        ks = rand(Poisson(γ/σ^2*x0/(exp(2*γ*Dt)-1)), n)
        return rand.(Gamma.(ks+δ/2, 1/β))
    end
end

transition_CIR(n::Integer, Dt::Real, x0::Real, δ, γ, σ) = rCIR(n::Integer, Dt::Real, x0::Real, δ, γ, σ)

function rec_transition_CIR(Dts, x, δ, γ, σ)
    x_new = transition_CIR(1, Dts[1], x[end], δ, γ, σ)
    if length(Dts) == 1
        return Float64[x; x_new]
    else
        return Float64[x; rec_transition_CIR(Dts[2:end], x_new, δ, γ, σ)]
    end
end

function generate_CIR_trajectory(times, x0, δ, γ, σ)
    Dts = diff(times)
    return rec_transition_CIR(Dts, [x0], δ, γ, σ)
end

function get_predictive_mixture_CIR_at_time(t, Λ_of_t, wms_of_t, δ::Ty, θ_of_t, γ::Ty, σ::Ty)  where Ty<:Number
    observation_times = Λ_of_t |> keys |> collect |> sort
    closest_observation_time = maximum(observation_times[observation_times.<t])
    return predict_CIR_params(wms_of_t[closest_observation_time], δ, θ_of_t[closest_observation_time], γ, σ, Λ_of_t[closest_observation_time], t-closest_observation_time)
end
