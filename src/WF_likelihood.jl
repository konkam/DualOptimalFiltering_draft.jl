using IterTools

function logμπh_WF(α::AbstractArray{T, 1}, m::Union{AbstractArray{U, 1}, Tuple}, y::AbstractArray{U, 1}) where {T <: Real, U <: Integer}
    ## Needs to be written for multiple observations too
    sy = sum(y)
    sα = sum(α)
    return lfactorial(sy) - sum(lfactorial.(y)) + sum(log_pochammer.(α .+ m, y)) - log_pochammer(sα + sum(m), sy)
end

function compute_next_Λ_max(current_Λ_max::Array{T, 1}, y::Array{T, 1}) where T <: Integer
    return t_WF(y, current_Λ_max)
end

#Keep to single observations for the moment.
# function compute_next_Λ_max(current_Λ_max::Array{T, 1}, y::Array{T, 2}) where T <: Integer
#     return current_Λ_max .+ vec(sum(y, dims = 1))
# end

function Λprime_i_from_Λimax(Λi_max::Union{AbstractArray{U, 1}, Tuple}) where U <: Integer
    return indices_of_tree_below(Λi_max)
end

function Λprime_i_from_Λ(Λ)
    return IterTools.distinct(Iterators.flatten(indices_of_tree_below(m) for m in Λ))
end


function Λi_from_Λprime_im1(yi::Union{AbstractArray{U, 1}, Tuple}, Λprime_im1) where U <: Integer
    return t_WF(yi, Λprime_im1)
end

function t_WF(y::Array{T, 1}, m::Union{AbstractArray{U, 1}, Tuple}) where {T <: Real, U <: Integer}
    return m .+ y
end

function t_WF(y::Array{T, 1}, Λ)  where {T <: Real, U <: Integer}
    return (t_WF(y, m) for m in Λ)
end

function update_logwms_to_i_from_log_wms_prime_im1!(α::Array{T, 1}, logwms, logwms_prime_im1, Λ_prime_im1, yi::Array{U, 1}) where {T <: Real, U <: Integer}
    # Λ_prime_im1 = Λ_from_Λ_max(Λ_prime_im1_max)
    Λ_i = t_WF(yi, Λ_prime_im1)

    for n in Λ_prime_im1
        shifted_idn = n .+ 1
        m = t_WF(yi, n)
        shifted_idm = m .+ 1

        logwms[shifted_idm...] = logμπh_WF(α, n, yi) + logwms_prime_im1[shifted_idn...]
    end

    norm_factor = logsumexp(logwms[(m.+1)...] for m in Λ_i)

    for m in Λ_i
        shifted_idm = m .+ 1
        logwms[shifted_idm...] -= norm_factor
    end
end

function update_logwms_prime_to_i_from_logwms_i!(sα::Real, logwms_i, logwms_prime, Λi, Λ_prime_i_max::Array{U, 1}, Δt_ip1::Real, log_ν_dict::Dict{Tuple{Int64, Int64}, Float64}, log_Cmmi_dict::Dict{Tuple{Int64, Int64}, Float64}, log_binomial_coeff_dict::Dict{Tuple{Int64, Int64}, Float64}) where U <: Integer
    Λ_prime_i = Λ_from_Λ_max(Λ_prime_i_max)

    #This is collecting the weight received by each new components.

    for m in Λ_prime_i
        shifted_id_m = m .+ 1
        logwms_prime[shifted_id_m...] = logsumexp(logpmn_precomputed(n, m, sum(n), sum(m), Δt_ip1, sα, log_ν_dict, log_Cmmi_dict, log_binomial_coeff_dict) + logwms_i[(n .+ 1)...] for n in Λi if all(n .>= m))
    end
    return
end

function update_logwms_prime_to_i_from_logwms_i2!(sα::Real, logwms_i, logwms_prime, Λi, Δt_ip1::Real, log_ν_dict::Dict{Tuple{Int64, Int64}, Float64}, log_Cmmi_dict::Dict{Tuple{Int64, Int64}, Float64}, log_binomial_coeff_dict::Dict{Tuple{Int64, Int64}, Float64}) where U <: Integer
    # Λprime_i_max = Λ_max_from_Λ(pruned_Λ)
    # Λprime_i_sup = Λ_from_Λ_max(Λprime_i_max)
    # for m in Λprime_i
    #     shifted_id_m = m .+ 1
    #     logwms_prime[shifted_id_m...] = 0.
    # end
    fill!(logwms_prime, -Inf)

#This is taking each component and propagating it downwards in the tree.

    for n in Λi
        shifted_id_n = n .+ 1
        for m in indices_of_tree_below(n)
            i = n .- m
            si = sum(i)
            sn = sum(n)

            shifted_id_m = m .+ 1
            logwms_prime[shifted_id_m...] = logaddexp(logwms_prime[shifted_id_m...], logwms_i[shifted_id_n...] + logpmmi_precomputed(i, n, sn, si, Δt_ip1, sα, log_ν_dict, log_Cmmi_dict, log_binomial_coeff_dict))
        end
    end
    return
end



function WF_loglikelihood_adaptive_precomputation(α, data_; silence = false)
    # println("filter_WF_mem2")

    @assert length(α) == length(data_[collect(keys(data_))[1]])
    Δts = keys(data_) |> collect |> sort |> diff |> unique
    if length(Δts) > 1
        test_equal_spacing_of_observations(data_; override = false)
    end
    Δt = mean(Δts)

    sα = sum(α)
    times = keys(data_) |> collect |> sort
    data_one_obs = Dict(t => vec(data_[t]) for t in times)
    # nmax = maximum_number_of_components_WF(data_)
    nmax = sum(data_one_obs |> values)

    log_ν_dict = Dict{Tuple{Int64, Int64}, Float64}()
    log_Cmmi_dict = Dict{Tuple{Int64, Int64}, Float64}()
    log_binomial_coeff_dict = Dict{Tuple{Int64, Int64}, Float64}()

    logw = fill(-Inf,nmax...)
    logw_prime = fill(-Inf,nmax...)

    μν_prime_im1 = Array{Float64,1}(undef, length(times))

    #Prior
    Λprime_im1_max = zeros(Int64, length(α))
    Λprime_im1 = [Λprime_im1_max]
    logw_prime[(Λprime_im1_max .+ 1)...] = 0.

    for i in 1:(length(times)-1)
        if (!silence)
            println("Step index: $i")
        end

        ti = times[i]
        yi = data_one_obs[times[i]]
        t_ip1 = times[i+1]
        Δt_ip1 = t_ip1 - ti

        #Log likelihood
        # println("Log likelihood")

        # println("Λprime_im1_max = $Λprime_im1_max")
        # println("Λprime_im1 = $(collect(Λprime_im1))")

        μν_prime_im1[i] = logsumexp(logw_prime[(m .+ 1)...] + logμπh_WF(α, m, yi) for m in Λprime_im1)

        #Hidden state updating
        # println("Hidden state updating")
        Λi_max = t_WF(yi, Λprime_im1_max)
        Λi = Λi_from_Λprime_im1(yi, Λprime_im1)
        Λprime_i_max = Λi_max
        Λprime_i = Λprime_i_from_Λimax(Λi_max)

        # println("Λi = $(collect(Λi))")


        #Update
        # println("Weights updating")
        update_logwms_to_i_from_log_wms_prime_im1!(α, logw, logw_prime, Λprime_im1, yi)
        # println("logw = $(collect(logw[(m .+ 1)...] for m in Λi) |> vec)")

        #Prediction
        # println("Precomputation for prediction")
        precompute_next_terms!(sum(Λprime_im1_max), sum(Λi_max), log_ν_dict, log_Cmmi_dict, log_binomial_coeff_dict, sα, Δt)

        # println("Prediction")
        update_logwms_prime_to_i_from_logwms_i!(sα, logw, logw_prime, Λi, Λprime_i_max, Δt_ip1, log_ν_dict, log_Cmmi_dict, log_binomial_coeff_dict)

        # last_sm_max = maximum(sum.(Λ_pruned))
        # new_sm_max = last_sm_max + sum(data_one_obs[times[k+1]])
        #
        # if sm_max_so_far < new_sm_max
        #     precompute_next_terms!(sm_max_so_far, new_sm_max, log_ν_dict, log_Cmmi_dict, log_binomial_coeff_dict, sα, Δt)
        #     sm_max_so_far = max(sm_max_so_far,new_sm_max)
        # end

        # For next iteration
        Λprime_im1_max = Λprime_i_max
        Λprime_im1 = Λ_from_Λ_max(Λprime_im1_max)
    end

    last_yi = data_one_obs[times[end]]
    μν_prime_im1[end] = logsumexp(logw_prime[(m .+ 1)...] + logμπh_WF(α, m, last_yi) for m in Λprime_im1)
    log_lik_terms = Dict(zip(times, cumsum(μν_prime_im1)))

    return log_lik_terms

end

function log_likelihood_WF_keep_fixed_number(α, data, fixed_number::Int64; silence = false)
    # println("filter_WF_mem2")

    prune_keeping_fixed_number_log_wms_(Λ_of_t, log_wms_of_t) =
        prune_keeping_fixed_number_log_wms(Λ_of_t, log_wms_of_t, fixed_number)

    WF_loglikelihood_adaptive_precomputation_pruning(α, data, prune_keeping_fixed_number_log_wms_; silence = silence)

end

function log_likelihood_WF_fixed_fraction(α, data, fraction::Float64; silence = false)
    # println("filter_WF_mem2")
    prune_keeping_fixed_fraction_log_wms_(Λ_of_t, log_wms_of_t) = prune_keeping_fixed_fraction_log_wms(Λ_of_t, log_wms_of_t, fraction)

    WF_loglikelihood_adaptive_precomputation_pruning(α, data, prune_keeping_fixed_fraction_log_wms_; silence = silence)

end

function log_likelihood_WF_keep_above_threshold(α, data, ε::Float64; silence = false)
    # println("filter_WF_mem2")

    prune_keeping_above_threshold_log_wms_(Λ_of_t, log_wms_of_t) = prune_keeping_above_threshold_log_wms(Λ_of_t, log_wms_of_t, ε)

    WF_loglikelihood_adaptive_precomputation_pruning(α, data, prune_keeping_above_threshold_log_wms_; silence = silence)

end

function sum_Λ_max_from_Λ(Λ)::Int64
    # n = zeros(Int64, length(Λ[1]))
    # for k in 1:length(Λ)
    #     n .= max.(n, Λ[k])
    # end
    # return n
    return maximum(sum(k) for k in Λ)
end

function Λ_max_from_Λ(Λ)
    n = zeros(Int64, length(collect(take(Λ, 1))[1]))
    for m in Λ
        n .= max.(n, m)
    end
    return n
end

function WF_loglikelihood_adaptive_precomputation_pruning(α, data_, do_the_pruning_log_wms::Function; silence = false)
    # println("filter_WF_mem2")

    ##From the pruned set of indices, there is no obvious cheap way to compute the set after propagation. Each component propagates downwards on the tree, but the number of active components. At the moment we make an set iterator, using the function IterTools.distinct

    @assert length(α) == length(data_[collect(keys(data_))[1]])
    Δts = keys(data_) |> collect |> sort |> diff |> unique
    if length(Δts) > 1
        test_equal_spacing_of_observations(data_; override = false)
    end
    Δt = mean(Δts)

    sα = sum(α)
    times = keys(data_) |> collect |> sort
    data_one_obs = Dict(t => vec(data_[t]) for t in times)
    # nmax = maximum_number_of_components_WF(data_)
    nmax = sum(data_one_obs |> values)

    log_ν_dict = Dict{Tuple{Int64, Int64}, Float64}()
    log_Cmmi_dict = Dict{Tuple{Int64, Int64}, Float64}()
    log_binomial_coeff_dict = Dict{Tuple{Int64, Int64}, Float64}()

    logw = zeros(nmax...)
    logw_prime = zeros(nmax...)

    μν_prime_im1 = Array{Float64,1}(undef, length(times))

    #Prior
    # Λprime_im1_max = zeros(Int64, length(α))
    Λprime_im1 = [zeros(Int64, length(α))]
    logw_prime[(zeros(Int64, length(α)) .+ 1)...] = 0.
    sum_Λ_max_last = 0

    for i in 1:(length(times)-1)
        if (!silence)
            println("Step index: $i")
        end

        ti = times[i]
        yi = data_one_obs[times[i]]
        t_ip1 = times[i+1]
        Δt_ip1 = t_ip1 - ti

        #Log likelihood
        # println("Log likelihood")

        # println("Λprime_im1_max = $Λprime_im1_max")
        # println("Λprime_im1 = $(collect(Λprime_im1))")

        μν_prime_im1[i] = logsumexp(logw_prime[(m .+ 1)...] + logμπh_WF(α, m, yi) for m in Λprime_im1)

        #Hidden state updating
        # println("Hidden state updating")
        # Λi_max = t_WF(yi, Λprime_im1_max)
        Λi = Λi_from_Λprime_im1(yi, Λprime_im1)

        # println("Λi = $(collect(Λi))")

        #Update
        # println("Weights updating")
        update_logwms_to_i_from_log_wms_prime_im1!(α, logw, logw_prime, Λprime_im1, yi)
        # println("logw = $(collect(logw[(m .+ 1)...] for m in Λi) |> vec)")

        #Pruning

        pruned_Λ, pruned_log_wms = do_the_pruning_log_wms(Λi, logw[(m .+ 1)...] for m in Λi)
        sum_Λ_max = sum_Λ_max_from_Λ(pruned_Λ)

        if sum_Λ_max > sum_Λ_max_last
            # println("Precomputation for prediction")
            precompute_next_terms!(sum_Λ_max_last, sum_Λ_max, log_ν_dict, log_Cmmi_dict, log_binomial_coeff_dict, sα, Δt)
            sum_Λ_max_last = sum_Λ_max
        end

        #Prediction
        # println("Prediction")
        update_logwms_prime_to_i_from_logwms_i2!(sα, logw, logw_prime, pruned_Λ, Δt_ip1, log_ν_dict, log_Cmmi_dict, log_binomial_coeff_dict)

        # last_sm_max = maximum(sum.(Λ_pruned))
        # new_sm_max = last_sm_max + sum(data_one_obs[times[k+1]])
        #
        # if sm_max_so_far < new_sm_max
        #     precompute_next_terms!(sm_max_so_far, new_sm_max, log_ν_dict, log_Cmmi_dict, log_binomial_coeff_dict, sα, Δt)
        #     sm_max_so_far = max(sm_max_so_far,new_sm_max)
        # end

        # For next iteration
        # Λprime_im1_max = Λ_max_from_Λ(pruned_Λ)
        Λprime_im1 = Λprime_i_from_Λ(pruned_Λ)
        #This Λprime_im1 contains components with 0 weight.
    end

    last_yi = data_one_obs[times[end]]
    μν_prime_im1[end] = logsumexp(logw_prime[(m .+ 1)...] + logμπh_WF(α, m, last_yi) for m in Λprime_im1)
    log_lik_terms = Dict(zip(times, cumsum(μν_prime_im1)))

    return log_lik_terms

end
