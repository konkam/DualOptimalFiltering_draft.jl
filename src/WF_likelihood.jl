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

function Λi_from_Λprime_im1_max(yi::Union{AbstractArray{U, 1}, Tuple}, Λprime_im1_max::Union{AbstractArray{U, 1}, Tuple}) where U <: Integer
    return indices_of_tree_below(t_WF(yi, Λprime_im1_max))
end

function t_WF(y::Array{T, 1}, m::Union{AbstractArray{U, 1}, Tuple}) where {T <: Real, U <: Integer}
    return m .+ y
end

function t_WF(y::Array{T, 1}, Λ)  where {T <: Real, U <: Integer}
    return (t_WF(y, m) for m in Λ)
end

function update_logwms_to_i_from_log_wms_prime_im1!(α::Array{T, 1}, logwms, logwms_prime_im1, Λ_prime_im1_max::Array{U, 1}, yi::Array{U, 1}) where {T <: Real, U <: Integer}
    Λ_prime_im1 = Λ_from_Λ_max(Λ_prime_im1_max)
    Λ_i = Λ_from_Λ_max(t_WF(yi, Λ_prime_im1_max))
    for n in Λ_prime_im1

        shifted_idn = n .+ 1
        m = t_WF(yi,n)
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

    # for n in Λ
    #     shifted_id_n = n .+ 1
    #     for m in indices_of_tree_below(t_WF(n))
    #         shifted_id_m = m .+ 1
    #         current_logwms[shifted_id_m...] += current_logwms_hat[shifted_id_n...] +
    #
    #     end
    # end

    for m in Λ_prime_i
        shifted_id_m = m .+ 1
        logwms_prime[shifted_id_m...] = logsumexp(logpmn_precomputed(n, m, sum(n), sum(m), Δt_ip1, sα, log_ν_dict, log_Cmmi_dict, log_binomial_coeff_dict) + logwms_i[(n .+ 1)...] for n in Λi if all(n .>= m))
    end

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

    logw = zeros(nmax...)
    logw_prime = zeros(nmax...)

    μν_prime_im1 = Array{Float64,1}(undef, length(times))

    #Prior
    Λprime_im1_max = zeros(Int64, length(α))
    Λprime_im1 = [Λprime_im1_max]
    logw_prime[(Λprime_im1_max .+ 1)...] = 1.

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

        μν_prime_im1[i] = logsumexp(logw_prime[(m .+ 1)...] + logμπh_WF(α, m, yi) for m in Λprime_im1)

        #Hidden state updating
        # println("Hidden state updating")
        Λi_max = t_WF(yi, Λprime_im1_max)
        Λi = Λi_from_Λprime_im1_max(yi, Λprime_im1_max)
        Λprime_i_max = Λi_max
        Λprime_i = Λprime_i_from_Λimax(Λi_max)

        #Update
        # println("Weights updating")
        update_logwms_to_i_from_log_wms_prime_im1!(α, logw, logw_prime, Λprime_im1_max, yi)

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
