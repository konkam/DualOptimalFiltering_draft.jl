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

function next_Λ_from_Λ_prime_max(Λ_max::Union{AbstractArray{U, 1}, Tuple}) where U <: Integer
    return indices_of_tree_below(Λ_max)
end

function t_WF(y::Array{T, 1}, m::Union{AbstractArray{U, 1}, Tuple}) where {T <: Real, U <: Integer}
    return m .+ y
end

function t_WF(y::Array{T, 1}, Λ)  where {T <: Real, U <: Integer}
    return (t_WF(y, m) for m in Λ)
end

function next_logwms_from_log_wms_prime!(α::Array{T, 1}, current_logwms, current_logwms_prime, current_Λ_prime_max, y::Array{U, 1}) where {T <: Real, U <: Integer}
    Λ = Λ_from_Λ_max(current_Λ_prime_max)
    for m in Λ
        shifted_id = m .+ 1
        current_logwms[shifted_id...] = current_logwms_prime[shifted_id...] + logμπh_WF(α, m, y)
    end
    norm_factor = logsumexp(current_logwms[(m.+1)...] for m in Λ)
    for m in Λ
       current_logwms[(m.+1)...] -= norm_factor
    end
end

function next_log_wms_prime_from_log_wms!(sα::Real, current_logwms, current_logwms_prime, current_Λ, Λ_prime_max::Array{U, 1}, y::Array{U, 1}, t::Real, log_ν_dict::Dict{Tuple{Int64, Int64}, Float64}, log_Cmmi_dict::Dict{Tuple{Int64, Int64}, Float64}, log_binomial_coeff_dict::Dict{Tuple{Int64, Int64}, Float64}) where U <: Integer
    Λ_new = Λ_from_Λ_max(Λ_prime_max)

    # for n in Λ
    #     shifted_id_n = n .+ 1
    #     for m in indices_of_tree_below(t_WF(n))
    #         shifted_id_m = m .+ 1
    #         current_logwms[shifted_id_m...] += current_logwms_hat[shifted_id_n...] +
    #
    #     end
    # end

    for m in Λ_new
        shifted_id_m = m .+ 1
        current_logwms_prime[shifted_id_m...] = logsumexp(logpmn_precomputed(t_WF(y, n), m, sum(t_WF(y, n)), sum(m), t, sα, log_ν_dict, log_Cmmi_dict, log_binomial_coeff_dict) + current_logwms[(n .+ 1)...] for n in current_Λ if all(t_WF(y, n) .>= m))
    end

end

function WF_likelihood_adaptive_precomputation(α, data, do_the_pruning::Function; silence = false)
    # println("filter_WF_mem2")

    @assert length(α) == length(data[collect(keys(data))[1]])
    Δts = keys(data) |> collect |> sort |> diff |> unique
    if length(Δts) > 1
        test_equal_spacing_of_observations(data; override = false)
    end
    Δt = mean(Δts)

    nmax = maximum_number_of_components_WF(data)

    log_ν_dict = Dict{Tuple{Int64, Int64}, Float64}()
    log_Cmmi_dict = Dict{Tuple{Int64, Int64}, Float64}()
    log_binomial_coeff_dict = Dict{Tuple{Int64, Int64}, Float64}()

    logw = zeros(nmax)
    logw_prime = zeros(nmax)

    sα = sum(α)
    times = keys(data) |> collect |> sort
    # Λ_of_t = Dict()
    # wms_of_t = Dict()
    log_lik_terms = Dict()
    y0 = data[times[1]]
    log_lik_terms[times[1]] = logμπh_WF(α, zeros(Int64, length(α)), y0)

    #Prior
    Λ_minus1_prime_max = zeros(Int64, length(α))
    Λ_minus1_prime = [Λ_minus1_prime_max]

    #1st update
    Λ_max= compute_next_Λ_max(Λ_minus1_prime_max, y0)
    Λ = next_Λ_from_Λ_prime(Λ_minus1_prime, y0, t_WF)
    next_logwms_from_log_wms_prime!(α, logw, Λ_minus1_prime_max, y0)
    # Should be equivalent to logw[y0...] = 1.

    #1st prediction
    Λ_prime_max = Λ_max
    Λ_prime = next_Λ_from_Λ_prime_max(Λ_prime_max)

    precompute_next_terms!(0, Λ_prime_max, log_ν_dict, log_Cmmi_dict, log_binomial_coeff_dict, sα, Δt)
    next_log_wms_prime_from_log_wms!(sα, logw, logw_prime, Λ, Λ_prime_max, times[2]-times[1], log_ν_dict, log_Cmmi_dict, log_binomial_coeff_dict)

    log_lik_terms[times[2]] = log_lik_terms[times[2]] + logsumexp(logμπh_WF(α, n, data[times[2]]) for n in Λ_prime)

    for i in 2:(length(times)-1)
        if (!silence)
            println("Step index: $k")
            println("Number of components: $(length(filtered_Λ))")
        end

        t = times[i]
        next_t = times[i+1]
        Δt = next_t-t

        #Update
        Λ_max= compute_next_Λ_max(Λ_prime_max, data[times[i]])
        Λ = next_Λ_from_Λ_prime(Λ_minus1_prime, data[times[i]], t_WF)
        next_logwms_from_log_wms_prime!(α, logw, Λ_prime_max, data[times[i]])

        #Prediction
        precompute_next_terms!(Λ_prime_max, Λ_max, log_ν_dict, log_Cmmi_dict, log_binomial_coeff_dict, sα, Δt)
        Λ_prime_max = Λ_max
        Λ_prime = next_Λ_from_Λ_prime_max(Λ_prime_max)

        next_log_wms_prime_from_log_wms!(sα, logw, logw_prime, Λ, Λ_prime_max, Δt, log_ν_dict, log_Cmmi_dict, log_binomial_coeff_dict)

        # last_sm_max = maximum(sum.(Λ_pruned))
        # new_sm_max = last_sm_max + sum(data[times[k+1]])
        #
        # if sm_max_so_far < new_sm_max
        #     precompute_next_terms!(sm_max_so_far, new_sm_max, log_ν_dict, log_Cmmi_dict, log_binomial_coeff_dict, sα, Δt)
        #     sm_max_so_far = max(sm_max_so_far,new_sm_max)
        # end


        log_lik_terms[next_t] = log_lik_terms[t] + logsumexp(logμπh_WF(α, n, data[next_t]) for n in Λ_prime_max)

    end

    return log_lik_terms

end
