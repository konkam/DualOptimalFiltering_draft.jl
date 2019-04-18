
function Λ_tilde_prime_k_from_Λ_tilde_k_CIR(Λ_tilde_k)
    return 0:maximum(Λ_tilde_k)
end

function Λ_tilde_k_from_Λ_tilde_prime_kp1_CIR(yk, Λ_tilde_prime_kp1)
    return t_CIR(yk, Λ_tilde_prime_kp1)
end

function θ_tilde_prime_k_from_θ_tilde_k_CIR(θ_tilde_k, Δt, γ, σ)
    return θ_prime_from_θ_CIR(θ_tilde_k, Δt, γ, σ)
end

function θ_tilde_k_from_θ_tilde_prime_kp1(yk, θ_tilde_prime_kp1; λ = 1)
    return T_CIR(yk, θ_tilde_prime_kp1; λ = λ)
end

function pmn_CIR(m, n, p)
    return pdf(Binomial(m, p), n)
end


function pmn_CIR(m, n, Δt, θ, σ, γ)
    p = γ/σ^2*1/(θ*exp(2*γ*Δt) + γ/σ^2 - θ)
    return pmn_CIR(m, n, p)
end
function logpmn_CIR(m, n, p)
    return logpdf(Binomial(m, p), n)
end


function logpmn_CIR(m, n, Δt, θ, σ, γ)
    p = γ/σ^2*1/(θ*exp(2*γ*Δt) + γ/σ^2 - θ)
    return logpmn_CIR(m, n, p)
end


function logCmn_CIR(m, n, θback_k, θ_k, α, β)
    return SpecialFunctions.lgamma(α) - SpecialFunctions.lgamma(α+m) - SpecialFunctions.lgamma(α+n) - α*log(β) + (α+m)*log(θback_k) + (α+n)*log(θ_k) + SpecialFunctions.lgamma(α+m+n) - (α+m+n) * log(θback_k + θ_k-β)
end


function wms_tilde_kp1_from_wms_tilde_kp2(wms_tilde_kp2, Λ_tilde_prime_kp2, θ_tilde_kp1, θ_tilde_prime_kp2, ykp1, Δk, α, γ, σ, λ)
    wms_tilde_kp1 = zeros(t_CIR(ykp1, maximum(Λ_tilde_prime_kp2))+1)
    p = γ/σ^2*1/(θ_tilde_kp1*exp(2*γ*Δk) + γ/σ^2 - θ_tilde_kp1)
    # println(p)
    for k in 1:length(Λ_tilde_prime_kp2)
        n = Λ_tilde_prime_kp2[k]
        t_ykp1_n = t_CIR(ykp1, n)

        wn_mu_n = wms_tilde_kp2[k] * μπh(n, θ_tilde_prime_kp2, α, ykp1)
        for m in 0:t_ykp1_n
            idx = m+1
            # Careful, wms_tilde_kp2[k] has index k because we assume that the weights and the indices are in the same order.
            wms_tilde_kp1[idx] += wn_mu_n * pmn_CIR(t_ykp1_n, m, p)
        end
    end
    return wms_tilde_kp1
end

function logwms_tilde_kp1_from_logwms_tilde_kp2(logwms_tilde_kp2, Λ_tilde_prime_kp2, θ_tilde_kp1, θ_tilde_prime_kp2, ykp1, Δk, α, γ, σ, λ)
    logwms_tilde_kp1 = fill(-Inf, maximum(t_CIR(ykp1, maximum(Λ_tilde_prime_kp2)))+1)
    p = γ/σ^2*1/(θ_tilde_kp1*exp(2*γ*Δk) + γ/σ^2 - θ_tilde_kp1)
    # println(p)
    for k in 1:length(Λ_tilde_prime_kp2)
        n = Λ_tilde_prime_kp2[k]
        t_ykp1_n = t_CIR(ykp1, n)

        log_wn_mu_n = logwms_tilde_kp2[k]  + logμπh(n, θ_tilde_prime_kp2, α, ykp1; λ = λ )
        for m in 0:t_ykp1_n
            idx = m+1
            # Careful, logwms_tilde_kp2[k] has index k because we assume that the weights and the indices are in the same order.
            logwms_tilde_kp1[idx] = logaddexp(logwms_tilde_kp1[idx], log_wn_mu_n + logpmn_CIR(t_ykp1_n, m, p))
        end
    end
    return logwms_tilde_kp1
end

# function update_logweights_cost_to_go_CIR(logwms_tilde_kp2, Λ_tilde_prime_kp2, θ_tilde_prime_kp2, α, ykp1, λ)
#     return Float64[logwms_tilde_kp2[k]  + logμπh(Λ_tilde_prime_kp2[k], θ_tilde_prime_kp2, α, ykp1; λ = λ) for k in eachindex(Λ_tilde_prime_kp2)]
# end

function update_logweights_cost_to_go_CIR(logwms_tilde_kp2, Λ_tilde_prime_kp2, θ_tilde_prime_kp2, α, ykp1, λ)
    return update_logweights_cost_to_go(logwms_tilde_kp2, Λ_tilde_prime_kp2, θ_tilde_prime_kp2, ykp1, (α = α, λ = λ), (m, θ, y, params) -> logμπh(m, θ, params[:α], y; λ = params[:λ]))
end

function update_logweights_cost_to_go_CIR_precomputed(logwms_tilde_kp2, Λ_tilde_prime_kp2, θ_tilde_prime_kp2, α, ykp1, λ, precomputed_lgamma_α, precomputed_lfactorial)
    return Float64[logwms_tilde_kp2[k]  + logμπh(Λ_tilde_prime_kp2[k], θ_tilde_prime_kp2, α, ykp1; λ = λ) for k in eachindex(Λ_tilde_prime_kp2)]
end

function predict_logweights_cost_to_go_CIR(updated_logwms_tilde_kp2, Λ_tilde_prime_kp2, θ_tilde_kp1, θ_tilde_prime_kp2, ykp1, Δk, α, γ, σ, λ)
    logwms_tilde_kp1 = fill(-Inf, maximum(t_CIR(ykp1, maximum(Λ_tilde_prime_kp2)))+1)

    p = γ/σ^2*1/(θ_tilde_kp1*exp(2*γ*Δk) + γ/σ^2 - θ_tilde_kp1)

    for k in 1:length(Λ_tilde_prime_kp2)
        n = Λ_tilde_prime_kp2[k]
        t_ykp1_n = t_CIR(ykp1, n)

        for m in 0:t_ykp1_n
            idx = m+1
            # Careful, logwms_tilde_kp2[k] has index k because we assume that the weights and the indices are in the same order.
            logwms_tilde_kp1[idx] = logaddexp(logwms_tilde_kp1[idx], updated_logwms_tilde_kp2[k] + logpmn_CIR(t_ykp1_n, m, p))
        end
    end
    return logwms_tilde_kp1
end

function logwms_tilde_kp1_from_logwms_tilde_kp2_pruning(logwms_tilde_kp2, Λ_tilde_prime_kp2, θ_tilde_kp1, θ_tilde_prime_kp2, ykp1, Δk, α, γ, σ, λ; pruning_function::Function, return_indices = false)

    updated_logwms_tilde_kp2 = update_logweights_cost_to_go_CIR(logwms_tilde_kp2, Λ_tilde_prime_kp2, θ_tilde_prime_kp2, α, ykp1, λ)

    Λ_tilde_prime_kp2_pruned, updated_logwms_tilde_kp2_pruned = pruning_function(Λ_tilde_prime_kp2, updated_logwms_tilde_kp2)

    logwms_tilde_kp1 = predict_logweights_cost_to_go_CIR(updated_logwms_tilde_kp2_pruned, Λ_tilde_prime_kp2_pruned, θ_tilde_kp1, θ_tilde_prime_kp2, ykp1, Δk, α, γ, σ, λ)

    if return_indices
        return logwms_tilde_kp1, 0:maximum(t_CIR(ykp1, maximum(Λ_tilde_prime_kp2_pruned)))
    else
        return logwms_tilde_kp1
    end
end

function logwms_tilde_kp1_from_logwms_tilde_kp2_precomputed(logwms_tilde_kp2, Λ_tilde_prime_kp2, θ_tilde_kp1, θ_tilde_prime_kp2, ykp1, Δk, α, γ, σ, λ, precomputed_lgamma_α, precomputed_lfactorial)
    logwms_tilde_kp1 = fill(-Inf, maximum(t_CIR(ykp1, maximum(Λ_tilde_prime_kp2)))+1)
    p = γ/σ^2*1/(θ_tilde_kp1*exp(2*γ*Δk) + γ/σ^2 - θ_tilde_kp1)
    # println(p)
    for k in 1:length(Λ_tilde_prime_kp2)
        n = Λ_tilde_prime_kp2[k]
        t_ykp1_n = t_CIR(ykp1, n)

        log_wn_mu_n = logwms_tilde_kp2[k] + logμπh_precomputed(n, θ_tilde_prime_kp2, α, ykp1, precomputed_lgamma_α, precomputed_lfactorial; λ = λ )
        for m in 0:t_ykp1_n
            idx = m+1
            # Careful, logwms_tilde_kp2[k] has index k because we assume that the weights and the indices are in the same order.
            # @show n, m
            logwms_tilde_kp1[idx] = logaddexp(logwms_tilde_kp1[idx], log_wn_mu_n + logpmn_CIR(t_ykp1_n, m, p))
        end
    end
    return logwms_tilde_kp1
end

function logwms_tilde_kp1_from_logwms_tilde_kp2_precomputed_pruning(logwms_tilde_kp2, Λ_tilde_prime_kp2, θ_tilde_kp1, θ_tilde_prime_kp2, ykp1, Δk, α, γ, σ, λ, precomputed_lgamma_α, precomputed_lfactorial; pruning_function::Function, return_indices = false)

        updated_logwms_tilde_kp2 = update_logweights_cost_to_go_CIR_precomputed(logwms_tilde_kp2, Λ_tilde_prime_kp2, θ_tilde_prime_kp2, α, ykp1, λ, precomputed_lgamma_α, precomputed_lfactorial)

        Λ_tilde_prime_kp2_pruned, updated_logwms_tilde_kp2_pruned = pruning_function(Λ_tilde_prime_kp2, updated_logwms_tilde_kp2)

        logwms_tilde_kp1 = predict_logweights_cost_to_go_CIR(updated_logwms_tilde_kp2_pruned, Λ_tilde_prime_kp2_pruned, θ_tilde_kp1, θ_tilde_prime_kp2, ykp1, Δk, α, γ, σ, λ)

        if return_indices
            return logwms_tilde_kp1, 0:maximum(t_CIR(ykp1, maximum(Λ_tilde_prime_kp2_pruned)))
        else
            return logwms_tilde_kp1
        end
    end


function logwms_tilde_kp1_from_logwms_tilde_kp2_arb(logwms_tilde_kp2, Λ_tilde_prime_kp2, θ_tilde_kp1, θ_tilde_prime_kp2, ykp1, Δk, α, γ, σ, λ)
    logwms_tilde_kp1 = fill(RR(-Inf), maximum(t_CIR(ykp1, maximum(Λ_tilde_prime_kp2)))+1)
    p = γ/σ^2*1/(θ_tilde_kp1*exp(2*γ*Δk) + γ/σ^2 - θ_tilde_kp1)
    # println(p)
    for k in 1:length(Λ_tilde_prime_kp2)
        n = Λ_tilde_prime_kp2[k]
        t_ykp1_n = t_CIR(ykp1, n)

        log_wn_mu_n = logwms_tilde_kp2[k]  + logμπh_arb(n, θ_tilde_prime_kp2, α, ykp1; λ = λ )
        for m in 0:t_ykp1_n
            idx = m+1
            # Careful, logwms_tilde_kp2[k] has index k because we assume that the weights and the indices are in the same order.
            logwms_tilde_kp1[idx] = logaddexp(logwms_tilde_kp1[idx], log_wn_mu_n + logpmn_CIR(t_ykp1_n, m, p))
        end
    end
    return logwms_tilde_kp1
end

function compute_all_cost_to_go_functions_CIR(δ, γ, σ, λ, data; silence = false)
    times = data |> keys |> collect |> sort
    reversed_times = reverse(times)
    α = δ/2#Alternative parametrisation
    θ0_CIR = γ/σ^2
    Λ_tilde_prime_of_t = Dict()
    wms_tilde_of_t = Dict()
    θ_tilde_prime_of_t = Dict()

    yT = data[times[end]]

    Λ_tilde_prime_kp2 = [0]
    Λ_tilde_kp1 = t_CIR(yT, Λ_tilde_prime_kp2)
    θ_tilde_prime_kp2 = θ0_CIR
    wms_tilde_kp2 = [1.]

    for k in 2:length(reversed_times)
        if !silence
            println("(Cost to go) Step index: $k")
            println("Number of components: $(length(Λ_tilde_kp1))")
        end        # Change of notation for clarity
        prev_t = reversed_times[k-1]
        t = reversed_times[k]
        Δk = prev_t - t
        ykp1 = data[prev_t]

        # New weight computation
        θ_tilde_kp1 = θ_tilde_k_from_θ_tilde_prime_kp1(ykp1, θ_tilde_prime_kp2)

        wms_tilde_kp1 = wms_tilde_kp1_from_wms_tilde_kp2(wms_tilde_kp2, Λ_tilde_prime_kp2, θ_tilde_kp1, θ_tilde_prime_kp2, ykp1, Δk, α, γ, σ, λ)

        #Storage of the results
        θ_tilde_prime_kp1 = θ_tilde_prime_k_from_θ_tilde_k_CIR(θ_tilde_kp1, Δk, γ, σ)
        Λ_tilde_prime_kp1 = Λ_tilde_prime_k_from_Λ_tilde_k_CIR(Λ_tilde_kp1)

        yk = data[t]
        Λ_tilde_kp = Λ_tilde_k_from_Λ_tilde_prime_kp1_CIR(yk, Λ_tilde_prime_kp1) #Not stored, but better for consistency of notations.

        θ_tilde_prime_of_t[prev_t] = θ_tilde_prime_kp1
        Λ_tilde_prime_of_t[prev_t] = Λ_tilde_prime_kp1
        wms_tilde_of_t[prev_t] = wms_tilde_kp1

        #Preparation of next iteration
        θ_tilde_prime_kp2 = θ_tilde_prime_kp1
        wms_tilde_kp2 = wms_tilde_kp1
        Λ_tilde_prime_kp2 = Λ_tilde_prime_kp1
        Λ_tilde_kp1 = Λ_tilde_kp
    end

    return Λ_tilde_prime_of_t, wms_tilde_of_t, θ_tilde_prime_of_t
end

function compute_all_cost_to_go_functions_CIR_pruning(δ, γ, σ, λ, data, pruning_function::Function; silence = false)
    times = data |> keys |> collect |> sort
    reversed_times = reverse(times)
    α = δ/2#Alternative parametrisation
    θ0_CIR = γ/σ^2
    Λ_tilde_prime_of_t = Dict()
    wms_tilde_of_t = Dict()
    θ_tilde_prime_of_t = Dict()

    yT = data[times[end]]

    Λ_tilde_prime_kp2 = [0]
    Λ_tilde_kp1 = t_CIR(yT, Λ_tilde_prime_kp2)
    θ_tilde_prime_kp2 = θ0_CIR
    wms_tilde_kp2 = [1.]

    for k in 2:length(reversed_times)
        if !silence
            println("(Cost to go) Step index: $k")
            println("Number of components: $(length(Λ_tilde_kp1))")
        end        # Change of notation for clarity
        prev_t = reversed_times[k-1]
        t = reversed_times[k]
        Δk = prev_t - t
        ykp1 = data[prev_t]

        # New weight computation
        θ_tilde_kp1 = θ_tilde_k_from_θ_tilde_prime_kp1(ykp1, θ_tilde_prime_kp2)

        # pruned_Λ_tilde_prime_kp2, pruned_wms_tilde_kp2 = pruning_function(Λ_tilde_prime_kp2, wms_tilde_kp2)
        logwms_tilde_kp1, Λ_tilde_prime_kp1 = logwms_tilde_kp1_from_logwms_tilde_kp2_pruning(log.(wms_tilde_kp2), Λ_tilde_prime_kp2, θ_tilde_kp1, θ_tilde_prime_kp2, ykp1, Δk, α, γ, σ, λ; pruning_function = pruning_function, return_indices = true)

        wms_tilde_kp1 = exp.(logwms_tilde_kp1)

        #Storage of the results
        θ_tilde_prime_kp1 = θ_tilde_prime_k_from_θ_tilde_k_CIR(θ_tilde_kp1, Δk, γ, σ)
        # Λ_tilde_prime_kp1 = Λ_tilde_prime_k_from_Λ_tilde_k_CIR(Λ_tilde_kp1)

        yk = data[t]
        Λ_tilde_kp = Λ_tilde_k_from_Λ_tilde_prime_kp1_CIR(yk, Λ_tilde_prime_kp1) #Not stored, but better for consistency of notations.

        θ_tilde_prime_of_t[prev_t] = θ_tilde_prime_kp1
        Λ_tilde_prime_of_t[prev_t] = Λ_tilde_prime_kp1
        wms_tilde_of_t[prev_t] = wms_tilde_kp1

        #Preparation of next iteration
        θ_tilde_prime_kp2 = θ_tilde_prime_kp1
        wms_tilde_kp2 = wms_tilde_kp1
        Λ_tilde_prime_kp2 = Λ_tilde_prime_kp1
        Λ_tilde_kp1 = Λ_tilde_kp
    end

    return Λ_tilde_prime_of_t, wms_tilde_of_t, θ_tilde_prime_of_t
end

function compute_all_log_cost_to_go_functions_CIR_pruning(δ, γ, σ, λ, data, pruning_function::Function; silence = false)
    times = data |> keys |> collect |> sort
    reversed_times = reverse(times)
    α = δ/2#Alternative parametrisation
    θ0_CIR = γ/σ^2
    Λ_tilde_prime_of_t = Dict()
    logwms_tilde_of_t = Dict()
    θ_tilde_prime_of_t = Dict()

    yT = data[times[end]]

    Λ_tilde_prime_kp2 = [0]
    Λ_tilde_kp1 = t_CIR(yT, Λ_tilde_prime_kp2)
    θ_tilde_prime_kp2 = θ0_CIR
    logwms_tilde_kp2 = [0.]

    for k in 2:length(reversed_times)
        if !silence
            println("(Cost to go) Step index: $k")
            println("Number of components: $(length(Λ_tilde_kp1))")
        end
        # Change of notation for clarity
        prev_t = reversed_times[k-1]
        t = reversed_times[k]
        Δk = prev_t - t
        ykp1 = data[prev_t]

        # New weight computation
        θ_tilde_kp1 = θ_tilde_k_from_θ_tilde_prime_kp1(ykp1, θ_tilde_prime_kp2)

        logwms_tilde_kp1, Λ_tilde_prime_kp1 = logwms_tilde_kp1_from_logwms_tilde_kp2_pruning(logwms_tilde_kp2, Λ_tilde_prime_kp2, θ_tilde_kp1, θ_tilde_prime_kp2, ykp1, Δk, α, γ, σ, λ; pruning_function = pruning_function, return_indices = true)

        #Storage of the results
        θ_tilde_prime_kp1 = θ_tilde_prime_k_from_θ_tilde_k_CIR(θ_tilde_kp1, Δk, γ, σ)

        yk = data[t]
        Λ_tilde_kp = Λ_tilde_k_from_Λ_tilde_prime_kp1_CIR(yk, Λ_tilde_prime_kp1) #Not stored, but better for consistency of notations.

        θ_tilde_prime_of_t[prev_t] = θ_tilde_prime_kp1
        Λ_tilde_prime_of_t[prev_t] = Λ_tilde_prime_kp1
        logwms_tilde_of_t[prev_t] = logwms_tilde_kp1

        #Preparation of next iteration
        θ_tilde_prime_kp2 = θ_tilde_prime_kp1
        logwms_tilde_kp2 = logwms_tilde_kp1
        Λ_tilde_prime_kp2 = Λ_tilde_prime_kp1
        Λ_tilde_kp1 = Λ_tilde_kp
    end

    return Λ_tilde_prime_of_t, logwms_tilde_of_t, θ_tilde_prime_of_t
end

function compute_all_log_cost_to_go_functions_CIR_pruning_precomputed(δ, γ, σ, λ, data, pruning_function::Function; silence = false)



    times = data |> keys |> collect |> sort
    reversed_times = reverse(times)
    α = δ/2#Alternative parametrisation
    θ0_CIR = γ/σ^2


    precomputed_lgamma_α = DualOptimalFiltering.precompute_lgamma_α(α, data)
    precomputed_lfactorial = DualOptimalFiltering.precompute_lfactorial(data)

    Λ_tilde_prime_of_t = Dict()
    logwms_tilde_of_t = Dict()
    θ_tilde_prime_of_t = Dict()

    yT = data[times[end]]

    Λ_tilde_prime_kp2 = [0]
    Λ_tilde_kp1 = t_CIR(yT, Λ_tilde_prime_kp2)
    θ_tilde_prime_kp2 = θ0_CIR
    logwms_tilde_kp2 = [0.]

    for k in 2:length(reversed_times)
        if !silence
            println("(Cost to go) Step index: $k")
            println("Number of components: $(length(Λ_tilde_kp1))")
        end
        # Change of notation for clarity
        prev_t = reversed_times[k-1]
        t = reversed_times[k]
        Δk = prev_t - t
        ykp1 = data[prev_t]

        # New weight computation
        θ_tilde_kp1 = θ_tilde_k_from_θ_tilde_prime_kp1(ykp1, θ_tilde_prime_kp2)

        logwms_tilde_kp1, Λ_tilde_prime_kp1 = logwms_tilde_kp1_from_logwms_tilde_kp2_precomputed_pruning(logwms_tilde_kp2, Λ_tilde_prime_kp2, θ_tilde_kp1, θ_tilde_prime_kp2, ykp1, Δk, α, γ, σ, λ, precomputed_lgamma_α, precomputed_lfactorial; pruning_function = pruning_function, return_indices = true)

        #Storage of the results
        θ_tilde_prime_kp1 = θ_tilde_prime_k_from_θ_tilde_k_CIR(θ_tilde_kp1, Δk, γ, σ)

        yk = data[t]
        Λ_tilde_kp = Λ_tilde_k_from_Λ_tilde_prime_kp1_CIR(yk, Λ_tilde_prime_kp1) #Not stored, but better for consistency of notations.

        θ_tilde_prime_of_t[prev_t] = θ_tilde_prime_kp1
        Λ_tilde_prime_of_t[prev_t] = Λ_tilde_prime_kp1
        logwms_tilde_of_t[prev_t] = logwms_tilde_kp1

        #Preparation of next iteration
        θ_tilde_prime_kp2 = θ_tilde_prime_kp1
        logwms_tilde_kp2 = logwms_tilde_kp1
        Λ_tilde_prime_kp2 = Λ_tilde_prime_kp1
        Λ_tilde_kp1 = Λ_tilde_kp
    end

    return Λ_tilde_prime_of_t, logwms_tilde_of_t, θ_tilde_prime_of_t
end

compute_all_log_cost_to_go_functions_CIR(δ, γ, σ, λ, data; silence = false) = compute_all_log_cost_to_go_functions_CIR_pruning(δ, γ, σ, λ, data, (x,y) -> (x,y); silence = silence)

function d_CIR(m::Integer, n::Integer)
    return m .+ n
end
function e_CIR(θ1::Real, θ2::Real, β::Real)
    return θ1 + θ2 - β
end

function CIR_smoothing(δ, γ, σ, λ, data; silence = false)
    β = γ/σ^2
    α = δ/2#Alternative parametrisation


    if !silence
        println("Filtering")
    end
    Λ_of_t, wms_of_t, θ_of_t = filter_CIR_logscale_internals(δ, γ, σ, λ, data; silence = silence)
    if !silence
        println("Cost to go")
    end
    Λ_tilde_prime_of_t, wms_tilde_of_t, θ_tilde_prime_of_t = compute_all_cost_to_go_functions_CIR(δ, γ, σ, λ, data)

    times = Λ_of_t |> keys |> collect |> sort

    Λ_of_t_smooth = Dict()
    wms_of_t_smooth = Dict()
    θ_of_t_smooth = Dict()

    Λ_weights = Array{Float64,1}(undef, 2*sum(sum(values(data))))

    for k in 1:(length(times)-1)
        fill!(Λ_weights, 0.)

        Λk = Λ_of_t[times[k]]
        wk = wms_of_t[times[k]]
        Λ_tilde_prime_kp1 = Λ_tilde_prime_of_t[times[k+1]]
        w_tilde_kp1 = wms_tilde_of_t[times[k+1]]
        for i in eachindex(Λk)#There is a sum because multiple m,n pairs could give the same d_CIR(m,n), hence their weight should sum
            n = Λk[i]
            for j in eachindex(Λ_tilde_prime_kp1)
                m = Λ_tilde_prime_kp1[j]
                Λ_weights[d_CIR(m, n)] += w_tilde_kp1[j]*wk[i]*exp(logCmn_CIR(m, n, θ_tilde_prime_of_t[times[k+1]], θ_of_t[times[k]], α, β))
            end
        end
        Λ_weights = normalise(Λ_weights)

        Λ_of_t_smooth[times[k]] = [n for n in eachindex(Λ_weights) if Λ_weights[n] > 0.]
        wms_of_t_smooth[times[k]] = Λ_weights[Λ_weights .> 0.]
        θ_of_t_smooth[times[k]] = e_CIR(θ_tilde_prime_of_t[times[k+1]], θ_of_t[times[k]], β)
    end

        #The last smoothing distribution is a filtering distribution
        Λ_of_t_smooth[times[end]] = Λ_of_t[times[end]]
        wms_of_t_smooth[times[end]] = wms_of_t[times[end]]
        θ_of_t_smooth[times[end]] = θ_of_t[times[end]]

    return Λ_of_t_smooth, wms_of_t_smooth, θ_of_t_smooth

end


function merge_filtering_and_cost_to_go_logweights_CIR(Λ_of_t, logwms_of_t, θ_of_t, Λ_tilde_prime_of_t, logwms_tilde_of_t, θ_tilde_prime_of_t, data, α, β)
    times = Λ_of_t |> keys |> collect |> sort

    Λ_of_t_smooth = Dict()
    wms_of_t_smooth = Dict()
    θ_of_t_smooth = Dict()

    log_Λ_weights = Array{Float64,1}(undef, 2*sum(sum(values(data))))

    for k in 1:(length(times)-1)
        fill!(log_Λ_weights, -Inf)

        Λk = Λ_of_t[times[k]]
        logwk = logwms_of_t[times[k]]
        Λ_tilde_prime_kp1 = Λ_tilde_prime_of_t[times[k+1]]
        logw_tilde_kp1 = logwms_tilde_of_t[times[k+1]]
        for i in eachindex(Λk)
            n = Λk[i]
            for j in eachindex(Λ_tilde_prime_kp1)
                m = Λ_tilde_prime_kp1[j]
                log_Λ_weights[d_CIR(m, n)] = logaddexp(log_Λ_weights[d_CIR(m, n)], logw_tilde_kp1[j] + logwk[i] + logCmn_CIR(m, n, θ_tilde_prime_of_t[times[k+1]], θ_of_t[times[k]], α, β))
            end
        end
        Λ_weights = exp.(log_Λ_weights .- logsumexp(log_Λ_weights))

        Λ_of_t_smooth[times[k]] = [n for n in eachindex(Λ_weights) if Λ_weights[n] > 0.]
        wms_of_t_smooth[times[k]] = Λ_weights[Λ_weights .> 0.]
        θ_of_t_smooth[times[k]] = e_CIR(θ_tilde_prime_of_t[times[k+1]], θ_of_t[times[k]], β)
    end

        #The last smoothing distribution is a filtering distribution
        Λ_of_t_smooth[times[end]] = Λ_of_t[times[end]]
        wms_of_t_smooth[times[end]] = exp.(logwms_of_t[times[end]])
        θ_of_t_smooth[times[end]] = θ_of_t[times[end]]

    return Λ_of_t_smooth, wms_of_t_smooth, θ_of_t_smooth
end

function CIR_smoothing_logscale_internals(δ, γ, σ, λ, data; silence = false)
    β = γ/σ^2
    α = δ/2#Alternative parametrisation

    if !silence
        println("Filtering")
    end
    Λ_of_t, logwms_of_t, θ_of_t = filter_CIR_logweights(δ, γ, σ, λ, data; silence = silence)
    if !silence
        println("Cost to go")
    end
    Λ_tilde_prime_of_t, logwms_tilde_of_t, θ_tilde_prime_of_t = compute_all_log_cost_to_go_functions_CIR(δ, γ, σ, λ, data; silence = silence)

    return merge_filtering_and_cost_to_go_logweights_CIR(Λ_of_t, logwms_of_t, θ_of_t, Λ_tilde_prime_of_t, logwms_tilde_of_t, θ_tilde_prime_of_t, data, α, β)

end
