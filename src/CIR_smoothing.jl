
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


function wms_tilde_kp1_from_wms_tilde_kp2(wms_tilde_kp2, Λ_tilde_kp1, θ_tilde_kp1, θ_tilde_prime_kp2, ykp1, Δk, α, γ, σ, λ)
    wms_tilde_kp1 = zeros(maximum(Λ_tilde_kp1)+1)
    p = γ/σ^2*1/(θ_tilde_kp1*exp(2*γ*Δk) + γ/σ^2 - θ_tilde_kp1)
    for k in 1:length(Λ_tilde_kp1)
        n = Λ_tilde_kp1[k]
        for m in 0:n
            idx = m+1
            # Careful, wms_tilde_kp2[k] has index k because we assume that the weights and the indices are in the same order.
            wms_tilde_kp1[idx] += wms_tilde_kp2[k] * μπh(n, θ_tilde_prime_kp2, α, ykp1) * pmn_CIR(n, m, p)
        end
    end
    return wms_tilde_kp1
end

function compute_all_cost_to_go_functions_CIR(δ, γ, σ, λ, data)
    times = data |> keys |> collect |> sort
    reversed_times = reverse(times)
    α = δ/2#Alternative parametrisation
    θ0_CIR = γ/σ^2
    Λ_tilde_prime_of_t = Dict()
    wms_tilde_of_t = Dict()
    θ_tilde_prime_of_t = Dict()

    yT = data[times[end]]

    Λ_tilde_kp1 = t_CIR(yT, 0)
    θ_tilde_prime_kp2 = θ0_CIR
    wms_tilde_kp2 = 1.

    for k in 2:length(reversed_times)
        # Change of notation for clarity
        prev_t = reversed_times[k-1]
        Δk = prev_t - times[k]
        ykp1 = data[prev_t]

        # New weight computation
        θ_tilde_kp1 = θ_tilde_k_from_θ_tilde_prime_kp1(ykp1, θ_tilde_prime_kp2)

        wms_tilde_kp1 = wms_tilde_kp1_from_wms_tilde_kp2(wms_tilde_kp2, Λ_tilde_kp1, θ_tilde_kp1, θ_tilde_prime_kp2, ykp1, Δk, α, γ, σ, λ)

        #Storage of the results
        θ_tilde_prime_kp1 = θ_tilde_prime_k_from_θ_tilde_k_CIR(θ_tilde_kp1, Δk, γ, σ)
        Λ_tilde_prime_kp1 = Λ_tilde_prime_k_from_Λ_tilde_k_CIR(Λ_tilde_kp1)

        θ_tilde_prime_of_t[prev_t] = θ_tilde_prime_kp1
        Λ_tilde_prime_of_t[prev_t] = Λ_tilde_prime_kp1
        wms_tilde_of_t[prev_t] = wms_tilde_kp1

        #Preparation of next iteration
        θ_tilde_prime_kp2 = θ_tilde_prime_kp1
        wms_tilde_kp2 = wms_tilde_kp1
        Λ_tilde_kp1 = Λ_tilde_k_from_Λ_tilde_prime_kp1_CIR(ykp1, Λ_tilde_prime_kp2)
    end
end
