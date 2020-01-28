using IterTools


function Λ_tilde_prime_k_from_Λ_tilde_k_WF(Λ_tilde_k)
    return Λprime_i_from_Λ(Λ_tilde_k)
end

function Λ_tilde_k_from_Λ_tilde_prime_kp1_WF(yk, Λ_tilde_prime_kp1)
    return t_WF(yk, Λ_tilde_prime_kp1)
end

function WF_backpropagation_for_one_m_precomputed(m::Array{Int64,1}, α, sα::Ty, t::Ty, y_kp1, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff; wm = 1) where {Ty<:Number}
    # gm = map(x -> 0:x, m) |> vec |> x -> Iterators.product(x...)
    logweights_dict = WF_backpropagation_for_one_m_precomputed_logweights(m, α, sα, t, y_kp1, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff; logwm = log(wm))

    Dict( collect(n) => exp(logweights_dict[n]) for n in keys(logweights_dict) ) |> Accumulator

end

function WF_backpropagation_for_one_m_precomputed_logweights(m::Array{Int64,1}, α, sα::Ty, t::Ty, y_kp1, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff; logwm = 0) where {Ty<:Number}
    # gm = map(x -> 0:x, m) |> vec |> x -> Iterators.product(x...)
    logμm_θ = logμπh_WF(α, m, y_kp1)
    t_ykp1_m::Array{Int64,1} = t_WF(y_kp1, m)
    gm = indices_of_tree_below(t_ykp1_m)
    sm = sum(t_ykp1_m)

    function fun_n(n)
        i = t_ykp1_m.-n
        # println(i)
        si = sum(i)
        return logwm + logμm_θ  + logpmmi_precomputed(i, t_ykp1_m, sm, si, t, sα, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff)
    end

    Dict( collect(n) => fun_n(n) for n in gm ) |> Accumulator

end

function update_logweights_cost_to_go_WF(logwms_tilde_kp2, Λ_tilde_prime_kp2, α, ykp1)
    update_logweights_cost_to_go(logwms_tilde_kp2, Λ_tilde_prime_kp2, 1., ykp1, (α = α,), (m, θ, y, params) -> logμπh_WF(params[:α], m, y))
end

function predict_logweights_cost_to_go_WF_one_m(m::Array{Int64,1}, α, sα, Δt, y_kp1, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff; logwm = 0)
    # The indices are updated inside the prediction function !!
    t_ykp1_m::Array{Int64,1} = t_WF(y_kp1, m)
    gm = indices_of_tree_below(t_ykp1_m)
    sm = sum(t_ykp1_m)
    function fun_n(n)
        i = t_ykp1_m.-n
        # println(i)
        si = sum(i)
        return logwm + logpmmi_precomputed(i, t_ykp1_m, sm, si, Δt, sα, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff)
    end

    Dict( collect(n) => fun_n(n) for n in gm ) |> Accumulator
end

function predict_logweights_cost_to_go_WF(updated_logwms_tilde_kp2, Λ_tilde_prime_kp2, α, sα, y_kp1, Δt, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff)

    res = Accumulator{Array{Int64,1}, Float64}()
    for k in eachindex(Λ_tilde_prime_kp2)
        res = merge(logaddexp, res, predict_logweights_cost_to_go_WF_one_m(Λ_tilde_prime_kp2[k], α, sα, Δt, y_kp1, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff; logwm = updated_logwms_tilde_kp2[k]))
    end

    ks = keys(res) |> collect

    return ks, [res[k] for k in ks]

end

function logwms_tilde_kp1_from_logwms_tilde_kp2_WF_pruning(logwms_tilde_kp2::Array{Ty,1}, α, sα::Ty, Λ_tilde_prime_kp2::Array{Array{Int64,1},1}, Δt::Ty, y_kp1, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff, pruning_function::Function; return_indices = true) where {Ty<:Number}

    #only the weights are updated, and not the indices, because the indices are updated inside the prediction function


    updated_logwms_tilde_kp2::Array{Float64,1} = update_logweights_cost_to_go_WF(logwms_tilde_kp2, Λ_tilde_prime_kp2, α, y_kp1)

    Λ_tilde_prime_kp2_pruned::Array{Array{Int64,1},1}, updated_logwms_tilde_kp2_pruned::Array{Float64,1} = pruning_function(Λ_tilde_prime_kp2, updated_logwms_tilde_kp2)

    Λ_tilde_prime_kp1::Array{Array{Int64,1},1}, logwms_tilde_kp1::Array{Float64,1} = predict_logweights_cost_to_go_WF(updated_logwms_tilde_kp2, Λ_tilde_prime_kp2, α, sα, y_kp1, Δt, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff)

    if return_indices
        return Λ_tilde_prime_kp1, logwms_tilde_kp1
    else
        return logwms_tilde_kp1
    end

end

function wms_tilde_kp1_from_wms_tilde_kp2_WF(wms_tilde_kp2::Array{Ty,1}, α, sα::Ty, Λ_tilde_prime_kp2::Array{Array{Int64,1},1}, t::Ty, y_kp1, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff) where {Ty<:Number}

    res = Accumulator{Array{Int64,1}, Float64}()

    for k in eachindex(Λ_tilde_prime_kp2)
        res = merge(res, WF_backpropagation_for_one_m_precomputed(Λ_tilde_prime_kp2[k], α, sα, t, y_kp1, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff; wm = wms_tilde_kp2[k]))
    end

    ks = keys(res) |> collect

    return ks, [res[k] for k in ks]

end

function logwms_tilde_kp1_from_logwms_tilde_kp2_WF(logwms_tilde_kp2::Array{Ty,1}, α, sα::Ty, Λ_tilde_prime_kp2::Array{Array{Int64,1},1}, t::Ty, y_kp1, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff) where {Ty<:Number}

    res = Accumulator{Array{Int64,1}, Float64}()

    for k in eachindex(Λ_tilde_prime_kp2)
        res = merge(logaddexp, res, WF_backpropagation_for_one_m_precomputed_logweights(Λ_tilde_prime_kp2[k], α, sα, t, y_kp1, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff; logwm = logwms_tilde_kp2[k]))
    end#Check that we are merging logdicts !!

    ks = keys(res) |> collect

    return ks, [res[k] for k in ks]

end

function compute_all_cost_to_go_functions_WF(α, data, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff; silence = false)
    times = data |> keys |> collect |> sort
    reversed_times = reverse(times)

    Λ_tilde_prime_of_t = Dict()
    wms_tilde_of_t = Dict()
    sα = sum(α)

    yT = data[times[end]]

    Λ_tilde_prime_kp2 = [zeros(Int64, length(α))]
    Λ_tilde_kp1 = t_WF(yT, Λ_tilde_prime_kp2)
    wms_tilde_kp2 = [1.]

    for k in 2:length(reversed_times)
        if !silence
            println("(Cost to go) Step index: $k")
            println("Number of components: $(length(Λ_tilde_prime_kp2))")
        end
        # Change of notation for clarity
        prev_t = reversed_times[k-1]
        t = reversed_times[k]
        Δk = prev_t - t
        ykp1 = data[prev_t]

        # @show k
        # @show wms_tilde_kp2
        # New weight computation

        Λ_tilde_prime_kp1, wms_tilde_kp1 = wms_tilde_kp1_from_wms_tilde_kp2_WF(wms_tilde_kp2, α, sα, Λ_tilde_prime_kp2, Δk, ykp1, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff)

        #Storage of the results

        # Λ_tilde_prime_kp1 = Λ_tilde_prime_k_from_Λ_tilde_k_WF(Λ_tilde_kp1)

        yk = data[t]
        Λ_tilde_kp = Λ_tilde_k_from_Λ_tilde_prime_kp1_WF(yk, Λ_tilde_prime_kp1) #Not stored, but better for consistency of notations.

        Λ_tilde_prime_of_t[prev_t] = Λ_tilde_prime_kp1
        wms_tilde_of_t[prev_t] = wms_tilde_kp1

        #Preparation of next iteration
        wms_tilde_kp2 = wms_tilde_kp1
        Λ_tilde_prime_kp2 = Λ_tilde_prime_kp1
        Λ_tilde_kp1 = Λ_tilde_kp
    end

    return Λ_tilde_prime_of_t, wms_tilde_of_t
end

function compute_all_cost_to_go_functions_WF_adaptive_precomputation_ar(α, data, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff, sm_max_so_far; silence = false)
    times = data |> keys |> collect |> sort
    reversed_times = reverse(times)

    @assert length(α) == length(data[collect(keys(data))[1]])
    Δt = assert_constant_time_step_and_compute_it(data)


    Λ_tilde_prime_of_t = Dict()
    wms_tilde_of_t = Dict()
    sα = sum(α)

    yT = data[times[end]]

    Λ_tilde_prime_kp2 = [zeros(Int64, length(α))]
    Λ_tilde_kp1 = t_WF(yT, Λ_tilde_prime_kp2)
    wms_tilde_kp2 = [1.]

    new_sm_max = maximum(sum.(Λ_tilde_kp1))
    precompute_next_terms_ar!(sm_max_so_far, new_sm_max, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff, sα, Δt)
    sm_max_so_far = max(sm_max_so_far,new_sm_max)

    for k in 2:length(reversed_times)
        if !silence
            println("(Cost to go) Step index: $k")
            println("Number of components: $(length(Λ_tilde_prime_kp2))")
        end
        # Change of notation for clarity
        prev_t = reversed_times[k-1]
        t = reversed_times[k]
        Δk = prev_t - t
        ykp1 = data[prev_t]

        last_sm_max = maximum(sum.(Λ_tilde_kp1))
        new_sm_max = last_sm_max + sum(ykp1)

        if sm_max_so_far < new_sm_max
            precompute_next_terms_ar!(sm_max_so_far, new_sm_max, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff, sα, Δt)
            sm_max_so_far = max(sm_max_so_far,new_sm_max)
        end

        # @show k
        # @show wms_tilde_kp2
        # New weight computation

        Λ_tilde_prime_kp1, wms_tilde_kp1 = wms_tilde_kp1_from_wms_tilde_kp2_WF(wms_tilde_kp2, α, sα, Λ_tilde_prime_kp2, Δk, ykp1, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff)

        #Storage of the results

        # Λ_tilde_prime_kp1 = Λ_tilde_prime_k_from_Λ_tilde_k_WF(Λ_tilde_kp1)

        yk = data[t]
        Λ_tilde_kp = Λ_tilde_k_from_Λ_tilde_prime_kp1_WF(yk, Λ_tilde_prime_kp1) #Not stored, but better for consistency of notations.

        Λ_tilde_prime_of_t[prev_t] = Λ_tilde_prime_kp1
        wms_tilde_of_t[prev_t] = wms_tilde_kp1

        #Preparation of next iteration
        wms_tilde_kp2 = wms_tilde_kp1
        Λ_tilde_prime_kp2 = Λ_tilde_prime_kp1
        Λ_tilde_kp1 = Λ_tilde_kp
    end

    return Λ_tilde_prime_of_t, wms_tilde_of_t
end

function compute_all_log_cost_to_go_functions_WF_adaptive_precomputation_ar(α, data, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff, sm_max_so_far; silence = false)
    times = data |> keys |> collect |> sort
    reversed_times = reverse(times)

    @assert length(α) == length(data[collect(keys(data))[1]])
    Δt = assert_constant_time_step_and_compute_it(data)


    Λ_tilde_prime_of_t = Dict()
    logwms_tilde_of_t = Dict()
    sα = sum(α)

    yT = data[times[end]]

    Λ_tilde_prime_kp2 = [zeros(Int64, length(α))]
    Λ_tilde_kp1 = collect(t_WF(yT, Λ_tilde_prime_kp2))
    logwms_tilde_kp2 = [0.]

    new_sm_max = maximum(sum.(Λ_tilde_kp1))
    precompute_next_terms_ar!(sm_max_so_far, new_sm_max, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff, sα, Δt)
    sm_max_so_far = max(sm_max_so_far,new_sm_max)

    for k in 2:length(reversed_times)
        if !silence
            println("(Cost to go) Step index: $k")
            println("Number of components: $(length(Λ_tilde_prime_kp2))")
        end
        # Change of notation for clarity
        prev_t = reversed_times[k-1]
        t = reversed_times[k]
        Δk = prev_t - t
        ykp1 = data[prev_t]

        last_sm_max = maximum(sum.(Λ_tilde_kp1))
        new_sm_max = last_sm_max + sum(ykp1)

        if sm_max_so_far < new_sm_max
            precompute_next_terms_ar!(sm_max_so_far, new_sm_max, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff, sα, Δt)
            sm_max_so_far = max(sm_max_so_far,new_sm_max)
        end

        # @show k
        # @show wms_tilde_kp2
        # New weight computation

        Λ_tilde_prime_kp1, logwms_tilde_kp1 = logwms_tilde_kp1_from_logwms_tilde_kp2_WF(logwms_tilde_kp2, α, sα, Λ_tilde_prime_kp2, Δk, ykp1, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff)

        #Storage of the results

        # Λ_tilde_prime_kp1 = Λ_tilde_prime_k_from_Λ_tilde_k_WF(Λ_tilde_kp1)

        yk = data[t]
        Λ_tilde_kp = Λ_tilde_k_from_Λ_tilde_prime_kp1_WF(yk, Λ_tilde_prime_kp1) #Not stored, but better for consistency of notations.

        Λ_tilde_prime_of_t[prev_t] = Λ_tilde_prime_kp1
        logwms_tilde_of_t[prev_t] = logwms_tilde_kp1

        #Preparation of next iteration
        logwms_tilde_kp2 = logwms_tilde_kp1
        Λ_tilde_prime_kp2 = Λ_tilde_prime_kp1
        Λ_tilde_kp1 = Λ_tilde_kp
    end

    return Λ_tilde_prime_of_t, logwms_tilde_of_t
end


function d_WF(m, n)
    return m .+ n
end

function WF_smoothing(α, data; silence = false)

    # Re-using precomputed terms
    if !silence
        println("Filtering")
    end
    Λ_of_t, wms_of_t, log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset, sm_max_so_far = filter_WF_adaptive_precomputation_ar(α, data, (x, y) -> (x, y); silence = silence, return_precomputed_terms = true)
    if !silence
        println("Cost to go")
    end

    data_1D = DualOptimalFiltering_proof.prepare_WF_dat_1D_2D(data)[1]


    Λ_tilde_prime_of_t, wms_tilde_of_t = compute_all_cost_to_go_functions_WF_adaptive_precomputation_ar(α, data_1D, log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset, sm_max_so_far; silence = silence)

    times = Λ_of_t |> keys |> collect |> sort

    Λ_of_t_smooth = Dict()
    wms_of_t_smooth = Dict()

    # wms_smooth = Array{Float64,1}(undef, 2*sum(sum(values(data))))
    # Λ_smooth = Array{Float64,1}(undef, 2*sum(sum(values(data))))

    for k in 1:(length(times)-1)
        # fill!(wms_smooth, 0.)
        # cnt = 1
        wms_smooth = Accumulator(Dict{Array{Int64,1}, Float64}())

        Λk = Λ_of_t[times[k]]
        wk = wms_of_t[times[k]]
        Λ_tilde_prime_kp1 = Λ_tilde_prime_of_t[times[k+1]]
        w_tilde_kp1 = wms_tilde_of_t[times[k+1]]
        for i in eachindex(Λk)
            n = Λk[i]
            for j in eachindex(Λ_tilde_prime_kp1)
                m = Λ_tilde_prime_kp1[j]
                # @show m, n
                # @show d_WF(m,n)
                wms_smooth[d_WF(m,n)] += w_tilde_kp1[j]*wk[i]
            end
        end
        wms_smooth = normalise(wms_smooth)

        Λ_of_t_smooth[times[k]] = [key for key in keys(wms_smooth) if wms_smooth[key] > 0]
        wms_of_t_smooth[times[k]] = [wms_smooth[key] for key in Λ_of_t_smooth[times[k]]]
    end

        #The last smoothing distribution is a filtering distribution
        Λ_of_t_smooth[times[end]] = Λ_of_t[times[end]]
        wms_of_t_smooth[times[end]] = wms_of_t[times[end]]

    return Λ_of_t_smooth, wms_of_t_smooth

end

function WF_smoothing_log_internals(α, data; silence = false)

    # Re-using precomputed terms
    if !silence
        println("Filtering")
    end
    Λ_of_t, wms_of_t, log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset, sm_max_so_far = filter_WF_adaptive_precomputation_ar(α, data, (x, y) -> (x, y); silence = silence, return_precomputed_terms = true)
    if !silence
        println("Cost to go")
    end

    data_1D = DualOptimalFiltering_proof.prepare_WF_dat_1D_2D(data)[1]


    Λ_tilde_prime_of_t, logwms_tilde_of_t = compute_all_log_cost_to_go_functions_WF_adaptive_precomputation_ar(α, data_1D, log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset, sm_max_so_far; silence = silence)

return merge_filtering_and_cost_to_go_logweights_WF(Λ_of_t, wms_of_t, Λ_tilde_prime_of_t, logwms_tilde_of_t)

end

function compute_all_log_cost_to_go_functions_WF_adaptive_precomputation_ar_pruning(α, data, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff, sm_max_so_far, pruning_function::Function; silence = false)
    times = data |> keys |> collect |> sort
    reversed_times = reverse(times)

    @assert length(α) == length(data[collect(keys(data))[1]])
    Δt = assert_constant_time_step_and_compute_it(data)


    Λ_tilde_prime_of_t = Dict()
    logwms_tilde_of_t = Dict()
    sα = sum(α)

    yT = data[times[end]]

    Λ_tilde_prime_kp2 = [zeros(Int64, length(α))]
    Λ_tilde_kp1 = collect(t_WF(yT, Λ_tilde_prime_kp2))
    logwms_tilde_kp2 = [0.]

    new_sm_max = maximum(sum.(Λ_tilde_kp1))
    precompute_next_terms_ar!(sm_max_so_far, new_sm_max, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff, sα, Δt)
    sm_max_so_far = max(sm_max_so_far, new_sm_max)

    for k in 2:length(reversed_times)
        if !silence
            println("(Cost to go) Step index: $k")
            println("Number of components: $(length(Λ_tilde_prime_kp2))")
        end
        # Change of notation for clarity
        prev_t = reversed_times[k-1]
        t = reversed_times[k]
        Δk = prev_t - t
        ykp1 = data[prev_t]

        last_sm_max = maximum(sum.(Λ_tilde_kp1))
        new_sm_max = last_sm_max + sum(ykp1)

        if sm_max_so_far < new_sm_max
            precompute_next_terms_ar!(sm_max_so_far, new_sm_max, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff, sα, Δt)
            sm_max_so_far = max(sm_max_so_far,new_sm_max)
        end

        # @show k
        # @show wms_tilde_kp2
        # New weight computation

        Λ_tilde_prime_kp1, logwms_tilde_kp1 = logwms_tilde_kp1_from_logwms_tilde_kp2_WF_pruning(logwms_tilde_kp2, α, sα, Λ_tilde_prime_kp2, Δk, ykp1, precomputed_log_ν, precomputed_log_Cmmi, precomputed_log_binomial_coeff, pruning_function)

        #Storage of the results

        # Λ_tilde_prime_kp1 = Λ_tilde_prime_k_from_Λ_tilde_k_WF(Λ_tilde_kp1)

        yk = data[t]
        Λ_tilde_kp = Λ_tilde_k_from_Λ_tilde_prime_kp1_WF(yk, Λ_tilde_prime_kp1) #Not stored, but better for consistency of notations.

        Λ_tilde_prime_of_t[prev_t] = Λ_tilde_prime_kp1
        logwms_tilde_of_t[prev_t] = logwms_tilde_kp1

        #Preparation of next iteration
        logwms_tilde_kp2 = logwms_tilde_kp1
        Λ_tilde_prime_kp2 = Λ_tilde_prime_kp1
        Λ_tilde_kp1 = Λ_tilde_kp
    end

    return Λ_tilde_prime_of_t, logwms_tilde_of_t
end

function WF_smoothing_pruning(α, data, pruning_function::Function; silence = false)

    # Re-using precomputed terms
    if !silence
        println("Filtering")
    end
    Λ_of_t, wms_of_t, log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset, sm_max_so_far = filter_WF_adaptive_precomputation_ar(α, data, pruning_function; silence = silence, return_precomputed_terms = true)
    if !silence
        println("Cost to go")
    end

    data_1D = DualOptimalFiltering_proof.prepare_WF_dat_1D_2D(data)[1]


    Λ_tilde_prime_of_t, logwms_tilde_of_t = compute_all_log_cost_to_go_functions_WF_adaptive_precomputation_ar_pruning(α, data_1D, log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset, sm_max_so_far, pruning_function; silence = silence)

    return merge_filtering_and_cost_to_go_logweights_WF(Λ_of_t, wms_of_t, Λ_tilde_prime_of_t, logwms_tilde_of_t)

end

function merge_filtering_and_cost_to_go_logweights_WF(Λ_of_t, wms_of_t, Λ_tilde_prime_of_t, logwms_tilde_of_t)

    times = Λ_of_t |> keys |> collect |> sort
    Λ_of_t_smooth = Dict()
    wms_of_t_smooth = Dict()

    for k in 1:(length(times)-1)
        # fill!(wms_smooth, 0.)
        # cnt = 1
        logwms_smooth = Accumulator(Dict{Array{Int64,1}, Float64}())

        Λk = Λ_of_t[times[k]]
        logwk = log.(wms_of_t[times[k]])
        Λ_tilde_prime_kp1 = Λ_tilde_prime_of_t[times[k+1]]
        logw_tilde_kp1 = logwms_tilde_of_t[times[k+1]]

        #Initialisation of the Accumulator
        for i in eachindex(Λk)
            n = Λk[i]
            for j in eachindex(Λ_tilde_prime_kp1)
                m = Λ_tilde_prime_kp1[j]
                # @show m, n
                # @show d_WF(m,n)
                logwms_smooth[d_WF(m,n)] = -Inf
            end
        end

        for i in eachindex(Λk)
            n = Λk[i]
            for j in eachindex(Λ_tilde_prime_kp1)
                m = Λ_tilde_prime_kp1[j]
                # @show m, n
                # @show d_WF(m,n)
                logwms_smooth[d_WF(m,n)] = logaddexp(logwms_smooth[d_WF(m,n)], logw_tilde_kp1[j] + logwk[i])
            end
        end

        logwms_smooth = normalise_logAccumulator(logwms_smooth)

        Λ_of_t_smooth[times[k]] = [key for key in keys(logwms_smooth) if logwms_smooth[key] > -Inf]
        wms_of_t_smooth[times[k]] = [exp(logwms_smooth[key]) for key in Λ_of_t_smooth[times[k]]]
    end

        #The last smoothing distribution is a filtering distribution
        Λ_of_t_smooth[times[end]] = Λ_of_t[times[end]]
        wms_of_t_smooth[times[end]] = wms_of_t[times[end]]

    return Λ_of_t_smooth, wms_of_t_smooth
end
