function keep_last_k(x, k)
    if length(x) <= k
        return x
    else
        return x[(end-k):end]
    end
end


function keep_above_threshold(Λ_of_t, wms_of_t, ε)
    Λ_of_t_kept = [Λ_of_t[i] for i in 1:length(Λ_of_t) if wms_of_t[i] >= ε]
    wms_of_t_kept = [wms_of_t[i] for i in 1:length(Λ_of_t) if wms_of_t[i] >= ε]
    return Λ_of_t_kept, wms_of_t_kept
end

function keep_fixed_number_of_weights(Λ_of_t, wms_of_t, k)
    last_w = wms_of_t |> sort |> x -> keep_last_k(x, k) |> x -> x[1] #smallest weight kept
    keep_above_threshold(Λ_of_t, wms_of_t, last_w)
end

function filter_WF_precomputed_keep_fixed_number(α, data, log_ν_dict::Dict{Tuple{Int64, Int64}, Float64}, log_Cmmi_dict::Dict{Tuple{Int64, Int64}, Float64}, precomputed_log_binomial_coefficients::Dict{Tuple{Int64, Int64}, Float64}, fixed_number::Int64)
    # println("filter_WF_mem2")

    function keep_fixed_number(Λ_of_t, wms_of_t)
        keep_fixed_number_of_weights(Λ_of_t, wms_of_t, fixed_number)
    end

    filter_WF_precomputed_pruning(α, data, log_ν_dict, log_Cmmi_dict, precomputed_log_binomial_coefficients, keep_fixed_number)

end

function filter_WF_precomputed_pruning(α, data, log_ν_dict::Dict{Tuple{Int64, Int64}, Float64}, log_Cmmi_dict::Dict{Tuple{Int64, Int64}, Float64}, precomputed_log_binomial_coefficients::Dict{Tuple{Int64, Int64}, Float64}, do_the_pruning::Function)
    # println("filter_WF_mem2")

    @assert length(α) == length(data[collect(keys(data))[1]])

    sα = sum(α)
    times = keys(data) |> collect |> sort
    Λ_of_t = Dict()
    wms_of_t = Dict()

    filtered_Λ, filtered_wms = update_WF_params([1.], α, [repeat([0], inner = length(α))], data[times[1]])
    Λ_pruned, wms_pruned = do_the_pruning(filtered_Λ, filtered_wms)

    Λ_of_t[times[1]] = filtered_Λ
    wms_of_t[times[1]] = filtered_wms

    for k in 1:(length(times)-1)
        println("Step index: $k")
        println("Number of components: $(length(filtered_Λ))")
        filtered_Λ, filtered_wms = get_next_filtering_distribution_precomputed(Λ_pruned, wms_pruned, times[k], times[k+1], α, sα, data[times[k+1]], log_ν_dict, log_Cmmi_dict, precomputed_log_binomial_coefficients)
        Λ_pruned, wms_pruned = do_the_pruning(filtered_Λ, filtered_wms)
        Λ_of_t[times[k+1]] = filtered_Λ
        wms_of_t[times[k+1]] = filtered_wms
    end

    return Λ_of_t, wms_of_t

end
