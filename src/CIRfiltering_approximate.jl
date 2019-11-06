function filter_CIR_keep_fixed_number(δ, γ, σ, λ, data, fixed_number::Int64; silence = false)
    # println("filter_WF_mem2")

    function prune_keeping_fixed_number(Λ_of_t, wms_of_t)
        Λ_of_t_kept, wms_of_t_kept = keep_fixed_number_of_weights(Λ_of_t, wms_of_t, fixed_number)
        return Λ_of_t_kept, normalise(wms_of_t_kept)
    end

    filter_CIR_pruning(δ, γ, σ, λ, data, prune_keeping_fixed_number; silence = silence)

end

function filter_CIR_keep_above_threshold(δ, γ, σ, λ, data, ε::Float64; silence = false)
    # println("filter_WF_mem2")

    function prune_keeping_above_threshold(Λ_of_t, wms_of_t)
        Λ_of_t_kept, wms_of_t_kept = keep_above_threshold(Λ_of_t, wms_of_t, ε)
        return Λ_of_t_kept, normalise(wms_of_t_kept)
    end

    filter_CIR_pruning(δ, γ, σ, λ, data, prune_keeping_above_threshold; silence = silence)

end

function filter_CIR_fixed_fraction(δ, γ, σ, λ, data, fraction::Float64; silence = false)
    # println("filter_WF_mem2")
    function prune_keeping_fixed_fraction(Λ_of_t, wms_of_t)
        Λ_of_t_kept, wms_of_t_kept = keep_fixed_fraction(Λ_of_t, wms_of_t, fraction)
        return Λ_of_t_kept, normalise(wms_of_t_kept)
    end

    filter_CIR_pruning(δ, γ, σ, λ, data, prune_keeping_fixed_fraction; silence = silence)

end

function filter_CIR_pruning(δ, γ, σ, λ, data, do_the_pruning::Function; silence = false)

    times = keys(data) |> collect |> sort
    Λ_of_t = Dict()
    wms_of_t = Dict()
    θ_of_t = Dict()

    filtered_θ, filtered_Λ, filtered_wms = update_CIR_params([1.], δ, γ/σ^2, λ, [0], data[0])
    pruned_Λ, pruned_wms = do_the_pruning(filtered_Λ, filtered_wms)

    Λ_of_t[0] = filtered_Λ
    wms_of_t[0] = filtered_wms # = 1.
    θ_of_t[0] = filtered_θ

    for k in 1:(length(times)-1)
        if (!silence)
            println("Step index: $k")
            println("Number of components: $(length(filtered_Λ))")
        end
        filtered_θ, filtered_Λ, filtered_wms = get_next_filtering_distribution(pruned_Λ, pruned_wms, filtered_θ, times[k], times[k+1], δ, γ, σ, λ, data[times[k+1]])
        pruned_Λ, pruned_wms = do_the_pruning(filtered_Λ, filtered_wms)
        Λ_of_t[times[k+1]] = filtered_Λ
        wms_of_t[times[k+1]] = filtered_wms
        θ_of_t[times[k+1]] = filtered_θ
    end

    return Λ_of_t, wms_of_t, θ_of_t

end

function filter_predict_CIR_pruning(δ, γ, σ, λ, data, do_the_pruning::Function; silence = false)

    times = keys(data) |> collect |> sort
    Λ_of_t = Dict()
    wms_of_t = Dict()
    θ_of_t = Dict()
    Λ_pred_of_t = Dict()
    wms_pred_of_t = Dict()
    θ_pred_of_t = Dict()

    filtered_θ, filtered_Λ, filtered_wms = update_CIR_params([1.], δ, γ/σ^2, λ, [0], data[0])
    pruned_Λ, pruned_wms = do_the_pruning(filtered_Λ, filtered_wms)

    Λ_of_t[0] = filtered_Λ
    wms_of_t[0] = filtered_wms # = 1.
    θ_of_t[0] = filtered_θ

    for k in 1:(length(times)-1)
        if (!silence)
            println("Step index: $k")
            println("Number of components: $(length(filtered_Λ))")
        end
        predicted_θ, predicted_Λ, predicted_wms = predict_CIR_params(filtered_wms, δ, filtered_θ, γ, σ, filtered_Λ, times[k+1]-times[k])
        filtered_θ, filtered_Λ, filtered_wms = update_CIR_params(predicted_wms, δ, predicted_θ, λ, predicted_Λ, data[times[k+1]])
        pruned_Λ, pruned_wms = do_the_pruning(filtered_Λ, filtered_wms)

        Λ_pred_of_t[times[k+1]] = predicted_Λ
        wms_pred_of_t[times[k+1]] = predicted_wms
        θ_pred_of_t[times[k+1]] = predicted_θ
        Λ_of_t[times[k+1]] = filtered_Λ
        wms_of_t[times[k+1]] = filtered_wms
        θ_of_t[times[k+1]] = filtered_θ
    end

    return Λ_of_t, wms_of_t, θ_of_t, Λ_pred_of_t, wms_pred_of_t, θ_pred_of_t

end

function filter_CIR_pruning_logweights(δ, γ, σ, λ, data, do_the_pruning::Function; silence = false)
    Λ_of_t, wms_of_t, θ_of_t = filter_CIR_pruning(δ, γ, σ, λ, data, do_the_pruning::Function; silence = silence)
    times = keys(data) |> collect |> sort
    logwms_of_t = Dict(zip(times, [log.(wms_of_t[t]) for t in times]))
    # @show keys(logwms_of_t)
    for t in times
        # @show length(Λ_of_t[t]), length(logwms_of_t[t])
        # if length(Λ_of_t[t]) !== length(logwms_of_t[t])
        #     error("Some weights were lost during log conversions")
        # end
        @assert !any(isnan.(logwms_of_t[t]))
        # @show t
        # @show sum(logwms_of_t[t])
    end
    return Λ_of_t, logwms_of_t, θ_of_t
end

#Finish this if it is ever needed.

# function filter_CIR_pruning_logweights(δ, γ, σ, λ, data, do_the_pruning::Function; silence = false)
#
#     times = keys(data) |> collect |> sort
#     Λ_of_t = Dict()
#     wms_of_t = Dict()
#     θ_of_t = Dict()
#
#     filtered_θ, filtered_Λ, filtered_wms = update_CIR_params([1.], δ, γ/σ^2, λ, [0], data[0])
#     pruned_Λ, pruned_wms = do_the_pruning(filtered_Λ, filtered_wms)
#
#     Λ_of_t[0] = filtered_Λ
#     wms_of_t[0] = filtered_wms # = 1.
#     θ_of_t[0] = filtered_θ
#
#     for k in 1:(length(times)-1)
#         if (!silence)
#             println("Step index: $k")
#             println("Number of components: $(length(filtered_Λ))")
#         end
#         filtered_θ, filtered_Λ, filtered_wms = get_next_filtering_distribution(pruned_Λ, pruned_wms, filtered_θ, times[k], times[k+1], δ, γ, σ, λ, data[times[k+1]])
#         pruned_Λ, pruned_wms = do_the_pruning(filtered_Λ, filtered_wms)
#         Λ_of_t[times[k+1]] = filtered_Λ
#         wms_of_t[times[k+1]] = filtered_wms
#         θ_of_t[times[k+1]] = filtered_θ
#     end
#
#     return Λ_of_t, wms_of_t, θ_of_t
#
# end
