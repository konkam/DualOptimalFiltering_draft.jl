function keep_above_threshold(Λ_of_t, wms_of_t, ε)
    Λ_of_t_kept = [Λ_of_t[i] for i in 1:length(Λ_of_t) if wms_of_t[i] >= ε]
    wms_of_t_kept = [wms_of_t[i] for i in 1:length(Λ_of_t) if wms_of_t[i] >= ε]
    return Λ_of_t_kept, wms_of_t_kept
end

function keep_fixed_number_of_weights(Λ_of_t, wms_of_t, k)
    last_w = wms_of_t |> sort |> x -> keep_last_k(x, k) |> x -> x[1] #smallest weight kept
    keep_above_threshold(Λ_of_t, wms_of_t, last_w)
end

function prune_keeping_fixed_number_log_wms(Λ_of_t, log_wms_of_t, fixed_number::Integer)
    Λ_of_t_kept, log_wms_of_t_kept = keep_fixed_number_of_weights(Λ_of_t, log_wms_of_t, fixed_number)
    return Λ_of_t_kept, log_wms_of_t_kept .- logsumexp(log_wms_of_t_kept)
end

function prune_keeping_above_threshold_log_wms(Λ_of_t, log_wms_of_t, ε::Real)
    Λ_of_t_kept, log_wms_of_t_kept = keep_above_threshold(Λ_of_t, log_wms_of_t, log(ε))
    return Λ_of_t_kept, log_wms_of_t_kept .- logsumexp(log_wms_of_t_kept)
end

function prune_keeping_fixed_fraction_log_wms(Λ_of_t, log_wms_of_t, fraction::Real)
    Λ_of_t_kept, wms_of_t_kept = keep_fixed_fraction(Λ_of_t, exp.(log_wms_of_t), fraction)
    return Λ_of_t_kept, log.(normalise(wms_of_t_kept))
end

function keep_fixed_number_of_logweights(Λ_of_t, log_wms_of_t, k)
    last_log_w = log_wms_of_t |> sort |> x -> keep_last_k(x, k) |> x -> x[1] #smallest weight kept
    keep_above_threshold(Λ_of_t, log_wms_of_t, last_log_w)
end

function keep_fixed_fraction(Λ_of_t, wms_of_t, fraction)
    #Could be made slightly faster by avoiding some inclusion tests.
    #Could even use the threshold
    sorted_wms_of_t = sort(wms_of_t)
    wms_of_t_kept_sorted = sorted_wms_of_t[sorted_wms_of_t |> cumsum |> x -> Int.(x.>= 1-fraction) |> x -> Bool.(x)]
    threshold = minimum(wms_of_t_kept_sorted)#If it is unique, then there should be no error
    keep_above_threshold(Λ_of_t, wms_of_t, threshold)

    # Λ_of_t_kept = [Λ_of_t[i] for i in 1:length(Λ_of_t) if wms_of_t[i] in wms_of_t_kept_sorted]
    # wms_of_t_kept = [wms_of_t[i] for i in 1:length(Λ_of_t) if wms_of_t[i] in wms_of_t_kept_sorted] #to preserve ordering
    # return Λ_of_t_kept,wms_of_t_kept
end
