function get_active_components(Λ_with_weights)
#     return (m for m in keys(Λ_with_weights) if Λ_with_weights[m]>0)
    return filter((m,w) -> w>0., Λ_with_weights)
end

function update_WF_params!(full_Λ_with_weights::Dict{Array{Int64,1},Float64}, active_Λ::Dict{Array{Int64,1},Float64}, α::Array{Ty,1}, y::Array{Int64,2}) where {Ty<:Number}
    ms = keys(active_Λ) |> collect
#     filtered_Λ, filtered_wms = update_WF_params(values(active_Λ) |> collect, α, keys(active_Λ) |> collect, y) #Still hybrid because profiling said so
    filtered_Λ, filtered_wms = update_WF_params([active_Λ[m] for m in ms], α, ms, y) #Still hybrid because profiling said so
    for m in keys(active_Λ)
        full_Λ_with_weights[m] = 0.
    end
    for k in 1:length(filtered_Λ)
        full_Λ_with_weights[filtered_Λ[k]] = filtered_wms[k]
    end
end

function WF_prediction_for_one_m_precomputed!(full_Λ_with_weights::Dict{Array{Int64,1},Float64}, m::Array{Int64,1}, sα::Ty, t::Ty, log_ν_dict::Dict{Tuple{Int64, Int64}, Float64}, log_Cmmi_dict::Dict{Tuple{Int64, Int64}, Float64}, precomputed_log_binomial_coefficients::Dict{Tuple{Int64, Int64}, Float64}; wm = 1) where {Ty<:Number}
    gm = map(x -> 0:x, m) |> vec |> x -> product(x...)
    sm = sum(m)
    full_Λ_with_weights[m] -= wm
    for n in gm
        i = m.-n
        si = sum(i)
        full_Λ_with_weights[collect(n)] += wm*(logpmmi_precomputed(i, m, sm, si, t, sα, log_ν_dict, log_Cmmi_dict, precomputed_log_binomial_coefficients) |> exp)
    end
end

function predict_WF_params_precomputed!(full_Λ_with_weights::Dict{Array{Int64,1},Float64}, sα::Ty, active_Λ::Dict{Array{Int64,1},Float64}, t::Ty, log_ν_dict::Dict{Tuple{Int64, Int64}, Float64}, log_Cmmi_dict::Dict{Tuple{Int64, Int64}, Float64}, precomputed_log_binomial_coefficients::Dict{Tuple{Int64, Int64}, Float64}) where {Ty<:Number}

    for m in keys(active_Λ)
        WF_prediction_for_one_m_precomputed!(full_Λ_with_weights, m, sα, t, log_ν_dict, log_Cmmi_dict, precomputed_log_binomial_coefficients; wm = active_Λ[m])
    end

end

function cmp_next_filtering_distribution_precomputed!(full_Λ_with_weights::Dict{Array{Int64,1},Float64}, active_Λ::Dict{Array{Int64,1},Float64}, current_time, next_time, α, sα, next_y, log_ν_dict::Dict{Tuple{Int64, Int64}, Float64}, log_Cmmi_dict::Dict{Tuple{Int64, Int64}, Float64}, precomputed_log_binomial_coefficients::Dict{Tuple{Int64, Int64}, Float64})
    predict_WF_params_precomputed!(full_Λ_with_weights, sα, active_Λ, next_time-current_time, log_ν_dict, log_Cmmi_dict, precomputed_log_binomial_coefficients)
    active_Λ = get_active_components(full_Λ_with_weights)
    update_WF_params!(full_Λ_with_weights, active_Λ, α, next_y)
end

function filter_WF_precomputed_preallocate(α, data, log_ν_dict::Dict{Tuple{Int64, Int64}, Float64}, log_Cmmi_dict::Dict{Tuple{Int64, Int64}, Float64}, precomputed_log_binomial_coefficients::Dict{Tuple{Int64, Int64}, Float64})
    # println("filter_WF_mem2")

    @assert length(α) == length(data[collect(keys(data))[1]])

    K = length(α)
    sα = sum(α)
    mmax = values(data) |> sum

    times = keys(data) |> collect |> sort

    gm = map(x -> 0:x, mmax) |> vec |> x -> product(x...)

    full_Λ_with_weights = Dict(collect(m) => 0. for m in gm)
    full_Λ_with_weights[repeat([0], inner = K) ] = 1.

    Λ_of_t = Dict()

    wms_of_t = Dict()

    active_Λ_dict = get_active_components(full_Λ_with_weights)

    update_WF_params!(full_Λ_with_weights, active_Λ_dict, α, data[times[1]])

    active_Λ_dict = get_active_components(full_Λ_with_weights)
    ks = keys(active_Λ_dict)
    Λ_of_t[times[1]] = ks
    wms_of_t[times[1]] = [active_Λ_dict[k] for k in ks]

    for k in 1:(length(times)-1)
        println("Step index: $(k)")
        println("Number of components: $(length(keys(active_Λ_dict)))")
        cmp_next_filtering_distribution_precomputed!(full_Λ_with_weights, active_Λ_dict, times[k], times[k+1], α, sα, data[times[k+1]], log_ν_dict, log_Cmmi_dict, precomputed_log_binomial_coefficients)
        active_Λ_dict = get_active_components(full_Λ_with_weights)
        ks = keys(active_Λ_dict) |> collect
        Λ_of_t[times[k+1]] = ks
        wms_of_t[times[k+1]] = [active_Λ_dict[k] for k in ks]
    end

    return Λ_of_t, wms_of_t

end
