function convert_from_mi_to_mn(dict_mi::Dict{Tuple{Int64, Int64}, Float64})
    #change of variable from (m, i) to (m,n) using n = m-i
    return Dict{Tuple{Int64, Int64}, Float64}((k[1], k[1]-k[2]) => dict_mi[k] for k in keys(dict_mi))
end

function precompute_termsn(data::Dict{Float64,Array{Int64,2}}, sα::Number; digits_after_comma_for_time_precision = 4)

    if (data |> keys |> collect |> sort |> diff |> x -> truncate_float.(x,14) |> unique |> length > 1)
        error("Think twice about precomputing all terms, as the time intervals are not equal")
    end

    println("Precomputing 3 times")
    @printf "%e" values(data) |> sum |> sum |> n -> n*(n-1)/2 |> BigFloat
    println(" terms")

    log_Cmn_dict = precompute_log_Cmmi(data, sα; digits_after_comma_for_time_precision = 14) |> convert_from_mi_to_mn
    precomputed_log_binomial_coefficientsn = precompute_log_binomial_coefficients(data) |> convert_from_mi_to_mn
    log_νn_dict = precompute_log_first_term(data, sα) |> convert_from_mi_to_mn

    return log_νn_dict, log_Cmn_dict, precomputed_log_binomial_coefficientsn
end

function loghypergeom_pdf_using_precomputedn(n::Array{Int64,1}, m::Array{Int64,1}, sn::Int64, sm::Int64, precomputed_log_binomial_coefficientsn::Dict{Tuple{Int64, Int64}, Float64})
    return sum(precomputed_log_binomial_coefficientsn[(m[k],n[k])] for k in 1:length(m)) - precomputed_log_binomial_coefficientsn[(sm, sn)]
end

function logpmn_raw_precomputed(n::Array{Int64,1}, m::Array{Int64,1}, sm::Int64, sn::Int64, t::Number, log_νn_dict::Dict{Tuple{Int64, Int64}, Float64}, log_Cmn_dict::Dict{Tuple{Int64, Int64}, Float64}, precomputed_log_binomial_coefficientsn::Dict{Tuple{Int64, Int64}, Float64})
    return log_νn_dict[(sm, sn)] + log_Cmn_dict[(sm, sn)]  + loghypergeom_pdf_using_precomputedn(n, m, sn, sm, precomputed_log_binomial_coefficientsn)
end

function logpmn_precomputed(n::Array{Int64,1}, m::Array{Int64,1}, sm::Int64, sn::Int64, t::Number, sα::Number, log_νn_dict::Dict{Tuple{Int64, Int64}, Float64}, log_Cmn_dict::Dict{Tuple{Int64, Int64}, Float64}, precomputed_log_binomial_coefficientsn::Dict{Tuple{Int64, Int64}, Float64})
    if n == m
        return -λm(sm, sα)*t
    else
        return logpmn_raw_precomputed(n, m, sm, sn, t, log_νn_dict, log_Cmn_dict, precomputed_log_binomial_coefficientsn)
    end
end



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

function WF_prediction_for_one_m_precomputedn!(full_Λ_with_weights::Dict{Array{Int64,1},Float64}, m::Array{Int64,1}, sα::Ty, t::Ty, log_νn_dict::Dict{Tuple{Int64, Int64}, Float64}, log_Cmn_dict::Dict{Tuple{Int64, Int64}, Float64}, precomputed_log_binomial_coefficientsn::Dict{Tuple{Int64, Int64}, Float64}; wm = 1) where {Ty<:Number}
    gm = map(x -> 0:x, m) |> vec |> x -> product(x...)
    sm = sum(m)
    full_Λ_with_weights[m] -= wm
    for n in gm
        sn = sum(n)
        full_Λ_with_weights[collect(n)] += wm*(logpmn_precomputed(collect(n), m, sm, sn, t, sα, log_νn_dict, log_Cmn_dict, precomputed_log_binomial_coefficientsn) |> exp)
    end
end

function predict_WF_params_precomputedn!(full_Λ_with_weights::Dict{Array{Int64,1},Float64}, sα::Ty, active_Λ::Dict{Array{Int64,1},Float64}, t::Ty, log_νn_dict::Dict{Tuple{Int64, Int64}, Float64}, log_Cmn_dict::Dict{Tuple{Int64, Int64}, Float64}, precomputed_log_binomial_coefficientsn::Dict{Tuple{Int64, Int64}, Float64}) where {Ty<:Number}

    for m in keys(active_Λ)
        WF_prediction_for_one_m_precomputedn!(full_Λ_with_weights, m, sα, t, log_νn_dict, log_Cmn_dict, precomputed_log_binomial_coefficientsn; wm = active_Λ[m])
    end

end

function cmp_next_filtering_distribution_precomputedn!(full_Λ_with_weights::Dict{Array{Int64,1},Float64}, active_Λ::Dict{Array{Int64,1},Float64}, current_time, next_time, α, sα, next_y, log_νn_dict::Dict{Tuple{Int64, Int64}, Float64}, log_Cmn_dict::Dict{Tuple{Int64, Int64}, Float64}, precomputed_log_binomial_coefficientsn::Dict{Tuple{Int64, Int64}, Float64})
    predict_WF_params_precomputedn!(full_Λ_with_weights, sα, active_Λ, next_time-current_time, log_νn_dict, log_Cmn_dict, precomputed_log_binomial_coefficientsn)
    active_Λ = get_active_components(full_Λ_with_weights)
    update_WF_params!(full_Λ_with_weights, active_Λ, α, next_y)
end

function filter_WF_precomputed_preallocaten(α, data, log_νn_dict::Dict{Tuple{Int64, Int64}, Float64}, log_Cmn_dict::Dict{Tuple{Int64, Int64}, Float64}, precomputed_log_binomial_coefficientsn::Dict{Tuple{Int64, Int64}, Float64})
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
        cmp_next_filtering_distribution_precomputedn!(full_Λ_with_weights, active_Λ_dict, times[k], times[k+1], α, sα, data[times[k+1]], log_νn_dict, log_Cmn_dict, precomputed_log_binomial_coefficientsn)
        active_Λ_dict = get_active_components(full_Λ_with_weights)
        ks = keys(active_Λ_dict) |> collect
        Λ_of_t[times[k+1]] = ks
        wms_of_t[times[k+1]] = [active_Λ_dict[k] for k in ks]
    end

    return Λ_of_t, wms_of_t

end
