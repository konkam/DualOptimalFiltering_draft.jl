using IterTools, DataStructures, Memoize, SpecialFunctions

function create_dirichlet_mixture(α::Array{T, 1}, Λ::Array{Array{U,1},1}) where {T <: Real, U <:Integer}
    α_mixt = Array{Array{T,1},1}(undef, length(Λ))
    for i in 1:length(Λ)
        α_mixt[i] = α .+ Λ[i]
    end
    return α_mixt
end

function update_WF_params_debug(wms::Array{Ty,1}, α::Array{Ty,1}, Λ::Array{Array{Int64,1},1}, y::Array{Int64,2}; debug = true) where Ty<:Number
    #y is a matrix of dimension J*K, with K the dimension of the process
    # and J the number of observations
    # Julia is in row major, so the first index indicates the row (index of observation)
    # and the second the column (index of the dimension) (as in matrix mathematical notation)
    @assert length(wms) == size(Λ, 1)

    nJ = sum(y, dims = 2) |> vec#sum_j=1^K n_ij
    nK = sum(y, dims = 1) |> vec#sum_i=1^J n_ij
    sy = sum(y)
    J = size(y,1)
    sα = sum(α)


    first_term = sum(lfactorial.(nJ) - sum(lfactorial.(y), dims = 2))


    function lpga(m::Array{Int64,1})
        sm = sum(m)
        second_term = SpecialFunctions.lgamma(sα + sm)
        third_term = sum(SpecialFunctions.lgamma.(α + m + nK))
        fourth_term = -SpecialFunctions.lgamma(sα + sm + sy)
        fifth_term = -sum(SpecialFunctions.lgamma.(α + m))
        return first_term + second_term + third_term + fourth_term + fifth_term
    end

    filter_ = wms .== 0

    lwms_hat = log.(wms) .+ map(lpga, Λ)

    wms_hat = exp.(lwms_hat)
    wms_hat[filter_] .= 0
    wms_hat = wms_hat |> normalise

    if debug&&any(isnan, wms_hat)
        println("NAs in update step")
        println("wms")
        println(wms)
        println("pga")
        println(map(pga, Λ))
        println("wms_hat")
        println(wms_hat)
    end

    return [m .+ nK for m in Λ], wms_hat
end

function update_WF_params(wms::Array{Ty,1}, α::Array{Ty,1}, Λ::Array{Array{Int64,1},1}, y::Array{Int64,2}) where {Ty<:Number}
    update_WF_params_debug(wms, α, Λ, y; debug = false)
end

function λm(sm::Int64, sα::Number)
    return sm * (sm + sα - 1)/2
end

@memoize function λm_mem(sm::Int64, sα::Number)
    return λm(sm::Int64, sα::Number)
end

function logλm(sm::Int64, sα::Number)
    return log(sm) + log(sm + sα - 1) - log(2)
end

@memoize function logλm_mem(sm::Int64, sα::Number)
    return logλm(sm::Int64, sα::Number)
end

function logfirst_term_pmmi(si::Int64, sm::Int64, sα::Number)
    return logλm.((sm-si+1):sm, sα) |> sum
end

function denominator_Cmmi(si::Int64, k::Int64, sm::Int64, sα::Number)
    #already checked that it works for k = 0 and k = si
    tuples_to_compute = Iterators.product(k, Iterators.flatten((0:(k-1), (k+1):si)))#all the k, h pairs involved
    return prod(λm(sm-t[1], sα) - λm(sm-t[2], sα) for t in tuples_to_compute)
end

@memoize function denominator_Cmmi_mem(si::Int64, k::Int64, sm::Int64, sα::Number)
    return denominator_Cmmi(si::Int64, k::Int64, sm::Int64, sα::Number)
end

function Cmmi_mem(sm::Int64, si::Int64, t::Number, sα::Number)
    return (-1)^si * sum( exp(-λm_mem(sm-k, sα)*t) / denominator_Cmmi_mem(si, k, sm, sα) for k in 0:si)
end

@memoize logCmmi_mem(sm::Int64, si::Int64, t::Number, sα::Number) = Cmmi_mem(sm, si, t, sα) |> log


function log_denominator_Cmmi_nosign(si::Int64, k::Int64, sm::Int64, sα::Number)
    if k==0
        return lfactorial(si) - si*log(2) + log_descending_fact_no0(2*sm + sα - 2, si)
    elseif k==si
        return lfactorial(si) - si*log(2) + log_descending_fact_no0(2*sm + sα - si-1, si)
    else
        return -1.0 .* si * log(2) + lfactorial(k) + lfactorial(si-k) + log_descending_fact_no0(2*sm + sα - 2*k - 2, si-k) + log_descending_fact_no0(2*sm + sα - k - 1, k)
    end
end

function sign_denominator_Cmmi(k::Int64)
    if iseven(k)
        return 1
    else
        return -1
    end
end

sign_term_Cmmi(si::Int64) = sign_denominator_Cmmi(si)

function logCmmi_overflow_safe(sm::Int64, si::Int64, t::Number, sα::Number)
    tmp = [-λm(sm-k, sα)*t - log_denominator_Cmmi_nosign(si, k, sm, sα) for k in 0:si]
    max_tmp = maximum(tmp)
    return sign_term_Cmmi(si)*sum(sign_denominator_Cmmi(k) * exp.(tmp[k+1] - max_tmp) for k in 0:si) |> log |> x -> x + maximum(tmp)
end

@memoize function logCmmi_mem_overflow_safe(sm::Int64, si::Int64, t::Number, sα::Number)
    return logCmmi_overflow_safe(sm::Int64, si::Int64, t::Number, sα::Number)
end

function logpmmi_raw_mem2(i::Array{Int64,1}, m::Array{Int64,1}, sm::Int64, si::Int64, t::Number, sα::Number)
#     return logfirst_term_pmmi_mem(si, sm, sα) + logCmmi_mem(sm, si, t, sα) + loghypergeom_pdf_mem(i, m, si, sm)
    return logfirst_term_pmmi(si, sm, sα) + logCmmi_mem_overflow_safe(sm, si, t, sα) + loghypergeom_pdf_mem(i, m, si, sm)
end

function logpmmi_mem2(i::Array{Int64,1}, m::Array{Int64,1}, sm::Int64, si::Int64, t::Number, sα::Number)
    if maximum(i)==0 #probably faster to replace by testing all is == 0
#         return -λm_mem(sm::Int64, sα::Number)*t
        return -λm(sm::Int64, sα::Number)*t
    else
        return logpmmi_raw_mem2(i, m, sm, si, t, sα)
    end
end

function Λ_from_Λ_max(Λ_max) where U <: Integer
    # return Base.Iterators.product((0:Λi_max for Λi_max in Λ_max)...)
    return Base.Iterators.product(map(N -> 0:N, Λ_max)...)
end

indices_of_tree_below(m::Union{AbstractArray{U, 1}, Tuple}) where U <: Integer =  Λ_from_Λ_max(m)

function WF_prediction_for_one_m_debug_mem2(m::Array{Int64,1}, sα::Ty, t::Ty; wm = 1, debug = true) where {Ty<:Number}
    # gm = map(x -> 0:x, m) |> vec |> x -> Iterators.product(x...)
    gm = indices_of_tree_below(m)

    function fun_n(n)
        i = m.-n
        si = sum(i)
        sm = sum(m)
        return wm*(logpmmi_mem2(i::Array{Int64,1}, m::Array{Int64,1}, sm::Int64, si::Int64, t::Number, sα::Number) |> exp)
    end

    Dict( collect(n) => fun_n(n) for n in gm ) |> Accumulator

end

function predict_WF_params_debug_mem2(wms::Array{Ty,1}, sα::Ty, Λ::Array{Array{Int64,1},1}, t::Ty; debug = true) where {Ty<:Number}

    res = Accumulator{Array{Int64,1}, Float64}()

    for k in 1:length(Λ)
        res = merge(res, WF_prediction_for_one_m_debug_mem2(Λ[k], sα, t; wm = wms[k], debug = false))
    end

    ks = keys(res) |> collect

    return ks, [res[k] for k in ks]

end

function get_next_filtering_distribution_mem2(current_Λ, current_wms, current_time, next_time, α, sα, next_y)
    predicted_Λ, predicted_wms = predict_WF_params_debug_mem2(current_wms, sα, current_Λ, next_time-current_time)
    filtered_Λ, filtered_wms = update_WF_params(predicted_wms, α, predicted_Λ, next_y)

    return filtered_Λ, filtered_wms
end

function filter_WF_mem2(α, data)
    # println("filter_WF_mem2")

    @assert length(α) == length(data[collect(keys(data))[1]])


    sα = sum(α)
    times = keys(data) |> collect |> sort
    Λ_of_t = Dict()
    wms_of_t = Dict()

# update_WF_params(wms::Array{Ty,1}, α::Array{Ty,1}, Λ::Array{Array{Int64,1},1}, y::Array{Int64,2}) where {Ty<:Number}
    filtered_Λ, filtered_wms = update_WF_params([1.], α, [repeat([0], inner = length(α))], data[times[1]])

    Λ_of_t[times[1]] = filtered_Λ
    wms_of_t[times[1]] = filtered_wms

    for k in 1:(length(times)-1)
        println("Step index: $k")
        println("Number of components: $(length(filtered_Λ))")
        filtered_Λ, filtered_wms = get_next_filtering_distribution_mem2(filtered_Λ, filtered_wms, times[k], times[k+1], α, sα, data[times[k+1]])
        mask = filtered_wms .!= 0.
        filtered_Λ = filtered_Λ[mask]
        filtered_wms = filtered_wms[mask]
        Λ_of_t[times[k+1]] = filtered_Λ
        wms_of_t[times[k+1]] = filtered_wms
    end

    return Λ_of_t, wms_of_t

end

filter_WF(α, data) = filter_WF_mem2(α, data)


function maximum_number_of_components_WF(data)
    return data |> values |> sum |> x -> prod(x .+ 1)
end
