using IterTools, DataStructures
using Memoize

function precompute_log_first_term(data::Dict{Float64,Array{Int64,2}}, sα::Number)
    smmax = values(data) |> sum |> sum
    log_ν_dict = Dict{Tuple{Int64,Int64}, Float64}()
    for sm in 1:smmax
        for si in 1:sm
            log_ν_dict[(sm,si)] = logfirst_term_pmmi(si, sm, sα)
        end
    end
    return log_ν_dict
end

function precompute_log_Cmmi(data::Dict{Float64,Array{Int64,2}}, sα::Number; digits_after_comma_for_time_precision = 4, override = false)
    smmax = values(data) |> sum |> sum
#     𝛿ts = keys(data) |> collect |> sort |> diff
    𝛿ts = keys(data) |> collect |> sort |> diff |> x -> truncate_float.(x, 4) |> unique

    if !override&&(length(𝛿ts)>1)
        error("the time intervals are not constant, it may not be optimal to pre-compute all the Cmmi")
    end
    log_Cmmi_mem_dict = Dict{Tuple{Int64,Int64}, Float64}()
    for sm in 1:smmax
        for si in 1:sm
            log_Cmmi_mem_dict[(sm, si)] = logCmmi_overflow_safe(sm, si, 𝛿ts[1], sα)
        end
    end
    return log_Cmmi_mem_dict
end

function precompute_log_binomial_coefficients(data::Dict{Float64,Array{Int64,2}})
    smmax = values(data) |> sum |> sum
    log_binomial_coeff_dict = Dict{Tuple{Int64,Int64}, Float64}()
    for sm in 0:smmax
        for si in 0:sm
            log_binomial_coeff_dict[(sm,si)] = log_binomial_safe_but_slow(sm, si)
        end
    end
    return log_binomial_coeff_dict
end

function precompute_terms(data::Dict{Float64,Array{Int64,2}}, sα::Number; digits_after_comma_for_time_precision = 4, override = false)

    if !override&&(data |> keys |> collect |> sort |> diff |> x -> truncate_float.(x,digits_after_comma_for_time_precision) |> unique |> length > 1)
        println(data |> keys |> collect |> sort |> diff |> x -> truncate_float.(x,digits_after_comma_for_time_precision) |> unique)
        error("Think twice about precomputing all terms, as the time intervals are not equal. You can go ahead using the option 'override = true.'")
    end

    println("Precomputing 3 times")
    @printf "%e" values(data) |> sum |> sum |> n -> n*(n-1)/2 |> BigFloat
    println(" terms")

    log_Cmmi_dict = precompute_log_Cmmi(data, sα; digits_after_comma_for_time_precision = digits_after_comma_for_time_precision, override = override)
    precomputed_log_binomial_coefficients = precompute_log_binomial_coefficients(data)
    log_ν_dict = precompute_log_first_term(data, sα)

    return log_ν_dict, log_Cmmi_dict, precomputed_log_binomial_coefficients
end

function loghypergeom_pdf_using_precomputed(i::Array{Int64,1}, m::Array{Int64,1}, si::Int64, sm::Int64, precomputed_log_binomial_coefficients::Dict{Tuple{Int64, Int64}, Float64})
    return sum(precomputed_log_binomial_coefficients[(m[k],i[k])] for k in 1:length(m)) - precomputed_log_binomial_coefficients[(sm, si)]
end

function logpmmi_raw_precomputed(i::Array{Int64,1}, m::Array{Int64,1}, sm::Int64, si::Int64, t::Number, log_ν_dict::Dict{Tuple{Int64, Int64}, Float64}, log_Cmmi_dict::Dict{Tuple{Int64, Int64}, Float64}, precomputed_log_binomial_coefficients::Dict{Tuple{Int64, Int64}, Float64})
    return log_ν_dict[(sm, si)] + log_Cmmi_dict[(sm, si)]  + loghypergeom_pdf_using_precomputed(i, m, si, sm, precomputed_log_binomial_coefficients)
end

function logpmmi_precomputed(i::Array{Int64,1}, m::Array{Int64,1}, sm::Int64, si::Int64, t::Number, sα::Number, log_ν_dict::Dict{Tuple{Int64, Int64}, Float64}, log_Cmmi_dict::Dict{Tuple{Int64, Int64}, Float64}, precomputed_log_binomial_coefficients::Dict{Tuple{Int64, Int64}, Float64})
    if maximum(i)==0
        return -λm(sm, sα)*t
    else
        return logpmmi_raw_precomputed(i, m, sm, si, t, log_ν_dict, log_Cmmi_dict, precomputed_log_binomial_coefficients)
    end
end

function WF_prediction_for_one_m_precomputed(m::Array{Int64,1}, sα::Ty, t::Ty, log_ν_dict::Dict{Tuple{Int64, Int64}, Float64}, log_Cmmi_dict::Dict{Tuple{Int64, Int64}, Float64}, precomputed_log_binomial_coefficients::Dict{Tuple{Int64, Int64}, Float64}; wm = 1) where {Ty<:Number}
    gm = map(x -> 0:x, m) |> vec |> x -> product(x...)

    function fun_n(n)
        i = m.-n
        si = sum(i)
        sm = sum(m)
        return wm*(logpmmi_precomputed(i, m, sm, si, t, sα, log_ν_dict, log_Cmmi_dict, precomputed_log_binomial_coefficients) |> exp)
    end

    Dict( collect(n) => fun_n(n) for n in gm ) |> Accumulator

end

function predict_WF_params_precomputed(wms::Array{Ty,1}, sα::Ty, Λ::Array{Array{Int64,1},1}, t::Ty, log_ν_dict::Dict{Tuple{Int64, Int64}, Float64}, log_Cmmi_dict::Dict{Tuple{Int64, Int64}, Float64}, precomputed_log_binomial_coefficients::Dict{Tuple{Int64, Int64}, Float64}; wm = 1) where {Ty<:Number}

    res = Accumulator(Array{Int64,1}, Float64)

    for k in 1:length(Λ)
        res = merge(res, WF_prediction_for_one_m_precomputed(Λ[k], sα, t, log_ν_dict, log_Cmmi_dict, precomputed_log_binomial_coefficients; wm = wms[k]))
    end

    ks = keys(res) |> collect

    return ks, [res[k] for k in ks]

end

function get_next_filtering_distribution_precomputed(current_Λ, current_wms, current_time, next_time, α, sα, next_y, log_ν_dict::Dict{Tuple{Int64, Int64}, Float64}, log_Cmmi_dict::Dict{Tuple{Int64, Int64}, Float64}, precomputed_log_binomial_coefficients::Dict{Tuple{Int64, Int64}, Float64})
    predicted_Λ, predicted_wms = predict_WF_params_precomputed(current_wms, sα, current_Λ, next_time-current_time, log_ν_dict, log_Cmmi_dict, precomputed_log_binomial_coefficients)
    filtered_Λ, filtered_wms = update_WF_params(predicted_wms, α, predicted_Λ, next_y)

    return filtered_Λ, filtered_wms
end

function filter_WF_precomputed(α, data, log_ν_dict::Dict{Tuple{Int64, Int64}, Float64}, log_Cmmi_dict::Dict{Tuple{Int64, Int64}, Float64}, precomputed_log_binomial_coefficients::Dict{Tuple{Int64, Int64}, Float64}; silence = false)
    # println("filter_WF_mem2")

    @assert length(α) == length(data[collect(keys(data))[1]])


    sα = sum(α)
    times = keys(data) |> collect |> sort
    Λ_of_t = Dict()
    wms_of_t = Dict()

    filtered_Λ, filtered_wms = update_WF_params([1.], α, [repeat([0], inner = length(α))], data[times[1]])

    Λ_of_t[times[1]] = filtered_Λ
    wms_of_t[times[1]] = filtered_wms

    for k in 1:(length(times)-1)
        if (!silence)
            println("Step index: $k")
            println("Number of components: $(length(filtered_Λ))")
        end
        filtered_Λ, filtered_wms = get_next_filtering_distribution_precomputed(filtered_Λ, filtered_wms, times[k], times[k+1], α, sα, data[times[k+1]], log_ν_dict, log_Cmmi_dict, precomputed_log_binomial_coefficients)
        mask = filtered_wms .!= 0.
        filtered_Λ = filtered_Λ[mask]
        filtered_wms = filtered_wms[mask]
        Λ_of_t[times[k+1]] = filtered_Λ
        wms_of_t[times[k+1]] = filtered_wms
    end

    return Λ_of_t, wms_of_t

end

function get_log_dict(dict)
    k_dict = keys(dict)
    log_dict = Dict(zip(k_dict, Float64.(log.([DualOptimalFiltering.RR(dict[k]) for k in k_dict]) )))
    if any(isnan.(log_dict |> values |> collect))
        error("Log transformation of the dict returned at least one NaN, please check the precision of the values of the dict")
    else
        return log_dict
    end
end

function precompute_log_terms_arb(data::Dict{Float64,Array{Int64,2}}, sα::Number; digits_after_comma_for_time_precision = 4, override = false)
    ν_dict, Cmmi_dict, precomputed_binomial_coefficients = DualOptimalFiltering.precompute_terms_arb(data, sα; digits_after_comma_for_time_precision = digits_after_comma_for_time_precision, override = override)
    return get_log_dict(ν_dict), get_log_dict(Cmmi_dict), get_log_dict(precomputed_binomial_coefficients)
end

function get_predictive_mixture_at_time(t, Λ_of_t, wms_of_t, sα, log_ν_dict::Dict{Tuple{Int64, Int64}, Float64}, log_Cmmi_dict::Dict{Tuple{Int64, Int64}, Float64}, precomputed_log_binomial_coefficients::Dict{Tuple{Int64, Int64}, Float64})
    observation_times = Λ_of_t |> keys |> collect |> sort
    closest_observation_time = maximum(observation_times[observation_times.<t])
    return predict_WF_params_precomputed(wms_of_t[closest_observation_time], sα, Λ_of_t[closest_observation_time], t-closest_observation_time, log_ν_dict, log_Cmmi_dict, precomputed_log_binomial_coefficients)
end



#
#     In python hash(i)=i so it is very cheap but avoids collisions only for the case you have specified. In Julia hashing is more involved
#
# You can easily use hash(i)=i in Julia as well by defining a wrapper type for the keys, and when I do so I find that performance in the original benchmark is increased by almost a factor of 5, which makes it almost 3x faster than Python for me (in Julia 0.6):
#
# julia> struct FastHashInt{T<:Integer}; i::T; end
#
# julia> Base.:(==)(x::FastHashInt, y::FastHashInt) = x.i == y.i
#        Base.hash(x::FastHashInt, h::UInt) = xor(UInt(x.i), h)
#
# julia> function dict_perf()
#            n = 10^7
#            x = Dict{Int,Int}()
#            sizehint!(x, n)
#            for i = 1:n
#                x[i] = i
#            end
#            return x
#        end
# dict_perf (generic function with 1 method)
#
# julia> @time dict_perf(); @time dict_perf(); @time dict_perf();
#   1.754449 seconds (1.48 k allocations: 272.081 MiB, 4.54% gc time)
#   1.756465 seconds (11 allocations: 272.001 MiB, 4.38% gc time)
#   1.715037 seconds (11 allocations: 272.001 MiB, 1.10% gc time)
#
# julia> function dict_perf2()
#            n = 10^7
#            x = Dict{FastHashInt{Int},Int}()
#            sizehint!(x, n)
#            for i = 1:n
#                x[FastHashInt(i)] = i
#            end
#            return x
#        end
# dict_perf2 (generic function with 1 method)
#
# julia> @time dict_perf2(); @time dict_perf2(); @time dict_perf2();
#   0.376183 seconds (1.37 k allocations: 272.073 MiB, 5.45% gc time)
#   0.355044 seconds (11 allocations: 272.001 MiB, 3.26% gc time)
#   0.350325 seconds (11 allocations: 272.001 MiB, 2.91% gc time)
#
# The Python 2 version of this takes 1 second on my machine:
#
# In [1]: def dict_performance():
#     dic = dict()
#     for i in xrange(10000000):
#         dic[i] = i
#
# In [2]: %time dict_performance();
# CPU times: user 873 ms, sys: 188 ms, total: 1.06 s
# Wall time: 1.06 s
#
# (1.13 seconds in Python 3.) Update: In Julia 0.7, I get an additional factor of 2 speedup (0.18sec for dict_perf2).
#
# A moral of this story is that in cases where dictionary performance is critical, you might want a custom hashing function optimized for your use-case. The good news is that this is possible in pure Julia code.
# source: https://discourse.julialang.org/t/poor-time-performance-on-dict/9656/13
