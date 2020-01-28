using DualOptimalFiltering_proof, ExactWrightFisher
using StatsFuns, SpecialFunctions, Base.Iterators

# function λm(sm::Int64, sα::Number)
#     return sm * (sm + sα - 1)/2
# end
#
# function logλm(sm::Int64, sα::Number)
#     return log(sm) + log(sm + sα - 1) - log(2)
# end

function logfirst_term_pmmi_no_alloc(si::Int64, sm::Int64, sα::Number)
    ##### Equivalent to
    # logλm.((sm-si+1):sm, sα) |> sum
    # but the former version does one allocation
    res = 0
    for s in (sm-si+1):sm
        res += logλm.(s, sα)
    end
    return res
end

# function log_denominator_Cmmi_nosign(si::Int64, k::Int64, sm::Int64, sα::Number)
#     if k==0
#         return SpecialFunctions.logfactorial(si) - si*log(2) + log_descending_fact_no0(2*sm + sα - 2, si)
#     elseif k==si
#         return SpecialFunctions.logfactorial(si) - si*log(2) + log_descending_fact_no0(2*sm + sα - si-1, si)
#     else
#         return -1.0 .* si * log(2) + SpecialFunctions.logfactorial(k) + SpecialFunctions.logfactorial(si-k) + log_descending_fact_no0(2*sm + sα - 2*k - 2, si-k) + log_descending_fact_no0(2*sm + sα - k - 1, k)
#     end
# end
# take(cycle([1,-1]), 11)
function logCmmi_arb(sm::Int64, si::Int64, t::Number, sα::Number)::Float64
    # tmp = [-λm(sm-k, sα)*t - log_denominator_Cmmi_nosign(si, k, sm, sα) for k in 0:si]
    # max_tmp = maximum(tmp)
    return ExactWrightFisher.signed_logsumexp_arb((-λm(sm-k, sα)*t - log_denominator_Cmmi_nosign(si, k, sm, sα) for k in 0:si), (sign_denominator_Cmmi(k) for k in 0:si))[2]
    # return ExactWrightFisher.signed_logsumexp_arb((-λm(sm-k, sα)*t - log_denominator_Cmmi_nosign(si, k, sm, sα) for k in 0:si), take(cycle([1,-1]), si+1))[2]
    # return ExactWrightFisher.signed_logsumexp_arb(-λm.(sm .- (0:si), sα) .* t - log_denominator_Cmmi_nosign.(si, 0:si, sm, sα) , sign_denominator_Cmmi.(0:si) )[2]
end

function logCmmi(sm::Int64, si::Int64, t::Number, sα::Number)
    # tmp = [-λm(sm-k, sα)*t - log_denominator_Cmmi_nosign(si, k, sm, sα) for k in 0:si]
    # max_tmp = maximum(tmp)
    return ExactWrightFisher.signed_logsumexp(-λm.(sm .- (0:si), sα) .* t - log_denominator_Cmmi_nosign.(si, 0:si, sm, sα) , sign_denominator_Cmmi.(0:si) )
end

function precompute_next_terms!(last_sm_max::Integer, new_sm_max::Integer, log_ν_dict::Dict{Tuple{Int64,Int64},Float64}, log_Cmmi_dict::Dict{Tuple{Int64,Int64},Float64}, log_binomial_coeff_dict::Dict{Tuple{Int64,Int64},Float64}, sα, Δt)
    if last_sm_max == 0
        log_binomial_coeff_dict[(0,0)] = 0
    end
    for sm in (last_sm_max+1):new_sm_max
        for si in 1:sm
            log_ν_dict[(sm, si)] = logfirst_term_pmmi_no_alloc(si, sm, sα)
            log_Cmmi_dict[(sm, si)] = logCmmi_arb(sm, si, Δt, sα)
            log_binomial_coeff_dict[(sm,si)] = log_binomial_safe_but_slow(sm, si)
        end
        for si in 0:sm
            log_binomial_coeff_dict[(sm,si)] = log_binomial_safe_but_slow(sm, si)
        end
    end
end

function precompute_next_terms_ar!(last_sm_max::Integer, new_sm_max::Integer, log_ν_ar::Array{Float64,2}, log_Cmmi_ar::Array{Float64,2}, log_binomial_coeff_ar_offset::Array{Float64,2}, sα, Δt)
    if last_sm_max == 0
        log_binomial_coeff_ar_offset[1,1] = 0
    end
    for sm in (last_sm_max+1):new_sm_max
        for si in 1:sm
            log_ν_ar[sm, si] = logfirst_term_pmmi_no_alloc(si, sm, sα)
            log_Cmmi_ar[sm, si] = logCmmi_arb(sm, si, Δt, sα)
            log_binomial_coeff_ar_offset[sm+1,si+1] = log_binomial_safe_but_slow(sm, si)
        end
        for si in 0:sm
            log_binomial_coeff_ar_offset[sm+1,si+1] = log_binomial_safe_but_slow(sm, si)
        end
    end
end


function filter_WF_adaptive_precomputation(α, data, do_the_pruning::Function; silence = false)
    # println("filter_WF_mem2")

    @assert length(α) == length(data[collect(keys(data))[1]])
    Δt = assert_constant_time_step_and_compute_it(data)


    log_ν_dict = Dict{Tuple{Int64, Int64}, Float64}()
    log_Cmmi_dict = Dict{Tuple{Int64, Int64}, Float64}()
    log_binomial_coeff_dict = Dict{Tuple{Int64, Int64}, Float64}()

    sα = sum(α)
    times = keys(data) |> collect |> sort
    Λ_of_t = Dict()
    wms_of_t = Dict()

    filtered_Λ, filtered_wms = update_WF_params([1.], α, [repeat([0], inner = length(α))], data[times[1]])
    Λ_pruned, wms_pruned = do_the_pruning(filtered_Λ, filtered_wms)

    Λ_of_t[times[1]] = filtered_Λ
    wms_of_t[times[1]] = filtered_wms
    new_sm_max = maximum(sum.(Λ_pruned))
    precompute_next_terms!(0, new_sm_max, log_ν_dict, log_Cmmi_dict, log_binomial_coeff_dict, sα, Δt)
    sm_max_so_far = new_sm_max

    for k in 1:(length(times)-1)
        if (!silence)
            println("Step index: $k")
            println("Number of components: $(length(filtered_Λ))")
        end
        last_sm_max = maximum(sum.(Λ_pruned))
        new_sm_max = last_sm_max + sum(data[times[k+1]])

        if sm_max_so_far < new_sm_max
            precompute_next_terms!(sm_max_so_far, new_sm_max, log_ν_dict, log_Cmmi_dict, log_binomial_coeff_dict, sα, Δt)
            sm_max_so_far = max(sm_max_so_far,new_sm_max)
        end

        filtered_Λ, filtered_wms = get_next_filtering_distribution_precomputed(Λ_pruned, wms_pruned, times[k], times[k+1], α, sα, data[times[k+1]], log_ν_dict, log_Cmmi_dict, log_binomial_coeff_dict)
        Λ_pruned, wms_pruned = do_the_pruning(filtered_Λ, filtered_wms)
        Λ_of_t[times[k+1]] = filtered_Λ
        wms_of_t[times[k+1]] = filtered_wms
    end

    return Λ_of_t, wms_of_t

end

function filter_WF_adaptive_precomputation_ar(α, data, do_the_pruning::Function; silence = false, return_precomputed_terms = false)
    # println("filter_WF_mem2")

    @assert length(α) == length(data[collect(keys(data))[1]])
    Δt = assert_constant_time_step_and_compute_it(data)

    smmax = values(data) |> sum |> sum
    log_ν_ar = Array{Float64}(undef, smmax, smmax)
    log_Cmmi_ar = Array{Float64}(undef, smmax, smmax)
    log_binomial_coeff_ar_offset = Array{Float64}(undef,    smmax+1, smmax+1)

    sα = sum(α)
    times = keys(data) |> collect |> sort
    Λ_of_t = Dict()
    wms_of_t = Dict()

    filtered_Λ, filtered_wms = update_WF_params([1.], α, [repeat([0], inner = length(α))], data[times[1]])
    Λ_pruned, wms_pruned = do_the_pruning(filtered_Λ, filtered_wms)

    Λ_of_t[times[1]] = filtered_Λ
    wms_of_t[times[1]] = filtered_wms
    new_sm_max = maximum(sum.(Λ_pruned))
    precompute_next_terms_ar!(0, new_sm_max, log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset, sα, Δt)
    sm_max_so_far = new_sm_max

    for k in 1:(length(times)-1)
        if (!silence)
            println("Step index: $k")
            println("Number of components: $(length(filtered_Λ))")
        end
        last_sm_max = maximum(sum.(Λ_pruned))
        new_sm_max = last_sm_max + sum(data[times[k+1]])

        if sm_max_so_far < new_sm_max
            precompute_next_terms_ar!(sm_max_so_far, new_sm_max, log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset, sα, Δt)
            sm_max_so_far = max(sm_max_so_far,new_sm_max)
        end

        filtered_Λ, filtered_wms = get_next_filtering_distribution_precomputed(Λ_pruned, wms_pruned, times[k], times[k+1], α, sα, data[times[k+1]], log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset)
        Λ_pruned, wms_pruned = do_the_pruning(filtered_Λ, filtered_wms)
        Λ_of_t[times[k+1]] = filtered_Λ
        wms_of_t[times[k+1]] = filtered_wms
    end

    if return_precomputed_terms
        return Λ_of_t, wms_of_t, log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset, sm_max_so_far
    else
        return Λ_of_t, wms_of_t
    end

end

# using BenchmarkTools
# @btime logλm(10, 3.2)
# @btime logfirst_term_pmmi_no_alloc(10, 12, 3.2) # 0 allocation
# @btime log_denominator_Cmmi_nosign(10, 3, 12, 3.2) # 0 allocation
# @btime logCmmi_arb(10, 3, 12, 3.2) # 46 allocations
# @btime logCmmi(10, 3, 12, 3.2)
# @btime log_binomial_safe_but_slow(20, 6)
# @btime loghypergeom_pdf([2, 2, 2, 0], [10, 5, 3, 2], 6, 20)
