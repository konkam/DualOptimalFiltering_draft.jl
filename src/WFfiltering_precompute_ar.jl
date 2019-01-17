function precompute_log_first_term_ar(data::Dict{Float64,Array{Int64,2}}, sα::Number)
    smmax = values(data) |> sum |> sum
    log_ν_ar = Array{Float64}(undef, smmax, smmax)
    for sm in 1:smmax
        for si in 1:sm
            log_ν_ar[sm,si] = logfirst_term_pmmi(si, sm, sα)
        end
    end
    return log_ν_ar
end

function precompute_log_Cmmi_ar(data::Dict{Float64,Array{Int64,2}}, sα::Number; digits_after_comma_for_time_precision = 4, override = false)
    smmax = values(data) |> sum |> sum
#     𝛿ts = keys(data) |> collect |> sort |> diff
    𝛿ts = keys(data) |> collect |> sort |> diff |> x -> truncate_float.(x, 4) |> unique

    if !override&&(length(𝛿ts)>1)
        error("the time intervals are not constant, it may not be optimal to pre-compute all the Cmmi")
    end
    log_Cmmi_mem_ar = Array{Float64}(undef, smmax, smmax)
    for sm in 1:smmax
        for si in 1:sm
            log_Cmmi_mem_ar[sm, si] = logCmmi_overflow_safe(sm, si, 𝛿ts[1], sα)
        end
    end
    return log_Cmmi_mem_ar
end

function precompute_log_binomial_coefficients_ar(data::Dict{Float64,Array{Int64,2}})
    smmax = values(data) |> sum |> sum
    log_binomial_coeff_ar_offset = Array{Float64}(undef, smmax+1, smmax+1)
    for sm in 0:smmax
        for si in 0:sm
            log_binomial_coeff_ar_offset[sm+1,si+1] = log_binomial_safe_but_slow(sm, si)
        end
    end
    return log_binomial_coeff_ar_offset
end

function precompute_terms_ar(data::Dict{Float64,Array{Int64,2}}, sα::Number; digits_after_comma_for_time_precision = 4, override = false)

    test_equal_spacing_of_observations(data, override = override, digits_after_comma_for_time_precision = 4)

    println("Precomputing 3 times")
    @printf "%e" values(data) |> sum |> sum |> n -> n*(n-1)/2 |> BigFloat
    println(" terms")

    log_Cmmi_ar = precompute_log_Cmmi_ar(data, sα; digits_after_comma_for_time_precision = digits_after_comma_for_time_precision, override = override)
    log_binomial_coeff_ar_offset = precompute_log_binomial_coefficients_ar(data)
    log_ν_ar = precompute_log_first_term_ar(data, sα)

    return log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset
end

function loghypergeom_pdf_using_precomputed(i, m, si::Integer, sm::Integer, log_binomial_coeff_ar_offset::Array{Float64,2})
    return sum(log_binomial_coeff_ar_offset[m[k]+1,i[k]+1] for k in 1:length(m)) - log_binomial_coeff_ar_offset[sm+1, si+1]
end

function logpmmi_raw_precomputed(i, m, sm::Integer, si::Integer, t::Number, log_ν_ar::Array{Float64,2}, log_Cmmi_ar::Array{Float64,2}, log_binomial_coeff_ar_offset::Array{Float64,2})
    #Would return an error when called on sm = 0, but this should never occur.
    return log_ν_ar[sm, si] + log_Cmmi_ar[sm, si]  + loghypergeom_pdf_using_precomputed(i, m, si, sm, log_binomial_coeff_ar_offset)
end
