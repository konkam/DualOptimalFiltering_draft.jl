using Nemo

ZZ = FlintZZ
nbits = 256*4 # precision needed to be augmented because with a time step of 0.004, gave 0
RR = RealField(nbits)
R(x::Number) = RR(x)
R(x::Vector{T}) where T <: Number = RR.(x)

gamma_arb(x) = Nemo.gamma(RR(x))

function normalise_arb(x)
    return x .* [1 / sum(x)]
end

function λm_arb(sm::Int64, sα::Number)
    return RR(sm * (sm + sα - 1)/2)
end


function first_term_pmmi_arb(si::Int64, sm::Int64, sα::Number)
    # return λm_arb.((sm-si+1):sm, sα) |> prod
    return prod(λm_arb(s, sα) for s in (sm-si+1):sm)
end


function precompute_first_term_arb(data::Dict{Float64,Array{Int64,2}}, sα::Number)
    smmax = values(data) |> sum |> sum
    ν_dict = Dict{Tuple{Int64,Int64}, Nemo.arb}()
    for sm in 1:smmax
        for si in 1:sm
            ν_dict[(sm,si)] = first_term_pmmi_arb(si, sm, sα)
        end
    end
    return ν_dict
end

function denominator_Cmmi_arb(si::Int64, k::Int64, sm::Int64, sα::Number)
    #already checked that it works for k = 0 and k = si
    # tuples_to_compute = product(k, chain(0:(k-1), (k+1):si))#all the k, h pairs involved
    tuples_to_compute = Iterators.product(k, Iterators.flatten((0:(k-1), (k+1):si)))#all the k, h pairs involved
    return prod(λm_arb(sm-t[1], sα) - λm_arb(sm-t[2], sα) for t in tuples_to_compute)
end


function Cmmi_arb(sm::Int64, si::Int64, t::Number, sα::Number)
    if(iseven(si))
        sgn = 1
    else
        sgn = -1
    end
    return sgn * sum( exp(-λm_arb(sm-k, sα)*t) / denominator_Cmmi_arb(si, k, sm, sα) for k in 0:si)
end

function precompute_Cmmi_arb(data::Dict{Float64,Array{Int64,2}}, sα::Number; digits_after_comma_for_time_precision = 4, override = false)
    smmax = values(data) |> sum |> sum
#     𝛿ts = keys(data) |> collect |> sort |> diff
    𝛿ts = keys(data) |> collect |> sort |> diff |> x -> truncate_float.(x, 4) |> unique

    if !override&&(length(𝛿ts)>1)
        error("the time intervals are not constant, it may not be optimal to pre-compute all the Cmmi")
    end
    Cmmi_mem_dict = Dict{Tuple{Int64,Int64}, Nemo.arb}()
    for sm in 1:smmax
        for si in 1:sm
            Cmmi_mem_dict[(sm, si)] = Cmmi_arb(sm, si, 𝛿ts[1], sα)
        end
    end
    return Cmmi_mem_dict
end

function precompute_binomial_coefficients_arb(data::Dict{Float64,Array{Int64,2}})
    smmax = values(data) |> sum |> sum
    binomial_coeff_dict = Dict{Tuple{Int64,Int64}, Nemo.fmpz}()
    for sm in 0:smmax
        for si in 0:sm
            binomial_coeff_dict[(sm,si)] = Nemo.binom(sm, si)
        end
    end
    return binomial_coeff_dict
end

function precompute_terms_arb(data::Dict{Float64,Array{Int64,2}}, sα::Number; digits_after_comma_for_time_precision = 4, override = false)

    if !override&&(data |> keys |> collect |> sort |> diff |> x -> truncate_float.(x,digits_after_comma_for_time_precision) |> unique |> length > 1)
        println(data |> keys |> collect |> sort |> diff |> x -> truncate_float.(x,digits_after_comma_for_time_precision) |> unique)
        error("Think twice about precomputing all terms, as the time intervals are not equal. You can go ahead using the option 'override = true.'")
    end

    println("Precomputing 3 times")
    @printf "%e" values(data) |> sum |> sum |> n -> n*(n-1)/2 |> BigFloat
    println(" terms")

    Cmmi_dict = precompute_Cmmi_arb(data, sα; digits_after_comma_for_time_precision = digits_after_comma_for_time_precision, override = override)
    precomputed_binomial_coefficients = precompute_binomial_coefficients_arb(data)
    ν_dict = precompute_first_term_arb(data, sα)

    return ν_dict, Cmmi_dict, precomputed_binomial_coefficients
end

function hypergeom_pdf_using_precomputed_arb(i::Array{Int64,1}, m::Array{Int64,1}, si::Int64, sm::Int64, precomputed_binomial_coefficients_arb::Dict{Tuple{Int64, Int64}, Nemo.fmpz})
    return prod(precomputed_binomial_coefficients_arb[(m[k],i[k])] for k in 1:length(m))*RR(1.)/precomputed_binomial_coefficients_arb[(sm, si)]
end

function pmmi_raw_precomputed_arb(i::Array{Int64,1}, m::Array{Int64,1}, sm::Int64, si::Int64, t::Number, ν_dict_arb::Dict{Tuple{Int64, Int64}, Nemo.arb}, Cmmi_dict_arb::Dict{Tuple{Int64, Int64}, Nemo.arb}, precomputed_binomial_coefficients_arb::Dict{Tuple{Int64, Int64}, Nemo.fmpz})
    return ν_dict_arb[(sm, si)]*Cmmi_dict_arb[(sm, si)]*hypergeom_pdf_using_precomputed_arb(i, m, si, sm, precomputed_binomial_coefficients_arb)
end

function pmmi_precomputed_arb(i::Array{Int64,1}, m::Array{Int64,1}, sm::Int64, si::Int64, t::Number, sα::Number, ν_dict_arb::Dict{Tuple{Int64, Int64}, Nemo.arb}, Cmmi_dict_arb::Dict{Tuple{Int64, Int64}, Nemo.arb}, precomputed_binomial_coefficients_arb::Dict{Tuple{Int64, Int64}, Nemo.fmpz})
    if maximum(i)==0
        return -λm_arb(sm, sα)*t
    else
        return pmmi_raw_precomputed_arb(i, m, sm, si, t, ν_dict_arb, Cmmi_dict_arb, precomputed_binomial_coefficients_arb)
    end
end

function WF_prediction_for_one_m_precomputed_arb(m::Array{Int64,1}, sα::Ty, t::Ty, ν_dict_arb::Dict{Tuple{Int64, Int64}, Nemo.arb}, Cmmi_dict_arb::Dict{Tuple{Int64, Int64}, Nemo.arb}, precomputed_binomial_coefficients_arb::Dict{Tuple{Int64, Int64}, Nemo.fmpz}; wm = 1) where {Ty<:Number}
    gm = map(x -> 0:x, m) |> vec |> x -> Iterators.product(x...)

    function fun_n(n)
        i = m.-n
        si = sum(i)
        sm = sum(m)
        return wm*pmmi_precomputed_arb(i, m, sm, si, t, sα, ν_dict_arb, Cmmi_dict_arb, precomputed_binomial_coefficients_arb)
    end

    Dict( collect(n) => fun_n(n) for n in gm ) #|> Accumulator

end

function inc_arb!(ct::Dict{Array{Int64,1},Nemo.arb}, x, a::Nemo.arb)
    if(haskey(ct, x))
        ct[x] += a
    else
        ct[x] = a
    end
end
function merge_arb!(ct::Dict{Array{Int64,1},Nemo.arb}, other::Dict{Array{Int64,1},Nemo.arb})
    for (x, v) in other
        inc_arb!(ct, x, v)
    end
    ct
end

function predict_WF_params_precomputed_arb(wms::Array{Nemo.arb,1}, sα::Number, Λ::Array{Array{Int64,1},1}, t::Number, ν_dict_arb::Dict{Tuple{Int64, Int64}, Nemo.arb}, Cmmi_dict_arb::Dict{Tuple{Int64, Int64}, Nemo.arb}, precomputed_binomial_coefficients_arb::Dict{Tuple{Int64, Int64}, Nemo.fmpz})

    res = Dict{Array{Int64,1},Nemo.arb}()

    for k in 1:length(Λ)
        merge_arb!(res, WF_prediction_for_one_m_precomputed_arb(Λ[k], sα, t, ν_dict_arb, Cmmi_dict_arb, precomputed_binomial_coefficients_arb; wm = wms[k]))
    end

    ks = keys(res) |> collect

    return ks, [res[k] for k in ks]

end

function update_WF_params_arb(wms::Array{Nemo.arb,1}, α::Array{Ty,1}, Λ::Array{Array{Int64,1},1}, y::Array{Int64,2}) where Ty<:Number
    #y is a matrix of dimension J*K, with K the dimension of the process
    # and J the number of observations
    # Julia is in row major, so the first index indicates the row (index of observation)
    # and the second the column (index of the dimension) (as in matrix mathematical notation)
    @assert length(wms) == size(Λ, 1)

    nJ = sum(y, dims = 2) |> vec#sum_j=1^K n_ij
    nK = sum(y, dims = 1) |> vec#sum_i=1^J n_ij
    sy = sum(y)
    J = size(y, 1)
    sα = sum(α)

    first_term = prod(Nemo.fac.(nJ))*RR(1.)/prod(Nemo.fac.(y))

    function pga(m::Array{Int64,1})
        sm = sum(m)
        second_term = gamma_arb(sα + sm)
        third_term = prod(gamma_arb.(α + m + nK))
        fourth_term = gamma_arb(sα + sm + sy)
        fifth_term = prod(gamma_arb.(α + m))
        return first_term*second_term*third_term/(fourth_term*fifth_term)
    end

     wms_hat = normalise_arb(wms .* map(pga, Λ))

    return [m .+ nK for m in Λ], wms_hat
end

function get_next_filtering_distribution_precomputed_arb(current_Λ, current_wms, current_time, next_time, α, sα, next_y, ν_dict_arb::Dict{Tuple{Int64, Int64}, Nemo.arb}, Cmmi_dict_arb::Dict{Tuple{Int64, Int64}, Nemo.arb}, precomputed_binomial_coefficients_arb::Dict{Tuple{Int64, Int64}, Nemo.fmpz})
    predicted_Λ, predicted_wms = predict_WF_params_precomputed_arb(current_wms, sα, current_Λ, next_time-current_time, ν_dict_arb, Cmmi_dict_arb, precomputed_binomial_coefficients_arb)
    filtered_Λ, filtered_wms = update_WF_params_arb(predicted_wms, α, predicted_Λ, next_y)

    return filtered_Λ, filtered_wms
end

function filter_WF_precomputed_arb(α, data, ν_dict_arb::Dict{Tuple{Int64, Int64}, Nemo.arb}, Cmmi_dict_arb::Dict{Tuple{Int64, Int64}, Nemo.arb}, precomputed_binomial_coefficients_arb::Dict{Tuple{Int64, Int64}, Nemo.fmpz})
    # println("filter_WF_mem2")

    @assert length(α) == length(data[collect(keys(data))[1]])


    sα = sum(α)
    times = keys(data) |> collect |> sort
    Λ_of_t = Dict()
    wms_of_t = Dict()

    filtered_Λ, filtered_wms = update_WF_params_arb([RR(1.)], α, [repeat([0], inner = length(α))], data[times[1]])

    Λ_of_t[times[1]] = filtered_Λ
    wms_of_t[times[1]] = filtered_wms

    for k in 1:(length(times)-1)
        println("Step index: $k")
        println("Number of components: $(length(filtered_Λ))")
        filtered_Λ, filtered_wms = get_next_filtering_distribution_precomputed_arb(filtered_Λ, filtered_wms, times[k], times[k+1], α, sα, data[times[k+1]], ν_dict_arb, Cmmi_dict_arb, precomputed_binomial_coefficients_arb)
        # mask = filtered_wms .!= 0.
        # filtered_Λ = filtered_Λ[mask]
        # filtered_wms = filtered_wms[mask]
        Λ_of_t[times[k+1]] = filtered_Λ
        wms_of_t[times[k+1]] = filtered_wms
    end

    return Λ_of_t, wms_of_t

end
