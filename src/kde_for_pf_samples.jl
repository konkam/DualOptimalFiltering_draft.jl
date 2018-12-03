using KernelEstimator

"""
    create_gamma_kde_mixture_parameters(smp::Array{Float64,1})

This is using Chen, Song Xi. "Probability density function estimation using gamma kernels." Annals of the Institute of Statistical Mathematics 52, no. 3 (2000): 471-480.
Practically, given the bandwidth bw, the kernels are Gamma(α = x/bw+1, β = 1/bw) in the shape, rate parameterisation.
Returns the list of α for all kernels and the common β.


# Examples
```julia-repl
julia> bar([1, 2], [1, 2])
1
```
"""
function create_gamma_kde_mixture_parameters_one_β(smp::Array{T,1}) where T <: Real
    bw = bwlscv(smp, gammakernel)
    # bw = bwlcv(smp, gammakernel)
    if bw==0
        stop("bandwidth estimation by least square cross validation failed")
    else
        return smp ./ bw .+ 1, 1/bw
    end
end

function create_gamma_kde_mixture_parameters(smp::Array{T,1}) where T <: Real
    α_list, β = create_gamma_kde_mixture_parameters_one_β(smp)

    return α_list, repeat([β], length(α_list))
end


function create_gamma_mixture_density_αβ(α_list::AbstractArray{T,1}, β_list::AbstractArray{U,1}, wms::AbstractArray{V,1}) where {T <: Real, U <: Real, V <: Real}
    #use 1/θ because of the way the Gamma distribution is parameterised in Julia Distributions.jl
    function res(x::Real)
        sum(wms[i] * pdf(Gamma(α_list[i], 1/β_list[i]), x) for i in 1:length(wms))
    end
    return res
end
