using QuadGK, Optim

function Hellinger_dist_1D(d, dref, infbound, supbound)
    f(x) = sqrt(d(x)*dref(x))
#     f(74.9)
    res = quadgk(f, infbound, supbound)
    return (1-res[1], res[2])
end

function L2_dist_1D(pdf1, pdf2, infbound, supbound)
    squared_res = CvM_distance(pdf1, pdf2, infbound, supbound)
    #using a conservative error bound
    return (sqrt(squared_res[1]), squared_res[2])
end

L2_dist_1D(pdf1, pdf2) = L2_dist_1D(pdf1, pdf2, 0, Inf)

# function KS_distance(cdf_1, cdf_2)
#     f(x) = -abs(cdf_1(x) - cdf_2(x))
#     # return result = optimize(f, 0., 1.) |> Optim.minimum |> abs
#     return result = optimize(x -> f(first(x), BFGS()), [0.5]) |> Optim.minimum |> abs
# end


function CvM_distance(cdf_1, cdf_2, infbound, supbound)
     f(x) = (cdf_1(x) - cdf_2(x))^2
#     f(74.9)
    res = quadgk(f, infbound, supbound)
    return res
end

function CvM_distance_singularities(cdf_1::Function, cdf_2::Function, singularities::Array{T,1}, infbound::Number, supbound::Number) where T <: Number
     f(x) = (cdf_1(x) - cdf_2(x))^2
#     f(74.9)
    res = quadgk_singularities(f, singularities, infbound, supbound)
    return res
end

function quadgk_singularities(f::Function, singularities::Array{T,1}, infbound::Number, supbound::Number) where T <: Number
    unique_sorted_singularities = singularities |> unique |> sort |> x -> x[(x.>infbound).&(x.<supbound)]
    total, prec = quadgk(f, infbound, unique_sorted_singularities[1])
    for i in 1:(length(unique_sorted_singularities)-1)
        tmp = quadgk(f, unique_sorted_singularities[i], unique_sorted_singularities[i+1])
        total += tmp[1]
        prec = max(prec, tmp[2])
    end
    tmp = quadgk(f, unique_sorted_singularities[end], supbound)
    total += tmp[1]
    prec = max(prec, tmp[2])
    return total, prec
end
