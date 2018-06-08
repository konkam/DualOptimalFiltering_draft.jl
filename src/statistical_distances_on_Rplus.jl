using QuadGK

function Hellinger_dist_1D(d, dref, infbound, supbound)
    f(x) = sqrt(d(x)*dref(x))
#     f(74.9)
    res = quadgk(f, infbound, supbound)
    return (1-res[1], res[2])
end
