using RCall, Pkg

R"""Rcpp::sourceCpp($(joinpath(dirname(pathof(DualOptimalFiltering)), "dirichlet_mixture_gsl.cpp")))
# create_dirichlet_mixture_gsl = function(alpha, lambda, weights){
#     lambda_vec = unlist(lambda)
#     dirichlet_mixture = function(x){
#         ddirichlet_mixture_gsl_alpha_m_arma(x, alpha, weights, lambda_vec)
#     }
#     return(dirichlet_mixture)
# }
create_dirichlet_mixture_gsl_alpha = function(alpha, weights){
    dirichlet_mixture = function(x){
        ddirichlet_mixture_gsl_arma(x, alpha, weights)
    }
    return(dirichlet_mixture)
}
"""

R"compute_hellinger_distance_between_two_mixtures_of_Dirichlet_alpha_R = function (alpha_1, weights_1, alpha_2, weights_2, K){
    f1 = create_dirichlet_mixture_gsl_alpha(alpha_1, weights_1)
    f2 = create_dirichlet_mixture_gsl_alpha(alpha_2, weights_2)
    S <- SimplicialCubature::CanonicalSimplex(K-1)
    f1_simplex = function(x){f1(c(x, 1-sum(x)))}
    f2_simplex = function(x){f2(c(x, 1-sum(x)))}

    f_to_integrate = function(x){
        sqrt(f1_simplex(x)*f2_simplex(x))
    }

    res = SimplicialCubature::adaptIntegrateSimplex(f_to_integrate, S, maxEvals = 10^6,  tol = 1e-8)
    return(c(1-res$integral, res$estAbsError))
}"

# R"compute_hellinger_distance_between_two_mixtures_of_Dirichlet = function (alpha, lambda_1, weights_1, lambda_2, weights_2){
#     f1 = create_dirichlet_mixture_gsl(alpha, lambda_1, weights_1)
#     f2 = create_dirichlet_mixture_gsl(alpha, lambda_2, weights_2)
#     n = length(alpha)
#     S <- SimplicialCubature::CanonicalSimplex(n-1)
#     f1_simplex = function(x){f1(c(x, 1-sum(x)))}
#     f2_simplex = function(x){f2(c(x, 1-sum(x)))}
#
#     f_to_integrate = function(x){
#         sqrt(f1_simplex(x)*f2_simplex(x))
#     }
#
#     res = SimplicialCubature::adaptIntegrateSimplex(f_to_integrate, S, maxEvals = 10^6,  tol = 1e-8)
#     return(c(1-res$integral, res$estAbsError))
# }"

R"compute_L2_distance_between_two_mixtures_of_Dirichlet_alpha = function (alpha_1, weights_1, alpha_2, weights_2, K){

    f1 = create_dirichlet_mixture_gsl_alpha(alpha_1, weights_1)
    f2 = create_dirichlet_mixture_gsl_alpha(alpha_2, weights_2)
    n = K
    S <- SimplicialCubature::CanonicalSimplex(n-1)
    f1_simplex = function(x){f1(c(x, 1-sum(x)))}
    f2_simplex = function(x){f2(c(x, 1-sum(x)))}

    f_to_integrate = function(x){
        (f1_simplex(x)-f2_simplex(x))^2
    }

    res = SimplicialCubature::adaptIntegrateSimplex(f_to_integrate, S, maxEvals = 10^6)
    return(sqrt(c(res$integral, res$estAbsError)))
}"


# R"compute_L2_distance_between_two_mixtures_of_Dirichlet = function (alpha, lambda_1, weights_1, lambda_2, weights_2){
#     f1 = create_dirichlet_mixture_gsl(alpha, lambda_1, weights_1)
#     f2 = create_dirichlet_mixture_gsl(alpha, lambda_2, weights_2)
#     n = length(alpha)
#     S <- SimplicialCubature::CanonicalSimplex(n-1)
#     f1_simplex = function(x){f1(c(x, 1-sum(x)))}
#     f2_simplex = function(x){f2(c(x, 1-sum(x)))}
#
#     f_to_integrate = function(x){
#         (f1_simplex(x)-f2_simplex(x))^2
#     }
#
#     res = SimplicialCubature::adaptIntegrateSimplex(f_to_integrate, S, maxEvals = 10^6)
#     return(c(res$integral, res$estAbsError))
# }"

# function compute_hellinger_distance_between_two_Dirichlet_mixtures(α, Λ1, weights_1, Λ2, weights_2)
#     R"compute_hellinger_distance_between_two_mixtures_of_Dirichlet($α, $Λ1, $weights_1, $Λ2, $weights_2)"
# end

function compute_hellinger_distance_between_two_Dirichlet_mixtures_alpha(α1, weights_1, α2, weights_2, K::Int64)
    R"""Rcpp::sourceCpp($(joinpath(dirname(pathof(DualOptimalFiltering)), "dirichlet_mixture_gsl.cpp")))
    create_dirichlet_mixture_gsl_alpha = function(alpha, weights){
        dirichlet_mixture = function(x){
            ddirichlet_mixture_gsl_arma(x, alpha, weights)
        }
        return(dirichlet_mixture)
    }
    """

    R"compute_hellinger_distance_between_two_mixtures_of_Dirichlet_alpha_R = function (alpha_1, weights_1, alpha_2, weights_2, K){
        f1 = create_dirichlet_mixture_gsl_alpha(alpha_1, weights_1)
        f2 = create_dirichlet_mixture_gsl_alpha(alpha_2, weights_2)
        S <- SimplicialCubature::CanonicalSimplex(K-1)
        f1_simplex = function(x){f1(c(x, 1-sum(x)))}
        f2_simplex = function(x){f2(c(x, 1-sum(x)))}

        f_to_integrate = function(x){
            sqrt(f1_simplex(x)*f2_simplex(x))
        }

        res = SimplicialCubature::adaptIntegrateSimplex(f_to_integrate, S, maxEvals = 10^6,  tol = 1e-8)
        return(c(1-res$integral, res$estAbsError))
    }"
    R"compute_hellinger_distance_between_two_mixtures_of_Dirichlet_alpha_R($α1, $weights_1, $α2, $weights_2, $K)"
end

# function compute_L2_distance_between_two_Dirichlet_mixtures(α, Λ1, weights_1, Λ2, weights_2)
#     R"compute_L2_distance_between_two_mixtures_of_Dirichlet($α, $Λ1, $weights_1, $Λ2, $weights_2)"
# end

function compute_L2_distance_between_two_Dirichlet_mixtures_alpha(α1, weights_1, α2, weights_2, K::Int64)
    R"""Rcpp::sourceCpp($(joinpath(dirname(pathof(DualOptimalFiltering)), "dirichlet_mixture_gsl.cpp")))
    create_dirichlet_mixture_gsl_alpha = function(alpha, weights){
        dirichlet_mixture = function(x){
            ddirichlet_mixture_gsl_arma(x, alpha, weights)
        }
        return(dirichlet_mixture)
    }
    """
    R"compute_L2_distance_between_two_mixtures_of_Dirichlet_alpha = function (alpha_1, weights_1, alpha_2, weights_2, K){

        f1 = create_dirichlet_mixture_gsl_alpha(alpha_1, weights_1)
        f2 = create_dirichlet_mixture_gsl_alpha(alpha_2, weights_2)
        n = K
        S <- SimplicialCubature::CanonicalSimplex(n-1)
        f1_simplex = function(x){f1(c(x, 1-sum(x)))}
        f2_simplex = function(x){f2(c(x, 1-sum(x)))}

        f_to_integrate = function(x){
            (f1_simplex(x)-f2_simplex(x))^2
        }

        #conservative error bound
        res = SimplicialCubature::adaptIntegrateSimplex(f_to_integrate, S, maxEvals = 10^6)
        return(c(sqrt(res$integral), sqrt(res$estAbsError)))
    }"
    R"compute_L2_distance_between_two_mixtures_of_Dirichlet_alpha($α1, $weights_1, $α2, $weights_2, $K)"
end

function αΛ_to_α(α::Array{T,1}, Λ) where T<:Number
    it = Base.Iterators.cycle(α)
    # st = start(it)
    # st = iterate(it)[2]
    st = 1
    res = Float64.(Λ |> flat2)
    for i in eachindex(res)
        # (ii, st) = next(it, st)
        (ii, st) = iterate(it, st)
        res[i] += ii
    end
    return res
end

function compute_hellinger_distance_between_two_Dirichlet_mixtures(α, Λ1, weights_1, Λ2, weights_2)
    compute_hellinger_distance_between_two_Dirichlet_mixtures_alpha(αΛ_to_α(α, Λ1), weights_1, αΛ_to_α(α, Λ2), weights_2, length(α))
end

function compute_L2_distance_between_two_Dirichlet_mixtures(α, Λ1, weights_1, Λ2, weights_2)
    compute_L2_distance_between_two_Dirichlet_mixtures_alpha(αΛ_to_α(α, Λ1), weights_1, αΛ_to_α(α, Λ2), weights_2, length(α))
end
