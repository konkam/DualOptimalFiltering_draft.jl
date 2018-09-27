@testset "test high level statistical distances" begin
    res = DualOptimalFiltering.compute_hellinger_distance_between_two_Dirichlet_mixtures(ones(3), [[1,1,1], [1,1,1]], [0.5,0.5], [[1,1,3], [1,2,1]], [0.2,0.8])
    @test res[1] ≈ 1.974305e-02 atol=10.0^(-5)
    @test res[2] < 10.0^(-6)
    res = DualOptimalFiltering.compute_L2_distance_between_two_Dirichlet_mixtures(ones(3), [[1,1,1], [1,1,1]], [0.5,0.5], [[1,1,3], [1,2,1]], [0.2,0.8])
    @test res[1] ≈ 2.812121e-01 atol=10.0^(-5)
    @test res[2] < 10.0^(-5)
end;

res = DualOptimalFiltering.αΛ_to_α(1:3 |> collect, [[4,5,6], [1,3,2]])
# println(res)
for i in 1:length(res)
    @test  res[i] ==  [5, 7, 9, 2, 5, 5][i]
end

using RCall
@testset "Test RCpp functions" begin
    R"library(tidyverse)
    x = c(0.2401053, 0.3672329, 0.3926618)"
    res = R"ddirichlet_mixture_gsl_arma(x, rep(rep(1.1,3),6), 6 %>% rep(1./.,.))" |> Float64
    @test res ≈ 2.226413 atol=10.0^(-5)
    res = R"ddirichlet_gsl_arma(x , rep(1.1,3))"  |> Float64
    @test res ≈ 2.226413 atol=10.0^(-5)
    # R"set.seed(0)
    # lambda = rpois(30, lambda = 3)"
    # res = R"ddirichlet_mixture_gsl_alpha_m_arma(x, rep(1.1,3), 1/10, lambda)"  |> Float64
    # @test res ≈ 0.1436751 atol=10.0^(-5)
end;
