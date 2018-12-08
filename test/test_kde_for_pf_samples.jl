using Distributions, Random

@testset "test Gamma kde function" begin
    @test 1 == 1
    res = DualOptimalFiltering.create_gamma_kde_mixture_parameters_one_β([1,4,6])
    for i in 1:length(res[1])
        @test res[1][i] ≈ [1.46703, 2.86814, 3.80221][i] atol=10^(-5)
    end
    @test res[2] ≈ 0.4670348611479834 atol=10^(-10)
    res = DualOptimalFiltering.create_gamma_kde_mixture_parameters([1,4,6])
    @test length(res[1]) == length(res[2])
end;

@testset "test Dirichlet kde function" begin
    @test 1 == 1
    Random.seed!(0);
    xdata = rand(Dirichlet([0.3,5.,2.3]), 5)
    @test_nowarn create_Dirichlet_kde_mixture_parameters(xdata)
end;
