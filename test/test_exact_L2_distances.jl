@testset "test exact L2 Gamma distance functions" begin
    # @test_warn AssertionError DualOptimalFiltering.log_int_prod_2_Gammas(0.5, 0.5, 0.5, 1)
    @test DualOptimalFiltering.log_int_prod_2_Gammas(1, 1, 1, 1) == -log(2)
    @test DualOptimalFiltering.log_int_prod_2_Gammas(2, 0.5, 1, 0.5) == log(0.5^2*0.5 )
    @test DualOptimalFiltering.int_prod_2_Gammas(1, 1, 1, 1) == 0.5
    @test DualOptimalFiltering.int_prod_2_Gammas(2, 0.5, 1, 0.5) ≈ 0.5^2*0.5

    @test_nowarn DualOptimalFiltering.log_L2_dist_Gamma_mixtures(log.([0.2,0.3,0.5]), [1,2,3], [1,2,4], log.([0.6,0.4]), [1,2], [4,2])
    @test !isfinite(DualOptimalFiltering.log_L2_dist_Gamma_mixtures(log.([0.2,0.3,0.5]), [1,2,3], [1,2,4], log.([0.2,0.3,0.5]), [1,2,3], [1,2,4]))

    Random.seed!(0)
    δ = 3.
    γ = 3.
    σ = 0.5
    λ = 1.
    θ = γ/σ^2
    X = DualOptimalFiltering.generate_CIR_trajectory(range(0, stop = 2, length = 20), δ, γ, σ, λ);
    Y = map(λ -> rand(Poisson(λ),10), X);
    data = Dict(zip(range(0, stop = 2, length = 20), Y))
    Λ_of_t, wms_of_t, θ_of_t = DualOptimalFiltering.filter_CIR_debug(δ, γ, σ, λ, data);
    Λ_of_t1, wms_of_t1, θ_of_t1 = DualOptimalFiltering.filter_CIR_keep_fixed_number(δ, γ, σ, λ, data, 1);

    time_grid = data |> keys |> collect |> sort
    logw1 = log.(wms_of_t[time_grid[20]])
    αlist_1, βlist_1 = DualOptimalFiltering.create_mixture_parameters(δ, θ, Λ_of_t[time_grid[20]][isfinite.(logw1)], wms_of_t[time_grid[20]][isfinite.(logw1)])

    logw2 = log.(wms_of_t1[time_grid[20]])
    αlist_2, βlist_2 = DualOptimalFiltering.create_mixture_parameters(δ, θ, Λ_of_t1[time_grid[20]], wms_of_t1[time_grid[20]])

    @test_nowarn DualOptimalFiltering.log_L2_dist_Gamma_mixtures(logw1[isfinite.(logw1)], αlist_1, βlist_1, logw2, αlist_2, βlist_2)

    cc = DualOptimalFiltering.create_gamma_mixture_pdf(δ, θ, Λ_of_t[time_grid[20]][isfinite.(logw1)], wms_of_t[time_grid[20]][isfinite.(logw1)])

    cc1 = DualOptimalFiltering.create_gamma_mixture_pdf(δ, θ, Λ_of_t1[time_grid[20]], wms_of_t1[time_grid[20]])

    res = DualOptimalFiltering.L2_dist_1D(cc, cc1)
    @test exp(DualOptimalFiltering.log_L2_dist_Gamma_mixtures(logw1[isfinite.(logw1)], αlist_1, βlist_1, logw2, αlist_2, βlist_2)) ≈ res[1] atol=res[2]
end;
#
# @testset "test exact L2 distance functions" begin
#
#     @test DualOptimalFiltering.lmvbeta([1,1]) == 0.0
#     @test DualOptimalFiltering.log_int_prod_2_Dir([1,1], [1,1]) == 0.0
#     @test DualOptimalFiltering.log_int_prod_2_Dir([1,1,1], [1,1,1]) ≈ 0.6931471805599453
#
#     @test_nowarn DualOptimalFiltering.log_L2_dist_Dirichlet_mixtures(log.([0.2,0.3,0.5]), [[1,2,3],[1,2,4],[5,3,2]], log.([0.6,0.4]), [[1,2,1],[4,2,3]])
#     @test !isfinite(DualOptimalFiltering.log_L2_dist_Dirichlet_mixtures(log.([0.2,0.3,0.5]), [[1,2,3],[1,2,4],[5,3,2]], log.([0.2,0.3,0.5]), [[1,2,3],[1,2,4],[5,3,2]])
# )
# end;
