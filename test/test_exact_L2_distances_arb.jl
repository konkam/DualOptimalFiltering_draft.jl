using ExactWrightFisher

# @testset "test exact L2 Gamma distance functions with arbitrary precision" begin
#     # @test_warn AssertionError DualOptimalFiltering_proof.log_int_prod_2_Gammas(0.5, 0.5, 0.5, 1)
#     @test DualOptimalFiltering_proof.prod_2_gammas_arb(1, 1, 1, 1) == 0.5
#     @test DualOptimalFiltering_proof.prod_2_gammas_arb(2, 0.5, 1, 0.5) == 0.125
#     @test DualOptimalFiltering_proof.int_prod_2_Gammas(1, 1, 1, 1) == 0.5
#     @test DualOptimalFiltering_proof.int_prod_2_Gammas(2, 0.5, 1, 0.5) ≈ 0.5^2*0.5
#
#     @test_nowarn DualOptimalFiltering_proof.L2_dist_Gamma_mixtures_arb([0.2,0.3,0.5], [1,2,3], [1,2,4], [0.6,0.4], [1,2], [4,2])
#     @test Float64(DualOptimalFiltering_proof.L2_dist_Gamma_mixtures_arb([0.2,0.3,0.5], [1,2,3], [1,2,4], [0.2,0.3,0.5], [1,2,3], [1,2,4])) ≈ 0. atol = 10^(-20)
#
#     Random.seed!(0)
#     δ = 3.
#     γ = 3.
#     σ = 0.5
#     λ = 1.
#     θ = γ/σ^2
#     X = DualOptimalFiltering_proof.generate_CIR_trajectory(range(0, stop = 2, length = 20), δ, γ, σ, λ);
#     Y = map(λ -> rand(Poisson(λ),10), X);
#     data_ = Dict(zip(range(0, stop = 2, length = 20), Y))
#     Λ_of_t, wms_of_t, θ_of_t = DualOptimalFiltering_proof.filter_CIR_debug(δ, γ, σ, λ, data_);
#     Λ_of_t1, wms_of_t1, θ_of_t1 = DualOptimalFiltering_proof.filter_CIR_keep_fixed_number(δ, γ, σ, λ, data_, 1);
#
#     time_grid = data_ |> keys |> collect |> sort
#     # logw1 = log.(wms_of_t[time_grid[20]])
#     αlist_1, βlist_1 = DualOptimalFiltering_proof.create_gamma_mixture_parameters(δ, θ_of_t[time_grid[20]], Λ_of_t[time_grid[20]])
#
#     # logw2 = log.(wms_of_t1[time_grid[20]])
#     αlist_2, βlist_2 = DualOptimalFiltering_proof.create_gamma_mixture_parameters(δ, θ_of_t1[time_grid[20]], Λ_of_t1[time_grid[20]])
#
#     @test_nowarn DualOptimalFiltering_proof.L2_dist_Gamma_mixtures_arb(wms_of_t[time_grid[20]], αlist_1, βlist_1, wms_of_t1[time_grid[20]], αlist_2, βlist_2)
#
#     cc = DualOptimalFiltering_proof.create_gamma_mixture_pdf(δ, θ_of_t[time_grid[20]], Λ_of_t[time_grid[20]], wms_of_t[time_grid[20]])
#
#     cc1 = DualOptimalFiltering_proof.create_gamma_mixture_pdf(δ, θ_of_t1[time_grid[20]], Λ_of_t1[time_grid[20]], wms_of_t1[time_grid[20]])
#
#     res = DualOptimalFiltering_proof.L2_dist_1D(cc, cc1)
#     @test Float64(DualOptimalFiltering_proof.L2_dist_Gamma_mixtures_arb(wms_of_t[time_grid[20]], αlist_1, βlist_1, wms_of_t1[time_grid[20]], αlist_2, βlist_2)) ≈ res[1] atol=res[2]
# end;

@testset "test exact L2 distance functions for Dirichlet mixtures" begin

    @test DualOptimalFiltering_proof.mvbeta_arb([1,1]) == 1.
    @test DualOptimalFiltering_proof.int_prod_2_Dir_arb([1,1], [1,1]) == 1.
    @test DualOptimalFiltering_proof.int_prod_2_Dir_arb([1,1,1], [1,1,1]) == 2.

    @test_nowarn DualOptimalFiltering_proof.L2_dist_Dirichlet_mixtures_arb([0.2,0.3,0.5], [[1,2,3],[1,2,4],[5,3,2]], [0.6,0.4], [[1,2,1],[4,2,3]])
    @test Float64(DualOptimalFiltering_proof.L2_dist_Dirichlet_mixtures_arb([0.2,0.3,0.5], [[1,2,3],[1,2,4],[5,3,2]], [0.2,0.3,0.5], [[1,2,3],[1,2,4],[5,3,2]])) ≈ 0 atol=10^(-20)

    Random.seed!(0);
    α_vec = [1.2, 1.4, 1.3]
    # α = α_vec
    K = length(α_vec)
    Pop_size_WF3 = 10
    Nparts = 100
    time_grid_WF3 = range(0, stop = 1, length = 5)
    wfchain = Wright_Fisher_K_dim_exact_trajectory([0.2, 0.4, 0.4], time_grid_WF3, α_vec)
    wfobs_WF3 = [rand(Multinomial(Pop_size_WF3, wfchain[:,k])) for k in 1:size(wfchain,2)] |> l -> hcat(l...)
    data__WF3 = Dict(zip(time_grid_WF3 , [wfobs_WF3[:,t]' |> collect for t in 1:size(wfobs_WF3,2)]))

    log_ν_dict_arb, log_Cmmi_dict_arb, log_binomial_coeff_dict_arb = DualOptimalFiltering_proof.precompute_log_terms_arb(data__WF3, sum(α_vec); digits_after_comma_for_time_precision = 4)


    Λ_of_t_arb, wms_of_t_arb = DualOptimalFiltering_proof.filter_WF_precomputed(α_vec, data__WF3, log_ν_dict_arb, log_Cmmi_dict_arb, log_binomial_coeff_dict_arb)

    times = keys(Λ_of_t_arb) |> collect |> sort

    Λ_of_t, wms_of_t = DualOptimalFiltering_proof.filter_WF_precomputed_keep_fixed_number(α_vec, data__WF3, log_ν_dict_arb, log_Cmmi_dict_arb, log_binomial_coeff_dict_arb, 30)

    # res = DualOptimalFiltering_proof.compute_L2_distance_between_two_Dirichlet_mixtures(α_vec, Λ_of_t_arb[times[4]], wms_of_t_arb[times[4]], Λ_of_t[times[4]], wms_of_t[times[4]])
    #
    # @test Float64(DualOptimalFiltering_proof.L2_dist_Dirichlet_mixtures_arb(wms_of_t_arb[times[4]], DualOptimalFiltering_proof.create_dirichlet_mixture(α_vec, Λ_of_t_arb[times[4]]), wms_of_t[times[4]],  DualOptimalFiltering_proof.create_dirichlet_mixture(α_vec, Λ_of_t[times[4]]))) ≈ res[1] atol=res[2]

    res = DualOptimalFiltering_proof.compute_L2_distance_between_two_Dirichlet_mixtures(α_vec, Λ_of_t_arb[times[3]][1:50], wms_of_t_arb[times[3]][1:50], Λ_of_t[times[3]][1:50], wms_of_t[times[3]][1:50])

    @test exp(DualOptimalFiltering_proof.log_L2_dist_Dirichlet_mixtures(log.(wms_of_t_arb[times[3]][1:50]), DualOptimalFiltering_proof.create_dirichlet_mixture(α_vec, Λ_of_t_arb[times[3]][1:50]), log.(wms_of_t[times[3]][1:50]),  DualOptimalFiltering_proof.create_dirichlet_mixture(α_vec, Λ_of_t[times[3]][1:50]))) ≈ res[1] atol=res[2]

    @test Float64(DualOptimalFiltering_proof.L2_dist_Dirichlet_mixtures_arb(wms_of_t_arb[times[3]][1:50], DualOptimalFiltering_proof.create_dirichlet_mixture(α_vec, Λ_of_t_arb[times[3]][1:50]), wms_of_t[times[3]][1:50],  DualOptimalFiltering_proof.create_dirichlet_mixture(α_vec, Λ_of_t[times[3]][1:50]))) ≈ res[1] atol=res[2]

end;
