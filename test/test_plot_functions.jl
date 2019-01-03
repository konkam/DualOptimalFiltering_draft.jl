@testset "WF diagnostic plot functions" begin
    Random.seed!(4)
    α = ones(4)
    wfchain = rand(Dirichlet(4,0.3)) |> z-> DualOptimalFiltering.wright_fisher_PD1(z, 1.5, 50, 3)[:,2:end]*10 |> x -> round.(x) |> x -> Int64.(x)
    data = Dict(zip(range(0, stop = 5, length = size(wfchain,2)), [collect(wfchain[:,t:t]') for t in 1:size(wfchain,2)]))
    log_ν_dict_arb, log_Cmmi_dict_arb, log_binomial_coeff_dict_arb = DualOptimalFiltering.precompute_log_terms_arb(data, sum(α); digits_after_comma_for_time_precision = 4)

    Λ_of_t, wms_of_t = DualOptimalFiltering.filter_WF_precomputed_keep_fixed_number(α, data, log_ν_dict_arb, log_Cmmi_dict_arb, log_binomial_coeff_dict_arb, 30)

    marginal_CI = DualOptimalFiltering.compute_marginal_CI(α, Λ_of_t, wms_of_t; mass = 0.95)
    @test_nowarn DualOptimalFiltering.plot_marginal_posterior_credible_interval_and_data_given_marginalCI_with_hidden_state(α, Λ_of_t, wms_of_t, data, wfchain, marginal_CI)

end;

@testset "CIR diagnostic plot functions" begin
    Random.seed!(1)

    δ = 3.
    γ = 2.5
    σ = 4.
    Nobs = 10
    dt_CIR = 0.011
    # max_time = .1
    # max_time = 0.001
    Nsteps_CIR = 20
    λ = 1.

    time_grid_CIR = [k*dt_CIR for k in 0:(Nsteps_CIR-1)]
    X_CIR = generate_CIR_trajectory(time_grid_CIR, 3, δ, γ, σ)
    Y_CIR = map(λ -> rand(Poisson(λ), Nobs), X_CIR);
    data_CIR = Dict(zip(time_grid_CIR, Y_CIR))
    Λ_of_t_CIR, wms_of_t_CIR, θ_of_t_CIR = filter_CIR(δ, γ, σ, λ, data_CIR);

    @test_nowarn DualOptimalFiltering.plot_data_and_posterior_distribution(δ, θ_of_t_CIR, Λ_of_t_CIR, wms_of_t_CIR, data_CIR, X_CIR)

end;
