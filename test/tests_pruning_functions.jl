@testset "Pruning functions tests" begin
    srand(4)
    α = ones(4)
    wfchain = rand(Dirichlet(4,0.3)) |> z-> DualOptimalFiltering.wright_fisher_PD1(z, 1.5, 50, 3)[:,2:end]*10 |> x -> round.(x) |> x -> Int64.(x)
    data = Dict(zip(linspace(0, 5, size(wfchain,2)), [wfchain[:,t:t]' for t in 1:size(wfchain,2)]))
    log_ν_dict_arb, log_Cmmi_dict_arb, precomputed_log_binomial_coefficients_arb = DualOptimalFiltering.precompute_log_terms_arb(data, sum(α); digits_after_comma_for_time_precision = 4)

    Λ_of_t, wms_of_t = DualOptimalFiltering.filter_WF_precomputed_keep_fixed_number(α, data, log_ν_dict_arb, log_Cmmi_dict_arb, precomputed_log_binomial_coefficients_arb, 30)
    times = Λ_of_t |> keys |> collect |> sort
    Λ_of_t_kept, wms_of_t_kept = DualOptimalFiltering.keep_above_threshold(Λ_of_t[times[end]], wms_of_t[times[end]], 0.001)
    @test minimum(wms_of_t_kept) >= 0.001
    Λ_of_t_kept, wms_of_t_kept = DualOptimalFiltering.keep_fixed_number_of_weights(Λ_of_t[times[end]], wms_of_t[times[end]], 200)
    @test length(wms_of_t_kept |> unique) <= 200
    Λ_of_t_kept, wms_of_t_kept = DualOptimalFiltering.keep_fixed_number_of_weights(Λ_of_t[times[end]], wms_of_t[times[end]], 2)
    @test length(wms_of_t_kept |> unique) <= 2
    Λ_of_t_kept, wms_of_t_kept = DualOptimalFiltering.keep_fixed_number_of_weights(Λ_of_t[times[end]], wms_of_t[times[end]], 1)
    @test length(wms_of_t_kept |> unique) <= 1
    Λ_of_t_kept, wms_of_t_kept = DualOptimalFiltering.keep_fixed_fraction(Λ_of_t[times[end]], wms_of_t[times[end]], 0.95)
    @test sum(wms_of_t_kept) >= 0.95
    Λ_of_t_kept, wms_of_t_kept = DualOptimalFiltering.keep_fixed_fraction(Λ_of_t[times[end]], wms_of_t[times[end]], 0.99)
    @test sum(wms_of_t_kept) >= 0.99


end;
