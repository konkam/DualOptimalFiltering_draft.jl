@testset "Pruning functions tests" begin
    Random.seed!(4)
    α = ones(4)
    wfchain = rand(Dirichlet(4,0.3)) |> z-> DualOptimalFiltering_proof.wright_fisher_PD1(z, 1.5, 50, 3)[:,2:end]*10 |> x -> round.(x) |> x -> Int64.(x)
    data = Dict(zip(range(0, stop = 5, length = size(wfchain,2)), [collect(wfchain[:,t:t]') for t in 1:size(wfchain,2)]))
    log_ν_dict_arb, log_Cmmi_dict_arb, log_binomial_coeff_dict_arb = DualOptimalFiltering_proof.precompute_log_terms_arb(data, sum(α); digits_after_comma_for_time_precision = 4)

    Λ_of_t, wms_of_t = DualOptimalFiltering_proof.filter_WF_precomputed_keep_fixed_number(α, data, log_ν_dict_arb, log_Cmmi_dict_arb, log_binomial_coeff_dict_arb, 30)
    times = Λ_of_t |> keys |> collect |> sort
    Λ_of_t_kept, wms_of_t_kept = DualOptimalFiltering_proof.keep_above_threshold(Λ_of_t[times[end]], wms_of_t[times[end]], 0.001)
    @test minimum(wms_of_t_kept) >= 0.001
    Λ_of_t_kept, wms_of_t_kept = DualOptimalFiltering_proof.keep_fixed_number_of_weights(Λ_of_t[times[end]], wms_of_t[times[end]], 200)
    @test length(wms_of_t_kept |> unique) <= 200
    Λ_of_t_kept, wms_of_t_kept = DualOptimalFiltering_proof.keep_fixed_number_of_weights(Λ_of_t[times[end]], wms_of_t[times[end]], 2)
    @test length(wms_of_t_kept |> unique) <= 2
    Λ_of_t_kept, wms_of_t_kept = DualOptimalFiltering_proof.keep_fixed_number_of_weights(Λ_of_t[times[end]], wms_of_t[times[end]], 1)
    @test length(wms_of_t_kept |> unique) <= 1
    Λ_of_t_kept, wms_of_t_kept = DualOptimalFiltering_proof.keep_fixed_fraction(Λ_of_t[times[end]], wms_of_t[times[end]], 0.95)
    @test sum(wms_of_t_kept) >= 0.95
    Λ_of_t_kept, wms_of_t_kept = DualOptimalFiltering_proof.keep_fixed_fraction(Λ_of_t[times[end]], wms_of_t[times[end]], 0.99)
    @test sum(wms_of_t_kept) >= 0.99

    Λ_of_t_kept, wms_of_t_kept = DualOptimalFiltering_proof.keep_fixed_fraction_logw(Λ_of_t[times[end]], log.(wms_of_t[times[end]]), 0.99)
    @test logsumexp(wms_of_t_kept) >= log(0.99)


end;
