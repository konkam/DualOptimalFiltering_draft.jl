@testset "test adaptive precomputation intermediate functions" begin

    @test_nowarn log(DualOptimalFiltering.λm(3,3.1)) ≈ DualOptimalFiltering.logλm(3,3.1)
    @test_nowarn DualOptimalFiltering.logfirst_term_pmmi_no_alloc(10, 15, 4.2)
    @test_nowarn DualOptimalFiltering.logCmmi_arb(10, 8, 0.4, 2.1)
    @test_nowarn DualOptimalFiltering.logCmmi(10, 8, 0.4, 2.1)
    @test Float64(DualOptimalFiltering.logCmmi_arb(10, 8, 0.4, 2.1)) ≈ DualOptimalFiltering.logCmmi(10, 8, 0.4, 2.1)[2]
    log_ν_dict = Dict{Tuple{Int64,Int64},Float64}()
    log_Cmmi_dict = Dict{Tuple{Int64,Int64},Float64}()
    log_binomial_coeff_dict = Dict{Tuple{Int64,Int64},Float64}()
    @test_nowarn DualOptimalFiltering.precompute_next_terms!(10, 12, log_ν_dict, log_Cmmi_dict, log_binomial_coeff_dict, 3.1, 0.2)

    log_ν_dict = Dict{Tuple{Int64,Int64},Float64}()
    log_Cmmi_dict = Dict{Tuple{Int64,Int64},Float64}()
    log_binomial_coeff_dict = Dict{Tuple{Int64,Int64},Float64}()
    @test_nowarn DualOptimalFiltering.precompute_next_terms!(0, 12, log_ν_dict, log_Cmmi_dict, log_binomial_coeff_dict, 3.1, 0.2)
    @test_nowarn DualOptimalFiltering.logpmn_precomputed([2, 4, 3], [0, 1, 0], 9, 1, 0.2, 1.2, log_ν_dict, log_Cmmi_dict, log_binomial_coeff_dict)
    @test DualOptimalFiltering.logpmn_precomputed([2, 4, 3], [0, 1, 0], 9, 1, 0.2, 1.2, log_ν_dict, log_Cmmi_dict, log_binomial_coeff_dict) == DualOptimalFiltering.logpmmi_precomputed([2,3,3], [2, 4, 3], 9, 8, 0.2, 1.2, log_ν_dict, log_Cmmi_dict, log_binomial_coeff_dict)
end;

@testset "test adaptive precomputation filtering" begin

    Random.seed!(4)
    wfchain = rand(Dirichlet(4,0.3)) |> z-> DualOptimalFiltering.wright_fisher_PD1(z, 1.5, 50, 3)[:,2:end]*10 |> x -> round.(x) |> x -> Int64.(x)
    data = Dict(zip(range(0, stop = 5, length = size(wfchain,2)), [collect(collect(wfchain[:,t:t]')) for t in 1:size(wfchain,2)]))
    # Λ_of_t, wms_of_t = DualOptimalFiltering.filter_WF_mem2(ones(4), data)
    α = ones(4)

    ν_dict, Cmmi_dict, precomputed_binomial_coefficients = DualOptimalFiltering.precompute_terms(data, sum(α); digits_after_comma_for_time_precision = 4)
    Λ_of_t, wms_of_t = DualOptimalFiltering.filter_WF_precomputed(α, data, ν_dict, Cmmi_dict, precomputed_binomial_coefficients)

    Λ_of_t_adaptive, wms_of_t_adaptive = DualOptimalFiltering.filter_WF_adaptive_precomputation(α, data, (x,y) -> (x,y))
    @test length(keys(Λ_of_t)) == 3
    @test length(keys(wms_of_t)) == 3
    # @test_throws AssertionError filter_WF(α, data)
    times = keys(Λ_of_t) |> collect |> sort
    for i in eachindex(keys(Λ_of_t))
        @test Λ_of_t[times[i]] == Λ_of_t_adaptive[times[i]]
        for j in eachindex( wms_of_t[times[i]])
            @test wms_of_t[times[i]][j] ≈ wms_of_t_adaptive[times[i]][j]
        end
    end
end;

@testset "test adaptive precomputation approx filtering functions" begin

    Random.seed!(4)
    wfchain = rand(Dirichlet(4,0.3)) |> z-> DualOptimalFiltering.wright_fisher_PD1(z, 1.5, 50, 3)[:,2:end]*10 |> x -> round.(x) |> x -> Int64.(x)
    data = Dict(zip(range(0, stop = 5, length = size(wfchain,2)), [collect(collect(wfchain[:,t:t]')) for t in 1:size(wfchain,2)]))
    # Λ_of_t, wms_of_t = DualOptimalFiltering.filter_WF_mem2(ones(4), data)
    α = ones(4)

    @test_nowarn DualOptimalFiltering.filter_WF_adaptive_precomputation_keep_fixed_number(α, data, 5; silence = false)

    @test_nowarn DualOptimalFiltering.filter_WF_adaptive_precomputation_keep_above_threshold(α, data, 0.01; silence = false)

    @test_nowarn DualOptimalFiltering.filter_WF_adaptive_precomputation_keep_fixed_fraction(α, data, 0.95; silence = false)
end;
