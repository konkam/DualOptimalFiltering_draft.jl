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
end;

@testset "test adaptive precomputation filtering" begin

    Random.seed!(4)
    wfchain = rand(Dirichlet(4,0.3)) |> z-> DualOptimalFiltering.wright_fisher_PD1(z, 1.5, 50, 3)[:,2:end]*10 |> x -> round.(x) |> x -> Int64.(x)
    data = Dict(zip(range(0, stop = 5, length = size(wfchain,2)), [collect(collect(wfchain[:,t:t]')) for t in 1:size(wfchain,2)]))
    Λ_of_t, wms_of_t = DualOptimalFiltering.filter_WF_mem2(ones(4), data)
    Λ_of_t_adaptive, wms_of_t_adaptive = DualOptimalFiltering.filter_WF_adaptive_precomputation(ones(4), data, (x,y) -> (x,y))
    @test length(keys(Λ_of_t)) == 3
    @test length(keys(wms_of_t)) == 3
    @test_throws AssertionError filter_WF(ones(2), data)
    times = keys(Λ_of_t) |> collect |> sort
    for i in 1:length(keys(Λ_of_t))
        @test Λ_of_t[times[i]] == Λ_of_t_adaptive[times[i]]
        @test wms_of_t[times[i]] == wms_of_t_adaptive[times[i]]
    end
end;
