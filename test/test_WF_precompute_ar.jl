using Nemo

@testset "WF filtering precomputing tests" begin


    K = 3
    α = ones(K)
    sα = sum(α)
    Pop_size = 15
    Ntimes = 3
    Random.seed!(4)
    wfchain = rand(Dirichlet(K,0.3)) |> z-> DualOptimalFiltering_proof.wright_fisher_PD1(z, 1.5, 50, Ntimes)[:,2:end]
    wfobs = [rand(Multinomial(Pop_size, wfchain[:,k])) for k in 1:size(wfchain,2)] |> l -> hcat(l...)
    data = Dict(zip(range(0, stop = 5, length = size(wfobs,2)) , [collect(wfobs[:,t:t]') for t in 1:size(wfobs,2)]))

    @test_nowarn DualOptimalFiltering_proof.precompute_log_first_term_ar(data, sα)

    @test_nowarn DualOptimalFiltering_proof.precompute_log_Cmmi_ar(data, sα)

    @test_nowarn DualOptimalFiltering_proof.precompute_log_binomial_coefficients_ar(data)


    # @test typeof(DualOptimalFiltering_proof.precompute_first_term_arb(data, 2.1)) == Dict{Tuple{Int64,Int64},Nemo.arb}
    # @test typeof(DualOptimalFiltering_proof.precompute_Cmmi_arb(data, 2.1; digits_after_comma_for_time_precision = 4)) == Dict{Tuple{Int64,Int64},Nemo.arb}
    # @test typeof(DualOptimalFiltering_proof.precompute_binomial_coefficients_arb(data))  == Dict{Tuple{Int64,Int64},Nemo.fmpz}

    log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset = DualOptimalFiltering_proof.precompute_terms_ar(data, sum(α); digits_after_comma_for_time_precision = 4)

    log_ν_dict, log_Cmmi_dict, log_binomial_coeff_dict = DualOptimalFiltering_proof.precompute_terms(data, sum(α); digits_after_comma_for_time_precision = 4)
    # @test typeof(ν_dict_arb) == Dict{Tuple{Int64,Int64},Nemo.arb}
    # @test typeof(Cmmi_dict_arb) == Dict{Tuple{Int64,Int64},Nemo.arb}
    # @test typeof(precomputed_binomial_coefficients_arb) == Dict{Tuple{Int64,Int64},Nemo.fmpz}

    @test  DualOptimalFiltering_proof.loghypergeom_pdf_using_precomputed([5,4], [6,15], 9, 21, log_binomial_coeff_ar_offset) == DualOptimalFiltering_proof.loghypergeom_pdf_using_precomputed([5,4], [6,15], 9, 21, log_binomial_coeff_dict)
    @test DualOptimalFiltering_proof.logpmmi_raw_precomputed([5,4], [6,15], 21, 9, 1., log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset) == DualOptimalFiltering_proof.logpmmi_raw_precomputed([5,4], [6,15], 21, 9, 1., log_ν_dict, log_Cmmi_dict, log_binomial_coeff_dict)

    @test DualOptimalFiltering_proof.logpmn_precomputed([2, 4, 3], [0, 1, 0], 9, 1, 0.2, 1.2, log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset) == DualOptimalFiltering_proof.logpmn_precomputed([2, 4, 3], [0, 1, 0], 9, 1, 0.2, 1.2, log_ν_dict, log_Cmmi_dict, log_binomial_coeff_dict)

    @test DualOptimalFiltering_proof.logpmmi_precomputed([2,3,3], [2, 4, 3], 9, 8, 0.2, 1.2, log_ν_dict, log_Cmmi_dict, log_binomial_coeff_dict) == DualOptimalFiltering_proof.logpmmi_precomputed([2,3,3], [2, 4, 3], 9, 8, 0.2, 1.2, log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset)

end;
