@testset "WF smoothing helper functions" begin
    Random.seed!(4)
    wfchain = rand(Dirichlet(4,0.3)) |> z-> DualOptimalFiltering_proof.wright_fisher_PD1(z, 1.5, 50, 3)[:,2:end]*10 |> x -> round.(x) |> x -> Int64.(x)
    data = Dict(zip(range(0, stop = 5, length = size(wfchain,2)), [collect(collect(wfchain[:,t:t]')) for t in 1:size(wfchain,2)]))
    # Λ_of_t, wms_of_t = DualOptimalFiltering_proof.filter_WF_mem2(ones(4), data)
    α = ones(4)

    data_1D = DualOptimalFiltering_proof.prepare_WF_dat_1D_2D(data)[1]

    log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset = DualOptimalFiltering_proof.precompute_terms_ar(data, sum(α); digits_after_comma_for_time_precision = 4)

    @test length(DualOptimalFiltering_proof.Λ_tilde_prime_k_from_Λ_tilde_k_WF([[10,5,3]])) == prod([10,5,3] .+ 1)

    @test collect(DualOptimalFiltering_proof.Λ_tilde_k_from_Λ_tilde_prime_kp1_WF([2,3,4], [[10,5,3]])) == Array{Int64,1}[[12, 8, 7]]

    @test_nowarn DualOptimalFiltering_proof.WF_backpropagation_for_one_m_precomputed([1,2,1], 0.2*ones(3), sum(0.2*ones(3)), 0.5, [2,3,4], log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset; wm = 1)

    @test_nowarn DualOptimalFiltering_proof.update_logweights_cost_to_go_WF(log.([0.2,0.3,0.4,0.1]), [[1,3,1],[7,4,1],[2,1,5],[3,2,4]], 0.2*ones(3), [2,3,4])





    res =  DualOptimalFiltering_proof.wms_tilde_kp1_from_wms_tilde_kp2_WF([0.6, 0.4], 0.2*ones(3), sum(0.2*ones(3)), [[1,2,1],[1,2,1]], 0.5, [2,3,4], log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset) |> (x-> Dict(zip(x[1], x[2])))

    ref =  DualOptimalFiltering_proof.WF_backpropagation_for_one_m_precomputed([1,2,1], 0.2*ones(3), sum(0.2*ones(3)), 0.5, [2,3,4], log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset; wm = 1)

    for k in keys(ref)
        @test res[k] ≈ ref[k]
    end

    @test_nowarn DualOptimalFiltering_proof.wms_tilde_kp1_from_wms_tilde_kp2_WF([0.6, 0.4], 0.2*ones(3), sum(0.2*ones(3)), [[1,2,1],[1,5,3]], 0.5, [2,3,4], log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset)

    ref_tmp = DualOptimalFiltering_proof.wms_tilde_kp1_from_wms_tilde_kp2_WF([0.6, 0.4], 0.2*ones(3), sum(0.2*ones(3)), [[1,2,1],[1,5,3]], 0.5, [2,3,4], log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset)

    ref = Dict(zip(ref_tmp[1], ref_tmp[2]))

    res_tmp =  DualOptimalFiltering_proof.logwms_tilde_kp1_from_logwms_tilde_kp2_WF(log.([0.6, 0.4]), 0.2*ones(3), sum(0.2*ones(3)), [[1,2,1],[1,5,3]], 0.5, [2,3,4], log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset)

    res = Dict(zip(res_tmp[1], res_tmp[2]))

    for k in keys(ref)
        @test exp(res[k]) ≈ ref[k]
    end

    res_tmp =  DualOptimalFiltering_proof.logwms_tilde_kp1_from_logwms_tilde_kp2_WF_pruning(log.([0.6, 0.4]), 0.2*ones(3), sum(0.2*ones(3)), [[1,2,1],[1,5,3]], 0.5, [2,3,4], log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset, (x, y) -> (x, y))

    res = Dict(zip(res_tmp[1], res_tmp[2]))

    for k in keys(ref)
        @test exp(res[k]) ≈ ref[k]
    end

     updated_logwms_tilde_kp2 = DualOptimalFiltering_proof.update_logweights_cost_to_go_WF(log.([0.6, 0.4]), [[1,2,1],[1,5,3]], 0.2*ones(3), [2,3,4])

     res_tmp = DualOptimalFiltering_proof.predict_logweights_cost_to_go_WF(updated_logwms_tilde_kp2, [[1,2,1],[1,5,3]], 0.2*ones(3), sum(0.2*ones(3)), [2,3,4], 0.5, log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset)

    res = Dict(zip(res_tmp[1], res_tmp[2]))

     for k in keys(ref)
         @test exp(res[k]) ≈ ref[k]
     end

    @test_nowarn DualOptimalFiltering_proof.compute_all_cost_to_go_functions_WF(α, data_1D, log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset)

    ref = DualOptimalFiltering_proof.compute_all_cost_to_go_functions_WF(α, data_1D, log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset)

    res = DualOptimalFiltering_proof.compute_all_log_cost_to_go_functions_WF_adaptive_precomputation_ar(α, data_1D, log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset, 0)

    for k in keys(ref[2])
        for l in eachindex(ref[2][k])
            @test exp(res[2][k][l]) ≈ ref[2][k][l]
        end
    end

    res = DualOptimalFiltering_proof.compute_all_log_cost_to_go_functions_WF_adaptive_precomputation_ar_pruning(α, data_1D, log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset, 0, (x,y) -> (x,y))

    for k in keys(ref[2])
        for l in eachindex(ref[2][k])
            @test exp(res[2][k][l]) ≈ ref[2][k][l]
        end
    end

    #sort of a rough test of the pre-computation, some -Inf or NA should propagate to the results if pre-computation was failing
    smmax = values(data) |> sum |> sum
    log_ν_ar = Array{Float64}(undef, smmax, smmax).*-Inf
    log_Cmmi_ar = Array{Float64}(undef, smmax, smmax).*-Inf
    log_binomial_coeff_ar_offset = Array{Float64}(undef,    smmax+1, smmax+1).*-Inf

    res = DualOptimalFiltering_proof.compute_all_cost_to_go_functions_WF_adaptive_precomputation_ar(α, data_1D, log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset, 0)

    summary_ = res[2] |> values |> x -> sum.(x) |> sum

    @test !isnan(summary_)
    @test isfinite(summary_)

    @test_nowarn DualOptimalFiltering_proof.WF_smoothing(α, data; silence = false)

    ref = DualOptimalFiltering_proof.WF_smoothing(α, data; silence = false)
    res = DualOptimalFiltering_proof.WF_smoothing_log_internals(α, data; silence = false)

    for k in keys(ref[2])
        refk = Dict(zip(ref[1][k], ref[2][k]))
        resk = Dict(zip(res[1][k], res[2][k]))
        for l in keys(refk)
            @test resk[l] ≈ refk[l] atol=10^(-14)
        end
    end

    res = DualOptimalFiltering_proof.WF_smoothing_pruning(α, data, (x,y)->(x,y); silence = false)


    for k in keys(ref[2])
        refk = Dict(zip(ref[1][k], ref[2][k]))
        resk = Dict(zip(res[1][k], res[2][k]))
        for l in keys(refk)
            @test resk[l] ≈ refk[l] atol=10^(-14)
        end
    end
end;
