@testset "test the wf likelihood functions" begin

    @test_nowarn DualOptimalFiltering.logμπh_WF(ones(3), [0,0,0], [2,3,1])
    @test_nowarn DualOptimalFiltering.logμπh_WF(ones(3), [1,5,2], [2,3,1])

    @test DualOptimalFiltering.compute_next_Λ_max([2,1,3], [1,1,1]) == [3,2,4]
    # @test DualOptimalFiltering.compute_next_Λ_max([2,1,3], Int64.(zeros(2,3))) == [2,1,3]

    function simulate_WF3_data()
        K = 3
        α = ones(K)
        sα = sum(α)
        Pop_size_WF3 = 15
        Ntimes_WF3 = 3
        time_step_WF3 = 0.1
        Random.seed!(4)
        wfchain_WF3 = rand(Dirichlet(K,0.3)) |> z-> DualOptimalFiltering.wright_fisher_PD1(z, 1.5, 50, Ntimes_WF3)[:,2:end]
        wfobs_WF3 = [rand(Multinomial(Pop_size_WF3, wfchain_WF3[:,k])) for k in 1:size(wfchain_WF3,2)] |> l -> hcat(l...)
        time_grid_WF3 = [k*time_step_WF3 for k in 0:(Ntimes_WF3-1)]
        data_WF3 = Dict(zip(time_grid_WF3 , [wfobs_WF3[:,t] for t in 1:size(wfobs_WF3,2)]))
        return data_WF3, wfchain_WF3, α
    end
    data, wfchain_WF3, α = simulate_WF3_data()
    times = data |> keys |> collect |> sort

    @test vec(collect(DualOptimalFiltering.Λprime_i_from_Λimax([0,1,0]))) == [(0, 0, 0), (0, 1, 0)]

    @test vec(collect(DualOptimalFiltering.Λprime_i_from_Λ([[0,1,0], [0,0,0]]))) == [(0, 0, 0), (0, 1, 0)]

    @test DualOptimalFiltering.t_WF(data[times[1]] |> vec, [0,0,0]) == data[times[1]]
    @test collect(DualOptimalFiltering.t_WF(data[times[1]] |> vec, [[0,0,0]])) == Array{Int64,1}[data[times[1]] |> vec]
    @test_nowarn DualOptimalFiltering.t_WF(data[times[1]], [[0,0,0], [1,2,1]])
    @test_nowarn DualOptimalFiltering.Λi_from_Λprime_im1(data[times[1]], [[0,0,0], [1,2,1]])

    current_logw = -0.5*ones(5,5,5)
    current_logw_prime = -0.5*ones(5,5,5)
    current_Λ_max = [2,1,3]
    current_Λ = DualOptimalFiltering.Λ_from_Λ_max(current_Λ_max)
    y = [2,3,1]

    @test_nowarn DualOptimalFiltering.update_logwms_to_i_from_log_wms_prime_im1!(α, current_logw, current_logw_prime, current_Λ, y)

    log_ν_dict = Dict{Tuple{Int64,Int64},Float64}()
    log_Cmmi_dict = Dict{Tuple{Int64,Int64},Float64}()
    log_binomial_coeff_dict = Dict{Tuple{Int64,Int64},Float64}()
    @test_nowarn DualOptimalFiltering.precompute_next_terms!(0, 12, log_ν_dict, log_Cmmi_dict, log_binomial_coeff_dict, 3.1, 0.2)

    current_logw_prime_tmp = deepcopy(current_logw_prime)

    @test_nowarn DualOptimalFiltering.predict_logwms_prime_to_i_from_logwms_i!(1.2, current_logw, current_logw_prime, DualOptimalFiltering.Λ_from_Λ_max(current_Λ_max), current_Λ_max, 0.4, log_ν_dict, log_Cmmi_dict, log_binomial_coeff_dict)

    @test_nowarn DualOptimalFiltering.predict_logwms_prime_to_i_from_logwms_i2!(1.2, current_logw, current_logw_prime_tmp, DualOptimalFiltering.Λ_from_Λ_max(current_Λ_max), 0.4, log_ν_dict, log_Cmmi_dict, log_binomial_coeff_dict)

    for m in DualOptimalFiltering.Λ_from_Λ_max(current_Λ_max)
        @test current_logw_prime_tmp[(m .+ 1)...] == current_logw_prime_tmp[(m .+ 1)...]
    end

    @test_nowarn DualOptimalFiltering.WF_loglikelihood_adaptive_precomputation(α, data; silence = false)

    @test DualOptimalFiltering.sum_Λ_max_from_Λ(DualOptimalFiltering.Λ_from_Λ_max(current_Λ_max)) == 6

    @test DualOptimalFiltering.Λ_max_from_Λ(DualOptimalFiltering.Λ_from_Λ_max(current_Λ_max)) == [2,1,3]

    res = DualOptimalFiltering.WF_loglikelihood_adaptive_precomputation_pruning(α, data, (x,y) -> (x,y); silence = false)

    ref = DualOptimalFiltering.WF_loglikelihood_adaptive_precomputation(α, data; silence = false)

    for t in times
        @test res[t] ≈ ref[t]
    end

    # @test [res[t] for t in times] == [ref[t] for t in times]

    @test_nowarn DualOptimalFiltering.log_likelihood_WF_keep_fixed_number(α, data, 3; silence = false)

    @test_nowarn DualOptimalFiltering.log_likelihood_WF_fixed_fraction(α, data, 0.9; silence = false)

    @test_nowarn DualOptimalFiltering.log_likelihood_WF_keep_above_threshold(α, data, 0.001; silence = false)

end;


@testset "test the wf likelihood functions with array storage for precomputed coeffs" begin

    function simulate_WF3_data()
        K = 3
        α = ones(K)
        sα = sum(α)
        Pop_size_WF3 = 15
        Ntimes_WF3 = 3
        time_step_WF3 = 0.1
        Random.seed!(4)
        wfchain_WF3 = rand(Dirichlet(K,0.3)) |> z-> DualOptimalFiltering.wright_fisher_PD1(z, 1.5, 50, Ntimes_WF3)[:,2:end]
        wfobs_WF3 = [rand(Multinomial(Pop_size_WF3, wfchain_WF3[:,k])) for k in 1:size(wfchain_WF3,2)] |> l -> hcat(l...)
        time_grid_WF3 = [k*time_step_WF3 for k in 0:(Ntimes_WF3-1)]
        data_WF3 = Dict(zip(time_grid_WF3 , [wfobs_WF3[:,t] for t in 1:size(wfobs_WF3,2)]))
        return data_WF3, wfchain_WF3, α
    end
    data, wfchain_WF3, α = simulate_WF3_data()
    times = data |> keys |> collect |> sort


    current_logw = -0.5*ones(5,5,5)
    current_logw_prime = -0.5*ones(5,5,5)
    current_Λ_max = [2,1,3]
    current_Λ = DualOptimalFiltering.Λ_from_Λ_max(current_Λ_max)
    y = [2,3,1]

    smmax = values(data) |> sum |> sum
    log_ν_ar = Array{Float64}(undef, smmax, smmax)
    log_Cmmi_ar = Array{Float64}(undef, smmax, smmax)
    log_binomial_coeff_ar_offset = Array{Float64}(undef, smmax+1, smmax+1)

    @test_nowarn DualOptimalFiltering.precompute_next_terms_ar!(0, 12, log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset, 3.1, 0.2)

    log_ν_dict = Dict{Tuple{Int64,Int64},Float64}()
    log_Cmmi_dict = Dict{Tuple{Int64,Int64},Float64}()
    log_binomial_coeff_dict = Dict{Tuple{Int64,Int64},Float64}()
    @test_nowarn DualOptimalFiltering.precompute_next_terms!(0, 12, log_ν_dict, log_Cmmi_dict, log_binomial_coeff_dict, 3.1, 0.2)

    for k in keys(log_ν_dict)
        @test log_ν_dict[k] == log_ν_ar[k...]
    end
    for k in keys(log_Cmmi_dict)
        @test log_Cmmi_dict[k] == log_Cmmi_ar[k...]
    end
    for k in keys(log_binomial_coeff_dict)
        @test log_binomial_coeff_dict[k] == log_binomial_coeff_ar_offset[(k .+ 1)...]
    end

    current_logw_prime_tmp = deepcopy(current_logw_prime)

    @test_nowarn DualOptimalFiltering.predict_logwms_prime_to_i_from_logwms_i!(1.2, current_logw, current_logw_prime, DualOptimalFiltering.Λ_from_Λ_max(current_Λ_max), current_Λ_max, 0.4, log_ν_dict, log_Cmmi_dict, log_binomial_coeff_dict)

    @test_nowarn DualOptimalFiltering.predict_logwms_prime_to_i_from_logwms_i!(1.2, current_logw, current_logw_prime_tmp, DualOptimalFiltering.Λ_from_Λ_max(current_Λ_max), current_Λ_max, 0.4, log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset)

    for m in DualOptimalFiltering.Λ_from_Λ_max(current_Λ_max)
        @test current_logw_prime_tmp[(m .+ 1)...] == current_logw_prime_tmp[(m .+ 1)...]
    end

    @test_nowarn DualOptimalFiltering.predict_logwms_prime_to_i_from_logwms_i2!(1.2, current_logw, current_logw_prime_tmp, DualOptimalFiltering.Λ_from_Λ_max(current_Λ_max), 0.4, log_ν_dict, log_Cmmi_dict, log_binomial_coeff_dict)

    @test_nowarn DualOptimalFiltering.predict_logwms_prime_to_i_from_logwms_i2!(1.2, current_logw, current_logw_prime_tmp, DualOptimalFiltering.Λ_from_Λ_max(current_Λ_max), 0.4, log_ν_ar, log_Cmmi_ar, log_binomial_coeff_ar_offset)

    for m in DualOptimalFiltering.Λ_from_Λ_max(current_Λ_max)
        @test current_logw_prime_tmp[(m .+ 1)...] == current_logw_prime_tmp[(m .+ 1)...]
    end

    ref = DualOptimalFiltering.WF_loglikelihood_adaptive_precomputation(α, data; silence = false)

    @test DualOptimalFiltering.WF_loglikelihood_adaptive_precomputation_ar(α, data; silence = false) == ref

    res = DualOptimalFiltering.WF_loglikelihood_adaptive_precomputation_pruning_ar(α, data, (x,y) -> (x,y); silence = false)

    ref = DualOptimalFiltering.WF_loglikelihood_adaptive_precomputation(α, data; silence = false)

    for t in times
        @test res[t] ≈ ref[t]
    end

    res2 = DualOptimalFiltering.WF_loglikelihood_from_adaptive_filtering(α, data, (x, y)-> (x, y); silence = false)

    for t in times
        @test res2[t] ≈ ref[t]
    end

    # @test [res[t] for t in times] == [ref[t] for t in times]

    @test_nowarn DualOptimalFiltering.log_likelihood_WF_keep_fixed_number(α, data, 3; silence = false)

    @test_nowarn DualOptimalFiltering.log_likelihood_WF_fixed_fraction(α, data, 0.9; silence = false)

    @test_nowarn DualOptimalFiltering.log_likelihood_WF_keep_above_threshold(α, data, 0.001; silence = false)

end;
