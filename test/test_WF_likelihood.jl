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
        Ntimes_WF3 = 10
        time_step_WF3 = 0.1
        Random.seed!(4)
        wfchain_WF3 = rand(Dirichlet(K,0.3)) |> z-> DualOptimalFiltering.wright_fisher_PD1(z, 1.5, 50, Ntimes_WF3)[:,2:end]
        wfobs_WF3 = [rand(Multinomial(Pop_size_WF3, wfchain_WF3[:,k])) for k in 1:size(wfchain_WF3,2)] |> l -> hcat(l...)
        time_grid_WF3 = [k*time_step_WF3 for k in 0:(Ntimes_WF3-1)]
        data_WF3 = Dict(zip(time_grid_WF3 , [wfobs_WF3[:,t:t]' |> collect for t in 1:size(wfobs_WF3,2)]))
        return data_WF3, wfchain_WF3, α
    end
    data, wfchain_WF3, α = simulate_WF3_data()

    @test DualOptimalFiltering.t_WF(data[times[1]] |> vec, [0,0,0]) == [3, 9, 3]
    @test collect(DualOptimalFiltering.t_WF(data[times[1]] |> vec, [[0,0,0]])) == Array{Int64,1}[[3, 9, 3]]

    current_logw = -0.5*ones(5,5,5)
    current_logw_prime = -0.5*ones(5,5,5)
    current_Λ_max = [2,1,3]
    y = [2,3,1]

    @test_nowarn DualOptimalFiltering.next_logwms_from_log_wms_prime!(α, current_logw, current_logw_prime, current_Λ_max, y)

    log_ν_dict = Dict{Tuple{Int64,Int64},Float64}()
    log_Cmmi_dict = Dict{Tuple{Int64,Int64},Float64}()
    log_binomial_coeff_dict = Dict{Tuple{Int64,Int64},Float64}()
    @test_nowarn DualOptimalFiltering.precompute_next_terms!(0, 12, log_ν_dict, log_Cmmi_dict, log_binomial_coeff_dict, 3.1, 0.2)

    @test_nowarn DualOptimalFiltering.next_log_wms_prime_from_log_wms!(1.2, current_logw, current_logw_prime, DualOptimalFiltering.Λ_from_Λ_max(current_Λ_max), current_Λ_max, y, 0.4, log_ν_dict, log_Cmmi_dict, log_binomial_coeff_dict)

end;
