@testset "Test common utility functions" begin
    @test DualOptimalFiltering.normalise(1:4) == (1:4)/10
    tmp = DualOptimalFiltering.get_quantiles_from_mass(0.95)
    @test tmp[1] ≈ 0.025 atol=10.0^(-10)
    @test tmp[2] ≈ 0.975 atol=10.0^(-10)
    res = Array{Array{Float64,1},1}([[1,2,3],[1,2,3],[1,2,3]]) |> DualOptimalFiltering.flat2
    for i in 1:length(res)
        @test res[i] ≈ [1.0,  2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0][i]
    end
    res = [[1,2,3],[1,2,3],[1,2,3]] |> DualOptimalFiltering.flat2
    for i in 1:length(res)
        @test res[i] == [1,  2, 3, 1, 2, 3, 1, 2, 3][i]
    end
    ff = DualOptimalFiltering.create_gamma_mixture_pdf(1.0, 0.8, 1:3, (1:3)/sum(1:3))
    @test ff(0.3) ≈ 0.07920359132797428 atol=10.0^(-10)
    cc = DualOptimalFiltering.create_gamma_mixture_cdf(1.0, 0.8, 1:3, (1:3)/sum(1:3))
    @test cc(0.3) ≈ 0.01541885435694441 atol=10.0^(-10)

    function simulate_WF3_data()
        K = 3
        α = ones(K)
        sα = sum(α)
        Pop_size_WF3 = 15
        Ntimes_WF3 = 5
        time_step_WF3 = 0.1
        Random.seed!(4)
        wfchain_WF3 = rand(Dirichlet(K,0.3)) |> z-> DualOptimalFiltering.wright_fisher_PD1(z, 1.5, 50, Ntimes_WF3)[:,2:end]
        wfobs_WF3 = [rand(Multinomial(Pop_size_WF3, wfchain_WF3[:,k])) for k in 1:size(wfchain_WF3,2)] |> l -> hcat(l...)
        time_grid_WF3 = [k*time_step_WF3 for k in 0:(Ntimes_WF3-1)]
        data_WF3 = Dict(zip(time_grid_WF3 , [wfobs_WF3[:,t:t]' |> collect for t in 1:size(wfobs_WF3,2)]))
        return data_WF3, wfchain_WF3, α
    end
    data, wfchain_WF3, α = simulate_WF3_data()

    @test DualOptimalFiltering.test_equal_spacing_of_observations(data; override = false, digits_after_comma_for_time_precision = 4) == nothing
    data[6.0] = zeros(2,2)
    @test_throws ErrorException DualOptimalFiltering.test_equal_spacing_of_observations(data; override = false, digits_after_comma_for_time_precision = 4)

    @test_nowarn DualOptimalFiltering.log_pochammer(0.5, 5)
    @test DualOptimalFiltering.log_pochammer(0.5, 0) == 0
    @test DualOptimalFiltering.log_pochammer(0.5, 1) == log(0.5)
end;
