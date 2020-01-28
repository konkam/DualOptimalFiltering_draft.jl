using DataStructures, StatsFuns, SpecialFunctions

@testset "Test common utility functions" begin
    @test DualOptimalFiltering_proof.normalise(1:4) == (1:4)/10

    @test_throws ErrorException DualOptimalFiltering_proof.normalise(Float64[])

    @test DualOptimalFiltering_proof.lognormalise(log.(1:4)) |> logsumexp ≈ 0. atol=10^(-15)
    @test_throws ErrorException DualOptimalFiltering_proof.lognormalise(Float64[])

    tmp = DualOptimalFiltering_proof.get_quantiles_from_mass(0.95)
    @test tmp[1] ≈ 0.025 atol=10.0^(-10)
    @test tmp[2] ≈ 0.975 atol=10.0^(-10)
    res = Array{Array{Float64,1},1}([[1,2,3],[1,2,3],[1,2,3]]) |> DualOptimalFiltering_proof.flat2
    for i in eachindex(res)
        @test res[i] ≈ [1.0,  2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0][i]
    end
    res = [[1,2,3],[1,2,3],[1,2,3]] |> DualOptimalFiltering_proof.flat2
    for i in eachindex(res)
        @test res[i] == [1,  2, 3, 1, 2, 3, 1, 2, 3][i]
    end
    ff = DualOptimalFiltering_proof.create_gamma_mixture_pdf(1.0, 0.8, 1:3, (1:3)/sum(1:3))
    @test ff(0.3) ≈ 0.07920359132797428 atol=10.0^(-10)
    cc = DualOptimalFiltering_proof.create_gamma_mixture_cdf(1.0, 0.8, 1:3, (1:3)/sum(1:3))
    @test cc(0.3) ≈ 0.01541885435694441 atol=10.0^(-10)

    function simulate_WF3_data()
        K = 3
        α = ones(K)
        sα = sum(α)
        Pop_size_WF3 = 15
        Ntimes_WF3 = 5
        time_step_WF3 = 0.1
        Random.seed!(4)
        wfchain_WF3 = rand(Dirichlet(K,0.3)) |> z-> DualOptimalFiltering_proof.wright_fisher_PD1(z, 1.5, 50, Ntimes_WF3)[:,2:end]
        wfobs_WF3 = [rand(Multinomial(Pop_size_WF3, wfchain_WF3[:,k])) for k in 1:size(wfchain_WF3,2)] |> l -> hcat(l...)
        time_grid_WF3 = [k*time_step_WF3 for k in 0:(Ntimes_WF3-1)]
        data_WF3 = Dict(zip(time_grid_WF3 , [wfobs_WF3[:,t:t]' |> collect for t in 1:size(wfobs_WF3,2)]))
        return data_WF3, wfchain_WF3, α
    end
    data, wfchain_WF3, α = simulate_WF3_data()

    @test DualOptimalFiltering_proof.assert_constant_time_step_and_compute_it(data) ≈ 0.1

    @test DualOptimalFiltering_proof.test_equal_spacing_of_observations(data; override = false, digits_after_comma_for_time_precision = 4) == nothing
    data[6.0] = zeros(2,2)
    @test_throws ErrorException DualOptimalFiltering_proof.test_equal_spacing_of_observations(data; override = false, digits_after_comma_for_time_precision = 4)

    @test_nowarn DualOptimalFiltering_proof.log_pochammer(0.5, 5)
    @test DualOptimalFiltering_proof.log_pochammer(0.5, 0) == 0
    @test DualOptimalFiltering_proof.log_pochammer(0.5, 1) == log(0.5)
    @test  DualOptimalFiltering_proof.log_pochammer_rec(0.5, 5) == DualOptimalFiltering_proof.log_pochammer(0.5, 5)
    @test  DualOptimalFiltering_proof.log_pochammer_rec(1.5, 50) == DualOptimalFiltering_proof.log_pochammer(1.5, 50)

    for k in 1:10
        Random.seed!(k)
        x = sample(1:1000, 20)
        n = sample(1:10)
        @test sort(DualOptimalFiltering_proof.kmax(x, k)) == DualOptimalFiltering_proof.kmax_safe_but_slow(x, k)
    end

    @test DualOptimalFiltering_proof.kmax_safe_but_slow(1:3, 3) == 1:3
    @test DualOptimalFiltering_proof.kmax(1:3, 3) == 1:3
    @test_throws ErrorException DualOptimalFiltering_proof.kmax_safe_but_slow(1:2, 3)
    @test_throws ErrorException DualOptimalFiltering_proof.kmax(1:2, 3)

    aa = aa = Accumulator(Dict(zip(["a", "b", "c"], [0.3,0.8,0.9])))

    @test DualOptimalFiltering_proof.normalise(aa) |> values |> sum == 1


    aa = Dict(i => j for (i, j) in zip(1:4, [[5,6],[2],[0.5,0.6,3],[2,2]]))
    @test_nowarn DualOptimalFiltering_proof.convert_weights_to_logweights(aa)
    @test_nowarn DualOptimalFiltering_proof.convert_logweights_to_weights(aa)

    res = aa |> DualOptimalFiltering_proof.convert_weights_to_logweights |> DualOptimalFiltering_proof.convert_logweights_to_weights
    for k in keys(aa)
        @test res[k] == aa[k]
    end
    res = aa |> DualOptimalFiltering_proof.convert_logweights_to_weights |> DualOptimalFiltering_proof.convert_weights_to_logweights
    for k in keys(aa)
        @test res[k] == aa[k]
    end

    @test DualOptimalFiltering_proof.lgamma_local(2) ≈ SpecialFunctions.logabsgamma(2)[1]

    @test DualOptimalFiltering_proof.lgamma_local(5) ≈ SpecialFunctions.logabsgamma(5)[1]
end;
