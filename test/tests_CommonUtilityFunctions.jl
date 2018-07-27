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
end;
