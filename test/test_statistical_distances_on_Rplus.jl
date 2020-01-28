using StatsBase, Distributions

@testset "test high level statistical distances" begin
    Random.seed!(0)
    smp = rand(Gamma(), 50)
    cc = DualOptimalFiltering_proof.create_gamma_mixture_cdf(1.0, 0.8, 1:3, (1:3)/sum(1:3))
    res = DualOptimalFiltering_proof.CvM_distance(cc, ff, 0, Inf)
    @test res[1] ≈ 0.9488374541739389 atol=10.0^(-10)
    @test res[2] ≈ 1.383536297199951e-8 atol=10.0^(-10)
    res = DualOptimalFiltering_proof.quadgk_singularities(sin, [0.5], 0, pi)
    @test res[1] ≈ 2 atol=res[2]
    res = DualOptimalFiltering_proof.quadgk_singularities(sin, [0.5, 0.5, 0.7, 0.9], 0, pi)
    @test res[1] ≈ 2 atol=res[2]
    res = DualOptimalFiltering_proof.quadgk_singularities(sin, [0.5, 0.5, 0.7, 0.9, 50], 0, pi)
    @test res[1] ≈ 2 atol=res[2]
end;
