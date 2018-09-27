@testset "test kernel function" begin
    @test DualOptimalFiltering.dirichletkernel_oneval([0.2,0.3,0.5], [0.16,  0.64,  0.2], 1.2) ≈ 2.3904148155639433 atol=10.0^(-5)
    @test DualOptimalFiltering.dirichletkernel_oneval([0.0,0.5,0.5], [0.16,  0.64,  0.2], 1.2) == 0
    Random.seed!(1)
    X  = rand(Dirichlet(ones(4)),10)'
    w = ones(size(X, 1))
    dirichletkernel([0.2,0.3,0.4, 0.1], X, 1.2, w, size(X, 1))
    res = [7.85877, 7.34332, 8.70397, 8.05658, 4.97756, 8.25494, 8.63194, 7.72478, 7.36807, 8.70078]
    for i in 1:length(w)
        @test w[i] .≈ res[i] atol=10.0^(-5)
    end
end;


@testset "test bandwidth selection functions" begin
    Random.seed!(1)
    X  = rand(Dirichlet(ones(4)),10)'
    @test DualOptimalFiltering.midrange(X) ≈ 0.4139298224327084  atol=10.0^(-5)
    @test  DualOptimalFiltering.lcv(X, dirichletkernel, DualOptimalFiltering.midrange(X), ones(size(X, 1)), size(X, 1)) ≈ -16.896937547749225  atol=10.0^(-5)
    @test DualOptimalFiltering.bwlcv(X, dirichletkernel) ≈ 0.4072186731744858  atol=10.0^(-5)
    @test DualOptimalFiltering.bwlcv_large_bounds(X, dirichletkernel) ≈ 1.082464656143667  atol=10.0^(-5)
    @test DualOptimalFiltering.minus_log_leaveoneout(X, dirichletkernel, DualOptimalFiltering.midrange(X), ones(size(X, 1)), size(X, 1)) ≈ -37.72556390035346  atol=10.0^(-5)
    @test DualOptimalFiltering.bwloo_large_bounds(X, dirichletkernel) ≈ 1.082464656143667  atol=10.0^(-5)
end;
