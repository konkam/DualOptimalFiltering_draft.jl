@testset "test exact L2 distance functions" begin
    # @test_warn AssertionError DualOptimalFiltering.log_int_prod_2_Gammas(0.5, 0.5, 0.5, 1)
    @test DualOptimalFiltering.log_int_prod_2_Gammas(1, 1, 1, 1) == -log(2)
    @test DualOptimalFiltering.log_int_prod_2_Gammas(2, 0.5, 1, 0.5) == log(0.5^2*0.5 )
    @test DualOptimalFiltering.int_prod_2_Gammas(1, 1, 1, 1) == 0.5
    @test DualOptimalFiltering.int_prod_2_Gammas(2, 0.5, 1, 0.5) ≈ 0.5^2*0.5

    @test_nowarn DualOptimalFiltering.log_L2_dist_Gamma_mixtures(log.([0.2,0.3,0.5]), [1,2,3], [1,2,4], log.([0.6,0.4]), [1,2], [4,2])
    @test !isfinite(DualOptimalFiltering.log_L2_dist_Gamma_mixtures(log.([0.2,0.3,0.5]), [1,2,3], [1,2,4], log.([0.2,0.3,0.5]), [1,2,3], [1,2,4]))

    @test DualOptimalFiltering.lmvbeta([1,1]) == 0.0
    @test DualOptimalFiltering.log_int_prod_2_Dir([1,1], [1,1]) == 0.0
    @test DualOptimalFiltering.log_int_prod_2_Dir([1,1,1], [1,1,1]) ≈ 0.6931471805599453

    @test_nowarn DualOptimalFiltering.log_L2_dist_Dirichlet_mixtures(log.([0.2,0.3,0.5]), [[1,2,3],[1,2,4],[5,3,2]], log.([0.6,0.4]), [[1,2,1],[4,2,3]])
    @test !isfinite(DualOptimalFiltering.log_L2_dist_Dirichlet_mixtures(log.([0.2,0.3,0.5]), [[1,2,3],[1,2,4],[5,3,2]], log.([0.2,0.3,0.5]), [[1,2,3],[1,2,4],[5,3,2]])
)
end;
