@testset "update WF Tests" begin
    srand(2)
    p = rand(Dirichlet(ones(5)*1./5))
    data = rand(Multinomial(10, p),7)'
    tmp = DualOptimalFiltering.update_WF_params_debug([1., 0.1], ones(5)*1./5, [0*collect(1:5), collect(1:5)], data)
    @test tmp[1] == [[4, 0, 17, 23, 26], [5, 2, 20, 27, 31]]
    # @test tmp[2] == [0.578617, 0.421383]
    @test tmp[2][1] ≈ 0.578617 atol=10.0^(-5)
    @test tmp[2][2] ≈ 0.421383 atol=10.0^(-5)
end;

@testset "predict WF Tests" begin
    tmp = DualOptimalFiltering.WF_prediction_for_one_m_debug_mem2([0,4,2,1], ones(4) |> sum, 1.5)
    # @test tmp[1] ≈ 1.072578883495754 atol=10.0^(-5)
    @test length(keys(tmp)) == 30
    @test sum(values(tmp)) ≈ 1.0 atol=10.0^(-5)

    tmp = DualOptimalFiltering.WF_prediction_for_one_m_debug_mem2([0,4,2,1], ones(4) |> sum, 1.5; wm = 0.5)
    @test sum(values(tmp)) ≈ 0.5 atol=10.0^(-5)

    Λ_test = [[6, 9], [5, 10], [4, 27], [3, 2], [9, 8], [1, 7], [5, 17], [8, 3], [2, 18], [3, 28]]
    num_m = length(Λ_test)


    tmp = DualOptimalFiltering.predict_WF_params_debug_mem2(rand(Dirichlet(ones(num_m)*1./num_m)), 2., Λ_test, 1.)
    @test sum(tmp[2]) ≈ 1.0 atol=10.0^(-5)
    # [@test tmp[3][k] ≈ [0.686096, 0.268465, 0.0420196, 0.0032884, 0.000128673, 2.01396e-6][k] atol=10.0^(-5) for k in 1:6]
end;


@test size(rand(Dirichlet(4,0.3)) |> z-> DualOptimalFiltering.wright_fisher_PD1(z, 1.5, 50, 4)[:,2:end]*10 |> x -> round.(x) |> x -> Int64.(x) ) == (4,4)


@testset "CIR filtering tests" begin
    srand(4)
    wfchain = rand(Dirichlet(4,0.3)) |> z-> DualOptimalFiltering.wright_fisher_PD1(z, 1.5, 50, 3)[:,2:end]*10 |> x -> round.(x) |> x -> Int64.(x)
    data = Dict(zip(linspace(0, 5, size(wfchain,2)), [wfchain[:,t:t]' for t in 1:size(wfchain,2)]))
    Λ_of_t, wms_of_t = DualOptimalFiltering.filter_WF_mem2(ones(4), data)
    @test length(keys(Λ_of_t)) == 3
    @test length(keys(wms_of_t)) == 3
    @test_throws AssertionError filter_WF(ones(2), data)
end;
