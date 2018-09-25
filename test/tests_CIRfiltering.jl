@test DualOptimalFiltering.create_mixture_hpi(0.5, 1., [3], [1.])(3) |> isreal

@test DualOptimalFiltering.compute_quantile_mixture_hpi(1.5, 0.4, [3,4], [0.3, 0.7], 0.05) |> isreal

@test DualOptimalFiltering.compute_quantile_mixture_hpi(1., 1., [0], [1], 0.025) ≈ quantile(Gamma(1./2 + 0, 1/1.),0.025) atol=10.0^(-5)

# DualOptimalFiltering.update_CIR_params([0.5, 0.5], 1., θ, 1., [0, 1], [5]) |> println

@testset "update CIR Tests" begin
    tmp = DualOptimalFiltering.update_CIR_params([0.5, 0.5], 1., 1.5, 1., [0, 1], [5])
    @test tmp[1] == 2.5
    @test tmp[2] == [5, 6]
    @test tmp[3][1] ≈ 0.131579 atol=10.0^(-5)
    @test tmp[3][2] ≈ 0.868421 atol=10.0^(-5)
    tmp = DualOptimalFiltering.update_CIR_params_logweights(log.([0.5, 0.5]), 1., 1.5, 1., [0, 1], [5])
    @test tmp[1] == 2.5
    @test tmp[2] == [5, 6]
    @test exp(tmp[3][1]) ≈ 0.131579 atol=10.0^(-5)
    @test exp(tmp[3][2]) ≈ 0.868421 atol=10.0^(-5)
end;

@testset "predict CIR Tests" begin
    tmp = DualOptimalFiltering.predict_CIR_params([1.], 1., 2., 1., 1., [5], 1.)
    @test tmp[1] ≈ 1.072578883495754 atol=10.0^(-5)
    @test tmp[2] == 0:5
    [@test tmp[3][k] ≈ [0.686096, 0.268465, 0.0420196, 0.0032884, 0.000128673, 2.01396e-6][k] atol=10.0^(-5) for k in 1:6]
    tmp = DualOptimalFiltering.predict_CIR_params_logweights(log.([1.]), 1., 2., 1., 1., [5], 1.)
    @test tmp[1] ≈ 1.072578883495754 atol=10.0^(-5)
    @test tmp[2] == 0:5
    [@test exp(tmp[3][k]) ≈ [0.686096, 0.268465, 0.0420196, 0.0032884, 0.000128673, 2.01396e-6][k] atol=10.0^(-5) for k in 1:6]
end;

@test length(DualOptimalFiltering.generate_CIR_trajectory(linspace(0,2,20), 3, 3., 0.5, 1)) == 20

@testset "CIR filtering tests" begin
    srand(0)
    X = DualOptimalFiltering.generate_CIR_trajectory(linspace(0,2,20), 3, 3., 0.5, 1);
    Y = map(λ -> rand(Poisson(λ),10), X);
    data = Dict(zip(linspace(0,2,20), Y))
    Λ_of_t, wms_of_t, θ_of_t = DualOptimalFiltering.filter_CIR_debug(3., 0.5, 1.,1.,data);
    [@test isreal(k) for k in Λ_of_t |> keys]
    [@test isinteger(sum(k)) for k in Λ_of_t |> values]
    [@test isreal(sum(k)) for k in wms_of_t |> values]
    [@test isreal(k) for k in θ_of_t |> values]
    Λ_of_t_logweights, logweights_of_t, θ_of_t_logweights = DualOptimalFiltering.filter_CIR_logweights(3., 0.5, 1.,1.,data);
    times = logweights_of_t |> keys |> collect |> sort
    @test maximum([maximum(Λ_of_t[t] - Λ_of_t_logweights[t]) for t in times]) == 0
    @test θ_of_t_logweights == θ_of_t
    for t in times
        for i in 1:length(logweights_of_t[t])
            @test logweights_of_t[t][i] ≈ wms_of_t[t][i] atol=10.0^(-10)
        end
    end
end;

@testset "CIR approximate filtering tests" begin
    srand(4)

    X = DualOptimalFiltering.generate_CIR_trajectory(linspace(0,2,20), 3, 3., 0.5, 1);
    Y = map(λ -> rand(Poisson(λ),10), X);
    data = Dict(zip(linspace(0,2,20), Y))


    Λ_of_t, wms_of_t, θ_of_t = DualOptimalFiltering.filter_CIR_keep_fixed_number(3., 0.5, 1.,1., data, 10);

    @test length(keys(Λ_of_t)) == 20
    @test length(keys(wms_of_t)) == 20

    Λ_of_t, wms_of_t = DualOptimalFiltering.filter_CIR_keep_above_threshold(3., 0.5, 1.,1., data, 0.0001)
    @test length(keys(Λ_of_t)) == 20
    @test length(keys(wms_of_t)) == 20

    Λ_of_t, wms_of_t = DualOptimalFiltering.filter_CIR_fixed_fraction(3., 0.5, 1.,1., data, .99)
    @test length(keys(Λ_of_t)) == 20
    @test length(keys(wms_of_t)) == 20
end;
