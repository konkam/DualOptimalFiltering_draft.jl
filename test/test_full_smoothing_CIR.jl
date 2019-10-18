
@testset "test high level statistical distances" begin
    @test_nowarn DualOptimalFiltering.θ_primeΔ(0.1, 1.1, 1.2)
    @test_nowarn DualOptimalFiltering.μmθk(3, 6, 1.2, 3.1, 1.4)

    Random.seed!(0)
    times_sim = range(0, stop = 20, length = 20)
    X = DualOptimalFiltering.generate_CIR_trajectory(times_sim, 3, 3., 0.5, 1);
    Y = map(λ -> rand(Poisson(λ),1), X);
    data = Dict(zip(times_sim, Y))
    δ, γ, σ, λ = 3., 0.5, 1., 1.
    Λ_of_t, logwms_of_t, θ_of_t, Λ_pred_of_t, logwms_pred_of_t, θ_pred_of_t = DualOptimalFiltering.filter_predict_CIR_logweights(δ, γ, σ, λ, data);

    i = 2
    times = Λ_of_t |> keys |> collect |> sort
    ti = times[i]
    tip1 = times[i+1]
    xip1 = 1.1

    predictive_dens_ip1 = DualOptimalFiltering.create_Gamma_mixture_density(δ, θ_pred_of_t[tip1], Λ_pred_of_t[tip1], exp.(logwms_pred_of_t[tip1]))

    # for i in eachindex(times)
    #     ti = times[i]
    #     tip1 = times[i+1]
    #     Δt = tip1-ti
    #     xip1 = 1.1
    #     println(DualOptimalFiltering.create_Gamma_mixture_density(δ, θ_pred_of_t[tip1], Λ_pred_of_t[tip1], exp.(logwms_pred_of_t[tip1]))(xip1)-DualOptimalFiltering.compute_normalisation_constant(xip1, θ_of_t[ti], DualOptimalFiltering.θ_primeΔ(Δt, γ, σ), exp.(logwms_of_t[ti]), Λ_of_t[ti], Δt, δ, γ, σ))
    # end

    @test_nowarn DualOptimalFiltering.backward_sampling_CIR(xip1, exp.(logwms_of_t[ti]), Λ_of_t[ti], predictive_dens_ip1(xip1), diff(times) |> mean, δ, γ, σ, θ_of_t[ti])

    @test_nowarn DualOptimalFiltering.backward_sampling_CIR_logw(xip1, logwms_of_t[ti], Λ_of_t[ti], predictive_dens_ip1(xip1), diff(times) |> mean, δ, γ, σ, θ_of_t[ti])

    pred_dens_val = predictive_dens_ip1(xip1)

    Δt = diff(times) |> mean
    θ_primeΔt = DualOptimalFiltering.θ_primeΔ(Δt, γ, σ)

    @test_nowarn DualOptimalFiltering.select_κM(xip1, 1.2, θ_primeΔt, 0.6, [0.2,0.3,0.2,0.1,0.1,0.05,0.05], 2:8, Δt, δ, γ, σ, pred_dens_val)

    @test_nowarn DualOptimalFiltering.select_κM(xip1, θ_of_t[ti], θ_primeΔt, 0.6, exp.(logwms_of_t[ti]), Λ_of_t[ti], Δt, δ, γ, σ, pred_dens_val)

    @test_nowarn DualOptimalFiltering.select_κM_logw(xip1, θ_of_t[ti], θ_primeΔt, 0.6, logwms_of_t[ti], Λ_of_t[ti], Δt, δ, γ, σ, pred_dens_val)

    predictive_dens_ip1_arb = DualOptimalFiltering.create_Gamma_mixture_density_arb(DualOptimalFiltering.RR(δ), DualOptimalFiltering.RR(θ_pred_of_t[tip1]), Λ_pred_of_t[tip1], exp.(DualOptimalFiltering.RR.(logwms_pred_of_t[tip1]) |> x -> x .- logsumexp(x)))

    pred_dens_val_arb = predictive_dens_ip1_arb(xip1)

    @test_nowarn DualOptimalFiltering.select_κM_logw_arb(DualOptimalFiltering.RR(xip1), θ_of_t[ti], θ_primeΔt, 0.6, DualOptimalFiltering.RR.(logwms_of_t[ti]) |> x -> x .- logsumexp(x), Λ_of_t[ti], Δt, δ, γ, σ, pred_dens_val_arb)

    @test_nowarn DualOptimalFiltering.backward_sampling_CIR(xip1, [1], [1], predictive_dens_ip1(xip1), Δt, δ, γ, σ, 1.)

    @test_nowarn DualOptimalFiltering.sample_1_trajectory_from_joint_smoothing_CIR(δ, γ, σ, Λ_of_t, logwms_of_t, θ_of_t, Λ_pred_of_t, logwms_pred_of_t, θ_pred_of_t, data)


    data = Dict(zip(range(0, stop = 2, length = 20), Y))
    δ, γ, σ, λ = 3., 0.5, 1., 1.
    Λ_of_t, logwms_of_t, θ_of_t, Λ_pred_of_t, logwms_pred_of_t, θ_pred_of_t = DualOptimalFiltering.filter_predict_CIR_logweights(δ, γ, σ, data);

    Random.seed!(0)
    @test_nowarn DualOptimalFiltering.sample_1_trajectory_from_joint_smoothing_CIR(δ, γ, σ, Λ_of_t, logwms_of_t, θ_of_t, Λ_pred_of_t, logwms_pred_of_t, θ_pred_of_t, data)


end;
