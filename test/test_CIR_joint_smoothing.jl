
@testset "test full smoothing CIR" begin

    δ = 3.1

    @test_nowarn DualOptimalFiltering.θ_primeΔ(0.1, 1.1, 1.2)
    @test_nowarn DualOptimalFiltering.μmθk(3, 6, 1.2, δ, 1.4)

    @test DualOptimalFiltering.NegativeBinomial_logpdf(5, 0.6, 0.2) ≈ logpdf(NegativeBinomial(0.6, 0.2), 5)

    @test DualOptimalFiltering.μmθk(3, 6, 1.2, δ, 1.4) ≈ DualOptimalFiltering.μmθk_slow(3, 6, 1.2, δ, 1.4)

    @test DualOptimalFiltering.logμmθk2(3, 6, 1.2, δ, 1.4) ≈ DualOptimalFiltering.logμmθk(3, 6, 1.2, δ, 1.4)
    @test DualOptimalFiltering.logμmθk2(3, 6, 1.2, δ, 1.4) ≈ DualOptimalFiltering.logμmθk3(3, 6, 1.2, δ, 1.4)
    @test DualOptimalFiltering.logμmθk2(3, 6, 1.2, δ, 1.4) ≈ DualOptimalFiltering.logμmθk4(3, 6, 1.2, δ, 1.4)
    @test DualOptimalFiltering.logμmθk2(0, 6, 1.2, δ, 1.4) ≈ DualOptimalFiltering.logμmθk3(0, 6, 1.2, δ, 1.4)
    @test DualOptimalFiltering.logμmθk2(0, 6, 1.2, δ, 1.4) ≈ DualOptimalFiltering.logμmθk4(0, 6, 1.2, δ, 1.4)

    @test_nowarn DualOptimalFiltering.precompute_log_pochammer_for_logμmθk(δ, 50, 50)

    precomputed_terms = DualOptimalFiltering.precompute_log_pochammer_for_logμmθk(δ, 50, 50)

    @test DualOptimalFiltering.log_pochammer_precomputed(5, 10, precomputed_terms) == DualOptimalFiltering.log_pochammer(δ/2+5, 10)

    @test DualOptimalFiltering.logμmθk2(3, 6, 1.2, δ, 1.4) ≈ DualOptimalFiltering.logμmθk5(3, 6, 1.2, δ, 1.4, precomputed_terms)
    @test DualOptimalFiltering.logμmθk2(0, 6, 1.2, δ, 1.4) ≈ DualOptimalFiltering.logμmθk5(0, 6, 1.2, δ, 1.4, precomputed_terms)


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

    Δt = times |> diff |> mean

    @test_nowarn DualOptimalFiltering.compute_normalisation_constant(xip1, θ_of_t[ti], DualOptimalFiltering.θ_primeΔ(Δt, γ, σ), exp.(logwms_of_t[ti]), Λ_of_t[ti], Δt, δ, γ, σ)

    ref = DualOptimalFiltering.compute_normalisation_constant(xip1, θ_of_t[ti], DualOptimalFiltering.θ_primeΔ(Δt, γ, σ), exp.(logwms_of_t[ti]), Λ_of_t[ti], Δt, δ, γ, σ)

    mmax = maximum(maximum.(values(Λ_of_t)))
    precomputed_terms = Array{Float64, 2}(undef, mmax+1, 10^3)

    res = DualOptimalFiltering.compute_normalisation_constant_adaptive_precomputation(xip1, θ_of_t[ti], DualOptimalFiltering.θ_primeΔ(Δt, γ, σ), exp.(logwms_of_t[ti]), Λ_of_t[ti], Δt, δ, γ, σ, precomputed_terms, 0, mmax)

    @test ref == res[1]


    @test_nowarn DualOptimalFiltering.backward_sampling_CIR(xip1, exp.(logwms_of_t[ti]), Λ_of_t[ti], predictive_dens_ip1(xip1), diff(times) |> mean, δ, γ, σ, θ_of_t[ti])

    Random.seed!(0)
    ref = DualOptimalFiltering.backward_sampling_CIR(xip1, exp.(logwms_of_t[ti]), Λ_of_t[ti], predictive_dens_ip1(xip1), diff(times) |> mean, δ, γ, σ, θ_of_t[ti])

    precomputed_terms = DualOptimalFiltering.precompute_log_pochammer_for_logμmθk(δ, 500, 500)


    Random.seed!(0)
    res = DualOptimalFiltering.backward_sampling_CIR_precomputed(xip1, exp.(logwms_of_t[ti]), Λ_of_t[ti], predictive_dens_ip1(xip1), diff(times) |> mean, δ, γ, σ, θ_of_t[ti], precomputed_terms)

    @test ref == res

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

    @test_nowarn DualOptimalFiltering.sample_1_trajectory_from_joint_smoothing_CIR_logweights(δ, γ, σ, Λ_of_t, logwms_of_t, θ_of_t, Λ_pred_of_t, logwms_pred_of_t, θ_pred_of_t, data)


    data = Dict(zip(range(0, stop = 2, length = 20), Y))
    δ, γ, σ, λ = 3., 0.5, 1., 1.
    Λ_of_t, logwms_of_t, θ_of_t, Λ_pred_of_t, logwms_pred_of_t, θ_pred_of_t = DualOptimalFiltering.filter_predict_CIR_logweights(δ, γ, σ, λ, data);

    Random.seed!(0)
    @test_nowarn DualOptimalFiltering.sample_1_trajectory_from_joint_smoothing_CIR_logweights(δ, γ, σ, Λ_of_t, logwms_of_t, θ_of_t, Λ_pred_of_t, logwms_pred_of_t, θ_pred_of_t, data)

    Random.seed!(0)
    ref = DualOptimalFiltering.sample_1_trajectory_from_joint_smoothing_CIR_logweights(δ, γ, σ, Λ_of_t, logwms_of_t, θ_of_t, Λ_pred_of_t, logwms_pred_of_t, θ_pred_of_t, data)

    Random.seed!(0)
    res = DualOptimalFiltering.sample_1_trajectory_from_joint_smoothing_CIR(δ, γ, σ, Λ_of_t, logwms_of_t |> DualOptimalFiltering.convert_logweights_to_weights, θ_of_t, Λ_pred_of_t, logwms_pred_of_t |> DualOptimalFiltering.convert_logweights_to_weights, θ_pred_of_t, data)

    @test ref == res

    Random.seed!(0)
    res = DualOptimalFiltering.sample_1_trajectory_from_joint_smoothing_CIR_precompute(δ, γ, σ, Λ_of_t, logwms_of_t |> DualOptimalFiltering.convert_logweights_to_weights, θ_of_t, Λ_pred_of_t, logwms_pred_of_t |> DualOptimalFiltering.convert_logweights_to_weights, θ_pred_of_t, data)
    @test ref == res


    ntraj = 100

    ## A graphical test. Commented out to allow tests to pass

    # using RCall
    # R"library(tidyverse)"
    # R"1:$ntraj %>%
    #     lapply(function(xx) tibble(idx = xx, x = as.numeric($times))) %>%
    #     bind_rows() %>%
    #     mutate(y =  $(vcat([DualOptimalFiltering.sample_1_trajectory_from_joint_smoothing_CIR_logweights(δ, γ, σ, Λ_of_t, logwms_of_t, θ_of_t, Λ_pred_of_t, logwms_pred_of_t, θ_pred_of_t, data) for k in 1:ntraj]...))) %>%
    #     ggplot(aes(x=x, y=y) ) +
    #       stat_density_2d(aes(fill = ..density..), geom = 'raster', contour = FALSE) +
    #       scale_fill_distiller(palette= 'Spectral', direction=1) +
    #       scale_x_continuous(expand = c(0, 0)) +
    #       scale_y_continuous(expand = c(0, 0)) +
    #       theme(
    #         legend.position='none'
    #       ) +
    #       geom_line(data = tibble(x = as.numeric($times), y = $X))
    # "
    #
    #
    #
    # R"1:$ntraj %>%
    #     lapply(function(xx) tibble(idx = xx, x = as.numeric($times))) %>%
    #     bind_rows() %>%
    #     mutate(y =  $(vcat([DualOptimalFiltering.sample_1_trajectory_from_joint_smoothing_CIR_logweights(δ, γ, σ, Λ_of_t, logwms_of_t, θ_of_t, Λ_pred_of_t, logwms_pred_of_t, θ_pred_of_t, data) for k in 1:ntraj]...))) %>%
    #     ggplot(aes(x=x, y=y, group = idx) ) +
    #     geom_line(colour = '#333333', alpha = 0.1) +
    #     theme_bw()  +
    #     geom_line(data = tibble(x = as.numeric($times), y = $X), aes(group = NULL)) +
    #     geom_point(data = tibble(x = as.numeric($times), y = $(vcat(Y...))), aes(group = NULL))
    # "

end;

@testset "stringent test for full smoothing CIR" begin

    function simulate_CIR_data(;Nsteps_CIR = 200)
        Random.seed!(2)

        δ = 3.
        γ = 2.5
        σ = 2.
        Nobs = 2
        dt_CIR = 0.011
        # max_time = .1
        # max_time = 0.001
        λ = 1.

        time_grid_CIR = [k*dt_CIR for k in 0:(Nsteps_CIR-1)]
        X_CIR = generate_CIR_trajectory(time_grid_CIR, 3, δ, γ, σ)
        Y_CIR = map(λ -> rand(Poisson(λ), Nobs), X_CIR);
        data_CIR = Dict(zip(time_grid_CIR, Y_CIR))
        return data_CIR, Y_CIR, X_CIR, time_grid_CIR, δ, γ, σ, λ
    end

    data_CIR, Y_CIR, X_CIR, times, δ, γ, σ, λ = simulate_CIR_data(;Nsteps_CIR = 800)

    θ_it_δγ_param = (δ, γ, σ)


    Λ_of_t, wms_of_t, θ_of_t = DualOptimalFiltering.filter_CIR_keep_fixed_number(δ, γ, σ, λ, data_CIR, 10; silence = false)

    fixed_number = 50

    function prune_keeping_fixed_number(Λ_of_t, wms_of_t)
        Λ_of_t_kept, wms_of_t_kept = DualOptimalFiltering.keep_fixed_number_of_weights(Λ_of_t, wms_of_t, fixed_number)
        return Λ_of_t_kept, DualOptimalFiltering.normalise(wms_of_t_kept)
    end


    Λ_of_t_pruned, wms_of_t_pruned =  DualOptimalFiltering.prune_all_dicts(Λ_of_t, wms_of_t, prune_keeping_fixed_number)


    @test_nowarn DualOptimalFiltering.sample_1_trajectory_from_joint_smoothing_CIR_precompute(θ_it_δγ_param[1], θ_it_δγ_param[2], θ_it_δγ_param[3], Λ_of_t_pruned, wms_of_t_pruned, θ_of_t, 1, 1, 1, data_CIR)

end;
