
@testset "CIR smoothing helper functions" begin
    @test DualOptimalFiltering.Λ_tilde_prime_k_from_Λ_tilde_k_CIR([10,5,3]) == 0:10
    @test DualOptimalFiltering.Λ_tilde_k_from_Λ_tilde_prime_kp1_CIR(5, [10,5,3]) == [15, 10, 8]
    @test DualOptimalFiltering.Λ_tilde_k_from_Λ_tilde_prime_kp1_CIR([5,6], [10,5,3]) == [21, 16, 14]
    @test_nowarn DualOptimalFiltering.θ_tilde_prime_k_from_θ_tilde_k_CIR(0.3, 1.2, 1.1, 0.7)
    @test DualOptimalFiltering.θ_tilde_k_from_θ_tilde_prime_kp1(5, 4.1) == 5.1
    @test DualOptimalFiltering.θ_tilde_k_from_θ_tilde_prime_kp1([5,6], 4.1) == 6.1
    @test_nowarn DualOptimalFiltering.pmn_CIR(5, 2, 0.5)
    @test_nowarn DualOptimalFiltering.pmn_CIR(5, 2, 0.2, 1.1, 1.2, 1.3)
    @test_nowarn DualOptimalFiltering.wms_tilde_kp1_from_wms_tilde_kp2([0.2,0.3,0.4,0.1], [3,2,4,7], 1.3, 1.1, [6], 0.4, 1.1, 1.3, 1.2, 1.)

    @test_nowarn DualOptimalFiltering.update_logweights_cost_to_go_CIR(log.([0.2,0.3,0.4,0.1]), [3,2,4,7], 1.3, 1.1, [6], 0.4)

    @test_nowarn DualOptimalFiltering.logCmn_CIR(3, 4, 1.2, 1.6, 0.2, 0.6)

    ref =  DualOptimalFiltering.wms_tilde_kp1_from_wms_tilde_kp2([0.2,0.3,0.4,0.1], [3,2,4,7], 1.3, 1.1, [6], 0.4, 1.1, 1.3, 1.2, 1.)
    res = DualOptimalFiltering.logwms_tilde_kp1_from_logwms_tilde_kp2(log.([0.2,0.3,0.4,0.1]), [3,2,4,7], 1.3, 1.1, [6], 0.4, 1.1, 1.3, 1.2, 1.)
    for k in eachindex(ref)
        @test ref[k] ≈ exp(res[k])
    end

    Random.seed!(1)

    δ = 3.
    γ = 2.5
    σ = 4.
    Nobs = 2
    dt = 0.011
    Nsteps = 10
    λ = 1.

    α = δ/2
    β = γ/σ^2

    function rec_rcCIR(Dts, x, δ, γ, σ)
        x_new = DualOptimalFiltering.rCIR(1, Dts[1], x[end], δ, γ, σ)
        if length(Dts) == 1
            return Float64[x; x_new]
        else
            return Float64[x; rec_rcCIR(Dts[2:end], x_new, δ, γ, σ)]
        end
    end

    function generate_CIR_trajectory2(times, x0, δ, γ, σ)
        θ1 = δ*σ^2
        θ2 = 2*γ
        θ3 = 2*σ
        Dts = diff(times)
        return rec_rcCIR(Dts, [x0], δ, γ, σ)
    end

    time_grid = [k*dt for k in 0:(Nsteps-1)]
    X = generate_CIR_trajectory2(time_grid, 3, δ*1.2, γ/1.2, σ*0.7)
    Y = map(λ -> rand(Poisson(λ), Nobs), X);
    data = zip(time_grid, Y) |> Dict;

    precomputed_lgamma_α = DualOptimalFiltering.precompute_lgamma_α(α, data)
    precomputed_lfactorial = DualOptimalFiltering.precompute_lfactorial(data)

    ref = DualOptimalFiltering.logwms_tilde_kp1_from_logwms_tilde_kp2_arb(log.([0.2,0.3,0.4,0.1]), [3,2,4,7], 1.3, 1.1, [6], dt, α, 1.3, 1.2, 1.)

    res = DualOptimalFiltering.logwms_tilde_kp1_from_logwms_tilde_kp2(log.([0.2,0.3,0.4,0.1]), [3,2,4,7], 1.3, 1.1, [6], dt, α, 1.3, 1.2, 1.)
    for k in eachindex(ref)
        @test Float64(ref[k]) ≈ res[k]
    end


    # identity(x,y) = (x, y)

    res = DualOptimalFiltering.logwms_tilde_kp1_from_logwms_tilde_kp2_pruning(log.([0.2,0.3,0.4,0.1]), [3,2,4,7], 1.3, 1.1, [6], dt, α, 1.3, 1.2, 1.; pruning_function = (x, y) -> (x, y))
    for k in eachindex(ref)
        @test Float64(ref[k]) ≈ res[k]
    end

    res = DualOptimalFiltering.logwms_tilde_kp1_from_logwms_tilde_kp2_precomputed(log.([0.2,0.3,0.4,0.1]), [3,2,4,7], 1.3, 1.1, [6], dt, α, 1.3, 1.2, 1., precomputed_lgamma_α, precomputed_lfactorial)

    for k in eachindex(ref)
        @test Float64(ref[k]) ≈ res[k]
    end

    res = DualOptimalFiltering.logwms_tilde_kp1_from_logwms_tilde_kp2_precomputed_pruning(log.([0.2,0.3,0.4,0.1]), [3,2,4,7], 1.3, 1.1, [6], dt, α, 1.3, 1.2, 1., precomputed_lgamma_α, precomputed_lfactorial; pruning_function = (x, y) -> (x, y))

    for k in eachindex(ref)
        @test Float64(ref[k]) ≈ res[k]
    end

    res = DualOptimalFiltering.logwms_tilde_kp1_from_logwms_tilde_kp2_precomputed_pruning(log.([0.2,0.3,0.4,0.1]), [3,2,4,7], 1.3, 1.1, [6], dt, α, 1.3, 1.2, 1., precomputed_lgamma_α, precomputed_lfactorial; pruning_function = (x, y) -> (x, y), return_indices = true)
    @test length(res[1]) == length(res[2])

end;

@testset "CIR smoothing tests" begin

    Random.seed!(1)

    δ = 3.
    γ = 2.5
    σ = 4.
    Nobs = 2
    dt = 0.011
    Nsteps = 10
    λ = 1.

    α = δ/2
    β = γ/σ^2

    function rec_rcCIR(Dts, x, δ, γ, σ)
        x_new = DualOptimalFiltering.rCIR(1, Dts[1], x[end], δ, γ, σ)
        if length(Dts) == 1
            return Float64[x; x_new]
        else
            return Float64[x; rec_rcCIR(Dts[2:end], x_new, δ, γ, σ)]
        end
    end

    function generate_CIR_trajectory2(times, x0, δ, γ, σ)
        θ1 = δ*σ^2
        θ2 = 2*γ
        θ3 = 2*σ
        Dts = diff(times)
        return rec_rcCIR(Dts, [x0], δ, γ, σ)
    end

    time_grid = [k*dt for k in 0:(Nsteps-1)]
    X = generate_CIR_trajectory2(time_grid, 3, δ*1.2, γ/1.2, σ*0.7)
    Y = map(λ -> rand(Poisson(λ), Nobs), X);
    data = zip(time_grid, Y) |> Dict;

    @test_nowarn DualOptimalFiltering.compute_all_cost_to_go_functions_CIR(1.2, 0.3, 0.6, 1., data)

    ref =  DualOptimalFiltering.compute_all_cost_to_go_functions_CIR(1.2, 0.3, 0.6, 1., data)
    res = DualOptimalFiltering.compute_all_cost_to_go_functions_CIR_pruning(1.2, 0.3, 0.6, 1., data, (x, y) -> (x,y))

    for k in keys(ref[2])
        for l in eachindex(ref[2][k])
            @test ref[2][k][l] .≈ res[2][k][l]
            @test ref[1][k][l] .≈ res[1][k][l]
        end
    end

    res2 = DualOptimalFiltering.compute_all_log_cost_to_go_functions_CIR_pruning(1.2, 0.3, 0.6, 1., data, (x, y) -> (x,y))

    for k in keys(ref[2])
        for l in eachindex(ref[2][k])
            # @show k, l, ref[2][k][l], exp(res2[2][k][l])
            @test ref[2][k][l] ≈ exp(res2[2][k][l])
            @test ref[1][k][l] ≈ res2[1][k][l]
        end
    end

    for k in time_grid[2:(end-1)]
        @test length(ref[2][k]) == 1 + sum([sum(data[i]) for i in time_grid if i >= k])
    end

    @show sum(sum.(values(data)))

    res2 = DualOptimalFiltering.compute_all_log_cost_to_go_functions_CIR_pruning_precomputed(1.2, 0.3, 0.6, 1., data, (x, y) -> (x,y))

    for k in keys(ref[2])
        for l in eachindex(ref[2][k])
            # @show k, l, ref[2][k][l], exp(res2[2][k][l])
            @test ref[2][k][l] ≈ exp(res2[2][k][l]) atol=10^(-15)
            @test ref[1][k][l] ≈ res2[1][k][l]
        end
    end

    ref = CIR_smoothing(δ, γ, σ, λ, data; silence = false)
    res = DualOptimalFiltering.CIR_smoothing_logscale_internals(δ, γ, σ, λ, data; silence = false)

    for k in time_grid
        for l in eachindex(ref[2][k])
            # @show k, l, ref[2][k][l], res[2][k][l]
            @test ref[2][k][l] ≈ res[2][k][l]
        end
    end
end;
