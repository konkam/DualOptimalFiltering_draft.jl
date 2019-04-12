
@testset "CIR approx smoothing tests" begin

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

    @test_nowarn DualOptimalFiltering.log_cost_to_go_CIR_keep_fixed_number(δ, γ, σ, λ, data, 10; silence = false)

    @test_nowarn DualOptimalFiltering.smooth_CIR_keep_fixed_number(δ, γ, σ, λ, data, 10; silence = false)

    @test_nowarn DualOptimalFiltering.log_cost_to_go_CIR_keep_above_threshold(δ, γ, σ, λ, data, -50; silence = false)

    @test_nowarn DualOptimalFiltering.smooth_CIR_keep_above_threshold(δ, γ, σ, λ, data, 0.00001, -50; silence = false)

    @test_nowarn DualOptimalFiltering.log_cost_to_go_CIR_keep_fixed_fraction(δ, γ, σ, λ, data, 0.99; silence = false)

    @test_nowarn DualOptimalFiltering.smooth_CIR_keep_fixed_fraction(δ, γ, σ, λ, data, 0.99; silence = false)



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

    # @test_nowarn DualOptimalFiltering.smooth_CIR_keep_fixed_number(δ, γ, σ, λ, data, 10; silence = false)
    #
    # @test_nowarn DualOptimalFiltering.smooth_CIR_keep_above_threshold(δ, γ, σ, λ, data, 0.01, 10^(-20); silence = false)
    #
    # @test_nowarn DualOptimalFiltering.smooth_CIR_keep_fixed_fraction(δ, γ, σ, λ, data, 0.9; silence = false)

end;
