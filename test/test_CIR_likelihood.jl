@testset "Testing the CIR specific function μπh, t and T" begin
    @test DualOptimalFiltering.logμπh(2, 1.2, 3.1, 4) ≈ -1.9336877133784256 atol=10.0^(-5)
    @test DualOptimalFiltering.logμπh(2, 1.2, 3.1, [4]) ≈ -1.9336877133784256 atol=10.0^(-5)
    @test DualOptimalFiltering.logμπh(2, 1.2, 3.1, [4, 6]) ≈ -4.261806748257223 atol=10.0^(-5)
    @test DualOptimalFiltering.logμπh([2], 1.2, 3.1, [4, 6])[1] ≈ -4.261806748257223 atol=10.0^(-5)
    tmp = DualOptimalFiltering.logμπh([2, 3], 1.2, 3.1, [4, 6])
    @test tmp[1] ≈ -4.261806748257223 atol = 10.0^(-7)
    @test tmp[2] ≈ -4.15718179717835 atol = 10.0^(-7)
    @test DualOptimalFiltering.logpmmi(8,2, .4, 1.2, 1.5, 1.4) ≈ -7.019424887074462 atol = 10.0^(-7)

    @test DualOptimalFiltering.t_CIR(5, 3) == 8
    @test DualOptimalFiltering.t_CIR([5], 3) == 8
    @test DualOptimalFiltering.t_CIR([5,6], 3) == 14
    tmp = DualOptimalFiltering.t_CIR(5, [3,2,1])
    for t in 1:3
        @test tmp[t] == [8,7,6][t]
    end
    tmp = DualOptimalFiltering.t_CIR([5], [3,2,1])
    for t in 1:3
        @test tmp[t] == [8,7,6][t]
    end
    tmp = DualOptimalFiltering.t_CIR([5,6], [3,2,1])
    for t in 1:3
        @test tmp[t] == [14, 13, 12][t]
    end
end;

@testset "Testing the transition functions" begin

    @test DualOptimalFiltering.Λ_prime_1D([3,2,1]) == 0:3

    tmp = DualOptimalFiltering.next_Λ_from_Λ_prime([3,2,1], 6, DualOptimalFiltering.t_CIR)
    for t in 1:3
        @test tmp[t] == [9, 8, 7][t]
    end
    tmp = DualOptimalFiltering.next_Λ_from_Λ_prime([3,2,1], [6], DualOptimalFiltering.t_CIR)
    for t in 1:3
        @test tmp[t] == [9, 8, 7][t]
    end
    tmp = DualOptimalFiltering.next_Λ_from_Λ_prime([3,2,1], [5,6], DualOptimalFiltering.t_CIR)
    for t in 1:3
        @test tmp[t] == [14, 13, 12][t]
    end

    @test DualOptimalFiltering.θ_prime_from_θ_CIR(1.4, 0.3, 1.2, 2.3) ≈ 0.38310545974007354 atol = 10.0^(-7)

    @test DualOptimalFiltering.θ_from_θ_prime(5, 0.3, DualOptimalFiltering.T_CIR)  == 1.3
    @test DualOptimalFiltering.θ_from_θ_prime([5], 0.3, DualOptimalFiltering.T_CIR)  == 1.3
    @test DualOptimalFiltering.θ_from_θ_prime([5,6], 0.3, DualOptimalFiltering.T_CIR)  == 2.3

    tmp = DualOptimalFiltering.next_log_wms_from_log_wms_prime(log.([0.5,0.25,0.25]), [1,4,2], 6, 1.3, 0.7)
    for t in 1:3
        @test tmp[t] ≈ [-1.74651, -0.506833, -1.49961][t] atol = 5*10.0^(-5)
    end
    tmp = DualOptimalFiltering.next_log_wms_from_log_wms_prime(log.([0.5,0.25,0.25]), [1,4,2], [6], 1.3, 0.7)
    for t in 1:3
        @test tmp[t] ≈ [-1.74651, -0.506833, -1.49961][t] atol = 5*10.0^(-5)
    end
    tmp = DualOptimalFiltering.next_log_wms_from_log_wms_prime(log.([0.5,0.25,0.25]), [1,4,2], [5,6], 1.3, 0.7)
    ref = [-1.956560511980828, -0.42974982819492613, -1.5702921541433446]
    for t in 1:3
        @test tmp[t] ≈ ref[t] atol = 5*10.0^(-5)
    end
    tmp = DualOptimalFiltering.next_log_wms_from_log_wms_prime(log.([0.5,0.25,0.25]), [1,4,2], [6,5], 1.3, 0.7)
    for t in 1:3
        @test tmp[t] ≈ ref[t] atol = 5*10.0^(-5)
    end
    tmp = DualOptimalFiltering.next_wms_prime_from_wms([0.5,0.25,0.25], [1,4,2], 0.2, 1.3, 1.1, 0.6)
    for t in 1:5
        @test tmp[t] ≈ [0.104581, 0.487464, 0.199507, 0.101034, 0.107413][t] atol = 5*10.0^(-5)
    end
    tmp = DualOptimalFiltering.next_log_wms_prime_from_log_wms(log.([0.5,0.25,0.25]), [1,4,2], 0.2, 1.3, 1.1, 0.6)
    for t in 1:5
        @test tmp[t] ≈ [-2.25779, -0.718539, -1.6119, -2.2923, -2.23107][t] atol = 5*10.0^(-5)
    end
    tmp = log.(DualOptimalFiltering.next_wms_prime_from_wms([0.5,0.25,0.25], [1,4,2], 0.2, 1.3, 1.1, 0.6))
    for t in 1:5
        @test tmp[t] ≈ [-2.25779, -0.718539, -1.6119, -2.2923, -2.23107][t] atol = 5*10.0^(-5)
    end

end;

@testset "Testing the likelihood functions" begin

    Random.seed!(1)

    δ = 3.
    γ = 2.5
    σ = 4.
    Nobs = 3
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

    tmp = DualOptimalFiltering.log_likelihood(δ, γ, σ, λ, data)

    for t in 1:length(time_grid)
        @test tmp[time_grid[t]] ≈ [-9.316739337010942, -15.159852198559214, -20.35995442574231, -25.857131056368978, -30.968165950652903, -37.72357636605707, -43.89643355575996, -53.41447961431761, -60.84516445861371, -67.55501659305918][t]  atol = 5*10.0^(-5)
    end


end;


@testset "Testing the transition functions" begin

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

    @test_nowarn tmp = DualOptimalFiltering.log_likelihood_CIR_fixed_fraction(δ, γ, σ, λ, data, 0.9; silence = false)
    @test_nowarn tmp = DualOptimalFiltering.log_likelihood_CIR_keep_above_threshold(δ, γ, σ, λ, data, 0.001; silence = false)
    @test_nowarn tmp = DualOptimalFiltering.log_likelihood_CIR_keep_fixed_number(δ, γ, σ, λ, data, 10; silence = false)

    for t in 1:length(time_grid)
        @test tmp[time_grid[t]] ≈ [-5.48387, -11.6933, -15.3573, -18.6661, -22.2757, -28.1024, -32.6409, -39.4623, -43.8618, -48.6812][t]  atol = 5*10.0^(-5)
    end


end;
