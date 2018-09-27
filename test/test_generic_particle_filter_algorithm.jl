using StatsFuns, Distributions

function rCIR(n::Integer, Dt::Real, x0::Real, δ, γ, σ)
    β = γ/σ^2*exp(2*γ*Dt)/(exp(2*γ*Dt)-1)
    if n == 1
        ks = rand(Poisson(γ/σ^2*x0/(exp(2*γ*Dt)-1)))
        return rand(Gamma(ks+δ/2, 1/β))
    else
        ks = rand(Poisson(γ/σ^2*x0/(exp(2*γ*Dt)-1)), n)
        return rand.(Gamma.(ks+δ/2, 1/β))
    end
end

function create_Mt(Δt, δ, γ, σ)
    function Mt(X)
#         Aind = DualOptimalFiltering.indices_from_multinomial_sample_slow(A)
        return rCIR.(1, Δt, X, δ, γ, σ)
    end
    return Mt
end

@testset "Test ESS functions" begin
    @test DualOptimalFiltering.ESS(repeat([1], inner = 10)./10) ≈ 10 atol=10.0^(-7)
    @test DualOptimalFiltering.logESS(repeat([1], inner = 10)./10 |> v -> log.(v)) ≈ log(10) atol=10.0^(-7)
end;

@testset "test particle filter algorithm for CIR process" begin

    Random.seed!(0)

    Δt = 0.1
    δ = 3.
    γ = 2.5
    σ = 4.
    Nobs = 2
    Nsteps = 4
    λ = 1.
    Nparts = 10
    α = δ/2
    β = γ/σ^2

    time_grid = [k*Δt for k in 0:(Nsteps-1)]
    times = [k*Δt for k in 0:(Nsteps-1)]
    X = generate_CIR_trajectory(time_grid, 3, δ*1.2, γ/1.2, σ*0.7)
    Y = map(λ -> rand(Poisson(λ), Nobs), X);
    data = zip(times, Y) |> Dict
    Mt = DualOptimalFiltering.create_transition_kernels_CIR(data, δ, γ, σ)
    Gt = DualOptimalFiltering.create_potential_functions_CIR(data)
    logGt = DualOptimalFiltering.create_log_potential_functions_CIR(data)
    RS(W) = rand(Categorical(W), length(W))

    Random.seed!(0)
    @test Mt[0.1](3) ≈ 8.418659447049441 atol=10.0^(-7)
    @test Mt[0.1](3.1) ≈ 2.1900629888259893 atol=10.0^(-7)
    @test Mt[0.2](3.1) ≈ 2.6844105017153863 atol=10.0^(-7)
    @test Mt[time_grid[3]](3.1) ≈ 1.3897782586244247 atol=10.0^(-7)

    @test Gt[0.1](3) ≈ 2.2129511996787992e-8 atol=10.0^(-7)
    @test Gt[0.1](3.1) ≈ 3.7273708205666865e-8 atol=10.0^(-7)
    @test Gt[0.2](3.1) ≈ 0.03877426525100398 atol=10.0^(-7)
    @test Gt[time_grid[3]](3.1) ≈ 0.03877426525100398 atol=10.0^(-7)

    Random.seed!(0)
    # srand(0)
    pf = DualOptimalFiltering.generic_particle_filtering(Mt, Gt, Nparts, RS)

    @test typeof(pf) == Dict{String,Array{Float64,2}}
    marginal_lik_factors = DualOptimalFiltering.marginal_likelihood_factors(pf)
    # println(marginal_lik_factors)
    res = [ 0.005063925135653128, 0.0013145849369714938, 0.014640244207811792, 0.0017270473953601316]
    for k in 1:Nsteps
        @test marginal_lik_factors[k] ≈ res[k] atol=10.0^(-7)
    end
    @test DualOptimalFiltering.marginal_likelihood(pf, DualOptimalFiltering.marginal_likelihood_factors)  ≈ prod(res) atol=10.0^(-7)
    @test length(DualOptimalFiltering.sample_from_filtering_distributions(pf, 10, 2)) == 10

    Random.seed!(0)
    pf_logweights = DualOptimalFiltering.generic_particle_filtering_logweights(Mt, logGt, Nparts, RS)

    @test typeof(pf_logweights) == Dict{String,Array{Float64,2}}
    marginal_loglik_factors = DualOptimalFiltering.marginal_loglikelihood_factors(pf_logweights)
    # println(marginal_loglik_factors)
    res = [ -5.285613377888339, -6.634234300460378, -4.223981089726635, -6.361342036441921]
    for k in 1:Nsteps
        @test marginal_loglik_factors[k] ≈ res[k] atol=5*10.0^(-5)
    end
    @test DualOptimalFiltering.marginal_loglikelihood(pf_logweights, DualOptimalFiltering.marginal_loglikelihood_factors) ≈ sum(res) atol=5*10.0^(-5)
    @test length(DualOptimalFiltering.sample_from_filtering_distributions_logweights(pf_logweights, 10, 2)) == 10
    #

    Random.seed!(0)
    pf_adaptive = DualOptimalFiltering.generic_particle_filtering_adaptive_resampling(Mt, Gt, Nparts, RS)

    marginal_lik_factors = DualOptimalFiltering.marginal_likelihood_factors_adaptive_resampling(pf_adaptive)
    # println(marginal_lik_factors)
    res = [0.005063925135653128, 0.0013145849369714936, 0.014640244207811792, 0.0020015945094952942]
    for k in 1:Nsteps
        @test marginal_lik_factors[k] ≈ res[k] atol=10.0^(-7)
    end
    @test DualOptimalFiltering.marginal_likelihood(pf_adaptive, DualOptimalFiltering.marginal_likelihood_factors) ≈ prod(res) atol=10.0^(-7)
    @test length(DualOptimalFiltering.sample_from_filtering_distributions(pf_adaptive, 10, 2)) == 10

    Random.seed!(0)
    pf_adaptive_logweights = DualOptimalFiltering.generic_particle_filtering_adaptive_resampling_logweights(Mt, logGt, Nparts, RS)
    marginal_loglik_factors = DualOptimalFiltering.marginal_loglikelihood_factors_adaptive_resampling(pf_adaptive_logweights)
    # println(marginal_loglik_factors)
    res = [ -5.285613377888339, -6.634234300460378, -4.223981089726635, -6.213811161313297]
    for k in 1:Nsteps
        @test marginal_loglik_factors[k] ≈ res[k] atol=5*10.0^(-5)
    end
    @test DualOptimalFiltering.marginal_loglikelihood(pf_adaptive_logweights, DualOptimalFiltering.marginal_loglikelihood_factors_adaptive_resampling) ≈ sum(res) atol=5*10.0^(-5)
    @test length(DualOptimalFiltering.sample_from_filtering_distributions_logweights(pf_adaptive_logweights, 10, 2)) == 10
end;
