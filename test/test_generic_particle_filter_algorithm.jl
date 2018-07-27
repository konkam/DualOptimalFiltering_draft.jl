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

    srand(0)

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

    srand(0)
    pf = DualOptimalFiltering.generic_particle_filtering(Mt, Gt, Nparts, RS)

    @test typeof(pf) == Dict{String,Array{Float64,2}}
    marginal_lik_factors = DualOptimalFiltering.marginal_likelihood_factors(pf)
    # println(marginal_lik_factors)
    res = [0.00506393, 0.000919112, 0.0075905, 0.00210651]
    for k in 1:Nsteps
        @test marginal_lik_factors[k] ≈ res[k] atol=10.0^(-7)
    end
    @test DualOptimalFiltering.marginal_likelihood(pf, DualOptimalFiltering.marginal_likelihood_factors)  ≈ prod(res) atol=10.0^(-7)
    @test length(DualOptimalFiltering.sample_from_filtering_distributions(pf, 10, 2)) == 10

    srand(0)
    pf_logweights = DualOptimalFiltering.generic_particle_filtering_logweights(Mt, logGt, Nparts, RS)

    @test typeof(pf_logweights) == Dict{String,Array{Float64,2}}
    marginal_loglik_factors = DualOptimalFiltering.marginal_loglikelihood_factors(pf_logweights)
    # println(marginal_loglik_factors)
    res = [-5.28561, -6.9921, -4.88086, -6.16272]
    for k in 1:Nsteps
        @test marginal_loglik_factors[k] ≈ res[k] atol=5*10.0^(-5)
    end
    @test DualOptimalFiltering.marginal_loglikelihood(pf_logweights, DualOptimalFiltering.marginal_loglikelihood_factors) ≈ sum(res) atol=5*10.0^(-5)
    @test length(DualOptimalFiltering.sample_from_filtering_distributions_logweights(pf_logweights, 10, 2)) == 10
    #

    srand(0)
    pf_adaptive = DualOptimalFiltering.generic_particle_filtering_adaptive_resampling(Mt, Gt, Nparts, RS)

    marginal_lik_factors = DualOptimalFiltering.marginal_likelihood_factors_adaptive_resampling(pf_adaptive)
    # println(marginal_lik_factors)
    res = [0.00506393, 0.000919112, 0.0075905, 0.00210651]
    for k in 1:Nsteps
        @test marginal_lik_factors[k] ≈ res[k] atol=10.0^(-7)
    end
    @test DualOptimalFiltering.marginal_likelihood(pf_adaptive, DualOptimalFiltering.marginal_likelihood_factors) ≈ prod(res) atol=10.0^(-7)
    @test length(DualOptimalFiltering.sample_from_filtering_distributions(pf_adaptive, 10, 2)) == 10

    srand(0)
    pf_adaptive_logweights = DualOptimalFiltering.generic_particle_filtering_adaptive_resampling_logweights(Mt, logGt, Nparts, RS)
    marginal_loglik_factors = DualOptimalFiltering.marginal_loglikelihood_factors_adaptive_resampling(pf_adaptive_logweights)
    # println(marginal_loglik_factors)
    res = [-5.28561, -6.9921, -4.88086, -6.16272]
    for k in 1:Nsteps
        @test marginal_loglik_factors[k] ≈ res[k] atol=5*10.0^(-5)
    end
    @test DualOptimalFiltering.marginal_loglikelihood(pf_adaptive_logweights, DualOptimalFiltering.marginal_loglikelihood_factors_adaptive_resampling) ≈ sum(res) atol=5*10.0^(-5)
    @test length(DualOptimalFiltering.sample_from_filtering_distributions_logweights(pf_adaptive_logweights, 10, 2)) == 10
end;
