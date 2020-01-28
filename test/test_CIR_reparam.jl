@testset "Testing the reparametrisation functions" begin
    δ, γ, σ = 1.1, 1.6, 2.3
    for pidx in eachindex([δ, γ, σ])
        @test [δ, γ, σ][pidx] ≈ DualOptimalFiltering_proof.inverse_reparam_CIR(DualOptimalFiltering_proof.reparam_CIR(δ, γ, σ)...)[pidx]
    end
    for pidx in eachindex([δ, γ, σ])
        @test [δ, γ, σ][pidx] ≈ DualOptimalFiltering_proof.reparam_CIR(DualOptimalFiltering_proof.inverse_reparam_CIR(δ, γ, σ)...)[pidx]
    end

    Random.seed!(0)
    times_sim = range(0, stop = 20, length = 20)
    X = DualOptimalFiltering_proof.generate_CIR_trajectory(times_sim, 3, 3., 0.5, 1);
    Y = map(λ -> rand(Poisson(λ),1), X);
    data = Dict(zip(times_sim, Y))
    δ, γ, σ, λ = 3., 0.5, 1., 1.
    a, b, σ_prime = DualOptimalFiltering_proof.reparam_CIR(δ, γ, σ)
    Λ_of_t, logwms_of_t, θ_of_t, Λ_pred_of_t, logwms_pred_of_t, θ_pred_of_t = DualOptimalFiltering_proof.filter_predict_CIR_logweights(δ, γ, σ, λ, data);

    times = Λ_of_t |> keys |> collect |> sort

    @test DualOptimalFiltering_proof.reparam_joint_loglikelihood_CIR(data, X, times, a, b, σ_prime, λ) == DualOptimalFiltering_proof.joint_loglikelihood_CIR(data, X, times, δ, γ, σ, λ)

    prior_a = truncated(Normal(5, 4), 0, Inf)
    prior_b = truncated(Normal(5, 4), 0, Inf)
    prior_σ_prime = truncated(Normal(5, 4), 0, Inf)
    prior_logpdf(ai, bi, σ_primei) = logpdf(prior_a, ai) + logpdf(prior_b, bi) + logpdf(prior_σ_prime, σ_primei)

    @test_nowarn DualOptimalFiltering_proof.joint_sampler_CIR_reparam_keep_fixed_number_precompute(data, λ, prior_logpdf, 10, 2; final_chain_length = 10)
end;
