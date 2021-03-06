using ExactWrightFisher, Distributions

@testset "test full inference CIR" begin
    Random.seed!(0)
    times_sim = range(0, stop = 20, length = 20)
    X = DualOptimalFiltering_proof.generate_CIR_trajectory(times_sim, 3, 3., 0.5, 1);
    Y = map(λ -> rand(Poisson(λ),1), X);
    data = Dict(zip(times_sim, Y))
    δ, γ, σ, λ = 3., 0.5, 1., 1.
    Λ_of_t, logwms_of_t, θ_of_t, Λ_pred_of_t, logwms_pred_of_t, θ_pred_of_t = DualOptimalFiltering_proof.filter_predict_CIR_logweights(δ, γ, σ, λ, data);

    times = Λ_of_t |> keys |> collect |> sort

    @test_nowarn DualOptimalFiltering_proof.hidden_signal_loglikelihood_CIR(X, times, δ, γ, σ)

    @test_nowarn DualOptimalFiltering_proof.emitted_data_conditional_loglikelihood_CIR(data, X, times, λ)

    @test_nowarn DualOptimalFiltering_proof.joint_loglikelihood_CIR(data, X, times, δ, γ, σ, λ)


    prior_δ = truncated(Normal(5, 4), 0, Inf)
    prior_γ = truncated(Normal(5, 4), 0, Inf)
    prior_σ = truncated(Normal(5, 4), 0, Inf)
    prior_logpdf(δi, γi, σi) = logpdf(prior_δ, δi) + logpdf(prior_γ, γi) + logpdf(prior_σ, σi)

    @test_nowarn DualOptimalFiltering_proof.joint_sampler_CIR(data, λ, prior_logpdf, [1.,1.,1.], 10, final_chain_length = 10)

    @test_nowarn DualOptimalFiltering_proof.joint_sampler_CIR_keep_fixed_number(data, λ, prior_logpdf, [1.,1.,1.], 10, 2; final_chain_length = 10)

    @test_nowarn DualOptimalFiltering_proof.joint_sampler_CIR_keep_fixed_number_nopred(data, λ, prior_logpdf, [1.,1.,1.], 10, 2; final_chain_length = 10)

    @test_nowarn DualOptimalFiltering_proof.joint_sampler_CIR_keep_fixed_number_precompute(data, λ, prior_logpdf, [1.,1.,1.], 10, 2; final_chain_length = 10)

end;
