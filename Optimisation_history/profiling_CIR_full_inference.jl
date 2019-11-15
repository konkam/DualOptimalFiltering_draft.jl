using Revise
using DualOptimalFiltering, Random, Distributions, RCall, DataFrames, DataFramesMeta, Optim, FeynmanKacParticleFilters, BenchmarkTools, ProfileView, MCMCDiagnostics

import DualOptimalFiltering.sample_1_trajectory_from_joint_smoothing_CIR, DualOptimalFiltering.draw_next_sample, DualOptimalFiltering.joint_loglikelihood_CIR, DualOptimalFiltering.hidden_signal_loglikelihood_CIR, DualOptimalFiltering.keep_fixed_number_of_weights, DualOptimalFiltering.normalise, DualOptimalFiltering.sample_1_trajectory_from_joint_smoothing_CIR_precompute, DualOptimalFiltering.inverse_reparam_CIR, DualOptimalFiltering.prune_all_dicts, DualOptimalFiltering.reparam_joint_loglikelihood_CIR

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

data_CIR, Y_CIR, X_CIR, times, δ, γ, σ, λ = simulate_CIR_data(;Nsteps_CIR = 1000)
a, b, σ_prime = DualOptimalFiltering.reparam_CIR(δ, γ, σ)
prior_a = Truncated(Normal(5, 4), 0, Inf)
prior_b = Truncated(Normal(5, 4), 0, Inf)
prior_σ_prime = Truncated(Normal(5, 4), 0, Inf)
const prior_logpdf(ai, bi, σ_primei)::Float64 = logpdf(prior_a, ai) + logpdf(prior_b, bi) + logpdf(prior_σ_prime, σ_primei)


const fixed_number = 10
function prune_keeping_fixed_number(Λ_of_t, wms_of_t)
    Λ_of_t_kept, wms_of_t_kept = keep_fixed_number_of_weights(Λ_of_t, wms_of_t, fixed_number)
    return Λ_of_t_kept, normalise(wms_of_t_kept)
end
do_the_pruning = prune_keeping_fixed_number
final_chain_length = 1000
silence = false
jump_sizes = (0.5, 0.5, 0.5)
θ_init = (1.,1.,1.)
niter = 10
data = data_CIR


trajectory_chain = Array{Float64,2}(undef, niter, length(data |> keys))
parameter_chain = Array{Float64,2}(undef, niter, 3)
times = data |> keys |> collect |> sort

function unnormalised_logposterior(a, b, σ_prime, trajectory)::Float64
    prior_contribution::Float64 = prior_logpdf(a, b, σ_prime)
    if isinf(prior_contribution) #Avoids likelihood computation in this case
        return prior_contribution
    else
        return prior_contribution + reparam_joint_loglikelihood_CIR(data, trajectory, times, a, b, σ_prime, λ)::Float64
    end
end

θ_it = θ_init

const Jtsym_rand(θ::Tuple{Float64,Float64,Float64})::Tuple{Float64,Float64,Float64} = rand.(Normal.(0, jump_sizes)) .+ θ
θ_it_δγ_param = inverse_reparam_CIR(θ_it...)
@time Λ_of_t, wms_of_t, θ_of_t = DualOptimalFiltering.filter_CIR_pruning(θ_it_δγ_param[1], θ_it_δγ_param[2], θ_it_δγ_param[3], λ, data, do_the_pruning, silence = true)
@time Λ_of_t_pruned, wms_of_t_pruned =  prune_all_dicts(Λ_of_t, wms_of_t, do_the_pruning)
@code_warntype prune_all_dicts(Λ_of_t, wms_of_t, do_the_pruning)
@time X_it = sample_1_trajectory_from_joint_smoothing_CIR_precompute(θ_it_δγ_param[1], θ_it_δγ_param[2], θ_it_δγ_param[3], Λ_of_t_pruned, wms_of_t_pruned, θ_of_t, 1, 1, 1, data)
@time θ_it = draw_next_sample(θ_it, Jtsym_rand, st -> unnormalised_logposterior(st[1], st[2], st[3], X_it))


@profview parameter_chain, trajectory_chain = DualOptimalFiltering.joint_sampler_CIR_reparam_keep_fixed_number_precompute(data_CIR, λ, prior_logpdf, 30, 10; final_chain_length = 1000, silence = false, jump_sizes = (.5, .5, 0.25))
