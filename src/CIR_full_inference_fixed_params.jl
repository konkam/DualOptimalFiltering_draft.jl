using Revise
using DualOptimalFiltering_proof, Random, Distributions, RCall, DataFrames, DataFramesMeta, Optim, FeynmanKacParticleFilters, BenchmarkTools, ProfileView, MCMCDiagnostics

R"library(tidyverse)"

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

a, b, σ_prime = DualOptimalFiltering_proof.reparam_CIR(δ, γ, σ)

prior_a = truncated(Normal(5, 4), 0, Inf)
prior_b = truncated(Normal(5, 4), 0, Inf)
prior_σ_prime = Normal(σ_prime, 0)
const prior_logpdf(ai, bi, σ_primei)::Float64 = logpdf(prior_a, ai) + logpdf(prior_b, bi)

prior_logpdf(a, b, σ_prime)

@time parameter_chain, trajectory_chain = DualOptimalFiltering_proof.joint_sampler_CIR_reparam_keep_fixed_number_precompute(data_CIR, λ, prior_logpdf, 5000, 20; final_chain_length = 1000, silence = false, jump_sizes = (.5, .5, 0.), θ_init = (1., 1., σ_prime))

println("Acceptance rate = $(parameter_chain[:,1] |> x -> length(unique(x))/length(x))")

R"$parameter_chain %>%
    as_tibble %>%
    mutate(iter = seq_along(V1)) %>%
    gather(variable, value, -iter) %>%
    ggplot(aes(x = iter, y = value)) +
    theme_bw() +
    facet_wrap(~variable, scale = 'free_y') +
    geom_line()"


prior = DataFrame(x = range(0.01, stop = 15, length = 50)) |>
        df -> @transform(df, typ = repeat(["Prior"], size(df,1)), a = pdf.(prior_a, :x), b = pdf.(prior_b, :x), σ_prime = pdf.(prior_σ_prime, :x))
    posterior = rename(parameter_chain |> DataFrame, Dict(zip([:x1, :x2, :x3], [:a, :b, :σ_prime])));
    params = DataFrame(variable = ["a", "b", "σ_prime"], value = [a, b, σ_prime]);
R"$posterior %>%
    as_tibble %>%
    gather(variable, value) %>%
    ggplot(aes(x = value, y = ..density..)) +
    theme_minimal() +
    facet_wrap(~variable) +
    geom_density(colour = 'blue') +
    geom_line(data = $prior %>%
    as_tibble %>%
    gather(variable, value, -x, -typ), aes(x = x, y = value)) +
    geom_vline(data = $params %>% as_tibble, aes(xintercept = value), colour = 'red')"

R"$trajectory_chain %>%
        t %>%
        as_tibble %>%
        mutate(times = $times) %>%
        gather(iter, value, -times) %>%
        ggplot() +
        theme_bw() +
        geom_line(aes(x = times, y = value, group = iter), colour = '#333333', alpha = 0.1) +
        geom_line(data = tibble(x = as.numeric($times), y = $X_CIR), aes(group = NULL, x = x, y = y), colour = 'red') +
        geom_point(data = $(hcat(Y_CIR...)) %>%
        t %>%
        as_tibble %>%
        mutate(times = $times) %>%
        gather(variable, value, -times), aes(group = NULL, x = times, y = value))"
