 using Revise, DualOptimalFiltering_proof, Random, Distributions, RCall, DataFrames, DataFramesMeta, Optim, FeynmanKacParticleFilters, BenchmarkTools, MCMCDiagnostics
 R"library(tidyverse)"

# R"pdf('test.pdf')
# plot(1:4)
# dev.off()
# "

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

data_CIR, Y_CIR, X_CIR, times, δ, γ, σ, λ = simulate_CIR_data(;Nsteps_CIR = 5000)

a, b, σ_prime = DualOptimalFiltering_proof.reparam_CIR(δ, γ, σ)

prior_a = truncated(Normal(5, 4), 0, Inf)
prior_b = truncated(Normal(5, 4), 0, Inf)
prior_σ_prime = truncated(Normal(5, 4), 0, Inf)
const prior_logpdf(ai, bi, σ_primei)::Float64 = logpdf(prior_a, ai) + logpdf(prior_b, bi) + logpdf(prior_σ_prime, σ_primei)


f = () -> DualOptimalFiltering_proof.param_sampler(X_CIR, times, prior_logpdf, [1.,1.,1.], 5000; final_chain_length = 1000, silence = false)

mcmc_chains = [f() for i in 1:3];

function cmp_acceptance_rate(param_chain)
    return param_chain[:,1]  |> x -> length(unique(x))/length(x)
end


println("Acceptance rate: $(cmp_acceptance_rate.(mcmc_chains) |> mean)")

println("Effective sample sizes: $([effective_sample_size(vcat([mcmc_chains[k][:,i] for k in 1:3]...)) for i in 1:3])")

println("Potential_scale_reduction: $([potential_scale_reduction([mcmc_chains[k][:,i] for k in 1:3]...) for i in 1:3])")

to_traceplot = mcmc_chains |>
#     x -> DataFrame.(x) |>
#     x -> map(y -> y', x) |>
#     first
    x -> map(y -> DataFrame(collect(y)), x) |>
    x -> map( (y, idy) -> @transform(y, chain_id = repeat([idy], size(y,1))), x, eachindex(x)) |>
    x -> vcat(x...);

R"
#pdf(file = 'traceplot.pdf')
$to_traceplot %>%
    as_tibble %>%
    setNames(c('a', 'b', 's_prime', 'chain_id')) %>%
    gather(param, value, -chain_id) %>%
    group_by(chain_id, param) %>%
    mutate(iter = seq_along(value)) %>%
    ggplot(aes(x = iter, y = value, colour = factor(chain_id), group = chain_id)) +
    theme_minimal() +
    facet_wrap(~param) +
    geom_line() +
    geom_hline(data = tibble(a = $a, b=$b, s_prime = $σ_prime) %>% gather(param, value), aes(yintercept = value), colour = 'red')
    #dev.off()
    "

R"$(vcat(mcmc_chains...)) %>%
        PerformanceAnalytics::chart.Correlation(., histogram=TRUE, pch=19) %>%
        ggsave(plot = ., filename = 'corplot.pdf')"

prior = DataFrame(x = range(0.01, stop = 15, length = 50)) |>
    df -> @transform(df, typ = repeat(["Prior"], size(df,1)), a = pdf.(prior_a, :x), b = pdf.(prior_b, :x), s_prime = pdf.(prior_σ_prime, :x))

function add_iteration_number(df)
    return @transform(df, iter = 1:size(df, 1))
end

posterior = mcmc_chains |>
    cs -> DataFrame.(cs) |>
    cs -> add_iteration_number.(cs) |>
    cs -> vcat(cs...) |>
    df -> @where(df, :iter .> maximum(:iter)/2) |>
    df -> rename(df, Dict(zip([:x1, :x2, :x3], [:a, :b, :s_prime])));
params = DataFrame(variable = ["a", "b", "s_prime"], value = [a, b, σ_prime]);



R"
pdf(file = 'prior_posterior_plot.pdf')
p = $(vcat(mcmc_chains...)) %>%
    as_tibble %>%
    setNames(c('a', 'b', 's_prime')) %>%
    gather(variable, value) %>%
    ggplot(aes(x = value, y = ..density..)) +
    theme_minimal() +
    facet_wrap(~variable, scales = 'free') +
    geom_density(colour = 'blue') +
    geom_line(data = $prior %>%
    as_tibble %>%
    gather(variable, value, -x, -typ), aes(x = x, y = value)) +
    geom_vline(data = $params %>% as_tibble, aes(xintercept = value), colour = 'red')
    plot(p)
    dev.off()
    "
