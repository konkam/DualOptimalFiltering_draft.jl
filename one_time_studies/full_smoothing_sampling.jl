using Revise, DualOptimalFiltering, Random, Distributions, RCall, DataFrames, DataFramesMeta, Optim, FeynmanKacParticleFilters, BenchmarkTools, MCMCDiagnostics
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

data_CIR, Y_CIR, X_CIR, times, δ, γ, σ, λ = simulate_CIR_data(;Nsteps_CIR = 50)

ntraj = 100

Λ_of_t, logwms_of_t, θ_of_t = DualOptimalFiltering.filter_CIR_logweights(δ, γ, σ, λ, data_CIR);


# A graphical test. Commented out to allow tests to pass

R"1:$ntraj %>%
    lapply(function(xx) tibble(idx = xx, x = as.numeric($times))) %>%
    bind_rows() %>%
    mutate(y =  $(vcat([DualOptimalFiltering.sample_1_trajectory_from_joint_smoothing_CIR_logweights(δ, γ, σ, Λ_of_t, logwms_of_t, θ_of_t, 1, 1, 1, data_CIR) for k in 1:ntraj]...))) %>%
    ggplot(aes(x=x, y=y) ) +
      stat_density_2d(aes(fill = ..density..), geom = 'raster', contour = FALSE) +
      scale_fill_distiller(palette= 'Spectral', direction=1) +
      scale_x_continuous(expand = c(0, 0)) +
      scale_y_continuous(expand = c(0, 0)) +
      theme(
        legend.position='none'
      ) +
      geom_line(data = tibble(x = as.numeric($times), y = $X_CIR))
"



R"1:$ntraj %>%
    lapply(function(xx) tibble(idx = xx, x = as.numeric($times))) %>%
    bind_rows() %>%
    mutate(y =  $(vcat([DualOptimalFiltering.sample_1_trajectory_from_joint_smoothing_CIR_logweights(δ, γ, σ, Λ_of_t, logwms_of_t, θ_of_t, 1, 1, 1, data_CIR) for k in 1:ntraj]...))) %>%
    ggplot(aes(x=x, y=y, group = idx) ) +
    geom_line(colour = '#333333', alpha = 0.1) +
    theme_bw()  +
    geom_line(data = tibble(x = as.numeric($times), y = $X_CIR), aes(group = NULL), colour = 'red') +
    geom_point(data = tibble(x = rep(as.numeric($times), $(length(Y_CIR[1]))), y = $(vcat(Y_CIR...))), aes(group = NULL))
"
