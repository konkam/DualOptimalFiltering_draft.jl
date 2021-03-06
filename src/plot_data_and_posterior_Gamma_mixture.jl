function plot_data_and_posterior_distribution(δ, θ_of_t, Λ_of_t, wms_of_t, data, X_CIR)
    times = keys(data) |> collect |> sort;
    psi_t = [DualOptimalFiltering_proof.create_Gamma_mixture_density(δ, θ_of_t[t], Λ_of_t[t], wms_of_t[t]) for t in keys(data) |> collect |> sort];
    expect_mixture = [sum(wms_of_t[t].*(δ/2 .+ Λ_of_t[t]) ./ θ_of_t[t]) for t in times]
    qt0025 = [DualOptimalFiltering_proof.compute_quantile_mixture_hpi(δ, θ_of_t[t], Λ_of_t[t], wms_of_t[t], 0.025) for t in keys(data) |> collect |> sort];
    qt0975 = [DualOptimalFiltering_proof.compute_quantile_mixture_hpi(δ, θ_of_t[t], Λ_of_t[t], wms_of_t[t], 0.975) for t in keys(data) |> collect |> sort];
    R"$([f.(range(0, stop = maximum(data |> values |> collect |> x -> vcat(x...)), length = 200)) for f in psi_t] |> x -> hcat(x...)) %>%
        as_tibble %>%
        mutate(x = $(range(0, stop = maximum(data |> values |> collect |> x -> vcat(x...)), length = 200))) %>%
        gather(time, value, -x) %>%
        mutate(time = gsub('V','',time) %>% as.numeric) %>%
        mutate(time = $(Λ_of_t |> keys |> collect |> sort) %>% unlist %>% (function(x) x[time])) %>%
        ggplot(aes(x = time, y = x)) +
            geom_raster(aes(fill = value^.25), interpolate= T, alpha = 1) +
            scale_fill_gradient(low = 'white', high = 'red') +
            geom_point(data = $([data[t] for t in times] |> X -> hcat(X...)') %>%
                as_tibble %>%
                mutate(time = seq_along(V1)) %>%
                gather(var, x, -time) %>%
            mutate(time = $(Λ_of_t |> keys |> collect |> sort) %>% unlist %>% (function(x) x[time])), alpha = 0.5) +
                #geom_line(data = tibble(time = $times, x = $expect_mixture), colour = 'blue') +
                geom_line(data = tibble(time = $times, x = $qt0025), colour = 'red') +
                geom_line(data = tibble(time = $times, x = $qt0975), colour = 'red') +
                geom_line(data = tibble(x = $X_CIR) %>% mutate(time = seq_along(x))%>%
                                    mutate(time = $(Λ_of_t |> keys |> collect |> sort) %>% unlist %>% (function(x) x[time])), colour = 'blue') +
            xlab('Time') +
            guides(fill=FALSE) +
            theme_minimal()
    "
end
