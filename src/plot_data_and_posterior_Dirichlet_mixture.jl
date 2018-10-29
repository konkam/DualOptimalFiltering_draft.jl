using Roots, IterTools, DataFrames, DataFramesMeta, RCall
R"library(tidyverse)"

function compute_quantile_beta_mixture(α, Λ, wms, margin, q::Float64)
    function Beta_mixture_cdf(x::Real)
        return sum(wms.*Float64[cdf(Beta(m[margin] + α[margin], sum(m .+ α) - m[margin] - α[margin]),x) for m in Λ])
    end
    return fzero(x -> Beta_mixture_cdf(x)-q, 0, 10^9)
end

function create_marginal_beta_mixture(α, Λ, wms, margin)
    function Beta_mixture(x::Real)
        return sum(wms.*Float64[pdf(Beta(m[margin] + α[margin], sum(m .+ α) - m[margin] - α[margin]),x) for m in Λ])
    end
    return Beta_mixture
end

function compute_marginal_CI(α, Λ_of_t, wms_of_t; mass = 0.95)
    qinfCI, qupCI = get_quantiles_from_mass(mass)
    K = length(α)
    times = Λ_of_t |> keys |> collect |> sort
    Iterators.product(times, 1:K) |>
        collect |>
        x -> map(pr -> DataFrame(time = pr[1],
            margin = pr[2],
            infCI = compute_quantile_beta_mixture(α, Λ_of_t[pr[1]], wms_of_t[pr[1]], pr[2], qinfCI),
            supCI = compute_quantile_beta_mixture(α, Λ_of_t[pr[1]], wms_of_t[pr[1]], pr[2], qupCI)), x) |>
        dflist -> vcat(dflist...)

end

function for_one_marginal(α, margin, Λ_of_t, wms_of_t)
    times = Λ_of_t |> keys |> collect |> sort

    marginal_psi_t = [create_marginal_beta_mixture(α, Λ_of_t[t], wms_of_t[t], margin) for t in times]
    [DataFrame(x = range(0, stop = 1, length = 100)) |> df -> @transform(df, dens = f.(:x)) for f in marginal_psi_t] |>
dfs -> map((dfi, ti) -> @transform(dfi, time = ti, variable = "x$margin"), dfs, times) |>
    dfs -> vcat(dfs...)
end

function marginal_densities(α, Λ_of_t, wms_of_t)
    K = Λ_of_t[0]... |> length
    map(margin -> for_one_marginal(α, margin, Λ_of_t, wms_of_t), 1:K) |>
    dfs -> vcat(dfs...)
end


function plot_marginal_posterior_credible_interval_and_data(α, Λ_of_t, wms_of_t, data_; mass = 0.95)
    marginal_CI = compute_marginal_CI(α, Λ_of_t, wms_of_t; mass = mass)

    plot_marginal_posterior_credible_interval_and_data_given_marginalCI(α, Λ_of_t, wms_of_t, data_, marginal_CI)
end

function plot_marginal_posterior_credible_interval_and_data_given_marginalCI(α, Λ_of_t, wms_of_t, data_, marginal_CI)

    R"$data_ %>%
        (function(lst){
            Reduce(rbind, lst) %>%
            cbind(names(lst)) %>%
            as_tibble %>%
            mutate_all(as.numeric) %>%
            setNames(c(paste('x', 1:(ncol(.)-1), sep=''), 't'))
        }) %>%
        mutate(pop_size = rowSums(.) - t) %>%
        gather(variable, n, -t, -pop_size) %>%
        ggplot(aes(x = t, y = n/pop_size)) +
        geom_raster(data = $(marginal_densities(α, Λ_of_t, wms_of_t)) %>%
                                as_tibble %>%
                                group_by(x, time),
            aes(x = time, fill = dens, y = x, colour = NULL), interpolate= T, alpha = 1) +
        scale_fill_gradient(low = 'white', high = 'red') +
        #viridis::scale_colour_viridis(discrete = T) +
        geom_line(colour = 'blue') +
        theme_minimal() +
        ylim(0,1) +
        ylab('Fraction') +
        facet_wrap(~variable) +
        geom_line(data = $marginal_CI %>% as_tibble %>% mutate(variable = paste('x', margin, sep='')), aes(x = time, y = infCI), colour = 'red', linetype = 'dashed') +
        geom_line(data = $marginal_CI %>% as_tibble %>% mutate(variable = paste('x', margin, sep='')), aes(x = time, y = supCI), colour = 'red', linetype = 'dashed')
    "
end


function plot_marginal_posterior_credible_interval_and_data_with_hidden_state(α, Λ_of_t, wms_of_t, data_, wfchain; mass = 0.95)
    marginal_CI = compute_marginal_CI(α, Λ_of_t, wms_of_t; mass = mass)

    plot_marginal_posterior_credible_interval_and_data_given_marginalCI_with_hidden_state(α, Λ_of_t, wms_of_t, data_, wfchain, marginal_CI)
end

function plot_marginal_posterior_credible_interval_and_data_given_marginalCI_with_hidden_state(α, Λ_of_t, wms_of_t, data_, wfchain, marginal_CI)
    times = Λ_of_t |> keys |> collect |> sort

    R"$data_ %>%
        (function(lst){
            Reduce(rbind, lst) %>%
            cbind(names(lst)) %>%
            as_tibble %>%
            mutate_all(as.numeric) %>%
            setNames(c(paste('x', 1:(ncol(.)-1), sep=''), 't'))
        }) %>%
        mutate(pop_size = rowSums(.) - t) %>%
        gather(variable, n, -t, -pop_size) %>%
        ggplot(aes(x = t, y = n/pop_size)) +
        geom_raster(data = $(marginal_densities(α, Λ_of_t, wms_of_t)) %>%
                                as_tibble %>%
                                group_by(x, time),
            aes(x = time, fill = dens, y = x, colour = NULL), interpolate= T, alpha = 1) +
        scale_fill_gradient(low = 'white', high = 'red', name = 'Posterior density') +
        #viridis::scale_colour_viridis(discrete = T) +
        geom_point(colour = 'black') +
        geom_line(data = $wfchain %>%
                            t %>%
                            as_tibble %>%
                            mutate(time = $times %>% unlist) %>%
                            gather(variable, x, -time) %>%
                            mutate(variable = gsub('V', 'x', variable)), aes(x = time, y = x),  colour = 'blue') +
        theme_minimal() +
        ylim(0,1) +
        ylab('Fraction') +
        facet_wrap(~variable) +
        geom_line(data = $marginal_CI %>% as_tibble %>% mutate(variable = paste('x', margin, sep='')), aes(x = time, y = infCI), colour = 'red', linetype = 'dashed') +
        geom_line(data = $marginal_CI %>% as_tibble %>% mutate(variable = paste('x', margin, sep='')), aes(x = time, y = supCI), colour = 'red', linetype = 'dashed') +
    xlab('Time')
    "
end
