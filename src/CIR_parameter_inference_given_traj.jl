function param_sampler(trajectory, times, prior_logpdf, θ_init, niter;    final_chain_length = 1000, silence = false)

    function unnormalised_logposterior(a, b, σ_prime)::Float64
        prior_contribution = prior_logpdf(a, b, σ_prime)
        if isinf(prior_contribution) #Avoids likelihood computation in this case
            return prior_contribution
        else
            δ, γ, σ = inverse_reparam_CIR(a, b, σ_prime)
            return prior_contribution + hidden_signal_loglikelihood_CIR(trajectory, times, δ, γ, σ)::Float64
        end
    end

    parameter_chain = Array{Float64,2}(undef, niter, 3)

    θ_it = θ_init

    Jtsym_rand(θ) = rand.(Normal.(0, (0.25, 0.25, 0.125))) .+ θ

    for it in 1:niter

        print_every = max(1, floor(niter/20))

        if !silence && mod(it, print_every) == 0
            @info "$it iterations out of $niter"
        end

        @debug "iteration $it"
        @debug "trajectory" X_it

        # Metropolis-Hastings step

        θ_it = draw_next_sample(θ_it, Jtsym_rand, st -> unnormalised_logposterior(st[1], st[2], st[3]))

        # Record iteration

        parameter_chain[it,:] = collect(θ_it)

    end

    return parameter_chain
end
