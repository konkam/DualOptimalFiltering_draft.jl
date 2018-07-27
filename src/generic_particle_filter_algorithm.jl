using StatsFuns

function generic_particle_filtering(Mt, Gt, N, RS)
    times::Array{Float64, 1} = Mt |> keys |> collect |> sort
    # Initialisation
    X = Array{Float64, 2}(N,length(times))
    w = Array{Float64, 2}(N,length(times))
    W = Array{Float64, 2}(N,length(times))
    A = Array{Int64, 1}(N)
    X[:,1] = rand(Mt[times[1]], N)
    w[:,1] =  Gt[times[1]].(X[:,1])
    W[:,1] = w[:,1] |> normalise

    #Filtering
    for t in 2:length(times)
        time::Float64 = times[t]
        A::Array{Int64, 1} = RS(W[:,t-1])
        X[:,t] = Mt[time](X[A,t-1])
        w[:,t] =  Gt[times[t]].(X[:,t])
        W[:,t] = w[:,t] |> normalise
    end

    return Dict("w" => w, "W" => W, "X" => X)

end

function generic_particle_filtering_logweights(Mt, logGt, N, RS)
    times::Array{Float64, 1} = Mt |> keys |> collect |> sort
    # Initialisation
    X = Array{Float64, 2}(N,length(times))
    logw = Array{Float64, 2}(N,length(times))
    logW = Array{Float64, 2}(N,length(times))
    A = Array{Int64, 1}(N)
    X[:,1] = rand(Mt[times[1]], N)
    logw[:,1] =  logGt[times[1]].(X[:,1])
    logW[:,1] = logw[:,1] - StatsFuns.logsumexp(logw[:,1])

    #Filtering
    for t in 2:length(times)
        time::Float64 = times[t]
        A::Array{Int64, 1} = RS(exp.(logW[:,t-1]))
        X[:,t] = Mt[time](X[A,t-1])
        logw[:,t] = logGt[times[t]].(X[:,t])
        logW[:,t] = logw[:,t] - StatsFuns.logsumexp(logw[:,t])
    end

    return Dict("logw" => logw, "logW" => logW, "X" => X)

end

function indices_from_multinomial_sample_slow(A)
    return [repeat([k], inner = A[k]) for k in 1:length(A)] |> x -> vcat(x...)
end

function ESS(W)
    return 1/(W.^2 |> sum)
end

function logESS(logW)
    return - logsumexp(2*logW)
end

function generic_particle_filtering_adaptive_resampling(Mt, Gt, N, RS)
    times::Array{Float64, 1} = Mt |> keys |> collect |> sort
    ESSmin = N/2 # typical choice
    # Initialisation
    X = Array{Float64, 2}(N,length(times))
    w = Array{Float64, 2}(N,length(times))
    W = Array{Float64, 2}(N,length(times))
    A = Array{Int64, 1}(N)
    ŵ = Array{Int64, 1}(N)
    resampled = Array{Bool, 1}(length(times))

    X[:,1] = rand(Mt[times[1]], N)
    w[:,1] =  Gt[times[1]].(X[:,1])
    W[:,1] = w[:,1] |> normalise
    resampled[1] = true #this is intended to make the likelihood computations work

    #Filtering
    for t in 2:length(times)
        time::Float64 = times[t]
        if ESS(W[:,t-1]) < ESSmin
            A::Array{Int64, 1} = RS(W[:,t-1])
            ŵ = 1
            resampled[t] = true
        else
            A = 1:N
            ŵ = w[:,t-1]
            resampled[t] = false
        end
        X[:,t] = Mt[time](X[A,t-1])
        w[:,t] =  ŵ .* Gt[times[t]].(X[:,t])
        W[:,t] = w[:,t] |> normalise
    end

    return Dict("w" => w, "W" => W, "X" => X, "resampled" => resampled)
end


function generic_particle_filtering_adaptive_resampling_logweights(Mt, logGt, N, RS)
    times::Array{Float64, 1} = Mt |> keys |> collect |> sort
    logESSmin = log(N) - log(2) # typical choice
    # Initialisation
    X = Array{Float64, 2}(N,length(times))
    logw = Array{Float64, 2}(N,length(times))
    logW = Array{Float64, 2}(N,length(times))
    A = Array{Int64, 1}(N)
    logŵ = Array{Float64, 1}(N)
    resampled = Array{Bool, 1}(length(times))

    X[:,1] = rand(Mt[times[1]], N)
    logw[:,1] =  logGt[times[1]].(X[:,1])
    logW[:,1] = logw[:,1] - StatsFuns.logsumexp(logw[:,1])
    resampled[1] = true #this is intended to make the likelihood computations work

    #Filtering
    for t in 2:length(times)
        time::Float64 = times[t]
        if logESS(logW[:,t-1]) < logESSmin
            A::Array{Int64, 1} = RS(exp.(logW[:,t-1]))
            logŵ .= 0.
            resampled[t] = true
        else
            A = 1:N
            logŵ = logw[:,t-1]
            resampled[t] = false
        end
        X[:,t] = Mt[time](X[A,t-1])
        logw[:,t] =  logŵ .+ logGt[times[t]].(X[:,t])
        logW[:,t] = logw[:,t] - StatsFuns.logsumexp(logw[:,t])
    end

    return Dict("logw" => logw, "logW" => logW, "X" => X, "resampled" => resampled)
end

function marginal_likelihood_factors(particle_filter_output)
    return mean(particle_filter_output["w"], 1) |> vec
end

function marginal_likelihood_factors_adaptive_resampling(particle_filter_output_adaptive_resampling)
    ntimes = size(particle_filter_output_adaptive_resampling["w"],2)
    resampled = particle_filter_output_adaptive_resampling["resampled"]
    w = particle_filter_output_adaptive_resampling["w"]
    res = Array{Float64,1}(ntimes)
    for t in 1:ntimes
        if (resampled[t])
            res[t] = mean(w[:, t])
        else
            res[t] = sum(w[:, t])/sum(w[:, t-1])
        end
    end
    return res
end

function marginal_likelihood(particle_filter_output, marginal_likelihood_factors_function)
    return marginal_likelihood_factors_function(particle_filter_output) |> prod
end

function marginal_loglikelihood_factors(particle_filter_output_logweights)
    logw = particle_filter_output_logweights["logw"]
    N = size(logw,1)
    mapslices(StatsFuns.logsumexp, logw, 1)
    return vec(mapslices(StatsFuns.logsumexp, logw, 1) .- log(N))
end


function marginal_loglikelihood_factors_adaptive_resampling(particle_filter_output_adaptive_resampling_logweights)
    ntimes = size(particle_filter_output_adaptive_resampling_logweights["logw"], 2)
    N = size(particle_filter_output_adaptive_resampling_logweights["logw"], 1)
    resampled = particle_filter_output_adaptive_resampling_logweights["resampled"]
    logw = particle_filter_output_adaptive_resampling_logweights["logw"]
    res = Array{Float64,1}(ntimes)
    for t in 1:ntimes
        if (resampled[t])
            res[t] = logsumexp(logw[:, t]) .- log(N)
        else
            res[t] = logsumexp(logw[:, t]) - logsumexp(logw[:, t-1])
        end
    end
    return res
end

function marginal_loglikelihood(particle_filter_output_logweights, marginal_loglikelihood_factors_fun)
    return marginal_loglikelihood_factors_fun(particle_filter_output_logweights) |> sum
end

function sample_from_filtering_distributions(particle_filter_output, nsamples, index::Integer)
    return particle_filter_output["X"][rand(Categorical(particle_filter_output["W"][:, index]), nsamples), index]
end

function sample_from_filtering_distributions_logweights(particle_filter_output_logweights, nsamples, index::Integer)
    return particle_filter_output_logweights["X"][rand(Categorical(exp.(particle_filter_output_logweights["logW"][:, index])), nsamples), index]
end

function create_potential_functions(data, potential)
    times = data |> keys |> collect |> sort
    return zip(times, [potential(data[t]) for t in times]) |> Dict
end

function create_potential_functions_CIR(data)
    function potential(y)
        return x -> prod(pdf.(Poisson(x), y))
    end
    return create_potential_functions(data, potential)
end

function create_log_potential_functions_CIR(data)
    function potential(y)
        return x -> sum(logpdf.(Poisson(x), y))
    end
    return create_potential_functions(data, potential)
end

function create_transition_kernels(data, transition_kernel, prior)
    times = data |> keys |> collect |> sort
    return zip(times, [prior; [transition_kernel(times[k]-times[k-1]) for k in 2:length(times)]]) |> Dict
end

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

function create_transition_kernels_CIR(data, δ, γ, σ)
    function create_Mt(Δt)
        function Mt(X)
    #         Aind = DualOptimalFiltering.indices_from_multinomial_sample_slow(A)
            return rCIR.(1, Δt, X, δ, γ, σ)
        end
        return Mt
    end
    prior = Gamma(δ/2, σ^2/γ)#parameterisation shape scale
    create_transition_kernels(data, create_Mt, prior)
end
