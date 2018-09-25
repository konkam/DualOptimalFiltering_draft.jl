using Distributions

function perform_immigration(z::Array{Float64,1}, V::Array{Float64,2})
  return vec(z' * V)
end

# function perform_mutation(z::Array{Float64,1}, U::Array{Float64,2})
function perform_mutation(z, U::Array{Float64,2})
  return vec(z' * U)
end

function perform_immigration_mutation(z::Array{Float64,1}, α::Float64, θ::Float64, K::Int64, N::Int64, r::Array{Float64,1}, p::Array{Float64,1})
  return z + 1 ./ N * 1 ./ 2. * ( (θ + α)/(K-1)*(1-z) - (θ + α)*z + α*p*sum(r) - α*r )
end

function perform_PD1_mutation(z::Array{Float64,1}, θ::Float64, K::Int64, N::Int64)
  return z + 1/(2*N) * ( -θ * z + θ/(K-1)*(1-z))
end

# function perform_resampling(z::Array{Float64,1}, N::Int64)
function perform_resampling(z, N::Int64)
  return 1 ./ N * rand(Multinomial(N,z |> collect))
end

# convert(::Type{Bool}, x::Array{Any}) = broadcast(Bool, x)

function compute_U(Θ::Float64, α::Float64, N::Int64, K::Int64)
  U = fill((Θ + α)/(2*N*(K-1)), K, K)

  for k in 1:K
    U[k,k] = 1 .- (Θ + α)/(2 .* N)
  end

  return U
end

function compute_U_simple(Θ::Float64, α::Float64, N::Int64, K::Int64)
  U = fill(Θ/(2*N*(K-1)), K, K)

  for k in 1:K
    U[k,k] = 1 .- Θ/(2 .* N)
  end

  return U
end

function r26(z_i::Float64, K::Int64)
  (1-z_i) * (1 - (1 - z_i)^(K-1))
end

function r_simple(z_i::Float64, K::Int64)
  1 - (1 - z_i)^K
end

function p27(z_j::Float64, z::Array{Float64,1}, K::Int64)
  (1-z_j)^K / sum((1-z).^K)
end

function compute_V(α::Float64, z::Array{Float64,1}, p::Array{Float64,1}, r::Array{Float64,1}, K::Int64, N::Int64)
  V = Array{Float64}(K,K)
  sump = sum(p)
  for j in 1:K
    for i in setdiff(1:K, j)
      if ( (r[i]==0.) || (p[j]==0.) )
         V[i,j] = 0.
      else
         V[i,j] = α * r[i] * p[j] / (2 * N * z[i])
       end
    end
  end
  # for k in 1:K
  #   V[k,k] = 1. - α * r[k] / (2*N*z[k]) * (sump - p[k])
  # end
  for k in 1:K
    V[k,k] = 1. - sum(Float64[V[k,j] for j in setdiff(1:K, k)])
  end
  return V
end

function compute_V_faster(α::Float64, z::Array{Float64,1}, p::Array{Float64,1}, r::Array{Float64,1}, K::Int64, N::Int64)
  V = Array{Float64}(K,K)
  # sump = sum(p)
  for j in 1:K
    # for i in setdiff(1:K, j)
    for i in 1:K
      if ( (r[i]==0.) || (p[j]==0.) )
         V[i,j] = 0.
      else
         V[i,j] = α * r[i] * p[j] / (2 * N * z[i])
       end
    end
  end
  # for k in 1:K
  #   V[k,k] = 1. - α * r[k] / (2*N*z[k]) * (sump - p[k])
  # end
  for k in 1:K
    # V[k,k] = 1. - sum(Float64[V[k,j] for j in setdiff(1:K, k)])
    V[k,k] = 1. - sum(V[k,:]) + V[k,k]
  end
  return V
end
#
# function compute_V_simple_faster(α::Float64, z::Array{Float64,1}, K::Int64, N::Int64)
#   r = [r_simple(z_i, K) for z_i in z]
#   p = [p27(z_j, z, K) for z_j in z]
#   return compute_V_faster(α, z, p, r, K, N)
# end

function compute_V26_27(α::Float64, z::Array{Float64,1}, K::Int64, N::Int64)
  r = [r26(z_i, K) for z_i in z]
  p = [p27(z_j, z, K) for z_j in z]
  return compute_V_faster(α, z, p, r, K, N)
end

function compute_V_simple(α::Float64, z::Array{Float64,1}, K::Int64, N::Int64)
  r = [r_simple(z_i, K) for z_i in z]
  p = [p27(z_j, z, K) for z_j in z]
  return compute_V_faster(α, z, p, r, K, N)
end

function wright_fisher_PD1_bare(z_init::Array{Float64,1}, θ::Float64, N::Int64, Nsteps::Int64)
  K = length(z_init)
  # println("1")/
  res = Array{Float64}(K, Nsteps + 1)
  res[:,1] = z_init

  # zstar = copy(z_init)
  counter = 0
  # println("2")
  while counter < Nsteps
    counter += 1

    zstar = perform_PD1_mutation(res[:, counter], θ, K, N)
    res[:, counter + 1] = perform_resampling(zstar, N)
  end
  return res
end

function wright_fisher_PD1_for(z_init::Array{Float64,1}, θ::Float64, N::Int64, Nsteps::Int64)
  K = length(z_init)
  res = Array{Float64}(K, Nsteps + 1)
  res[:,1] = z_init

  @simd for i in 1:Nsteps
     @inbounds zstar = perform_PD1_mutation(res[:, i], θ, K, N)
     @inbounds res[:, i + 1] = perform_resampling(zstar, N)
  end

  return res
end

function L_steps_forward_PD1(z_init::Array{Float64,1}, θ::Float64, K::Int64, N::Int64, L::Int64)
  zstar = z_init
  for i in 1:L
    zstar = perform_PD1_mutation(zstar, θ, K, N)
    zstar = perform_resampling(zstar, N)
  end
  return zstar
end


function wright_fisher_PD1_thin(z_init::Array{Float64,1}, θ::Float64, N::Int64, Nsteps::Int64, thin::Int64)
  K = length(z_init)
  res = Array{Float64}(K, Nsteps + 1)
  res[:,1] = z_init
  counter = 0
  while counter < Nsteps
    counter += 1
    res[:, counter + 1] = L_steps_forward_PD1(res[:, counter], θ::Float64, K::Int64, N::Int64, thin)
  end
  return res
end

function wright_fisher_PD1_warmup(z_init::Array{Float64,1}, θ::Float64, N::Int64, Nsteps::Int64, warmup::Int64)
  if warmup==0
    start = z_init
  else
    K = length(z_init)
    start = L_steps_forward_PD1(z_init::Array{Float64,1}, θ::Float64, K::Int64, N::Int64, warmup::Int64)
  end
  return wright_fisher_PD1_bare(start, θ, N, Nsteps)
end

function wright_fisher_PD1(z_init::Array{Float64,1}, θ::Float64, N::Int64, Nsteps::Int64; warmup::Int64 = 0, thin::Int64 = 1)
  if thin==1
    wright_fisher_PD1_warmup(z_init::Array{Float64,1}, θ::Float64, N::Int64, Nsteps::Int64, warmup::Int64)
  else
    if warmup==0
      start::Array{Float64,1} = z_init
    else
      K = length(z_init)
      start = L_steps_forward_PD1(z_init::Array{Float64,1}, θ::Float64, K::Int64, N::Int64, warmup*thin::Int64)
    end
    return wright_fisher_PD1_thin(start::Array{Float64,1}, θ::Float64, N::Int64, Nsteps::Int64, thin::Int64)
  end
end


function wright_fisher_PD2_firstorderN(z_init::Array{Float64,1}, α::Float64, θ::Float64, N::Int64, Nsteps::Int64)
  K = length(z_init)
  # U = compute_U(θ, α, N, K)
  # println("1")
  res = Array{Float64}(K, Nsteps + 1)
  res[:,1] = z_init

  counter = 0
  # println("2")
  while counter < Nsteps
    counter += 1

    r = [r26(z_i, K) for z_i in res[:, counter]]
    p = [p27(z_j, res[:, counter], K) for z_j in res[:, counter]]
    # println(res[:, counter])
    zstarstar = perform_immigration_mutation(res[:, counter], α, θ, K, N, r, p) #eq. (17)
    # println(zstarstar)
    res[:, counter + 1] = perform_resampling(zstarstar, N) #eq. (13)

  end
  return res
end


function wright_fisher_PD2(z_init::Array{Float64,1}, α::Float64, θ::Float64, N::Int64, Nsteps::Int64)
  K = length(z_init)

  U = compute_U(θ, α, N, K)

  # println("1")
  res = Array{Float64}(K, Nsteps + 1)
  res[:,1] = z_init

  counter = 0
  # println("2")
  while counter < Nsteps
    counter += 1

    # println(counter)
    # println(res[:, counter])
    V = compute_V26_27(α, res[:, counter], K, N)
    # println(V)
    zstar = perform_immigration(res[:, counter], V) #eq. (11)
    # println(zstar)
    # println(typeof(zstar))
    zstarstar = perform_mutation(zstar, U) #eq. (12)
    # println(zstarstar)

    res[:, counter + 1] = perform_resampling(zstarstar, N) #eq. (13)

  end
  return res
end

function wright_fisher_PD2_new_parametrisation_bare(z_init::Array{Float64,1}, α::Float64, θ::Float64, N::Int64, Nsteps::Int64)
  K = length(z_init)
  U_simple = compute_U_simple(θ, α, N, K)
  res = Array{Float64}(K, Nsteps + 1)
  res[:,1] = z_init
  counter = 0
  while counter < Nsteps
    counter += 1
    V_simple = compute_V_simple(α, res[:, counter], K, N)
    zstar = perform_immigration(res[:, counter], V_simple) #eq. (11)
    zstarstar = perform_mutation(zstar, U_simple) #eq. (12)
    res[:, counter + 1] = perform_resampling(zstarstar, N) #eq. (13)
  end
  return res
end


function L_steps_forward_PD2_new_parametrisation(z_init::Array{Float64,1}, α::Float64, U::Array{Float64,2}, K::Int64, N::Int64, L::Int64)
  zstar = z_init
  for i in 1:L
    V = compute_V_simple(α, zstar, K, N)
    zstar = perform_immigration(zstar, V) #eq. (11)
    zstar = perform_mutation(zstar, U) #eq. (12)
    zstar = perform_resampling(zstar, N) #eq. (13)
  end
  return zstar
end


function wright_fisher_PD2_new_parametrisation_thin(z_init::Array{Float64,1}, α::Float64, θ::Float64, N::Int64, Nsteps::Int64, thin::Int64)
  K = length(z_init)
  U = compute_U(θ, α, N, K)
  res = Array{Float64}(K, Nsteps + 1)
  res[:,1] = z_init
  counter = 0
  while counter < Nsteps
    counter += 1
    res[:, counter + 1] = L_steps_forward_PD2_new_parametrisation(res[:, counter], α::Float64, U::Array{Float64,2}, K::Int64, N::Int64, thin)
  end
  return res
end

function wright_fisher_PD2_new_parametrisation_warmup(z_init::Array{Float64,1}, α::Float64, θ::Float64, N::Int64, Nsteps::Int64, warmup::Int64)
  if warmup==0
    start = z_init
  else
    K = length(z_init)
    U = compute_U(θ, α, N, K)
    start = L_steps_forward_PD2_new_parametrisation(z_init::Array{Float64,1}, α::Float64, U::Array{Float64,2}, K::Int64, N::Int64, warmup::Int64)
  end
  return wright_fisher_PD2_new_parametrisation_bare(start, α::Float64, θ, N, Nsteps)
end

function wright_fisher_PD2_new_parametrisation(z_init::Array{Float64,1}, α::Float64, θ::Float64, N::Int64, Nsteps::Int64; warmup::Int64 = 0, thin::Int64 = 1)
  if thin==1
    wright_fisher_PD2_new_parametrisation_warmup(z_init::Array{Float64,1}, α::Float64, θ::Float64, N::Int64, Nsteps::Int64, warmup::Int64)
  else
    if warmup==0
      start = z_init
    else
      K = length(z_init)
      start = wright_fisher_PD2_new_parametrisation_bare(z_init::Array{Float64,1}, α::Float64, θ::Float64, N::Int64, warmup*thin::Int64)[:,end]
    end
    return wright_fisher_PD2_new_parametrisation_thin(start::Array{Float64,1}, α::Float64, θ::Float64, N::Int64, Nsteps::Int64, thin::Int64)
  end
end



# function wright_fisher_PD1(z_init::Array{Float64,1}, θ::Float64, N::Int64, Nsteps::Int64, thin::Int64)
#   K = length(z_init)
#
#   res = Array{Float64}(K, Nsteps + 1)
#
#   res[:,1] = z_init
#
#   counter = 0
#
#   while counter < Nsteps * thin
#     counter += 1
#
#     zstar = perform_PD1_mutation(res[:, counter], θ, K, N)
#     res[:, counter + 1] = perform_resampling(zstar, N)
#   end
#   return res[:,1:thin:end]
# end

##tests
# rand(Dirichlet(10,0.3)) |> z-> perform_immigration(z,compute_V26_27(1., z, 10, 100)) |> sum
# rand(Dirichlet(10,0.3)) |> z-> perform_mutation(z,compute_U(1.,1., 100, 10)) |> sum
# rand(Dirichlet(10,0.3)) |> z-> perform_PD1_mutation(z, 1., 10, 100) |> sum
# alpha = .5
# theta = 1.5
# K = 2
# N = 1000
# iter = 5*10^6
# thinplot = 100
# rand(Dirichlet(10,0.3)) |> z -> 1 ./ N * rand(Multinomial(N,z)) |> sum
#
# srand(1)
# rand(Dirichlet(K,0.3)) |> z -> wright_fisher_PD2_new_parametrisation(z, alpha, theta, N::Int64, 3)
# srand(1)
# rand(Dirichlet(K,0.3)) |> z -> wright_fisher_PD2(z, alpha, theta, N::Int64, 3)
# srand(2)
# rand(Dirichlet(K,0.3)) |> z -> wright_fisher_PD2_new_parametrisation(z, alpha, theta, N::Int64, 1000)[:,105:107]
# srand(2)
# rand(Dirichlet(K,0.3)) |> z -> wright_fisher_PD2(z, alpha, theta, N::Int64, 1000)[:,105:107]
# srand(1)
# U = compute_U(theta, alpha, N, K)
# rand(Dirichlet(K,0.3)) |> z -> L_steps_forward_PD2_new_parametrisation(z, alpha, U, K::Int64, N::Int64, 4)
# srand(1)
# rand(Dirichlet(K,0.3)) |> z -> wright_fisher_PD2_new_parametrisation(z, alpha, theta, N::Int64, 3, warmup = 1, thin = 2)
# srand(1)
# rand(Dirichlet(K,0.3)) |> z -> wright_fisher_PD2(z, alpha, theta, N::Int64, 8)
# srand(1)
# rand(Dirichlet(K,0.3)) |> z -> wright_fisher_PD2_new_parametrisation(z, alpha, theta, N::Int64, 100, warmup = 0, thin = 1)[:,5:10]
# srand(1)
# rand(Dirichlet(K,0.3)) |> z -> wright_fisher_PD2(z, 0.5, theta, N::Int64, 100)[:,5:10]
#
#
# alpha = .5
# theta = 1.5
# K = 2
# N = 100
# iter = 1*10^6
# thinplot = 100
# #
# code_to_profile() = chain = rand(Dirichlet(K,0.3)) |> z-> wright_fisher_PD2_new_parametrisation_bare(z, alpha, theta, N, 2*iter)[:,(end-iter):end]
#
# code_to_profile()
#
# using ProfileView
#
# function benchmark()
#     # Any setup code goes here.
#
#     # Run once, to force compilation.
#     println("======================= First run:")
#     srand(666)
#     @time code_to_profile()
#
#     # Run a second time, with profiling.
#     println("\n\n======================= Second run:")
#     srand(666)
#     Profile.init(delay=0.01)
#     Profile.clear()
#     # clear_malloc_data()
#     @profile @time code_to_profile()
#
#     # Write profile results to profile.bin.
#     r = Profile.retrieve()
#     f = open("profile.bin", "w")
#     serialize(f, r)
#     close(f)
# end
#
# benchmark()
#
# ProfileView.view()
