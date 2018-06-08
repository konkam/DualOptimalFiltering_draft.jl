using DualOptimalFiltering
using Base.Test

using Distributions
# Run tests
#@test 1 == 2

tic()
println("Testing common utility functions")
@time include("tests_CommonUtilityFunctions.jl")

# println("Testing CIR filtering")
# @time include("tests_CIRfiltering.jl")
#
#
# println("Testing WF filtering")
# @time include("tests_WFfiltering.jl")
#
# println("Testing pruning functions")
# @time include("tests_pruning_functions.jl")

println("Testing statistical distances on the simplex")
@time include("test_statistical_distances_on_the_simplex.jl")

# println("Testing Dirichlet Kernel Density Estimate")
# @time include("test_dirichlet_kde.jl")

toc()
