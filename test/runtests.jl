using Test, Distributions, Random
using DualOptimalFiltering
# Run tests
@time @test 1 == 1

println("Testing common utility functions")
@time include("tests_CommonUtilityFunctions.jl")

println("Testing CIR filtering")
@time include("tests_CIRfiltering.jl")

println("Testing WF filtering")
@time include("tests_WFfiltering.jl")

println("Testing pruning functions")
@time include("tests_pruning_functions.jl")

println("Testing statistical distances on the simplex")
@time include("test_statistical_distances_on_the_simplex.jl")

println("Testing Dirichlet Kernel Density Estimate")
@time include("test_dirichlet_kde.jl")


println("Testing the exact CIR likelihood functions")
@time include("test_CIR_likelihood.jl")

println("Testing the WF particle filtering functions")
@time include("test_WF_particle_filter.jl")

println("Testing the plot functions")
@time include("test_plot_functions.jl")

println("Testing the exact L2 distances formulas")
@time include("test_exact_L2_distances.jl")

println("Testing the kde functions")
@time include("test_kde_for_pf_samples.jl")
