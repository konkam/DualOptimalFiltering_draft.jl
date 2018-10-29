# using DualOptimalFiltering
# using Base.Test
#
# using Distributions
# # Run tests
# #@test 1 == 2
#
# tic()
# println("Testing common utility functions")
# @time include("tests_CommonUtilityFunctions.jl")
#
# println("Testing CIR filtering")
# @time include("tests_CIRfiltering.jl")
#
#
# println("Testing WF filtering")
# @time include("tests_WFfiltering.jl")
#
# println("Testing pruning functions")
# @time include("tests_pruning_functions.jl")
#
# println("Testing statistical distances on the simplex")
# @time include("test_statistical_distances_on_the_simplex.jl")
#
# println("Testing Dirichlet Kernel Density Estimate")
# @time include("test_dirichlet_kde.jl")
#
# println("Testing the generic particle filter")
# @time include("test_generic_particle_filter_algorithm.jl")
#
# println("Testing the exact CIR likelihood functions")
# @time include("test_CIR_likelihood.jl")
#
# toc()

# @elapsed begin

    using Test, Distributions, Random
    # using DataFrames, DataFramesMeta, DataStructures, Distributions, IterTools, KernelEstimator, Memoize, Nemo, Optim, QuadGK, RCall, Roots, StatsBase, StatsFuns
    using DualOptimalFiltering
    # Run tests
    @time @test 1 == 1

    # println("Testing common utility functions")
    # @time include("tests_CommonUtilityFunctions.jl")
    #
    # println("Testing CIR filtering")
    # @time include("tests_CIRfiltering.jl")

    println("Testing WF filtering")
    @time include("tests_WFfiltering.jl")

    # println("Testing pruning functions")
    # @time include("tests_pruning_functions.jl")
    #
    # println("Testing statistical distances on the simplex")
    # @time include("test_statistical_distances_on_the_simplex.jl")
    #
    # println("Testing Dirichlet Kernel Density Estimate")
    # @time include("test_dirichlet_kde.jl")
    #
    # println("Testing the generic particle filter")
    # @time include("test_generic_particle_filter_algorithm.jl")
    #
    # println("Testing the exact CIR likelihood functions")
    # @time include("test_CIR_likelihood.jl")
# end
