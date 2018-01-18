using DualOptimalFiltering
using Base.Test

using Distributions
# Run tests
#@test 1 == 2

tic()
println("Testing common utility functions")
@time include("tests_CommonUtilityFunctions.jl")

println("Testing CIR filtering")
@time include("tests_CIRfiltering.jl")

toc()
