module DualOptimalFiltering

# package code goes here

export generate_CIR_trajectory, filter_CIR, filter_WF

include("CIRfiltering.jl")
include("WFfiltering.jl")
include("CommonUtilityFunctions.jl")
include("finite_size_wright_fisher_simulation.jl")

end # module
