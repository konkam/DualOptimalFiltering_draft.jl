module DualOptimalFiltering

# package code goes here

export generate_CIR_trajectory, filter_CIR, filter_WF

include("CIRfiltering.jl")
include("WFfiltering.jl")
include("WFfiltering_precompute.jl")
include("WFfiltering_precomputed_approximate.jl")
# include("WFfiltering_precompute_preallocate.jl")
# include("WFfiltering_precompute_preallocate_n.jl")
include("WFfiltering_precompute_Nemo_arbitrary_precision.jl")
include("CommonUtilityFunctions.jl")
include("finite_size_wright_fisher_simulation.jl")

end # module
