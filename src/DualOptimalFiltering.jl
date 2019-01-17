module DualOptimalFiltering

# package code goes here
using Printf

export generate_CIR_trajectory, filter_CIR, filter_WF, bind_rows, Hellinger_dist_1D, dirichletkernel


include("CIRfiltering.jl")
include("CIRfiltering_approximate.jl")
include("WFfiltering.jl")
include("WFfiltering_precompute.jl")
include("WFfiltering_precomputed_approximate.jl")
# include("WFfiltering_precompute_preallocate.jl")
# include("WFfiltering_precompute_preallocate_n.jl")
include("WFfiltering_precompute_Nemo_arbitrary_precision.jl")
include("WFfiltering_precompute_ar.jl")
include("WFfiltering_adaptive_precomputation.jl")
include("WF_likelihood.jl")
include("CommonUtilityFunctions.jl")
include("pruning_functions.jl")
include("finite_size_wright_fisher_simulation.jl")
include("plot_data_and_posterior_Dirichlet_mixture.jl")
include("plot_data_and_posterior_Gamma_mixture.jl")
include("statistical_distances_on_the_simplex.jl")
include("statistical_distances_on_Rplus.jl")
include("dirichlet_kde.jl")
# include("generic_particle_filter_algorithm.jl")
include("CIR_likelihood.jl")
include("WF_particle_filter.jl")
include("exact_L2_distances.jl")
include("exact_L2_distances_arb.jl")
include("kde_for_pf_samples.jl")

end # module
