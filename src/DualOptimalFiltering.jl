module DualOptimalFiltering

# package code goes here
using Printf

export generate_CIR_trajectory, filter_CIR, filter_WF, bind_rows, Hellinger_dist_1D, dirichletkernel, CIR_smoothing, WF_smoothing, joint_sampler_CIR

include("general_smoothing_functions.jl")
include("CommonUtilityFunctions.jl")
include("CIRfiltering.jl")
include("CIRfiltering_approximate.jl")
include("WFfiltering.jl")
include("WFfiltering_precompute.jl")
include("WFfiltering_precomputed_approximate.jl")
include("WFfiltering_precompute_Nemo_arbitrary_precision.jl")
include("WFfiltering_precompute_ar.jl")
include("WFfiltering_adaptive_precomputation.jl")
include("WFfiltering_adaptive_precomputation_approx.jl")
include("WF_likelihood.jl")
include("WF_smoothing.jl")
include("pruning_functions.jl")
include("finite_size_wright_fisher_simulation.jl")
include("post_process_Dirichlet_mixture_posterior.jl")
# include("plot_data_and_posterior_Dirichlet_mixture.jl")
# include("plot_data_and_posterior_Gamma_mixture.jl")
# include("statistical_distances_on_the_simplex.jl")
include("statistical_distances_on_Rplus.jl")
include("dirichlet_kde.jl")
# # include("generic_particle_filter_algorithm.jl")
include("CIR_likelihood.jl")
include("CIR_smoothing.jl")
include("CIR_smoothing_approximate.jl")
include("WF_particle_filter.jl")
include("exact_L2_distances.jl")
include("exact_L2_distances_arb.jl")
include("kde_for_pf_samples.jl")
include("CIR_joint_smoothing.jl")
include("CIR_full_inference.jl")
include("MCMC_sampler.jl")
include("CIR_reparam.jl")
include("CIR_parameter_inference_given_traj.jl")
end # module
