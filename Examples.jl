include("SchmidtNumber.jl")

#======================================================
Examples for entanglement (i.e. Schmidt number > 1)
======================================================#

"""From arXiv:1310.3530, a bipartite state has a N = 2 symmetric extension
iff tr(rho_B)^2 >= tr(rho_AB^2) - 4 sqrt(det rho_AB)."""
rho = [1 0 0 -1; 0 1 1/2 0; 0 1/2 1 0; -1 0 0 1]
@time maximally_mixed_distance(rho, 2, 1, 2, ppt=false)
# This should be separable (i.e., objective value = 1):
@time maximally_mixed_distance(isotropic(2, 1/3), 2, 1, 2, ppt=true)
# This should be entangled (i.e., objective value < 1):
@time maximally_mixed_distance(isotropic(2, 1/3 + 0.01), 2, 1, 2, ppt=true)

#======================================================
Example for Schmidt number > 2)
======================================================#
@time maximally_mixed_distance(ghz(3), 3, 2, 2, ppt=true)