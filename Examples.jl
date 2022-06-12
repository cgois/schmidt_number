include("SchmidtNumber.jl")

#======================================================
Examples for entanglement (i.e. Schmidt number > 1)
======================================================#

"""From arXiv:1310.3530, a bipartite state has a N = 2 symmetric extension
iff tr(rho_B)^2 >= tr(rho_AB^2) - 4 sqrt(det rho_AB)."""
rho = [1 0 0 -1; 0 1 1/2 0; 0 1/2 1 0; -1 0 0 1]
@time maximally_mixed_distance(rho, 2, 1, 2, ppt=false)

"""Isotropic states are entangled for visibility above 1 / (d + 1)"""
function isotropic(d::Integer=2, vis::Float64=1.0)
    vis * ghz(d, parties=2, ket=false) + (1 - vis) * I(d^2) / (d^2)
end
# This should be separable (i.e., objective value = 1):
@time maximally_mixed_distance(isotropic(2, 1/3), 2, 1, 2, ppt=true)
# This should be entangled (i.e., objective value < 1):
@time maximally_mixed_distance(isotropic(2, 1/3 + 0.01), 2, 1, 2, ppt=true)

#======================================================
Example for Schmidt number > 2)
======================================================#
# This case returns 1 (feasibility), so it does not certify sn > 2.
# to that, we possibly need to increase `n`, but it gets too slow.
# @time maximally_mixed_distance(ghz(4), 4, 2, 2, ppt=true)