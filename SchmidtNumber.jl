using JuMP, ComplexOptInterface, SCS

include("BosonicSymmetry.jl")
include("Utils.jl")

"""Maximum visibility (w.r.t. random noise) s.t. the DPS criterion is certified.
An objective value of 1 means feasibility (unconclusive), and < 1 means entanglement."""
function maximally_mixed_distance(state, local_dim, sn=1, n::Integer=3; ppt::Bool=true)
    # Constants
    dim = size(state, 1)
    noise = eye(dim) / dim^2
    aux_dim = local_dim * sn # Dimension with auxiliary spaces A'B'
    dims = repeat([aux_dim], n + 1) # AA' dim. + `n` copies of BB'.
    P = kron(eye(aux_dim), symmetric_projection(aux_dim, n)) # Bosonic subspace projector.
    Qdim = aux_dim * binomial(n + aux_dim - 1, aux_dim - 1)  # Dim. extension w/ bosonic symmetries.
    entangling = kron(eye(local_dim), sn .* ghz(sn), eye(local_dim)) # Entangling between A' and B'.

    problem = Model(SCS.Optimizer)
    COI = ComplexOptInterface # Adds complex number support do JuMP.
    COI.add_all_bridges(problem)

    # Optimization variables
    @variable(problem, 0 <= vis <= 1)
    Q = @variable(problem, [1:Qdim, 1:Qdim] in ComplexOptInterface.HermitianPSDCone())
    if ppt
        # Dummy vars. to enforce PPT (not possible directly in ComplexOptInterface?).
        fulldim = prod(dims)
        PSD = @variable(problem, [1:fulldim, 1:fulldim] in ComplexOptInterface.HermitianPSDCone())
    end

    # Constraints
    noisy_state = vis * state + (1 - vis) * noise
    @expression(problem, lifted, P * Q * P')
    @expression(problem, reduced, ptrace(lifted, dims, collect(3:n+1)) * entangling)
    @constraint(problem, noisy_state .== ptrace(reduced, [local_dim, sn, sn, local_dim], [2, 3]))
    if ppt
        ssys = Int.(1:ceil(n / 2) + 1)
        @constraint(problem, PSD .== ptranspose(lifted, dims, ssys))
    end

    # Solution
    @objective(problem, Max, vis)
    @show problem
    optimize!(problem)
    @show solution_summary(problem, verbose=true)
    problem, Q
end