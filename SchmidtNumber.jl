using JuMP, ComplexOptInterface

include("BosonicSymmetry.jl")
include("Utils.jl")

"""Maximum visibility (w.r.t. random noise) s.t. the DPS criterion is certified.
An objective value of 1 means feasibility (unconclusive), and < 1 means entanglement."""
function maximally_mixed_distance(state, local_dim, sn=1, n::Integer=3;
                                  ppt::Bool=true,
                                  solver=SCS.Optimizer,
                                  params=nothing,
                                  precision="default")
    # Constants
    dim = size(state, 1)
    noise = eye(dim) / dim
    aux_dim = local_dim * sn # Dimension with auxiliary spaces A'B'
    dims = repeat([aux_dim], n + 1) # AA' dim. + `n` copies of BB'.
    P = kron(eye(aux_dim), symmetric_projection(aux_dim, n)) # Bosonic subspace projector.
    Qdim = aux_dim * binomial(n + aux_dim - 1, aux_dim - 1)  # Dim. extension w/ bosonic symmetries.
    entangling = kron(eye(local_dim), sqrt(sn) .* ghz(sn, ket=true), eye(local_dim)) # Entangling between A' and B'.

    problem = setsolver(solver, params=params, precision=precision)
    # Optimization variables
    @variable(problem, 0 <= vis <= 1)
    Q = @variable(problem, [1:Qdim, 1:Qdim] in ComplexOptInterface.HermitianPSDCone())

    # Constraints
    noisy_state = vis * state + (1 - vis) * noise
    @expression(problem, lifted, (P * Q) * P')
    @expression(problem, reduced, ptrace(lifted, dims, collect(3:n+1)))
    @constraint(problem, noisy_state .== entangling' * reduced * entangling)
    @constraint(problem, tr(reduced) == sn)
    if ppt
        ssys = Int.(1:ceil(n / 2) + 1)
        ispsd(problem, ptranspose(lifted, dims, ssys))
    end

    # Solution
    @objective(problem, Max, vis)
    optimize!(problem)
    @show solution_summary(problem, verbose=true)
    problem, Q
end