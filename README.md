# Schmidt number certification via symmetric extensions.
Generalization of the [symmetric extensions criterion](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.69.022308), with bosonic symmetry and PPT constraint, to certify Schmidt number. May also be used to certify entanglemen by setting `sn=1`. Similar to [this](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.124.200502).

Uses the `SCS` solver and `JuMP`/`ComplexOptInterface` for modelling, `Convex` for the partial trace and partial transpose operations, and `Combinatorics` to generate the projection onto the symmetric subspace, so these should be installed.

Check `Examples.jl` for... examples.

## Files
- `SchmidtNumber.jl` -- Implements symmetric extensions SDP with bosonic symmetries and PPT criterion.
- `BosonicSymmetry.jl` -- Functions to generate the projector onto/out from the symmetric subspace.
- `Utils.jl` -- Partial trace and partial transposition operations (from [QuantumInformation.jl](https://github.com/iitis/QuantumInformation.jl))
- `Examples.jl` -- Provides simple examples for isotropic and extendible states.