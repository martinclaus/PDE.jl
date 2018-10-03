__precompile__()

module PDE

using DataStructures: OrderedDict

# types
export Model, ModelState
export DiagnosticEquation, PrognosticEquation

# model functions
export iterate, iterate_heaps

# time stepping routines
export euler_forward

# operator

include("equations.jl")
include("model.jl")
include("integrator.jl")
include("operator.jl")

end
