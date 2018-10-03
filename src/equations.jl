abstract type Equation end

struct DiagnosticEquation <: Equation
    var::Symbol
    terms::Tuple
end

struct PrognosticEquation{T<:Function} <: Equation
    var::Symbol
    inc::Symbol
    next::Symbol
    terms::Tuple
    integrator::T

    function PrognosticEquation(var::Symbol, terms::Tuple, integrator::T) where {T<:Function}
        new{T}(var, var, np1_symbol(var), terms, integrator)
    end
end

is_diagnostic(eq) = false
is_diagnostic(eq::DiagnosticEquation) = true
is_prognostic(eq) = false
is_prognostic(eq::PrognosticEquation) = true
