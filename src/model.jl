## Types ##

struct ModelState{T<:AbstractArray, N<:Integer}
    Nx::N
    Ny::N
    dtype::Type

    # State variables
    state::OrderedDict{Symbol, T}

    # time increments
    inc::OrderedDict{Symbol, T}

    # diagnostic variables
    diagnostic::OrderedDict{Symbol, T}

    # inner constructor
    function ModelState{T,N}(Nx::N, Ny::N) where {T<:AbstractArray, N<:Integer}
        d = OrderedDict{Symbol, T}()
        new{T, N}(Nx, Ny, eltype(T), d, copy(d), copy(d))
    end
end
ModelState{T}(Nx::N, Ny::N) where {T<:AbstractArray, N<:Integer} = ModelState{T, N}(Nx, Ny)


function add_variable!(s::MS, reg::Symbol, var::Symbol) where {MS<:ModelState}
    getproperty(s, reg)[var] = zeros(eltype(valtype(getproperty(s, reg))), s.Nx, s.Ny)
end

function add_variable!(s::MS, eq::PrognosticEquation) where {MS<:ModelState}
    add_variable!(s, :state, eq.var)
    add_variable!(s, :inc, eq.var)
    add_variable!(s, :inc, np1_symbol(eq.var))
end

add_variable!(s::MS, eq::DiagnosticEquation) where {MS<:ModelState} = add_variable!(s, :diagnostic, eq.var)

@inline np1_symbol(s::Symbol) = Symbol(s, "_next")


struct Model{S<:ModelState}
    # model state
    s::S

    # set of equations
    equations::Tuple

    # scratch workspace
    scratch::Vector

    function Model(state::S, eqs::Tuple) where {S<:ModelState}
        for eq in eqs
            add_variable!(state, eq)
        end
        new{S}(state, eqs, [])
    end
end


## Model mechanics ##

function create_scratch(m::M) where {M<:Model}
    template_var = first(keys(m.s.state))
    push!(m.scratch, similar(m.s.state[template_var]))
end
n_scratch(m::M) where M<:Model = length(m.scratch)

function guarantee_n_scratch!(m::M, n::I) where {M<:Model, I<:Integer}
    if n_scratch(m) >= n
        return
    end
    for i in 1:n - n_scratch(m)
        create_scratch(m)
    end
end

@inline function step!(m::M) where {M<:Model}
    # setup workspace
    guarantee_n_scratch!(m, 1)
    scratch = m.scratch[1]

    # compute diagnostic variables
    for eq in m.equations
        is_diagnostic(eq) ? step!(eq, m.s, scratch::T) : continue
    end

    # prognostic variables increment
    for eq in m.equations
        is_prognostic(eq) ? step!(eq, m.s, scratch::T) : continue
    end
end

@inline step!(eq::DiagnosticEquation, s::S, scratch::T) where {E<:Equation, S<:ModelState, T<:AbstractArray} =
    step!(eq.var, eq.terms, s, :diagnostic, scratch)

@inline step!(eq::PrognosticEquation, s::S, scratch::T) where {S<:ModelState, T<:AbstractArray} =
    step!(eq.var, eq.terms, s, :inc, scratch)

@inline function step!(var::Symbol, terms::Tuple, s::S, reg::Symbol, scratch::T) where {S<:ModelState, T<:AbstractArray}
    if length(terms) == 0
        return
    end
    v_arr = getproperty(s, reg)[var]
    scratch .= 0
    for (i, term!) in enumerate(terms)
        if i == 1
            term!(v_arr, s)
        else
            term!(scratch, s)
            v_arr .+= scratch
        end
    end
end

@inline function advance!(m::M) where M<:Model
    map(advance!, x -> advance!(eq, m.s), m.equations)
end
@inline function advance!(eq::PrognosticEquation, s::S) where {S<:ModelState}
    s.state[eq.var] .= s.inc[eq.next];
end
@inline function advance!(eq::E, s::S) where {E<:Equation, S<:ModelState} end


## Model iteration (the full workflow of one iteration)

function iterate!(m::M, dt::R, nt::N) where {M<:Model, R<:Real, N<:Integer}
    for i in 1:nt
        iterate!(m, dt)
    end
end
@inline function iterate!(m::M, dt::R) where {M<:Model, R<:Real}
    step!(m)
    integrate!(m, dt)
    advance!(m)
end
@inline function iterate!(eq::E, s::S, scratch::T, dt::N) where {E<:Equation, S<:ModelState, T<:AbstractArray, N<:Real}
    step!(eq, s, scratch)
    integrate!(eq, s, dt)
    advance!(eq, s)
end
@inline function iterate_heaps!(m::M, dt::R) where {M<:Model, R<:Real}
    guarantee_n_scratch!(m, 1)
    scratch = m.scratch[1]

    for eq in m.equations
        iterate!(eq, m.s, scratch, dt)
    end
end

function iterate_heaps!(m::M, dt::R, nt::N) where {M<:Model, R<:Real, N<:Integer}
    for i in 1:nt
        iterate_heaps!(m, dt)
    end
end
