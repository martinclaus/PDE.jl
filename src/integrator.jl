## Time stepping methods ##

@inline euler_forward!(eq::PrognosticEquation, s, dt) =
    euler_forward!(s.inc[eq.next], s.state[eq.var], s.inc[eq.inc], dt)

@inline function euler_forward!(next::T, old::T, inc::T, dt) where {T}
    n, m = size(old)
    @inbounds for j in 1:m
        for i in 1:n
            next[i, j] = dt * inc[i, j] + old[i, j]
        end
    end
end

@inline integrate!(eq::PrognosticEquation, state, dt) = eq.integrator(eq, state, dt)
@inline function integrate!(eq, state, dt) end
@inline integrate!(model, dt) = map(eq -> integrate!(eq, model.s, dt), model.equations)
