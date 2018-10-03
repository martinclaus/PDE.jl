using PDE

## Define constants ##
const f = Float64(10^(-4))
const g = 1.
const H = g
const A = .1

## define terms in model equations ##
function coriolis_u!(result::T, v::T) where T<:AbstractArray
    i_v2u!(result, v, false)
    result .*= f
    return result
end
coriolis_u!(res, s::S) where {S<:ModelState} = coriolis_u!(res, s.state[:v])

function coriolis_v!(result::T, u::T) where T<:AbstractArray
    i_u2v!(result, u, false)
    result .*= -f
    return result
end
coriolis_v!(res, s::S) where {S<:ModelState} = coriolis_v!(res, s.state[:u])

function velocity_divergence!(result::T, u::T, v::T) where {T<:AbstractArray}
    divergence_2D!(result, u, v)
    result .*= -H
    return result
end
velocity_divergence!(res, s::S) where {S<:ModelState} = velocity_divergence!(res, s.state[:u], s.state[:v])

function divergence_2D!(result::T, u::T, v::T) where T<:AbstractArray
    dx_u!(result, u, false)
    dy_v!(result, v, true)
    return result
end
divergence_2D!(result, s::ModelState) = divergence_2D!(result, s.state[:u], s.state[:v])

function pressure_gradient!(result::T, η::T, diff_op!::Function) where T<:AbstractArray
    diff_op!(result, η, false)
    result .*= -g
    return result
end

zonal_pressure_gradient!(result, η) = pressure_gradient!(result, η, dx_η!)
zonal_pressure_gradient!(result, s::ModelState) = zonal_pressure_gradient!(result, s.state[:η])

meridional_pressure_gradient!(result, η) = pressure_gradient!(result, η, dy_η!)
meridional_pressure_gradient!(result, s::S) where {S<:ModelState} = meridional_pressure_gradient!(result, s.state[:η])

function harmonic_diffusion!(result::T, var::T, A::N) where {S<:ModelState, T<:AbstractArray, N<:Real}
    ∇²!(result, var, false)
    result .*= A
    return result
end
harmonic_diffusion_u!(res, s::S, A) where {S<:ModelState} = harmonic_diffusion!(res, s.state[:u], A)
harmonic_diffusion_v!(res, s::S, A) where {S<:ModelState} = harmonic_diffusion!(res, s.state[:v], A)

function eke!(result::T, u::T, v::T) where T<:AbstractArray
    n, m = size(u)
    @boundscheck checkbounds(result, n, m)
    for j in 1:m
        for i in 1:n
            result[i, j] = .25 * (u[i, j]^2 + u[cyclic_index(i+1, n), j]^2
                                  + v[i, j]^2 + v[i, cyclic_index(j+1, m)]^2)
        end
    end
    return result
end
eke!(result, s::S) where {S<:ModelState}= eke!(result, s.state[:u], s.state[:v])


## Define model equations ##

zonal_momentum_equation = PrognosticEquation(
    :u,
    (zonal_pressure_gradient!,
     (r, s) -> lateral_mixing_u!(r, s, A),
    coriolis_u!,),
    euler_forward!
)
meridional_momentum_equation = PrognosticEquation(
    :v,
    (meridional_pressure_gradient!,
     (r, s) -> lateral_mixing_v!(r, s, A),
     coriolis_v!,),
    euler_forward!
)
continuity_equation = PrognosticEquation(
    :η,
    (velocity_divergence!,),
    euler_forward!
)

eq_eke = DiagnosticEquation(
    :eke,
    (eke!,)
)

model_equations = (eq_eke, continuity_equation, zonal_momentum_equation, meridional_momentum_equation)

## create model instance ##
Nx = 360; Ny = 180; dt = 0.125

model = Model(ModelState{Array{Float64}}(Nx, Ny), model_equations)

## set initial conditions
model.s.state[:η][:] = exp.(-((Array(1:Nx) .- Nx/2.5).^2 .+ (transpose(Array(1:Ny)) .- Ny/2.5).^2) ./ 5^2)

## iterate the model for 1000 time steps using the heaps scheme
@timev iterate_heaps!(model, dt, 1000)
