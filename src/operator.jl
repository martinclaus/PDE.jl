## Operators for cartesian Arakawa C-Grid ##
# TODO: Needs to be generalized

@inline cyclic_index(i::Int, N::Int) = mod(i - 1, N) + 1

const add_default = true

@inline add_op(a, b) = a + b
@inline ass_op(a, b) = b

assign_op(add::Bool) = add ? add_op : ass_op


## Derivative operators for cartesian C-Grid ##

@inline function dx_u!(result::T, var::T, add::Bool=add_default) where {T<:AbstractArray}
    n, m = size(var)
    @boundscheck checkbounds(result, n, m)
    op = assign_op(add)
    for j in 1:m
        for i in 1:n
            @inbounds result[i, j] = op(result[i, j], var[cyclic_index(i+1, n), j] - var[i, j])
        end
    end
    return result
end
dx_u(var) = dx_u!(similar(var), var, false)

@inline function dy_u!(result::T, var::T, add::Bool=add_default) where {T<:AbstractArray}
    n, m = size(var)
    @boundscheck checkbounds(result, n, m)
    op = assign_op(add)
    for j in 1:m
        for i in 1:n
            @inbounds result[i, j] = op(result[i, j], var[i, j] - var[i, cyclic_index(j-1, m)])
        end
    end
    return result
end
dy_u(var) = dy_u!(similar(var), var, false)

@inline function dx_v!(result::T, var::T, add::Bool=add_default) where {T<:AbstractArray}
    n, m = size(var)
    @boundscheck checkbounds(result, n, m)
    op = assign_op(add)
    for j in 1:m
        for i in 1:n
            @inbounds result[i, j] = op(result[i, j], var[i, j] - var[cyclic_index(i-1, n), j])
        end
    end
    return result
end
dx_v(var) = dx_v!(similar(var), var, false)

@inline function dy_v!(result::T, var::T, add::Bool=add_default) where {T<:AbstractArray}
    n, m = size(var)
    @boundscheck checkbounds(result, n, m)
    op = assign_op(add)
    for j in 1:m
        for i in 1:n
            @inbounds result[i, j] = op(result[i, j], var[i, cyclic_index(j+1, m)] - var[i, j])
        end
    end
    return result
end
dy_v(var) = dy_v!(similar(var), var, false)

dx_η! = dx_v!
dx_η = dx_v
dy_η! = dy_u!
dy_η = dy_u

@inline function d2x!(result::T, var::T, add::Bool=add_default) where {T<:AbstractArray}
    n, m = size(var)
    @boundscheck checkbounds(result, n, m)
    op = assign_op(add)
    for j in 1:m
        for i in 1:n
            @inbounds result[i, j] = op(
                result[i, j],
                var[cyclic_index(i+1, n), j] - 2var[i, j] + var[cyclic_index(i-1, n), j])
        end
    end
    return result
end
d2x(var) = d2x!(similar(var), var, false)
d2x_u! = d2x!
d2x_u = d2x
d2x_v! = d2x!
d2x_v = d2x
d2x_η! = d2x!
d2x_η = d2x

@inline function d2y!(result::T, var::T, add::Bool=add_default) where {T<:AbstractArray}
    n, m = size(var)
    @boundscheck checkbounds(result, n, m)
    op = assign_op(add)
    for j in 1:m
        for i in 1:n
            @inbounds result[i, j] = op(
                result[i, j],
                var[i, cyclic_index(j+1, m)] - 2var[i, j] + var[i, cyclic_index(j-1, m)])
        end
    end
    return result
end
d2y(var) = d2y!(similar(var), var, false)
d2y_u! = d2y!
d2y_u = d2y
d2y_v! = d2y!
d2y_v = d2y
d2y_η! = d2y!
d2y_η = d2y


@inline function ∇²!(result::T, var::T, add::Bool=add_default) where {T<:AbstractArray}
    n, m = size(var)
    @boundscheck checkbounds(result, n, m)
    op = assign_op(add)
    @inbounds for j in 1:m
        for i in 1:n
            result[i, j] =  op(
                result[i, j],
                var[cyclic_index(i+1, n), j] - 2var[i, j] + var[cyclic_index(i-1, n), j]
                + var[i, cyclic_index(j+1, m)] - 2var[i, j] + var[i, cyclic_index(j-1, m)]
            )
        end
    end
    return result
end

## interpolation between grids (Arakawa C-Grid)

@inline function i_u2v!(result::T, var::T, add::Bool=add_default) where {T<:AbstractArray}
    n, m = size(var)
    @boundscheck checkbounds(result, n, m)
    op = assign_op(add)
    for j in 1:m
        for i in 1:n
            @inbounds result[i, j] = op(
                result[i, j],
                .25 * (var[i, j] + var[cyclic_index(i+1, n), cyclic_index(j-1, m)]
                + var[i, cyclic_index(j-1, m)] + var[cyclic_index(i+1, n), j])
            )
        end
    end
    return result
end
i_u2v!(var, add=add_default) = res -> i_u2v!(res, var, add)
i_u2v(var) = i_u2v!(similar(var), var, false)

@inline function i_v2u!(result::T, var::T, add::Bool=add_default) where {T<:AbstractArray}
    n, m = size(var)
    @boundscheck checkbounds(result, n, m)
    op = assign_op(add)
    for j in 1:m
        for i in 1:n
            @inbounds result[i, j] = op(
                result[i, j],
                .25 * (var[i, j] + var[cyclic_index(i-1, n), cyclic_index(j+1, m)]
                + var[cyclic_index(i-1, n), j] + var[i, cyclic_index(j+1, m)])
            )
        end
    end
    return result
end
i_v2u!(var, add=add_default) = res -> i_v2u!(res, var, add)
i_v2u(var) = i_v2u!(similar(var), var, false)

@inline function i_u2η!(result::T, var::T, add::Bool=add_default) where {T<:AbstractArray}
    n, m = size(var)
    @boundscheck checkbounds(result, n, m)
    op = assign_op(add)
    for j in 1:m
        for i in 1:n
            @inbounds result[i, j] = op(
                result[i, j],
                .5 * (var[i, j] + var[cyclic_index(i+1, n), j])
            )
        end
    end
    return result
end
i_u2η!(var, add=add_default) = res -> i_u2η!(res, var, add)
i_u2η(var) = i_u2η!(similar(var), var, false)

@inline function i_v2η!(result::T, var::T, add::Bool=add_default) where T<:AbstractArray
    n, m = size(var)
    @boundscheck checkbounds(result, n, m)
    op = assign_op(add)
    for j in 1:m
        for i in 1:n
            @inbounds result[i, j] = op(
                result[i, j],
                .5 * (var[i, j] + var[i, cyclic_index(j+1, m)])
            )
        end
    end
    return result
end
i_v2η!(var, add=add_default) = res -> i_v2η!(res, var, add)
i_v2η(var) = i_v2η!(similar(var), var, false)
