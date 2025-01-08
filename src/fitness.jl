
"""
An abstract type for a fitness function
measuring the quality of an output `y` of the objective function.

Fitness is used by the `AcquisitionFunction` to determine promising points for future evaluations.

All fitness functions *should* implement:
- (::CustomFitness)(y::AbstractVector{<:Real}) -> fitness::Real

An exception is the `NoFitness`, which can be used for problem without a well defined fitness.
In such case, an `AcquisitionFunction` which does not depend on `Fitness` must be used.

See also: [`NoFitness`](@ref), [`LinFitness`](@ref), [`NonlinFitness`](@ref), [`AcquisitionFunction`](@ref)
"""
abstract type Fitness end

"""
    NoFitness()

Placeholder for problems with no defined fitness.
    
`BossProblem` defined with `NoFitness` can only be solved with `AcquisitionFunction` not dependent on `Fitness`.
"""
struct NoFitness <: Fitness end

"""
    LinFitness(coefs::AbstractVector{<:Real})

Used to define a linear fitness function 
measuring the quality of an output `y` of the objective function.

May provide better performance than the more general `NonlinFitness`
as some acquisition functions can be calculated analytically with linear fitness
functions whereas this may not be possible with a nonlinear fitness function.

See also: [`NonlinFitness`](@ref)

# Example
A fitness function `f(y) = y[1] + a * y[2] + b * y[3]` can be defined as:
```julia-repl
julia> LinFitness([1., a, b])
```
"""
struct LinFitness{
    C<:AbstractVector{<:Real},
} <: Fitness
    coefs::C
end

(f::LinFitness)(y) = f.coefs' * y

"""
    NonlinFitness(fitness::Function)

Used to define a general nonlinear fitness function
measuring the quality of an output `y` of the objective function.

If your fitness function is linear, use `LinFitness` which may provide better performance.

See also: [`LinFitness`](@ref)

# Example
```julia-repl
julia> NonlinFitness(y -> cos(y[1]) + sin(y[2]))
```
"""
struct NonlinFitness <: Fitness
    fitness::Function
end

(f::NonlinFitness)(y) = f.fitness(y)