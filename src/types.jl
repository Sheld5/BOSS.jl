using Optim
using Evolutionary
using NLopt
using Turing
using AbstractGPs

const AbstractBounds = Tuple{<:AbstractArray{<:Real}, <:AbstractArray{<:Real}}


# - - - - - - - - Fitness Function - - - - - - - -

abstract type Fitness end

"""
Used to define a linear fitness function.

Provides better performance than using the more general `NonlinFitness`.

# Example
A fitness function 'f(y) = y[1] + a * y[2] + b * y[3]' can be defined as:
```julia-repl
julia> LinFitness([1., a, b])
```
"""
struct LinFitness{
    C<:AbstractArray{<:Real},
} <: Fitness
    coefs::C
end
(f::LinFitness)(y) = f.coefs' * y

"""
Used to define a fitness function.

If your fitness function is linear, use `LinFitness` for better performance.

# Example
```julia-repl
julia> NonlinFitness(y -> cos(y[1]) + sin(y[2]))
```
"""
struct NonlinFitness{
    F<:Base.Callable,
} <: Fitness
    fitness::F
end
(f::NonlinFitness)(y) = f.fitness(y)


# - - - - - - - - Surrogate Model - - - - - - - -

abstract type SurrogateModel end

abstract type Parametric <: SurrogateModel end

param_count(model::Parametric) = length(model.param_priors)

"""
Used to define a linear parametric surrogate model.

This model definition will provide better performance than the more general 'NonlinModel' in the future.
This feature is not implemented yet so it is equivalent to using `NonlinModel` for now.

The linear model is defined as
    ϕs = lift(x)
    y = [θs[i]' * ϕs[i] for i in 1:m]
where
    x = [x₁, ..., xₙ]
    y = [y₁, ..., yₘ]
    θs = [θ₁, ..., θₘ], θᵢ = [θᵢ₁, ..., θᵢₚ]
    ϕs = [ϕ₁, ..., ϕₘ], ϕᵢ = [ϕᵢ₁, ..., ϕᵢₚ]
     n, m, p ∈ R.

# Fields
  - lift:           A function `x::Vector{Float64} -> ϕs::Vector{Vector{Float64}}`
  - param_priors:   A vector of priors for all params [θ₁₁,θ₁₂,...,θ₁ₚ, θ₂₁,θ₂₂,...,θ₂ₚ, ..., θₘ₁,θₘ₂,...,θₘₚ].
"""
struct LinModel{
    L<:Base.Callable,
    D<:AbstractArray{<:Any},
} <: Parametric
    lift::L
    param_priors::D
end

"""
Used to define a parametric surrogate model.

If your model is linear, you can use `LinModel` which will provide better performance in the future. (Not yet implemented.)

# Fields
  - predict:        A function `x::Vector{Float64}, params::Vector{Float64} -> y::Vector{Float64}`
  - param_priors:   A vector of priors for each model parameter.
"""
struct NonlinModel{
    P<:Base.Callable,
    D<:AbstractArray,
} <: Parametric
    predict::P
    param_priors::D
end

"""
Used to define a nonparametric surrogate model (Gaussian Process).

# Fields
  - mean:                   A function `x::Vector{Float64} -> y::Vector{Float64}`. Zero-mean is used if mean is nothing.
  - kernel:                 An instance of `AbstractGPs.Kernel`.
  - length_scale_priors:    An array of length `y_dim` containing multivariate prior distributions of size `x_dim`.
"""
struct Nonparametric{
    M<:Union{Nothing, Base.Callable},
    K<:Kernel,
    D<:AbstractArray,
} <: SurrogateModel
    mean::M
    kernel::K
    length_scale_priors::D
end

"""
Used to define a semiparametric surrogate model (combination of a parametric model and GP).

# Fields
  - parametric:         An instance of `Parametric` model.
  - nonparametric:      An instance of `Nonparametric` model with zero-mean. (The parametric model is used as mean.)
"""
struct Semiparametric{
    P<:Parametric,
    N<:Nonparametric,
} <: SurrogateModel
    parametric::P
    nonparametric::N

    function Semiparametric(p::Parametric, n::Nonparametric)
        @assert isnothing(n.mean)
        new{typeof(p), typeof(n)}(p, n)
    end
end


# - - - - - - - - Model-Fit Algorithms - - - - - - - -

abstract type ModelFit end
struct MLE <: ModelFit end
struct BI <: ModelFit end

"""
Specifies the library/algorithm is used to estimate the model parameters.

Inherit this type to define a custom model-fitting algorithms.

Example: `struct CustomAlg <: ModelFitter{MLE} ... end` or `struct CustomAlg <: ModelFitter{BI} ... end`

Structures derived from this type have to implement the following method:
`estimate_parameters(model_fitter::CustomFitter, problem::OptimizationProblem; info::Bool) where {CustomFitter<:ModelFitter}`.

This method should return a named tuple `(θ = ..., length_scales = ..., noise_vars = ...)`
with either MLE model parameters (if `CustomAlg <: ModelFitter{MLE}`) or model parameter samples (if `CustomAlg <: ModelFitter{BI}`). 

Additionally, if the custom algorithm is of type `ModelFitter{BI}`, it has to implement the method
`sample_count(::CustomAlg)` giving the number of parameter samples returned from `estimate_parameters`.

See '\\src\\algorithms' for some implementations of `ModelFitter`.
"""
abstract type ModelFitter{T<:ModelFit} end

# Specific implementations of `ModelFitter` are in '\src\algorithms'.


# - - - - - - - - EI Maximization - - - - - - - -

"""
Specifies the library/algorithm is used for acquisition function optimization.

Inherit this type to define a custom acquisition maximizer.

Example: `struct CustomAlg <: AcquisitionMaximizer ... end`

Structures derived from this type have to implement the following method:
`maximize_acquisition(acq_maximizer::CustomAlg, problem::OptimizationProblem, acq::Base.Callable; info::Bool) where {CustomAlg <: AcquisitionMaximizer}`

This method should return the point of the input domain which maximizes the given acquisition function `acq`.

See '\\src\\algorithms' for some implementations of `AcquisitionMaximizer`.
"""
abstract type AcquisitionMaximizer end

# Specific implementations of `AcquisitionMaximizer` are in '\src\algorithms'.


# - - - - - - - - Termination Conditions - - - - - - - -

"""
Specifies the termination condition of the whole BOSS algorithm.

Inherit this type to define a custom termination condition.

Example: `struct CustomCond <: TermCond ... end`

Structures derived from this type have to implement the following method:
`(cond::CustomCond)(problem::OptimizationProblem) where {CustomCond <: TermCond}`

This method should return true to keep the optimization running and return false once the optimization is to be terminated.
"""
abstract type TermCond end

# Specific implementations of `TermCond` are in '\src\term_cond.jl'.


# - - - - - - - - Data - - - - - - - -

"""
The structures deriving this type contain all the data collected during the optimization
as well as the parameters and hyperparameters of the model.
"""
abstract type AbstractExperimentData{NUM<:Real} end

Base.length(data::AbstractExperimentData) = size(data.X)[2]

mutable struct ExperimentDataPrior{
    NUM<:Real,
    T<:AbstractMatrix{NUM},
} <: AbstractExperimentData{NUM}
    X::T
    Y::T
end

abstract type ExperimentDataPost{T<:ModelFit, NUM<:Real} <: AbstractExperimentData{NUM} end

mutable struct ExperimentDataMLE{
    NUM<:Real,
    T<:AbstractMatrix{NUM},
    P<:Union{Nothing, <:AbstractArray{NUM}},
    L<:Union{Nothing, <:AbstractMatrix{NUM}},
    N<:Union{Nothing, <:AbstractArray{NUM}},
} <: ExperimentDataPost{MLE, NUM}
    X::T
    Y::T
    θ::P
    length_scales::L
    noise_vars::N
end

mutable struct ExperimentDataBI{
    NUM<:Real,
    T<:AbstractMatrix{NUM},
    P<:Union{Nothing, <:AbstractMatrix{NUM}},
    L<:Union{Nothing, <:AbstractArray{<:AbstractMatrix{NUM}}},
    N<:Union{Nothing, <:AbstractMatrix{NUM}},
} <: ExperimentDataPost{BI, NUM}
    X::T
    Y::T
    θ::P
    length_scales::L
    noise_vars::N
end


# - - - - - - - - Optimization Problem - - - - - - - -

"""
This structure defines the whole optimization problem.
"""
mutable struct OptimizationProblem{
    NUM<:Real,
    Q<:Fitness,
    F<:Base.Callable,
    C<:AbstractArray{NUM},
    X<:Any,  # type depends on EI maximizer
    I<:AbstractArray{<:Bool},
    M<:SurrogateModel,
    N<:Any,
    D<:AbstractExperimentData{NUM},
}
    fitness::Q
    f::F
    cons::C
    domain::X
    discrete::I
    model::M
    noise_var_prior::N
    data::D
end

x_dim(problem::OptimizationProblem) = length(problem.discrete)
y_dim(problem::OptimizationProblem) = length(problem.cons)


# - - - - - - - - Boss Options - - - - - - - -

struct BossOptions
    info::Bool
    ϵ_samples::Int  # only for MLE, in case of BI ϵ_samples == sample_count(ModelFitterBI)
end