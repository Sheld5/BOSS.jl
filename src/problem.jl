
"""
    BossProblem(; kwargs...)

Defines the whole optimization problem for the BOSS algorithm.

# Problem Definition

    There is some (noisy) blackbox function `y = f(x) = f_true(x) + ϵ` where `ϵ ~ Normal`.

    We have some surrogate model `y = model(x) ≈ f_true(x)`
    describing our knowledge (or lack of it) about the blackbox function.

    We wish to find `x ∈ domain` such that `fitness(f(x))` is maximized
    while satisfying the constraints `f(x) <= y_max`.

# Keywords
- `fitness::Fitness`: The [`Fitness`](@ref) function.
- `f::Union{Function, Missing}`: The objective blackbox function.
- `model::SurrogateModel`: The [`SurrogateModel`](@ref).
- `data::ExperimentData`: The initial data of objective function evaluations. See [`ExperimentDataPrior`](@ref).
- `domain::Domain`: The [`Domain`](@ref) of `x`.
- `y_max`: The constraints on `y`. (See the definition above.)

See also: [`bo!`](@ref)
"""
@kwdef mutable struct BossProblem{
    F<:Any,
}
    fitness::Fitness = NoFitness()
    f::F
    model::SurrogateModel
    data::ExperimentData
    domain::Domain
    y_max::AbstractVector{<:Real} = fill(Inf, y_dim(data))
end

Base.copy(problem::BossProblem) = BossProblem(getfield.(Ref(problem), fieldnames(BossProblem))...)

"""
    slice(::BossProblem, slice::Int) -> ::BossProblem

Return a `BossProblem` for the given `slice` output dimension.

The returned `BossProblem` has a single output dimension,
`NoFitness` and `missing` objective function.
"""
function slice(problem::BossProblem, idx::Int)
    θ_slice_ = θ_slice(problem.model, idx)
    
    return BossProblem(
        NoFitness(),
        missing,
        slice(problem.model, idx),
        slice(problem.data, θ_slice_, idx),
        problem.domain,
        slice(problem.y_max, idx),
    )
end

"""
    model_posterior(::BossProblem) -> post

Return the posterior predictive distribution of the surrogate model with two methods:
- `post(x::AbstractVector{<:Real}) -> μs::AbstractVector{<:Real}, σs::AsbtractVector{<:Real}`
- `post(X::AbstractMatrix{<:Real}) -> μs::AbstractMatrix{<:Real}, Σs::AbstractArray{<:Real, 3}`

The first method takes a single point `x` of length `x_dim(::BossProblem)` from the `Domain`,
and returns the predictive means and deviations
of the corresponding output vector `y` of length `y_dim(::BossProblem)` such that:
- `μs, σs = post(x)` => `y ∼ product_distribution(Normal.(μs, σs))`
- `μs, σs = post(x)` => `y[i] ∼ Normal(μs[i], σs[i])`

The second method takes multiple points from the `Domain` as a column-wise matrix `X` of size `(x_dim, N)`,
and returns the joint predictive means and covariance matrices
of the corresponding output matrix `Y` of size `(y_dim, N)` such that:
- `μs, Σs = post(X)` => `transpose(Y) ∼ product_distribution(MvNormal.(eachcol(μs), eachslice(Σs; dims=3)))`
- `μs, Σs = post(X)` => `Y[i,:] ∼ MvNormal(μs[:,i], Σs[:,:,i])`

See also: [`model_posterior_slice`](@ref)
"""
model_posterior(prob::BossProblem) =
    model_posterior(prob.model, prob.data)

"""
    model_posterior_slice(::BossProblem, slice::Int) -> post

Return the posterior predictive distributions of the given output `slice` with two methods:
- `post(x::AbstractVector{<:Real}) -> μ::Real, σ::Real`
- `post(X::AbstractMatrix{<:Real}) -> μ::AbstractVector{<:Real}, Σ::AbstractMatrix{<:Real}`

The first method takes a single point `x` of length `x_dim(::BossProblem)` from the `Domain`,
and returns the predictive mean and deviation
of the corresponding output number `y` such that:
- `μ, σ = post(x)` => `y ∼ Normal(μ, σ)`

The second method takes multiple points from the `Domain as a column-wise matrix `X` of size `(x_dim, N)`,
and returns the joint predictive mean and covariance matrix
of the corresponding output vector `y` of length `N` such that:
- `μ, Σ = post(X)` => `y ∼ MvNormal(μ, Σ)`

In case one is only interested in predicting a certain output dimension,
using `model_posterior_slice` can be more efficient than `model_posterior`
(depending on the used `SurrogateModel`).

Note that `model_posterior_slice` can be used even if `sliceable(model) == false`.
It will, however, not provide any additional efficiency in such case.

See also: [`model_posterior`](@ref)
"""
model_posterior_slice(prob::BossProblem, slice::Int) =
    model_posterior_slice(prob.model, prob.data, slice)

"""
    result(problem) -> (x, y)

Return the best found point `(x, y)`.

Returns the point `(x, y)` from the dataset of the given problem
such that `y` satisfies the constraints and `fitness(y)` is maximized.
Returns nothing if the dataset is empty or if no feasible point is present.

Does not check whether `x` belongs to the domain as no exterior points
should be present in the dataset.
"""
function result(problem::BossProblem)
    X, Y = problem.data.X, problem.data.Y
    @assert size(X)[2] == size(Y)[2]
    isempty(X) && return nothing

    feasible = is_feasible.(eachcol(Y), Ref(problem.y_max))
    fitness = problem.fitness.(eachcol(Y))
    fitness[.!feasible] .= -Inf
    best = argmax(fitness)

    feasible[best] || return nothing
    return X[:,best], Y[:,best]
end
