using Turing

include("utils.jl")

# TODO docs, comments & example

abstract type ParamModel end

function (model::ParamModel)(params)
    return x -> model(x, params)
end

"""
Used to define a linear parametric model for the BOSS algorithm.
The model has to be linear in its parameters and have Gaussian parameter priors.
This model definition will provide better performance than the 'NonlinModel' option in the future. (Not yet implemented.)

The linear model is defined as
    ϕs = lift(x)
    y = [θs[i]' * ϕs[i] for i in 1:m]
where
    x = [x₁, ..., xₙ]
    y = [y₁, ..., yₘ]
    θs = [θ₁, ..., θₘ], θ_i = [θᵢ₁, ..., θᵢₚ]
    ϕs = [ϕ₁, ..., ϕₘ], ϕ_i = [ϕᵢ₁, ..., ϕᵢₚ]
     n, m, p ∈ R .

# Fields
  - lift:           A function: x::Vector{Float64} -> ϕs::Vector{Vector{Float64}}
  - param_priors:   A vector of priors for all params θᵢⱼ.
  - param_count:    The number of model parameters. Equal to 'm * p'.
"""
struct LinModel <: ParamModel
    lift::Function
    param_priors::Vector{Normal}
    param_count::Int
end

function (model::LinModel)(x, params)
    ϕs = model.lift(x)
    m = length(ϕs)

    ϕ_lens = length.(ϕs)
    θ_indices = vcat(0, partial_sums(ϕ_lens))
    
    y = [(params[θ_indices[i]+1:θ_indices[i+1]])' * ϕs[i] for i in 1:m]
    return y
end

"""
Used to define a parametric model for the BOSS algorithm.

# Fields
  - predict:        A function: x::Vector{Float64}, params::Vector{Float64} -> y::Vector{Float64}
  - param_priors:   A vector of priors for each model parameter.
  - param_count:    The number of model parameters. (The length of 'params'.)
"""
struct NonlinModel{D} <: ParamModel
    predict::Function
    param_priors::Vector{D}
    param_count::Int
end

function (m::NonlinModel)(x, params)
    return m.predict(x, params)
end

function convert(::Type{NonlinModel}, model::LinModel)
    return NonlinModel(
        (x, params) -> model(x, params),
        model.param_priors,
        model.param_count,
    )
end

# - - - - - - - - - - - - - - - - - - - - - - - -

function param_posterior(Φs, Y, model::LinModel, noise_priors)
    throw(ErrorException("Support for linear models not implemented yet."))
    # θ_posterior_(...)
end

function θ_posterior_(Φ, y, θ_prior, noise)
    # refactor for noise prior instead of a given noise value

    ω = 1 / noise
    μθ, Σθ = θ_prior
    inv_Σθ = inv(Σθ)

    Σθ_ = inv(inv_Σθ + ω * Φ' * Φ)
    μθ_ = Σθ_ * (inv_Σθ * μθ + ω * Φ' * y)

    return μθ_, Σθ_
end

function model_params_loglike(X, Y, model::ParamModel)
    xs = collect(eachrow(X))
    data_size = length(xs)

    function loglike(params, noise)
        μs = model(params).(xs)
        L_data = sum([logpdf(MvNormal(μs[i], noise), Y[i,:]) for i in 1:data_size])
        L_params = sum([logpdf(model.param_priors[i], params[i]) for i in 1:model.param_count])
        return L_data + L_params
    end

    return loglike
end

function fit_model_params_lbfgs(X, Y, model::ParamModel, noise_priors; y_dim, multistart, info=true, debug=false, min_param_value=1e-6)
    params_loglike = model_params_loglike(X, Y, model::ParamModel)
    noise_loglike = noise -> sum([logpdf(noise_priors[i], noise[i]) for i in 1:y_dim])
    
    function loglike(p)
        noise, params = p[1:y_dim], p[y_dim+1:end]
        return params_loglike(params, noise) + noise_loglike(noise)
    end

    starts = reduce(hcat, [generate_start_(model, noise_priors) for _ in 1:multistart])'
    
    p, _ = optim_params(loglike, starts; info, debug)
    noise, params = p[1:y_dim], p[y_dim+1:end]
    return params, noise
end

function generate_start_(model, noise_priors)
    return vcat([rand(d) for d in noise_priors], [rand(d) for d in model.param_priors])
end

# Sample from the posterior parameter distributions given the data 'X', 'Y'.
function sample_param_posterior(X, Y, par_model, noise_priors; y_dim, mc_settings::MCSettings)
    Turing.@model function prob_model(X, Y, model, noise_priors, y_dim, ::Type{V}=Float64) where {V}
        params = Vector{V}(undef, model.param_count)
        for i in 1:model.param_count
            params[i] ~ model.param_priors[i]
        end

        noise = Vector{V}(undef, y_dim)
        for i in 1:y_dim
            noise[i] ~ noise_priors[i]
        end
    
        for i in 1:size(X)[1]
            Y[i,:] ~ Distributions.MvNormal(model(X[i,:], params), noise)
        end
    end

    param_symbols = vcat([Symbol("params[$i]") for i in 1:par_model.param_count],
                         [Symbol("noise[$i]") for i in 1:y_dim])
    model = prob_model(X, Y, par_model, noise_priors, y_dim)
    samples = sample_params_nuts(model, param_symbols, mc_settings)
    
    params = collect(eachrow(reduce(hcat, samples[1:par_model.param_count])))
    noise = collect(eachrow(reduce(hcat, samples[par_model.param_count+1:end])))
    return params, noise
end