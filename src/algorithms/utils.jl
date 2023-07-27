using LatinHypercubeSampling

middle(domain::AbstractBounds) = [mean((l,u)) for (l,u) in zip(domain...)]

# Activation function to ensure positive optimization arguments.
softplus(x) = log(one(x) + exp(x))

function generate_starts_LHC(bounds::AbstractBounds, count::Int)
    @assert count > 1  # `randomLHC(count, dim)` returns NaNs if `count == 1`
    lb, ub = bounds
    x_dim = length(lb)
    starts = scaleLHC(randomLHC(count, x_dim), [(lb[i], ub[i]) for i in 1:x_dim]) |> transpose
    return starts
end

function random_start(bounds::AbstractBounds)
    lb, ub = bounds
    dim = length(lb)
    start = rand(dim) .* (ub .- lb) .+ lb
    return start
end

function sample_params_vec(model::Parametric, noise_var_priors::AbstractArray; softplus_noise_vars=false)
    θ = rand.(model.param_priors)
    noise_vars = rand.(noise_var_priors)
    
    softplus_noise_vars && (noise_vars = softplus.(noise_vars))
    return vcat(θ, noise_vars)
end
function sample_params_vec(model::Nonparametric, noise_var_priors::AbstractArray; softplus_noise_vars=false)
    λ = rand.(model.length_scale_priors)
    noise_vars = rand.(noise_var_priors)
    
    softplus_noise_vars && (noise_vars = softplus.(noise_vars))
    return vcat(reduce(vcat, λ), noise_vars)
end
function sample_params_vec(model::Semiparametric, noise_var_priors::AbstractArray; softplus_noise_vars=false)
    θ = rand.(model.parametric.param_priors)
    λ = rand.(model.nonparametric.length_scale_priors)
    noise_vars = rand.(noise_var_priors)
    
    softplus_noise_vars && (noise_vars = softplus.(noise_vars))
    return vcat(θ, reduce(vcat, λ), noise_vars)
end

function split_model_params(model::Parametric, params::AbstractArray{<:Real}; softplus_noise_vars=false)
    θ_len = length(model.param_priors)

    θ = params[1:θ_len]
    noise_vars = params[θ_len+1:end]

    softplus_noise_vars && (noise_vars = softplus.(noise_vars))
    return (θ=θ, noise_vars=noise_vars)
end
function split_model_params(model::Nonparametric, params::AbstractArray{<:Real}; softplus_noise_vars=false)
    x_dim = length(first(model.length_scale_priors))
    y_dim = length(model.length_scale_priors)
    λ_len = y_dim * x_dim
    
    length_scales = reshape(params[1:λ_len], x_dim, y_dim)
    noise_vars = params[λ_len+1:end]
    
    softplus_noise_vars && (noise_vars = softplus.(noise_vars))
    return (length_scales=length_scales, noise_vars=noise_vars)
end
function split_model_params(model::Semiparametric, params::AbstractArray{<:Real}; softplus_noise_vars=false)
    x_dim = length(first(model.nonparametric.length_scale_priors))
    y_dim = length(model.nonparametric.length_scale_priors)
    θ_len = length(model.parametric.param_priors)
    λ_len = y_dim * x_dim
    
    θ = params[1:θ_len]
    length_scales = reshape(params[θ_len+1:θ_len+λ_len], x_dim, y_dim)
    noise_vars = params[θ_len+λ_len+1:end]
    
    softplus_noise_vars && (noise_vars = softplus.(noise_vars))
    return (θ=θ, length_scales=length_scales, noise_vars=noise_vars)
end

# Maximize the given objective function with restarts.
function optimize_multistart(
    optimize::Base.Callable,  # arg, val = optimize(start)
    starts::AbstractMatrix{<:Real},
    parallel::Bool,
    info::Bool,
)   
    multistart = size(starts)[2]

    args = Vector{Vector{Float64}}(undef, multistart)
    vals = Vector{Float64}(undef, multistart)
    errors = Threads.Atomic{Int}(0)
    
    if parallel
        io_lock = Threads.SpinLock()
        Threads.@threads for i in 1:multistart
            try
                a, v = optimize(starts[:,i])
                args[i] = a
                vals[i] = v

            catch e
                if info
                    lock(io_lock)
                    try
                        warn_optim_err(e)
                    finally
                        unlock(io_lock)
                    end
                end
                Threads.atomic_add!(errors, 1)
                args[i] = Float64[]
                vals[i] = -Inf
            end
        end

    else
        for i in 1:multistart
            try
                a, v = optimize(starts[:,i])
                args[i] = a
                vals[i] = v

            catch e
                info && warn_optim_err(e)
                errors.value += 1
                args[i] = Float64[]
                vals[i] = -Inf
            end
        end
    end

    (errors.value == multistart) && throw(ErrorException("All acquisition optimization runs failed!"))
    info && (errors.value > 0) && @warn "$(errors.value)/$(multistart) acquisition optimization runs failed!\n"
    b = argmax(vals)
    return args[b], vals[b]
end

function warn_optim_err(e)
    @warn "Optimization error:"
    showerror(stderr, e); println(stderr)
end
