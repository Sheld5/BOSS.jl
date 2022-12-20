using Distributions
using Optim
using Turing
using Turing: Variational
using Zygote
using Evolutionary
using Suppressor
using NLopt
using ForwardDiff

# - - - - - - OPTIM - - - - - -

# Find ̂x that maximizes f(x) using parallel multistart gradient-based optimization.
function optim_Optim_multistart(f, starts, constraints=nothing; options=Optim.Options(), parallel=true, info=true, debug=false)
    multistart = size(starts)[2]

    args = Vector{Vector{Float64}}(undef, multistart)
    vals = Vector{Float64}(undef, multistart)
    convergence_errors = [0 for _ in 1:multistart]
    
    if parallel
        Threads.@threads for i in 1:multistart
            try
                a, v = optim_Optim(f, starts[:,i], constraints; options, info)
                args[i] = a
                vals[i] = v
            catch e
                debug && throw(e)
                convergence_errors[i] += 1
                args[i] = Float64[]
                vals[i] = -Inf
            end
        end
    else
        for i in 1:multistart
            try
                a, v = optim_Optim(f, starts[:,i], constraints; options, info)
                args[i] = a
                vals[i] = v
            catch e
                debug && throw(e)
                convergence_errors[i] += 1
                args[i] = Float64[]
                vals[i] = -Inf
            end
        end
    end

    errs = sum(convergence_errors)
    info && (errs > 0) && @warn "      $(sum(convergence_errors))/$(multistart) optimization runs failed!\n"
    (errs == multistart) && throw(ErrorException("All optimization runs failed."))

    opt_i = argmax(vals)
    return args[opt_i], vals[opt_i]
end

function optim_Optim(f, start, constraints::Nothing; options=Optim.Options(), info=false)
    opt_res = Optim.optimize(p -> -f(p), start, NelderMead(), options)  # TODO try changing alg
    info && check_optim_convergence(opt_res)
    return Optim.minimizer(opt_res), -Optim.minimum(opt_res)
end
function optim_Optim(f, start, constraints::Tuple; options=Optim.options(), info=false)
    # return optim_(f, start, TwiceDifferentiableConstraints(constraints...); options, info)
    opt_res = Optim.optimize(p -> -f(p), constraints..., start, Fminbox(LBFGS()), options)
    info && check_optim_convergence(opt_res)
    return Optim.minimizer(opt_res), -Optim.minimum(opt_res)
end
function optim_Optim(f, start, constraints::TwiceDifferentiableConstraints; options=Optim.Options(), info=false, alpha=1e-8)
    IPNewton_check_start_!(start, get_bounds(constraints), alpha; info)
    opt_res = @suppress Optim.optimize(p -> -f(p), constraints, start, IPNewton(), options)  # suppress "initial point not interior" warnings
    info && check_optim_convergence(opt_res)
    arg, val = Optim.minimizer(opt_res), -Optim.minimum(opt_res)
    arg, val
end
function optim_Optim(f, start, constraints; kwargs...)
    throw(ArgumentError("Constraints of type `$(typeof(constraints))` are not supported with Optim. Use `Optim.TwiceDifferentiableConstraints` instead."))
end

function check_optim_convergence(opt_res)
    Optim.x_converged(opt_res) || @warn "Optimization run did not converge!"
end

# IPNewton cannot handle `start == bound`.
# (https://julianlsolvers.github.io/Optim.jl/stable/#examples/generated/ipnewton_basics/#generic-nonlinear-constraints)
function IPNewton_check_start_!(start, bounds, alpha; info=true)
    lb, ub = bounds
    @assert all(ub .- lb .>= 2*alpha)
    @assert all(start .>= lb) && all(start .<= ub)

    lb_far = ((start .- lb) .>= alpha) 
    ub_far = ((ub .- start) .>= alpha)
    all(lb_far) && all(ub_far) && return start
    
    info && @warn "Start is too close to the domain bounds. Moving it further."
    for i in eachindex(start)
        lb_far[i] || (start[i] = lb[i] + alpha)
        ub_far[i] || (start[i] = ub[i] - alpha)
    end
    return start
end

# - - - - - - CMA-ES - - - - - -

# maximize f(x)
optim_cmaes(f, start; kwargs...) = optim_cmaes(f, Evolutionary.NoConstraints(), start; kwargs...)
optim_cmaes(f, bounds::Tuple, start; kwargs...) = optim_cmaes(f, Evolutionary.BoxConstraints(bounds...), start; kwargs...)
function optim_cmaes(f, constraints::Evolutionary.AbstractConstraints, start; options=Evolutionary.Options(; Evolutionary.default_options(CMAES())...), info=false)
    res = Evolutionary.optimize(x->-f(x), constraints, start, CMAES(), options)
    info && (Evolutionary.iterations(res) == options.iterations) && println("Warning: Maximum iterations reached while optimizing!")
    return res.minimizer, f(res.minimizer)
end
function optim_cmaes(f, cons::Optim.TwiceDifferentiableConstraints, start; kwargs...)
    throw(ArgumentError("Constraints of type `$(typeof(cons))` are not supported with CMAES. Use `Evolutionary.WorstFitnessConstraints` or any other constraints from the `Evolutionary` package instead."))
end

function optim_cmaes_multistart(f, constraints, starts; options=Evolutionary.Options(; Evolutionary.default_options(CMAES())...), parallel=true, info=false)
    multistart = size(starts)[2]

    args = Vector{Vector{Float64}}(undef, multistart)
    vals = Vector{Float64}(undef, multistart)

    if parallel
        Threads.@threads for i in 1:multistart
            a, v = optim_cmaes(f, constraints, starts[:,i]; options, info)
            args[i] = a
            vals[i] = v
        end
    else
        for i in 1:multistart
            a, v = optim_cmaes(f, constraints, starts[:,i]; options, info)
            args[i] = a
            vals[i] = v
        end
    end
    
    opt_i = argmax(vals)
    return args[opt_i], vals[opt_i]
end

# - - - - - - NLopt - - - - - -

function optim_NLopt_multistart(f, starts; optimizer=nothing, parallel=true, info=true)
    multistart = size(starts)[2]

    args = Vector{Vector{Float64}}(undef, multistart)
    vals = Vector{Float64}(undef, multistart)
    
    if parallel
        Threads.@threads for i in 1:multistart
            a, v = optim_NLopt(f, starts[:,i]; optimizer, info)
            args[i] = a
            vals[i] = v
        end
    else
        for i in 1:multistart
            a, v = optim_NLopt(f, starts[:,i]; optimizer, info)
            args[i] = a
            vals[i] = v
        end
    end

    opt_i = argmax(vals)
    return args[opt_i], vals[opt_i]
end

function optim_NLopt(f, start; optimizer=nothing, info=false)
    isnothing(optimizer) && (optimizer = Opt(:LD_MMA, length(start)))
    
    function f_nlopt(x::Vector, grad::Vector)
        if length(grad) > 0
            grad .= ForwardDiff.gradient(f, x)
        end
        return f(x)
    end
    
    optimizer.max_objective = f_nlopt
    val, arg, ret = NLopt.optimize(optimizer, start)
    info && check_nlopt_convergence(ret)
    return arg, val
end

function check_nlopt_convergence(ret)
    if ret != :XTOL_REACHED
        @warn "Optimization terminated with return value `$ret`."
    end
end

# - - - - - - NUTS SAMPLING - - - - - -

"""
Stores hyperparameters of the MC sampler.

Amount of drawn samples:    'chain_count * (warmup + leap_size * sample_count)'
Amount of used samples:     'chain_count * sample_count'

# Fields
  - warmup: The amount of initial unused 'warmup' samples in each chain.
  - sample_count: The amount of samples used from each chain.
  - chain_count: The amount of independent chains sampled.
  - leap_size: The "distance" between two following used samples in a chain. (To avoid correlated samples.)

In each chain;
    Firstly, the first 'warmup' samples are discarded.
    Then additional 'leap_size * sample_count' samples are drawn
    and each 'leap_size'-th of these samples is kept.
Finally, kept samples from all chains are joined and returned.
"""
struct MCSettings{S}
    sampler::S
    warmup::Int
    samples_in_chain::Int
    chain_count::Int
    leap_size::Int
end

sample_count(mc::MCSettings) = mc.chain_count * mc.samples_in_chain

# Sample parameters of the given probabilistic model (defined with Turing.jl) using parallel NUTS MC sampling.
# Other AD backends than Zygote cause issues: https://discourse.julialang.org/t/gaussian-process-regression-with-turing-gets-stuck/86892
function sample_params_turing(model, param_symbols, mc::MCSettings; adbackend=:zygote, parallel=true)
    Turing.setadbackend(adbackend)

    samples_in_chain = mc.warmup + (mc.leap_size * mc.samples_in_chain)
    if parallel
        chains = Turing.sample(model, mc.sampler, MCMCThreads(), samples_in_chain, mc.chain_count; progress=false)
    else
        chains = mapreduce(_ -> Turing.sample(model, mc.sampler, samples_in_chain; progress=false), chainscat, 1:mc.chain_count)
    end

    samples = [reduce(vcat, eachrow(chains[s][(mc.warmup+mc.leap_size):mc.leap_size:end,:])) for s in param_symbols]
end

# TODO unused
function sample_params_vi(model, samples; alg=ADVI{Turing.AdvancedVI.ForwardDiffAD{0}}(10, 1000))
    posterior = vi(model, alg)
    rand(posterior, samples) |> eachrow |> collect
end

# - - - - - - DATA GENERATION - - - - - -

# Sample from uniform distribution.
function uniform_sample(a, b, sample_size)
    distr = Product(Distributions.Uniform.(a, b))
    X = rand(distr, sample_size)
    return vec.(collect.(eachslice(X; dims=length(size(X)))))
end

# Sample from log-uniform distribution.
log_sample(a, b, sample_size) = exp.(uniform_sample(log.(a), log.(b), sample_size))

# Return points distributed evenly over a given logarithmic range.
function log_range(a, b, len)
    a = log10.(a)
    b = log10.(b)
    range = collect(LinRange(a, b, len))
    range = [10 .^ range[i] for i in 1:len]
    return range
end

# - - - - - - MODEL ERROR - - - - - -

# Calculate RMS error with given y values and models predictions for them.
function rms_error(preds, ys; N=nothing)
    isnothing(N) && (N = length(preds))
    return sqrt((1 / N) * sum((preds .- ys).^2))
end
# Calculate RMS error with given test data.
function rms_error(X, Y, model)
    dims = size(Y)[2]
    preds = reduce(hcat, model.(eachrow(X)))'
    return [rms_error(preds[:,i], Y[:,i]) for i in 1:dims]
end
# Calculate RMS error using uniformly sampled test data.
function rms_error(obj_func, model, a, b, sample_count)
    X = reduce(hcat, uniform_sample(a, b, sample_count))'
    Y = reduce(hcat, obj_func.(X))'
    return rms_error(X, Y, model)
end

function partial_sums(array)
    isempty(array) && return empty(array)
    
    s = zero(first(array))
    sums = [(s += val) for val in array]
    return sums
end

# - - - - - - OTHER - - - - - -

get_bounds(bounds::Tuple) = bounds
get_bounds(constraints::TwiceDifferentiableConstraints) = get_bounds_std_format_(constraints)
get_bounds(constraints::Evolutionary.AbstractConstraints) = get_bounds_std_format_(constraints)
get_bounds(constraints::MixedTypePenaltyConstraints) = get_bounds(constraints.penalty)

function get_bounds_std_format_(constraints)
    domain_lb = constraints.bounds.bx[1:2:end]
    domain_ub = constraints.bounds.bx[2:2:end]
    return domain_lb, domain_ub
end

function in_domain(x::AbstractVector, domain::Tuple)
    lb, ub = domain
    any(x .< lb) && return false
    any(x .> ub) && return false
    return true
end
in_domain(x::AbstractVector, domain::TwiceDifferentiableConstraints) = Optim.isinterior(domain, x)
in_domain(x::AbstractVector, domain::Evolutionary.AbstractConstraints) = Evolutionary.isfeasible(domain, x)

Evolutionary.isfeasible(c::MixedTypePenaltyConstraints, x) = Evolutionary.isfeasible(c.penalty, x)
