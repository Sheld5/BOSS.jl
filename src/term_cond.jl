"""
    IterLimit(iter_max::Int)

Terminates the BOSS algorithm after predefined number of iterations.

See also: [`BOSS.boss!`](@ref)
"""
mutable struct IterLimit <: TermCond
    iter::Int
    iter_max::Int
end
IterLimit(iter_max::Int) = IterLimit(0, iter_max)

function (cond::IterLimit)(problem::OptimizationProblem)
    (cond.iter >= cond.iter_max) && return false
    cond.iter += 1
    return true
end
