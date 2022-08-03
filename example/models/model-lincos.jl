using Distributions
using Soss

function m_lincos_(x, params)
    return params[1] * x[1] * safe_cos_(params[2] * x[1] + params[3]) + params[4]
end
function safe_cos_(x::Real)
    isinf(x) && return 0.
    return cos(x)
end

function prob_model_lincos_()
    return @model X, noise begin
        params ~ For(zeros(4)) do _
            Distributions.Normal(1., 1.)
        end

        Y ~ For(collect(eachrow(X))) do x
            Distributions.Normal(m_lincos_(x, params), noise[1])
        end
        return Y
    end
end

function model_lincos()
    return SSModel(m_lincos_, prob_model_lincos_())
end
