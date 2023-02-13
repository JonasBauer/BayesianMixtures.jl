# Axis-aligned multivariate normal (i.e., independent entries, i.e., diagonal covariance matrix) using conjugate prior.
module MN

module MNcmodel # submodule for component family definitions
export Theta, Data, log_marginal, new_theta, Theta_clear!, Theta_adjoin!, Theta_remove!,
       Hyperparameters, construct_hyperparameters, update_hyperparameters!

using Statistics
using SpecialFunctions

const Data = Array{Float64,1}

mutable struct Theta
    n::Int64                 # number of data points assigned to this cluster
    sum_x::Array{Float64,1}  # sum of the data points x assigned to this cluster
    sum_xx::Array{Float64,1} # sum of x.*x for the data points assigned to this cluster
    Theta(d) = (p=new(); p.n=0; p.sum_x=zeros(d); p.sum_xx=zeros(d); p)
end
new_theta(H) = Theta(H.d)
Theta_clear!(p) = (p.sum_x[:] .= 0.; p.sum_xx[:] .= 0.; p.n = 0)
Theta_adjoin!(p,x) = (for i=1:length(x); p.sum_x[i] += x[i]; p.sum_xx[i] += x[i]*x[i]; end; p.n += 1)
Theta_remove!(p,x) = (for i=1:length(x); p.sum_x[i] -= x[i]; p.sum_xx[i] -= x[i]*x[i]; end; p.n -= 1)



# In each dimension independently,
# X_1,...,X_n ~ Normal(mu,1/lambda) with Normal(mu|m,1/(c*lambda))Gamma(lambda|a,b) prior on mean=mu, precision=lambda.
function log_marginal(p,H)
    print(H)
    # Multinomial-Dirichlet-distribution: https://en.wikipedia.org/wiki/Dirichlet-multinomial_distributio
    logΓ = function (n) 
        res = 0
        for i=2:(n-1)
            res += log(i)
        end
        return res
    end
    logB = function (a,n)
        logΓ(a)+logΓ(n)-logΓ(a+n)
    end
    for k in 1:d
        res += sum(logΓ())+logB(p.sum_x[k]+H.α)-logB(H.α)
    end
    return res
end

function log_marginal(x,p,H)
    Theta_adjoin!(p,x)
    result = log_marginal(p,H)
    Theta_remove!(p,x)
    return result
end

mutable struct Hyperparameters
    d::Int64 
    α::Array{Float64,1}  # prior Dirichlet parameter for gene expression levels
end

function construct_hyperparameters(options)
    x = options.x
    n = length(x)
    d = length(x[1])
    α = repeat([1],d)
    return Hyperparameters(α)
end

function update_hyperparameters!(H,theta,list,t,x,z)
    #error("update_hyperparameters is not yet implemented.")
end

end #end module MNcmodel
using .MNcmodel

# Include generic code
include("generic.jl")

# Include core sampler code
include("coreConjugate.jl")

end # module MN



