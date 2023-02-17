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
    #lsum_gene::Array{Float64,1}  # log-sum of the data points x assigned to this cluster i.e. log(C_i!)
    sum_x::Array{Float64,1}
    Theta(d) = (p=new(); p.n=0; p.sum_x=zeros(d); p)
end

new_theta(H) = Theta(H.d)
Theta_clear!(p) = (p.sum_x .= 0; p.n = 0)
Theta_adjoin!(p,x) = (p.sum_x+=x; p.n += 1)
Theta_remove!(p,x) = (p.sum_x-=x; p.n -= 1)


# Marginal/compound distribution for one particular cluster specified by p (i.e. theta)
function log_marginal(p,H)
    # Multinomial-Dirichlet-distribution: https://en.wikipedia.org/wiki/Dirichlet-multinomial_distributio
    # logB = function (a,n)
    #     logΓ = function (n) 
    #         res = 0
    #         for i=2:(n-1)
    #             res += log(i)
    #         end
    #         return res
    #     end
    #     logΓ(a)+logΓ(n)-logΓ(a+n)
    # end
    print("\n P\n",p)
    log_marg = loggamma(sum(p.sum_x+H.β)) + loggamma(p.n+1) - loggamma(sum(p.sum_x+H.β)+p.n) +  sum(loggamma.(p.sum_x + H.β)) - H.lgamma_sum - sum(loggamma.(p.sum_x .+ 1))
    print("\n log-Marginal:\n",log_marg)
    return log_marg
end

function log_marginal(x,p,H)
    Theta_adjoin!(p,x)
    result = log_marginal(p,H)
    Theta_remove!(p,x)
    return result
end

mutable struct Hyperparameters
    d::Int64 
    β::Array{Float64,1}  # prior Dirichlet parameter for gene expression levels
    beta_sum::Float64
    lgamma_sum::Float64
end


function construct_hyperparameters(options)
    x = options.x
    n = length(x)
    d = length(x[1])
    β =  options.β
    if any(β .<= 0)
        @warn "β > 0 required and was thus set to repeat([1],G)"
        β; β = repeat([1],d)
    end
    beta_sum = sum(β)
    lgamma_sum = loggamma(beta_sum)
    
    return Hyperparameters(d,β,beta_sum,lgamma_sum)
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



