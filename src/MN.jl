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
    sum_gene::Array{Float64,1}  # log-sum of the data points x assigned to this cluster i.e. log(C_i!)
    log_sum::Float64
    Theta(d) = (p=new(); p.n=0; p.sum_gene=zeros(d); p.log_sum=0.; p)
end

new_theta(H) = Theta(H.d)
Theta_clear!(p) = (p.n = 0; p.sum_gene .= 0.0; p.log_sum=0)          # should be β from Dir-prior
Theta_adjoin!(p,x) = (p.log_sum += loggamma(sum(x)+1)-sum(loggamma.(x .+ 1)); p.sum_gene += x; p.n += 1)
Theta_remove!(p,x) = (p.log_sum -= loggamma(sum(x)+1)-sum(loggamma.(x .+ 1)); p.sum_gene -= x; p.n -= 1)


# Marginal/compound distribution for one particular cluster specified by p (i.e. theta)
function log_marginal(p,H)
    # Multinomial-Dirichlet-distribution: https://en.wikipedia.org/wiki/Dirichlet-multinomial_distributio

    # First trial
    #log_marg = loggamma(sum(p.sum_x+H.β)) + loggamma(p.n+1) - loggamma(sum(p.sum_x+H.β)+p.n) +  sum(loggamma.(p.sum_x + H.β)) - H.lgamma_sum - sum(loggamma.(p.sum_x .+ 1))
    # log(Product density from Wikipedia's Multinomial-Dirichlet compound)
    # Product density of compound with prior
    # log_marg = p.n*(H.lgamma_sum + loggamma(p.n+1) - loggamma(p.n+H.beta_sum) - H.sum_lgamma) + p.sum_lgamma
    # # Product density of compound with posterior
    # post_beta_sum = H.beta_sum+sum(p.sum_gene)
    # log_marg = p.n*(loggamma(post_beta_sum) + loggamma(p.n+1) - loggamma(p.n+post_beta_sum) - sum(loggamma.(H.β .+ p.sum_gene))) + p.sum_lgamma

    #print("\n log-Marginal:\n",log_marg)
    #print("\n P\n",p)

    # Marginal derived from likelihood x prior / posterior:
    log_marg = sum(loggamma.(p.sum_gene .+H.β))-H.sum_lgamma-loggamma(sum(p.sum_gene)+H.beta_sum)+H.lgamma_sum + p.log_sum 

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
    sum_lgamma::Float64
end


function construct_hyperparameters(options)
    x = options.x
    #n = length(x)
    d = length(x[1])
    β =  options.β
    if any(β .<= 0)
        #@warn "β > 0 required and was thus set to repeat([1],G)"
        β; β = repeat([1],d)
    end
    beta_sum = sum(β)
    sum_lgamma = sum(loggamma.(β))
    lgamma_sum = loggamma(beta_sum)
    return Hyperparameters(d,β,beta_sum,lgamma_sum, sum_lgamma)
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



