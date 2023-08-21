# Axis-aligned multivariate normal (i.e., independent entries, i.e., diagonal covariance matrix) using conjugate prior.
module NMN

module NMNcmodel # submodule for component family definitions
export Theta, Data, log_marginal, new_theta, Theta_clear!, Theta_adjoin!, Theta_remove!,
       Hyperparameters, construct_hyperparameters, update_hyperparameters!

using Statistics
using SpecialFunctions

const Data = Array{Float64,1}

mutable struct Theta
    n::Int64                 # number of data points assigned to this cluster
    sum_gene::Array{Float64,1}  # log-sum of the data points x assigned to this cluster i.e. log(C_i!)
    log_sum::Float64
    C0:: Int                 # Control parameter for overdispersion
    Theta(G, C0) = (p=new(); p.n=0; p.sum_gene=zeros(G+1); p.log_sum=0.; p.C0=C0; p)
end

new_theta(H) = Theta(H.G, H.C0)
Theta_clear!(p) = (p.n = 0; p.sum_gene .= 0.0; p.log_sum=0)          # should be β from Dir-prior
Theta_adjoin!(p,x) = (p.log_sum += loggamma(sum(x)+p.C0)-sum(loggamma.(vcat(p.C0, x .+ 1))); p.sum_gene += vcat(p.C0, x); p.n += 1)
Theta_remove!(p,x) = (p.log_sum -= loggamma(sum(x)+p.C0)-sum(loggamma.(vcat(p.C0, x .+ 1))); p.sum_gene -= vcat(p.C0, x); p.n -= 1)


function log_marginal(p,H)
    # Neg. Multinomial-Dirichlet-distribution: https://en.wikipedia.org/wiki/Dirichlet_negative_multinomial_distribution

    # Marginal derived from likelihood x prior / posterior:
    # Reference number in the Xournal formula:
    #                    1                       2                           3                       4             5           
    log_marg = sum(loggamma.(p.sum_gene .+H.β))-H.sum_lgamma-loggamma(sum(p.sum_gene)+H.beta_sum)+H.lgamma_sum+p.log_sum
    return log_marg
end

function log_marginal(x,p,H)
    Theta_adjoin!(p,x)
    result = log_marginal(p,H)
    print(result)
    Theta_remove!(p,x)
    return result
end

mutable struct Hyperparameters
    G::Int64 
    β::Array{Float64,1}  # prior Dirichlet parameter for gene expression levels
    beta_sum::Float64
    lgamma_sum::Float64
    sum_lgamma::Float64
    C0::Int64
end


function construct_hyperparameters(options)
    x = options.x
    #n = length(x)
    G = length(x[1])
    β =  options.β
    C0 = options.C0
    if any(β .<= 0)
        #@warn "β > 0 required and was thus set to repeat([1],G)"
        β = repeat([1],G+1)
    end
    beta_sum = sum(β)
    sum_lgamma = sum(loggamma.(β))
    lgamma_sum = loggamma(beta_sum)
    return Hyperparameters(G, β, beta_sum, lgamma_sum, sum_lgamma, C0)
end

function update_hyperparameters!(H,theta,list,t,x,z)
    #error("update_hyperparameters is not yet implemented.")
end

end #end module MNcmodel
using .NMNcmodel

# Include generic code
include("generic.jl")

# Include core sampler code
include("coreConjugate.jl")

end # module NMN



