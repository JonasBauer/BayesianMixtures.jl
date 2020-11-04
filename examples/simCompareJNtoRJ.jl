# Compare Jain-Neal sampler to Reversible Jump on simulated data of increasing dimension.
module SimCompareJNtoRJ

using Statistics
using Random
using BayesianMixtures
B = BayesianMixtures

# Settings
ds = [4,8,12]  # increasing dimensions
reset_random = true  # reset the random number generator before each run
save_figures = true  # save the figures to file
n_total = 1000000  # number of MCMC sweeps to run each algorithm

fignum = 0
for (i_d,d) in enumerate(ds)
    B.open_figure(9+i_d)
    for (label,mode,color) in [("Collapsed Jain-Neal","MVNaaC","g"),
                               ("Uncollapsed Jain-Neal","MVNaaN","r"),
                               ("RJMCMC","MVNaaRJ","b")]
    
        if reset_random; Random.seed!(0); end

        # Generate data
        n = 100
        xs = [randn(d) for i = 1:n]
        zs = [Int(ceil(rand()*3)) for i = 1:n]
        shift = (3/sqrt(d))*[-1,0,1]
        x = convert(Array{Array{Float64,1},1},[xs[i].+shift[zs[i]] for i = 1:n])
        mu = mean(x)  # sample mean
        v = mean([xi.*xi for xi in x]) - mu.*mu  # sample variance
        x = [((xi-mu)./sqrt.(v))::Array{Float64,1} for xi in x] # normalized to zero mean, unit variance

        # Run sampler
        println(label)
        options = B.options(mode,"MFM",x,n_total; n_keep=1000,t_max=20)
        result = B.run_sampler(options)

        # Traceplot of t
		global fignum
        B.open_figure(fignum+=1)
        t_show = 20  # number of seconds to show
        B.traceplot_timewise(result,t_show)
        B.PyPlot.title("$label, d = $d")
        if save_figures; B.PyPlot.savefig("simCompareJNtoRJ-traceplot-$mode-d=$d.png",dpi=200); end

        # Plot autocorrelation
        B.open_figure(9+i_d; clear_figure=false)
        t_show = 0.025*d  # number of seconds to show
        B.plot_autocorrelation(result,t_show; color=color,label=label)
        B.PyPlot.title("Autocorrelation, d = $d")
    end
    if i_d==1; B.PyPlot.legend(loc="upper right",fontsize=12); end
    if save_figures; B.PyPlot.savefig("simCompareJNtoRJ-autocorrelation-d=$d.png",dpi=200); end
end

end # module SimCompareJNtoRJ

