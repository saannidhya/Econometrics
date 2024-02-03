### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ a551bf77-bf1d-41b9-ba22-a0d80b13f177
begin
	using Pkg
	Pkg.add("Binscatters")
	Pkg.add("Optim")
	Pkg.add("StatsPlots")
	Pkg.add("FixedEffectModels")
	Pkg.add("GLM")	
	Pkg.add("Distributions")
	Pkg.add("DataFrames")
	Pkg.add("LinearAlgebra")
	Pkg.add("Statistics")
	Pkg.add("Plots")
	Pkg.add("ForwardDiff")
	Pkg.add("LineSearches")
end

# ╔═╡ 06f880f8-dd94-11eb-2930-7555dcfa1cde
using LinearAlgebra, Distributions, Statistics, DataFrames, Optim, StatsPlots, Plots, GLM, FixedEffectModels, Binscatters, ForwardDiff, LineSearches

# ╔═╡ 20efbda2-87ba-4ca3-8a36-ee3de4059a8c
md"""
### Model
Household $i$ residing in neighborhood $n$ receives direct utility:

```math
u_{in} = \beta_i^C \ln c_{in} + \beta_i^A A_n + \beta_i^gg\left(N_{wn},N_{bn}\right) + \xi_n + \varepsilon_{in}
```
where $c_{in}$ is household consumption, $A_n$ is the log of exogenous neighborhood amenities, $g\left(N_{wn},N_{bn}\right)$ represents the log of the value households place on endogenous amenities, which are a function of $N_{wn}$ and $N_{bn}$, the number of white and black households in the neighborhood respectively. $\xi_n$ is unobserved neighborhood quality and $\varepsilon_{in}$ is an idiosyncratic neighborhood preference.

Households consume one unit of housing. We normalize the price of non-housing consumption to 1 and let $R_n$ denote the price of housing in neighborhood $n$. Households thus choose consumption and neighborhoods subject to the budget constraint

```math
c_{in} \leq I_i - R_n
```

where $I_i$ is household income. Thus, household $i$ residing in neighborhood $n$ receives indirect utility

```math
v_{in} = \beta_i^C \ln(I_i - R_n) + \beta_i^A A_n + \beta_i^g g\left(N_{wn},N_{bn}\right) + \xi_{n\tau} + \varepsilon_{in}
```

Note that households' preferences are permitted to vary by type $\tau \in \{B,W\}$:

```math
\beta_i^k = \beta_0^k + z_i'\beta_\tau^k
```

for $k \in \{C,A,g\}$ and where $z_i$ is a $2 \times 1$ vector indicating household $i$'s type. We can thus rewrite indirect utility as:

```math
v_{in} = \beta_i^C \ln(I_i-R_n) + \delta_{n\tau} + \varepsilon_{in}
```
where $\delta_{n\tau} \equiv \xi_{n\tau} + \beta_\tau^A A_n + \beta_\tau^g g_n$.

We will assume that the endogenous amenities function takes the following simple form: 

```math
g\left(N_{wn},N_{bn}\right) = \frac{N_{wn}}{N_{bn}.}
```
Finally, the idiosyncratic shock $\varepsilon_{in}$ is assumed to have a Type-I extreme value distribution.

"""

# ╔═╡ ab51bc20-57aa-42e2-863a-59b8073c70fc
md"""
### Simulate data
"""

# ╔═╡ 9b9af415-ad45-45e9-b5f2-7e405998dc45
md"""
Define parameters:
"""

# ╔═╡ 6555bbf4-d99e-42e0-9058-47443a97a0e6
begin
	ss = 200000
	nss = 50
	β_c = .5
	β_cw = .2
	β_A = .3
	β_Aw = 0
	β_g = 0
	β_gw = 0
	β_ξ = 1 # All types have common value for unobserved neighborhood characteristics

end;

# ╔═╡ 347b5dee-a2bc-44bd-8808-d9ba5a36a70c
md"""
We use the `gendata()` function to simulate the data observed in multiple markets.
"""

# ╔═╡ abb519a8-3528-4b77-ab17-2a26ed0e8747
md"""
We will estimate this model via MLE. To do so, first note that we can write the predicted share of each household $i$ living in neighobrhood $n$ as:

```math
\pi_{in} = \frac{\exp((1-z_i)\delta_n^b + z_i\delta_n^w + (\beta_c + z_i\beta_{cw}) \ln(I_i - r_n)) }{\sum_k \exp((1-z_i)\delta_k^b + z_i\delta_k^w + (\beta_c + z_i\beta_{cw}) \ln(I_i - r_k))}
```
The first estimation step finds the vector $\boldsymbol{\theta} = (\beta_{c}, \beta_{cw}, {\delta}_{n}^b, \delta_n^w)$ that solves the following problem:

```math
\underset{\theta}{\max}\sum_{i=1}^{ss}\sum_{n=1}^{nss} \log(\pi_{in})1\{d_i = n\},
```
subject to,
```math
\pi_{in} = \pi_{in}^e
```
where $1\{d_i = n\}$ is an indicator equal to one if household $i$ lives in neighborhood $n$, and $\pi_{in}^e$ is the observed share of households in each neigbhrohood. 

For each trial of $\beta_{cw}$ and $\beta_{Aw}$, the constraint fully determines the value of ${\delta}_{n}$. Finding the values of ${\delta}_{n}$ that satisfy the constraint can be done quickly using our contraction mapping. 

We use 2SLS and our estimates $\hat{\delta}_n^\tau$ to estimate the remaining parameters.
"""

# ╔═╡ 14bbd6c9-d6ce-4b4e-af41-1abd79d58236
md"""
### Estimate preferences with IV
"""

# ╔═╡ 4da4cc9a-77bc-4102-97b1-66f0127f9496
β_A, β_Aw, β_g, β_gw

# ╔═╡ 20f4ac6f-0b42-4256-a5cb-36411c727bda
md"""
---
### Define functions used above
"""

# ╔═╡ be4e26a7-385f-4953-9507-8efa041028bf
function gendata(ss, nss, β_c, β_cw, β_A, β_Aw, β_g, β_gw, β_ξ)
	# Generate households -----
	HH_data_sim = DataFrame(White = rand((0,1),ss), Income = rand(Normal(45,9),ss))
	HH_data_sim.Black = (HH_data_sim.White .- 1) * (-1)
	for n in 1:nss
    	HH_data_sim[!,"ε_$n"] = rand(Gumbel(0,4),ss)
	end

	# Generate neighborhoods -----
	NN_data = DataFrame(Neighborhood = (1:nss), 
						ξ_b = rand(Normal(0, 4), nss),
						Z0 = ones(nss),
						Z1 = 5*randn(nss),
						Z2 = 5*randn(nss))
	NN_data.ξ_w = NN_data.ξ_b + .75*randn(nss)
	NN_data.A = .125(NN_data.ξ_w + NN_data.ξ_b) + NN_data.Z1 + randn(nss)
	NN_data.r = .125(NN_data.ξ_w + NN_data.ξ_b) + 3NN_data.Z2 + randn(nss)
	
	# Initialize parameters
	N_w0, N_b0 = ones(nss) * ss / 2, ones(nss) * ss / 2
	N_w1, N_b1 = zeros(nss), zeros(nss)
	diff = 1
	eps_N = 1 / 100
	
	count = 1
	
	while diff > eps_N && count < 40 # To simulate observed distributions, we need to account for the role of endogenous amenities:
	
		for n in 1:nss
			# Assign utility level for each household in each neighborhood:
			afford = HH_data_sim.Income .> NN_data.r[n]
			z = HH_data_sim[afford, "White"]
			HH_data_sim[!,"v_$n"] .= -Inf
			HH_data_sim[afford, "v_$n"] = (β_c .+ z*β_cw).*log.(HH_data_sim[afford, "Income"] .- NN_data.r[n]) + (β_A .+ z*β_Aw)*NN_data.A[n] + (β_g .+ z*β_gw)*N_w0[n]/N_b0[n] .+ NN_data.ξ_b[n] .+ z * (NN_data.ξ_w[n] - NN_data.ξ_b[n]) .+ HH_data_sim[afford, "ε_$n"]
		end

		# Find neighborhood that yields highest utility to each household:
		HH_data_sim.pref_n = vec([i[2] for i in findmax(Matrix(HH_data_sim[:, r"v_."]), dims = 2)[2]])

		# Calculate number of each type of household that prefers each neighborhood:
		for n in 1:nss
			N_w1[n] = sum(HH_data_sim.pref_n .* HH_data_sim.White .== n)
			N_b1[n] = sum(HH_data_sim.pref_n .* HH_data_sim.Black .== n)	
		end	

		diff = maximum(abs.(N_w0 ./ N_b0 - N_w1 ./ N_b1))
		N_w0 = copy(N_w1)
		N_b0 = copy(N_b1)
	
	count += 1
		
	end	
	
	### Clean household data:
	HH_data = HH_data_sim[:, Not(r"(v|ε)_.")]
	
	# Generate observed neighborhood data
	NN_data.share = zeros(nss)
	NN_data.pct_white = zeros(nss)
	NN_data.pct_black = zeros(nss)
	NN_data.N_w = zeros(nss)
	NN_data.N_b = zeros(nss)
	NN_data.Income_b = zeros(nss)
	NN_data.Income_w = zeros(nss)
	for n in 1:nss
	    NN_data.share[n] = mean(HH_data.pref_n .== n)
		NN_data.pct_white[n] = mean(HH_data.White[HH_data.pref_n .== n])
		NN_data.pct_black[n] = 1 .- NN_data.pct_white[n]
		NN_data.Income_b[n] = mean(HH_data.Income[HH_data.Black .* HH_data.pref_n .== n])
		NN_data.Income_w[n] = mean(HH_data.Income[HH_data.White .* HH_data.pref_n .== n])
	end
	NN_data.share_white = NN_data.share .* NN_data.pct_white / mean(HH_data.White)
	NN_data.share_black = NN_data.share .* NN_data.pct_black / mean(HH_data.Black)

	for n in 1:nss
		NN_data.N_w[n] = sum(HH_data.pref_n .* HH_data.White .== n)
		NN_data.N_b[n] = sum(HH_data.pref_n .* HH_data.Black .== n)	
	end	

	NN_data.g = NN_data.N_w ./ NN_data.N_b
	NN_data.δ_b = NN_data.ξ_b + β_A*NN_data.A + β_g*NN_data.g
	NN_data.δ_w = NN_data.ξ_w + (β_A+β_Aw)*NN_data.A + (β_g+β_gw)*NN_data.g
	
	#NN_data.Z3 = exp.(NN_data.Z1)
	#NN_data.Z4 = exp.(NN_data.Z2)
	NN_data.Z3 = residuals(reg(NN_data, @formula(g ~ A + r + ξ_b + ξ_w)), NN_data)
	
	return HH_data, NN_data

end
	

# ╔═╡ 88b54c13-aff4-44f7-8f4e-4dd8a74c68b9
begin
	HH_data1, NN_data1 = gendata(ss, nss, β_c, β_cw, β_A, β_Aw, β_g, β_gw, β_ξ)
end;

# ╔═╡ 7c9062d7-15a6-493a-9778-fb2c63d10c83
begin
	HH_data = HH_data1
	NN_data = NN_data1
end;

# ╔═╡ 84f9f088-83d0-426b-b070-a80e81358d6f
describe(HH_data)

# ╔═╡ 518010b6-97f4-4f33-bc24-da95845c8dfe
describe(NN_data)

# ╔═╡ 7d1dd7aa-103c-41ff-aa9f-7ea890c425b6
# IV estimation
begin
	foo_b = copy(NN_data)
	foo_w = copy(NN_data)
	foo_b.white = zeros(nrow(NN_data))
	foo_w.white = ones(nrow(NN_data))
	#foo_b.δhat = δ_bhat
	#foo_w.δhat = δ_what
	foo = [foo_b; foo_w]
	foo.δ = foo.δ_b .+ (foo.δ_w .- foo.δ_b) .* foo.white
	reg(foo, @formula(δ ~ ((g + A)*white ~ (Z1 + Z2 + Z3)*white)))
end

# ╔═╡ 1b4623fd-06a0-4ad1-b2eb-a529da0f7878
#= 
Takes as input a length 57 vector of parameters. The first 7 entries are the β parameters and the last 50 are the δ parameteres
=#

function predsharesblack(θ)

    # Assign inputs:
	β_c = θ[1]; β_cw = θ[2] 
	δ = θ[3:end]
    
    # Calculate predicted utility:    
    df = HH_data[HH_data.Black .== 1,:]
    pred_data = DataFrame()
    
	for n in 1:nss  # loop through neighborhoods, generating ss X nss matrix
        
		# Assign utility level for each household in each neighborhood:
        afford = df.Income .> NN_data.r[n]
		# Assign utility level for each household in each neighborhood:
        #pred_data[!, "$n"] = ones(nrow(df)).*(-Inf)
		pred_data[!, "$n"] = real(afford.*(β_c*log.(Complex.(df.Income .- NN_data.r[n])) .+ δ[n])/4 + (1 .- afford)*(-1000000))
	end 
    
    # Calculate predicted shares:
    pred_data[!,"Denom"] = sum(eachcol(exp.(pred_data)))
    pred_share = DataFrame()
    for n in 1:nss
    	pred_data[!, "Ratio_$n"] = exp.(pred_data[:, n]) ./ pred_data.Denom  # individual level choice probabilities  
        pred_share[!, "pred_share_$n"] = sum(eachrow(pred_data[:, "Ratio_$n"])) / size(df, 1)  # take average across individual choice probabilities to get predicted shares
    end

    # Return vector of predicted shares:
    return(transpose(Matrix(pred_share))) # nss x 1

end


# ╔═╡ 881dfde6-8302-4582-a2b1-e0c887ae9f62
#= 
Takes as input a length 57 vector of parameters. The first 7 entries are the β parameters and the last 50 are the δ parameteres
=#

function predshareswhite(θ)

    # Assign inputs:
	β_c = θ[1]; β_cw = θ[2] 
	δ = θ[3:end]
    
    # Calculate predicted utility:    
    df = HH_data[HH_data.White .== 1,:]
    pred_data = DataFrame()
    
	for n in 1:nss  # loop through neighborhoods, generating ss X nss matrix
        
		# Assign utility level for each household in each neighborhood:
        afford = df.Income .> NN_data.r[n]
		# Assign utility level for each household in each neighborhood:
        pred_data[!, "$n"] = real(afford.*(β_c*log.(Complex.(df.Income .- NN_data.r[n])) .+ δ[n])/4 + (1 .- afford)*(-1000000))
	end 
    
    # Calculate predicted shares:
    pred_data[!,"Denom"] = sum(eachcol(exp.(pred_data)))
    pred_share = DataFrame()
    for n in 1:nss
    	pred_data[!, "Ratio_$n"] = exp.(pred_data[:, n]) ./ pred_data.Denom  # individual level choice probabilities  
        pred_share[!, "pred_share_$n"] = sum(eachrow(pred_data[:, "Ratio_$n"])) / size(df, 1)  # take average across individual choice probabilities to get predicted shares
    end

    # Return vector of predicted shares:
    return(transpose(Matrix(pred_share))) # nss x 1

end


# ╔═╡ 2f8960c7-24cf-4cf7-b7d5-76c9664ddae4
#= 
Returns a vector of predicted mean utilities for black households
=#

function contractionblack(β)   
	df = NN_data
	β_c = β[1]; β_cw = β[2]
	eps = 1 / 10^4
    δ_0 = ones(nss)
    dist = 1
    count = 0
	
    while dist > eps && count <= 100
        δ_1 = δ_0 - log.(predsharesblack([β_c; β_cw; δ_0])) + log.(df.share_black)
        dist = maximum(abs.(δ_1 - δ_0))
        δ_0 = δ_1
        count += 1
    end
    return δ_0
end

# ╔═╡ 9a6250c7-9450-42f6-9550-d48bd3ff8ea4
#= 
Returns a vector of predicted mean utilities for white households. 
=#

function contractionwhite(β)   
	df = NN_data
	β_c = β[1]; β_cw = β[2]
    eps = 1 / 10^4
    δ_0 = ones(nss)
    dist = 1
    count = 0
	
    while dist > eps && count <= 100
        δ_1 = δ_0 - log.(predshareswhite([β_c; β_cw; δ_0])) + log.(df.share_white)
        dist = maximum(abs.(δ_1 - δ_0))
        δ_0 = δ_1
        count += 1
    end
    return δ_0
end

# ╔═╡ 5533d823-bd7f-4f8a-b91e-5f6b61ae1b3a
function individual_cp(β)

    # Assign inputs:
	δ_b = contractionblack(β)
	δ_w = contractionwhite(β)
	β_c = β[1]  
	β_cw = β[2]
    
    # Calculate predicted utility:    
    z = HH_data.White
    pred_data = DataFrame()
    
	for n in 1:nss  # loop through neighborhoods, generating ss X nss matrix
        
		# Assign utility level for each household in each neighborhood:
		# NEED TO SET UTILITY TO -Inf IF OUT OF BUDGET SET 
		afford = HH_data.Income .> NN_data.r[n]
		pred_data[:,"$n"] = afford .* real.(((1 .- z) * δ_b[n] .+ z * δ_w[n] .+ (β_c .+ β_cw*z) .* log.(Complex.(HH_data.Income .- NN_data.r[n])))/4) - (1 .- afford)*1000000

	end 
	
    # Calculate individual choice probabilities:
    pred_data[!,"Denom"] = sum(eachcol(exp.(pred_data)))
    pred_share = DataFrame()
    for n in 1:nss
    	pred_data[!, "prob_$n"] = exp.(pred_data[:, n]) ./ pred_data[:, "Denom"] #individual choice probabilities
	end
	
	return pred_data[:,r"prob_."]
	
end


# ╔═╡ e8095574-8b38-41c5-9ebb-afb0f149f287
function loglikelihood_fn(β)

	df = log.(individual_cp(β))  # individual level LOG choice probabilities  

	loglikelihood = 0
	for i in 1:ss
		d = HH_data[i, "pref_n"]
	loglikelihood += df[i, "prob_$d"]
	end
	println(loglikelihood)
	return (-1) * loglikelihood
	
end

# ╔═╡ 3308b9e9-9d26-4915-8d0b-9b3f87b5ded5
MLE_results = let
	initial_vals = [.25, 0]
	optimize(loglikelihood_fn, initial_vals, LBFGS(linesearch=LineSearches.HagerZhang()),  Optim.Options(show_trace = true, f_tol = 1e-8), autodiff=:forward)
end

# ╔═╡ 1fcf5281-56aa-40a5-a9ea-923bd96d8b4c
MLE_results.minimizer, [β_c, β_cw]

# ╔═╡ 56491759-2776-46a8-8f3a-46ad1aaebada
δ_bhat = contractionblack(MLE_results.minimizer); δ_what = contractionwhite(MLE_results.minimizer)

# ╔═╡ a5c0f004-fc82-495e-a8d8-5daa9483997f
ForwardDiff.jacobian(individual_cp,[.2,0])

# ╔═╡ 753962a0-c429-4280-aa7f-e0f613d91e89
#= 
Takes as inputs households' preferences β as well as a vector of mean utilities for each neighborhood 
as a function of β: δ_β. Returns a scalar moment.
=#

function Cov_m(θ)
	# Get predicted neighborhood shares conditional on being white
	s_black = predsharesblack(θ)
	pr_black = mean(HH_data.Black)
	
	# construct moments
	m_a = sum(NN_data.A .* NN_data.pct_black .* NN_data.share) - sum(NN_data.A .* s_black * pr_black)
	m_r = sum(NN_data.r .* NN_data.pct_black .* NN_data.share) - sum(NN_data.r .* s_black * pr_black)
	
	return [m_a, m_r]
end


# ╔═╡ 613f4fa7-b99d-422e-9872-d768d27faaed
# Are predicted δs correlated with true δs?
plot(contractionblack(NN_data),NN_data.δ_b, seriestype=:scatter)

# ╔═╡ 203aba34-9035-41d0-b796-32c60d4cebcc
plot(NN_data.δ_what,NN_data.δ_w, seriestype=:scatter)

# ╔═╡ Cell order:
# ╠═a551bf77-bf1d-41b9-ba22-a0d80b13f177
# ╠═06f880f8-dd94-11eb-2930-7555dcfa1cde
# ╟─20efbda2-87ba-4ca3-8a36-ee3de4059a8c
# ╟─ab51bc20-57aa-42e2-863a-59b8073c70fc
# ╟─9b9af415-ad45-45e9-b5f2-7e405998dc45
# ╠═6555bbf4-d99e-42e0-9058-47443a97a0e6
# ╟─347b5dee-a2bc-44bd-8808-d9ba5a36a70c
# ╠═88b54c13-aff4-44f7-8f4e-4dd8a74c68b9
# ╠═7c9062d7-15a6-493a-9778-fb2c63d10c83
# ╠═84f9f088-83d0-426b-b070-a80e81358d6f
# ╠═518010b6-97f4-4f33-bc24-da95845c8dfe
# ╟─abb519a8-3528-4b77-ab17-2a26ed0e8747
# ╠═3308b9e9-9d26-4915-8d0b-9b3f87b5ded5
# ╠═1fcf5281-56aa-40a5-a9ea-923bd96d8b4c
# ╟─14bbd6c9-d6ce-4b4e-af41-1abd79d58236
# ╠═56491759-2776-46a8-8f3a-46ad1aaebada
# ╠═4da4cc9a-77bc-4102-97b1-66f0127f9496
# ╠═7d1dd7aa-103c-41ff-aa9f-7ea890c425b6
# ╟─20f4ac6f-0b42-4256-a5cb-36411c727bda
# ╠═be4e26a7-385f-4953-9507-8efa041028bf
# ╟─5533d823-bd7f-4f8a-b91e-5f6b61ae1b3a
# ╟─e8095574-8b38-41c5-9ebb-afb0f149f287
# ╠═a5c0f004-fc82-495e-a8d8-5daa9483997f
# ╠═1b4623fd-06a0-4ad1-b2eb-a529da0f7878
# ╠═881dfde6-8302-4582-a2b1-e0c887ae9f62
# ╠═2f8960c7-24cf-4cf7-b7d5-76c9664ddae4
# ╠═9a6250c7-9450-42f6-9550-d48bd3ff8ea4
# ╠═753962a0-c429-4280-aa7f-e0f613d91e89
# ╠═613f4fa7-b99d-422e-9872-d768d27faaed
# ╠═203aba34-9035-41d0-b796-32c60d4cebcc
