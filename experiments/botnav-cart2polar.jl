
using Pkg
Pkg.activate(".")
Pkg.instantiate()

using Revise
using JLD2
using Optim
using ForwardDiff
using ProgressMeter
using LinearAlgebra
using Distributions
using StatsPlots
using Plots
default(label="", grid=false, linewidth=3, markersize=3, margin=15Plots.pt)

includet("../ModelPredictiveControllers.jl"); using. ModelPredictiveControllers
includet("../FreeEnergyAgents.jl"); using. FreeEnergyAgents
includet("../Robots.jl"); using. Robots
includet("../util.jl")


"""Experimental parameters"""

num_reps = 10

# Time
Δt = 0.2
len_trial = 20
tsteps = range(0, step=Δt, length=len_trial)
len_horizon = 5;

# Nonlinear observation
g(x::AbstractVector) = [sqrt(x[1]^2 + x[2]^2), atan(x[2],x[1])]

# Goal
z_star = [0.0, 1., 0.0, 0.0]
goal = (g(z_star), 0.5diagm(ones(2)))

# Parameters
σ = 1e-3
ρ = [1e-2, 1e-2]
η = 0.0

# Limits of controller
u_lims = (-1.0, 1.0)
opts = Optim.Options(time_limit=20)

# Define robot
fbot = FieldBot(g, ρ, σ=σ, Δt=Δt, control_lims=u_lims)

# Define agent
efeagent  = EFEAgent(goal, g, ρ, σ=σ; η=η, Δt=Δt, time_horizon=len_horizon)
mpcontrol = MPController(z_star, g, ρ, σ=σ; η=η, Δt=Δt, time_horizon=len_horizon)

# Initial state
z_0 = [0.0, -1., 0., 0.]

# Initial belief
m_0 = z_0
S_0 = 0.5diagm(ones(4));

""" Run experiments """

methods = ["EFE2", "EFE1", "EFER"]

global approximation = "ET2"
global add_ambiguity = true

for method in methods
     println(method)

     # Preallocate
     F = zeros(len_trial, num_reps)
     J = zeros(len_trial, num_reps)     

     @showprogress for nn in 1:num_reps

          z_est  = (zeros(4,len_trial), zeros(4,4,len_trial))
          z_sim  = zeros(4,len_trial)
          y_sim  = zeros(2,len_trial)
          u_sim  = zeros(2,len_trial)

          # Initial state
          z_sim[:,1] = z_0
          z_est[1][:,1] = z_0
          z_est[2][:,:,1] = S_0

          # Start recursion
          m_kmin1 = m_0
          S_kmin1 = S_0

          for k in 2:len_trial
    
               "Interact with environment"
           
               # Update system with selected control
               y_sim[:,k], z_sim[:,k] = update(fbot, z_sim[:,k-1], u_sim[:,k-1])
                          
               "State estimation"
               
               m_k_pred, S_k_pred = FreeEnergyAgents.predict(efeagent, m_kmin1, S_kmin1, u_sim[:,k-1])
               m_k,S_k = FreeEnergyAgents.correct(efeagent, y_sim[:,k], m_k_pred, S_k_pred, approx="ET2")
               
               # Compute model evidence
               F[k,nn] = FreeEnergyAgents.evidence(efeagent, y_sim[:,k], m_k_pred, S_k_pred, approx="ET2")
               J[k,nn] = -logpdf(MvNormal(goal[1], goal[2]), y_sim[:,k])
               
               # Store state estimates
               z_est[1][:,k] = m_k
               z_est[2][:,:,k] = S_k
               
               "Planning"

               if method == "EFE2"
                    approximation = "ET2"
                    add_ambiguity = true
               elseif method == "EFE1"
                    approximation = "ET1"
                    add_ambiguity = true
               elseif method == "EFER"
                    approximation = "ET2"
                    add_ambiguity = false
               else
                    error("Method unknown")
               end
               
               # Single-argument objective
               G(u::AbstractVector) = EFE(efeagent, u, (m_k,S_k), approx=approximation, add_ambiguity=add_ambiguity)
               
               # Call minimizer using constrained L-BFGS procedure
               results = Optim.optimize(G, u_lims[1], u_lims[2], zeros(2*len_horizon), Fminbox(LBFGS()), opts; autodiff=:forward)
               
               # Extract minimizing control
               policy = reshape(Optim.minimizer(results), (2,len_horizon))
               u_sim[:,k] = policy[:,1]
              
               # Update recursion
               m_kmin1 = m_k
               S_kmin1 = S_k
               
           end

          # Write results of current repetition to file
          jldsave("experiments/results/botnav-cart2polar-$method-$nn.jld2"; F, J, z_est, z_sim, u_sim, y_sim, goal, Δt, len_trial, len_horizon, u_lims, η, z_0, m_0, S_0)

     end

     # Write results of current method to file
     jldsave("experiments/results/botnav-cart2polar-$method.jld2"; F, J, goal, Δt, len_trial, len_horizon, u_lims, η, z_star, z_0, m_0, S_0)

end

xl = [-2., 2.]
yl = [-1.2, 2.8]
k = len_trial

p93 = plot(size=(500,500), legend=:topleft, aspect_ratio=:equal, ylabel="position y", xlabel="position x", xlims=xl, ylims=yl)
scatter!([0.0], [0.0], color="black", marker=:ltriangle, label="sensor station", markersize=8)
scatter!([z_0[1]], [z_0[2]], color="green", label="start state", markersize=8)
scatter!([z_star[1]], [z_star[2]], color="red", label="goal state", markersize=8)

plot!([z_sim[1,1:k]], [z_sim[2,1:k]], c="blue", marker=".", label="system states", alpha=1., markersize=5)

plot!(z_est[1][1,1:k], z_est[1][2,1:k], c="purple", marker=".", label="state estimates", alpha=1., markersize=5)
for j in 1:len_trial
    covellipse!(z_est[1][1:2,j], z_est[2][1:2,1:2,j], n_std=1, color="purple", linewidth=1, fillalpha=0.2)
end
plot!(dpi=300)