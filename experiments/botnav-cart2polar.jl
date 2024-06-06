
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

includet("../FreeEnergyAgents.jl"); using. FreeEnergyAgents
includet("../ModelPredictiveControllers.jl"); using. ModelPredictiveControllers
includet("../Robots.jl"); using. Robots
includet("../util.jl")


"""Experimental parameters"""

num_reps = 100

# Time
Δt = 0.2
len_trial = 50
tsteps = range(0, step=Δt, length=len_trial)
len_horizon = 5;

# Nonlinear observation
g(x::AbstractVector) = [sqrt(x[1]^2 + x[2]^2), atan(x[2],x[1])]

# Goal
z_star = [0.0, .5, 0.0, 0.0]
goal = (g(z_star), diagm(ones(2)))

# Parameters
σ = 1e-4
ρ = [1e-3, 1e-3]
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
z_0 = [0.0, -.5, 0., 0.]

# Initial belief
m_0 = z_0
S_0 = 0.5diagm(ones(4));

""" Run experiments """

methods = ["EFE2", "EFE1", "EFER", "MPC"]

for method in methods
     println(method)

     # Preallocate
     FE1 = zeros(len_trial-1, num_reps)
     FE2 = zeros(len_trial-1, num_reps)
     YGP = zeros(len_trial-1, num_reps)
     z_est  = (zeros(4,len_trial, num_reps), zeros(4,4,len_trial, num_reps))
     z_sim  = zeros(4,len_trial, num_reps)
     y_sim  = zeros(2,len_trial, num_reps)
     u_sim  = zeros(2,len_trial, num_reps)

     @showprogress for nn in 1:num_reps

          # Initial state
          z_sim[:,1,nn] = z_0

          # Start recursion
          m_kmin1 = m_0
          S_kmin1 = S_0

          for k in 2:len_trial
          
               "Interact with environment"
               
               # Update system with selected control
               y_sim[:,k,nn], z_sim[:,k,nn] = update(fbot, z_sim[:,k-1,nn], u_sim[:,k-1,nn])

               # Compute negative log-likelihood under goal prior
               YGP[k-1,nn] = -logpdf(MvNormal(goal[1], goal[2]), y_sim[:,k,nn])
               
               "State estimation"
               
               # Prediction step
               m_k_pred, S_k_pred = predict(efeagent, m_kmin1, S_kmin1, u_sim[:,k-1,nn])
               
               "Planning"
               
               # Single-argument objective
               if method == "EFE2"

                    m_k,S_k = correct(efeagent, y_sim[:,k,nn], m_k_pred, S_k_pred, approx="ET2")
                    G(u::AbstractVector) = EFE(efeagent, u, (m_k,S_k), approx="ET2")
                    FE1[k-1,nn] = evidence(efeagent, y_sim[:,k,nn], m_k_pred, S_k_pred, approx="ET1")
                    FE2[k-1,nn] = evidence(efeagent, y_sim[:,k,nn], m_k_pred, S_k_pred, approx="ET2")
               
               elseif method == "EFER"
                    
                    m_k,S_k = correct(efeagent, y_sim[:,k,nn], m_k_pred, S_k_pred, approx="ET2")
                    G(u::AbstractVector) = EFE(efeagent, u, (m_k,S_k), approx="ET2", add_ambiguity=false)
                    FE1[k-1,nn] = evidence(efeagent, y_sim[:,k,nn], m_k_pred, S_k_pred, approx="ET1")
                    FE2[k-1,nn] = evidence(efeagent, y_sim[:,k,nn], m_k_pred, S_k_pred, approx="ET2")

               elseif method == "EFE1"
                    
                    m_k,S_k = correct(efeagent, y_sim[:,k,nn], m_k_pred, S_k_pred; approx="ET1")
                    G(u::AbstractVector) = EFE(efeagent, u, (m_k,S_k), approx="ET1")
                    FE1[k-1,nn] = evidence(efeagent, y_sim[:,k,nn], m_k_pred, S_k_pred, approx="ET1")
                    FE2[k-1,nn] = evidence(efeagent, y_sim[:,k,nn], m_k_pred, S_k_pred, approx="ET2")

               elseif method == "MPC"
                    
                    m_k,S_k = correct(efeagent, y_sim[:,k,nn], m_k_pred, S_k_pred, approx="ET2")
                    G(u::AbstractVector) = objective(mpcontrol, u, (m_k,S_k))
                    FE1[k-1,nn] = evidence(mpcontrol, y_sim[:,k,nn], m_k_pred, S_k_pred, approx="ET1")
                    FE2[k-1,nn] = evidence(mpcontrol, y_sim[:,k,nn], m_k_pred, S_k_pred, approx="ET2")

               else
                    error("Method unknown")
               end

               # Store state estimates
               z_est[1][:,k,nn] = m_k
               z_est[2][:,:,k,nn] = S_k
               
               # Call minimizer using constrained L-BFGS procedure
               results = Optim.optimize(G, u_lims[1], u_lims[2], zeros(2*len_horizon), Fminbox(LBFGS()), opts; autodiff=:forward)
               
               # Extract minimizing control
               policy = reshape(Optim.minimizer(results), (2,len_horizon))
               u_sim[:,k,nn] = policy[:,1]
               
               # Update recursion
               m_kmin1 = m_k
               S_kmin1 = S_k
          
          end

          # Write results of current repetition to file
          # jldsave("experiments/results/botnav-cart2polar-$method-$nn.jld2"; FE, z_est, z_sim, u_sim, y_sim, goal, Δt, len_trial, len_horizon, u_lims, η, z_0, m_0, S_0)

     end

     # Write results of current method to file
     jldsave("experiments/results/botnav-cart2polar-$method.jld2"; FE1, FE2, YGP, z_est, z_sim, y_sim, u_sim, goal, Δt, len_trial, len_horizon, u_lims, η, z_0, m_0, S_0)

end