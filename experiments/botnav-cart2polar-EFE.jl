using Pkg
Pkg.activate(".")
Pkg.instantiate()

using Colors
using Optim
using ForwardDiff
using ProgressMeter
using LinearAlgebra
using ControlSystems
using Distributions
using StatsPlots
using Plots
default(label="", grid=false, linewidth=3, markersize=3, margin=15Plots.pt)

include("../agent.jl")
include("../system.jl")
include("../util.jl")


# Experimental parameters

# Time
Δt = 0.1
len_time = 10
tsteps = range(0, step=Δt, length=len_time)

# State transition matrix
A = [1. 0. Δt 0.;
     0. 1. 0. Δt;
     0. 0. 1. 0.;
     0. 0. 0. 1.]

# Control matrix
B = [0. 0.;
     0. 0.;
     Δt 0.;
     0. Δt]

# Process noise covariance matrix
ρ_1 = 1e0
ρ_2 = 1e0
Q = [Δt^3/3*ρ_1          0.0  Δt^2/2*ρ_1          0.0;
             0.0  Δt^3/3*ρ_2          0.0  Δt^2/2*ρ_2;
     Δt^2/2*ρ_1          0.0      Δt*ρ_1          0.0;
             0.0  Δt^2/2*ρ_2          0.0      Δt*ρ_2]

# Nonlinear observation
g(x::AbstractVector) = [sqrt(x[1]^2 + x[2]^2), atan(x[2],x[1])]

# Measurement noise
R = 1e0*diagm(ones(2));