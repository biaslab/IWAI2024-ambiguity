module Robots

using Distributions
using LinearAlgebra

export FieldBot, update

mutable struct FieldBot

    Dx :: Integer
    Du :: Integer
    Dy :: Integer
    Δt :: Float64

    g  :: Function
    A  :: Matrix{Float64}
    B  :: Matrix{Float64}
    Q  :: Matrix{Float64}
    R  :: Matrix{Float64}

    control_lims ::Tuple{Float64,Float64}

    function FieldBot(g::Function, ρ::Vector; σ::Float64=1.0, Δt::Float64=1.0, control_lims=(-1.,1.))

        Dx = 4
        Du = 2
        Dy = length(g(zeros(Dx)))

        # State transition
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
        Q = [Δt^3/3*ρ[1]          0.0  Δt^2/2*ρ[1]          0.0;
                     0.0  Δt^3/3*ρ[2]          0.0  Δt^2/2*ρ[2];
             Δt^2/2*ρ[1]          0.0      Δt*ρ[1]          0.0;
                     0.0  Δt^2/2*ρ[2]          0.0      Δt*ρ[2]]

        # Measurement noise covariance matrix
        R = diagm(ρ)
       
        return new(Dx,Du,Dy,Δt,g,A,B,Q,R,control_lims)
    end
end

function step(bot::FieldBot, z_kmin1, u_k)
    "Stochastic state transition"

    clamp!(u_k, bot.control_lims...)
    return rand(MvNormal(bot.A*z_kmin1 + bot.B*u_k, bot.Q))
end

function emit(bot::FieldBot, z_k)
    "Stochastic observation"

    return rand(MvNormal(bot.g(z_k), bot.R))
end

function update(bot::FieldBot, z_kmin1, u_k)
    "Update environment" 
     
    # State transition
    z_k = step(bot, z_kmin1, u_k)
     
    # Emit noisy observation
    y_k = emit(bot, z_k)
     
    return y_k, z_k
end

end