module FreeEnergyAgents

using LinearAlgebra
using Distributions
include("util.jl")

export EFEAgent, predict, correct, EFE, evidence, risk, ambiguity, planned_trajectory

mutable struct EFEAgent

    Dx :: Integer
    Du :: Integer
    Dy :: Integer
    Δt :: Float64

    g  :: Function
    A  :: Matrix{Float64}
    B  :: Matrix{Float64}
    Q  :: Matrix{Float64}
    R  :: Matrix{Float64}
    η  :: Float64

    goal :: Tuple{Vector,Matrix}
    time_horizon :: Integer

    function EFEAgent(goal::Tuple{Vector,Matrix}, 
                      g::Function, 
                      ρ::Vector; 
                      σ::Float64=1.0, 
                      η::Float64=1.0, 
                      Δt::Float64=1.0,
                      time_horizon::Integer=1)
        "Construct agent"

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
        R = diagm(σ^2*ones(Dy))
        
        return new(Dx,Du,Dy,Δt,g,A,B,Q,R,η,goal,time_horizon)
    end
end

function predict(agent::EFEAgent, m_kmin1, S_kmin1, u_kmin1)
    "Chapman-Kolmogorov for linear Gaussian state transition using known control u"
    
    m_k_pred = agent.A*m_kmin1 + agent.B*u_kmin1
    S_k_pred = agent.A*S_kmin1*agent.A' .+ agent.Q
    
    return m_k_pred, S_k_pred
end

function correct(agent::EFEAgent, y_k, m_k_pred, S_k_pred; approx="ET2")
    "Correction step based on Gaussian approximation to nonlinear measurement"
    
    if approx == "ET1"
        μ, Σ, Γ = ET1(m_k_pred, S_k_pred, agent.g, addmatrix=agent.R, forceHermitian=true)
    elseif approx == "ET2"
        μ, Σ, Γ = ET2(m_k_pred, S_k_pred, agent.g, addmatrix=agent.R, forceHermitian=true)
    elseif approx == "UT"
        μ, Σ, Γ = UT( m_k_pred, S_k_pred, agent.g, addmatrix=agent.R, forceHermitian=true)
    else
        error("Approximation method unknown.")
    end

    m_k = m_k_pred .+ Γ*inv(Σ)*(y_k - μ)
    S_k = S_k_pred .- Γ*inv(Σ)*Γ'

    return m_k, S_k
end

function condition_yx(m,S, dims::Integer=1)
    """
    Conditioning a Gaussian distribution.

    Appendix A(5), Särkkä (2013), Bayesian filtering & Smoothing.
    """

    m_a = m[1:dims]
    m_b = m[dims+1:end]

    S_A = S[1:dims, 1:dims]
    S_B = S[dims+1:end, dims+1:end]
    S_C = S[1:dims, dims+1:end]
    
    m_y(x) = m_b + S_C'*inv(S_A)*(x - m_a)
    S_y(x) = S_B - S_C'*inv(S_A)*S_C
    
    return m_y, S_y
end

function ambiguity(Σ,Γ,S)
    "Conditional entropy term within expected free energy"

    return 0.5*(size(Σ,1)*log(2π*ℯ) + logdet(Σ - Γ'*inv(S)*Γ))
 end

function risk(μ, Σ, goal)
    "Kullback-Leibler divergence term within expected free energy"
    
    m_star, S_star = goal
    D = length(m_star)
    
    L0 = cholesky(Σ).L
    L1 = cholesky(S_star).L
    
    M = inv(L1)*L0
    y = inv(L1)*(m_star - μ)
    
    return 0.5(sum(M[:].^2) - D + norm(y,1).^2 + 2*sum([log(L1[i,i]./L0[i,i]) for i in 1:D]))
end

function evidence(agent::EFEAgent, y_k, m_k, S_k; approx="ET2")
    "Marginal likelihood"
    
    # Gaussian approximation
    if approx == "ET1"
        μ, Σ, Γ = ET1(m_k, S_k, agent.g, addmatrix=agent.R, forceHermitian=true)
    elseif approx == "ET2"
        μ, Σ, Γ = ET2(m_k, S_k, agent.g, addmatrix=agent.R, forceHermitian=true)
    elseif approx == "UT"
        μ, Σ, Γ = UT( m_k, S_k, agent.g, addmatrix=agent.R, forceHermitian=true)
    else
        error("Approximation method unknown.")
    end

    return -logpdf(MvNormal(μ, Matrix(Σ)), y_k)
end

function EFE(agent::EFEAgent,
             u::AbstractVector, 
             state::Tuple{Vector{Float64}, Matrix{Float64}};
             approx::String="ET2",
             add_ambiguity::Bool=true)
    "Expected Free Energy"

    # Unpack parameters of current state
    m_tmin1, S_tmin1 = state

    # Start cumulative sum
    cEFE = 0.0
    for t in 1:agent.time_horizon

        # State transition p(z_t | u_t)
        m_t,S_t = predict(agent, m_tmin1, S_tmin1, u[(t-1)*2+1:2t])

        # Gaussian approximation
        if approx == "ET1"
            μ, Σ, Γ = ET1(m_t, S_t, agent.g, addmatrix=agent.R, forceHermitian=true)
        elseif approx == "ET2"
            μ, Σ, Γ = ET2(m_t, S_t, agent.g, addmatrix=agent.R, forceHermitian=true)
        elseif approx == "UT"
            μ, Σ, Γ = UT( m_t, S_t, agent.g, addmatrix=agent.R, forceHermitian=true)
        else
            error("Approximation method unknown.")
        end

        # Accumulate objective
        cEFE += risk(μ,Σ, agent.goal) + agent.η*u[t]^2
        if add_ambiguity; cEFE += ambiguity(Σ,Γ, S_t); end

        # Update state recursion
        m_tmin1 = m_t
        S_tmin1 = S_t
        
    end
    return cEFE
end

function planned_trajectory(agent::EFEAgent,
                            policy, 
                            current_state; 
                            approx="ET2")
    "Generate future states and observations"
    
    # Unpack parameters of current state
    m_tmin1, S_tmin1 = current_state
    
    # Track predicted observations
    z_m = zeros(4,  agent.time_horizon)
    z_S = zeros(4,4,agent.time_horizon)
    y_m = zeros(2,  agent.time_horizon)
    y_S = zeros(2,2,agent.time_horizon)
    
    for t in 1:agent.time_horizon
        
        # State transition
        z_m[:,t] = agent.A*m_tmin1 + agent.B*policy[:,t]
        z_S[:,:,t] = agent.A*S_tmin1*agent.A' + agent.Q
        
        # Gaussian approximation
        if approx == "ET1"
            y_m[:,t], y_S[:,:,t] = ET1(z_m[:,t], z_S[:,:,t], agent.g, addmatrix=agent.R, forceHermitian=true)
        elseif approx == "ET2"
            y_m[:,t], y_S[:,:,t] = ET2(z_m[:,t], z_S[:,:,t], agent.g, addmatrix=agent.R, forceHermitian=true)
        elseif approx == "UT"
            y_m[:,t], y_S[:,:,t] = UT(z_m[:,t], z_S[:,:,t], agent.g, addmatrix=agent.R, forceHermitian=true)
        else
            error("Approximation method unknown.")
        end
        
        # Update previous state
        m_tmin1 = z_m[:,t]
        S_tmin1 = z_S[:,:,t]
        
    end
    return (z_m, z_S), (y_m, y_S)
end

end