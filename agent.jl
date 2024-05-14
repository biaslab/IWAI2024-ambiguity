using LinearAlgebra
using Distributions

include("util.jl")


function predict(m_kmin1, S_kmin1, u_kmin1, A,B,Q)
    "Chapman-Kolmogorov for linear Gaussian state transition using known control u"
    m_k_pred = A*m_kmin1 + B*u_kmin1
    S_k_pred = A*S_kmin1*A' .+ Q
    return m_k_pred, S_k_pred
end

function correct(y_, m_k_pred, S_k_pred, g, R; approx="ET2")
    "Correction step based on second-order Taylor approximation to nonlinear measurement"
    if approx == "ET1"
        μ, Σ, Γ = ET1(m_k_pred, S_k_pred, g, addmatrix=R, forceHermitian=true)
    elseif approx == "ET2"
        μ, Σ, Γ = ET2(m_k_pred, S_k_pred, g, addmatrix=R, forceHermitian=true)
    elseif approx == "UT"
        μ, Σ, Γ = UT(m_k_pred, S_k_pred, g, addmatrix=R, forceHermitian=true)
    else
        error("Approximation method unknown.")
    end

    m_k = m_k_pred .+ Γ*inv(Σ)*(y_[:,k] - μ)
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
    "Epistemics-based term in Expected Free Energy functional."
    return 0.5*logdet(Σ - Γ'*inv(S)*Γ)
 end

function risk(μ, Σ, goal)
    "Goal-driven term in Expected Free Energy functional."
    
    m_star, S_star = goal
    k = length(m_star)
    
    L0 = cholesky(Σ).L
    L1 = cholesky(S_star).L
    
    M = inv(L1)*L0
    y = inv(L1)*(m_star - μ)
    
    return 0.5(sum(M[:].^2) - k + norm(y,1).^2 + 2*sum([log(L1[i,i]./L0[i,i]) for i in 1:k]))
end

function evidence(y,m,S; approx="ET1")
    "Marginal likelihood"
    
    # Gaussian approximation
    if approx == "ET1"
        μ, Σ, Γ = ET1(m_, S_t, g, addmatrix=R, forceHermitian=true)
    elseif approx == "ET2"
        μ, Σ, Γ = ET2(m_t, S_t, g, addmatrix=R, forceHermitian=true)
    elseif approx == "UT"
        μ, Σ, Γ = UT(m_t, S_t, g, addmatrix=R, forceHermitian=true)
    else
        error("Approximation method unknown.")
    end

    return pdf(MvNormal(μ, Matrix(Σ)), y)
end

function EFE(u::AbstractVector, 
             state::Tuple{Vector{Float64}, Matrix{Float64}}, 
             goal::Tuple{Vector{Float64}, Matrix{Float64}}; 
             v_u::Float64=1.0,
             time_horizon::Int64=1,
             approx="ET2")
    "Expected Free Energy"

    # Unpack parameters of current state
    m_tmin1, S_tmin1 = state

    # Start cumulative sum
    cEFE = 0.0
    for t in 1:time_horizon

        # State transition p(z_t | u_t)
        m_t = A*m_tmin1 + B*u[(t-1)*2+1:2t]
        S_t = A*S_tmin1*A' + Q

        # Gaussian approximation
        if approx == "ET1"
            μ, Σ, Γ = ET1(m_, S_t, g, addmatrix=R, forceHermitian=true)
        elseif approx == "ET2"
            μ, Σ, Γ = ET2(m_t, S_t, g, addmatrix=R, forceHermitian=true)
        elseif approx == "UT"
            μ, Σ, Γ = UT(m_t, S_t, g, addmatrix=R, forceHermitian=true)
        else
            error("Approximation method unknown.")
        end

        # Add to cumulative EFE
        cEFE += risk(μ,Σ, goal) + ambiguity(Σ,Γ, S_t) + 1/(v_u)*u[t]^2

        # Update state recursion
        m_tmin1 = m_t
        S_tmin1 = S_t
        
    end
    return cEFE
end

function planned_trajectory(policy, current_state; time_horizon=1, approx="ET2")
    "Generate future states and observations"
    
    # Unpack parameters of current state
    m_tmin1, S_tmin1 = current_state
    
    # Track predicted observations
    z_m = zeros(4,  time_horizon)
    z_S = zeros(4,4,time_horizon)
    y_m = zeros(2,  time_horizon)
    y_S = zeros(2,2,time_horizon)
    
    for t in 1:time_horizon
        
        # State transition
        z_m[:,t] = A*m_tmin1 + B*policy[:,t]
        z_S[:,:,t] = A*S_tmin1*A' + Q
        
        # Predicted observations
        # Gaussian approximation
        if approx == "ET1"
            y_m[:,t], y_S[:,:,t] = ET1(z_m[:,t], z_S[:,:,t], g, addmatrix=R, forceHermitian=true)
        elseif approx == "ET2"
            y_m[:,t], y_S[:,:,t] = ET2(z_m[:,t], z_S[:,:,t], g, addmatrix=R, forceHermitian=true)
        elseif approx == "UT"
            y_m[:,t], y_S[:,:,t] = UT(z_m[:,t], z_S[:,:,t], g, addmatrix=R, forceHermitian=true)
        else
            error("Approximation method unknown.")
        end
        
        # Update previous state
        m_tmin1 = z_m[:,t]
        S_tmin1 = z_S[:,:,t]
        
    end
    return (z_m, z_S), (y_m, y_S)
end