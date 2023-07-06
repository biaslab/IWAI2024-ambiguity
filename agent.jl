using LinearAlgebra
using Distributions
include("util.jl")


function predict(m_kmin1, S_kmin1, u_kmin1, A,B,Q)
    "Chapman-Kolmogorov for linear Gaussian state transition using known control u"
    m_k_pred = A*m_kmin1 + B*u_kmin1
    S_k_pred = A*S_kmin1*A' .+ Q
    return m_k_pred, S_k_pred
end

function correct_ET2(y_, m_k_pred, S_k_pred, g, R)
    "Correction step based on second-order Taylor approximation to nonlinear measurement"
    μ, Σ, Γ = ET2(m_k_pred, S_k_pred, g, addmatrix=R, forceHermitian=true)
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

function EFE(u::AbstractVector, 
             state::Tuple{Vector{Float64}, Matrix{Float64}}, 
             goal::Tuple{Vector{Float64}, Matrix{Float64}}; 
             v_u::Float64=1.0,
             time_horizon::Int64=1)
    "Expected Free Energy"

    # Unpack parameters of current state
    m_tmin1, S_tmin1 = state

    # Start cumulative sum
    cEFE = 0.0
    for t in 1:time_horizon

        # State transition p(z_t | u_t)
        m_t = A*m_tmin1 + B*u[(t-1)*2+1:2t]
        S_t = A*S_tmin1*A' + Q

        # Unscented transform moments
        μ, Σ, Γ = ET2(m_t, S_t, g, addmatrix=R, forceHermitian=true)

        # Cumulate EFE
        cEFE += risk(μ,Σ, goal) + ambiguity(Σ,Γ, S_t) + 1/(v_u)*u[t]^2

        # Update state recursion
        m_tmin1 = m_t
        S_tmin1 = S_t
        
    end
    return cEFE
end