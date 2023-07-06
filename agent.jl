using LinearAlgebra
using Distributions


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