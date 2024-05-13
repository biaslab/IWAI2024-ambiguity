using Distributions
using LinearAlgebra


function step(z_kmin1, u_k, A,B,Q)
    "Stochastic state transition"
     return rand(MvNormal(A*z_kmin1 + B*u_k, Q))
end

function emit(z_k,g,R)
    "Stochastic observation"
    return rand(MvNormal(g(z_k), R))
end

function update(z_kmin1, u_k, A,B,Q,R, g)
    "Update environment" 
     
     # State transition
     z_k = step(z_kmin1,u_k,A,B,Q)
     
     # Emit noisy observation
     y_k = emit(z_k,g,R)
     
     return y_k, z_k
 end