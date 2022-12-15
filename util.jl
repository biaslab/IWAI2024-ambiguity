using LinearAlgebra



function polar2cart(z; l=1.0)
    "Map angles to Cartesian space"

    # Position of first mass
    x = l*sin.(z[1])
    y = l*-cos.(z[1])
    
    return (x,y)
end

function sqrtm(M::AbstractMatrix)
    "https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix"

    if size(M) !== (2,2); error("Matrix must be 2x2"); end

    A,C,B,D = M

    # Determinant
    δ = A*D - B*C
    s = sqrt(δ)

    # Trace
    τ = A+D
    t = sqrt(τ + 2s)

    return 1/t*(M+s*Matrix{eltype(M)}(I,2,2))
end

function project2posdef!(S::AbstractMatrix)
    L,V = eigen(S)
    return V*diagm(max.(0.0,L))*V'
end

function sigma_points(m::AbstractVector, P::AbstractMatrix; α=1e-3, κ=0.0)
    
    # Number of sigma points depends on dimensionality
    N = size(P,2)
    
    # Compute scaling parameter
    λ = α^2*(N+κ)-N
    
    # Square root of covariance matrix through Babylonian method
    sP = sqrtm(P)
    
    # Preallocate
    sigma = Matrix(undef,N,2N+1)
    
    # First point is mean
    sigma[:,1] = m
    
    # Positive
    for n = 1:N
        sigma[:,1+n] = m + sqrt(N+λ)*sP[:,n]
    end
            
    # Negative
    for n = 1:N
        sigma[:,1+N+n] = m - sqrt(N+λ)*sP[:,n]
    end
    
    return sigma
end

function ut_weights(; α=1e-3, β=2.0, κ=0.0, N=1)
    
    # Compute scaling parameter
    λ = α^2*(N+κ)-N
    
    # Preallocate
    Wm = Vector(undef, 2N+1)
    Wc = Vector(undef, 2N+1)
    
    # Zero-order weights
    Wm[1] = λ/(N+λ)
    Wc[1] = λ/(N+λ) + (1-α^2+β)
    
    for n = 2:(2N+1)
        Wm[n] = 1/(2(N+λ))
        Wc[n] = 1/(2(N+λ))
    end
    return Wm,Wc
end

function UT(m::AbstractVector, P::AbstractMatrix, g; Q=nothing, D=1, α=1e-3, β=2.0, κ=0.0)
    "Algorithm 5.12 in 'Bayesian filtering & smoothing'"
    
    # Number of sigma points depends on dimensionality
    N = size(P,2)
    
    # Compute constant weigths
    Wm, Wc = ut_weights(α=α, β=β, κ=κ, N=N)
    
    # Form sigma points
    sp = sigma_points(m,P, α=α, κ=κ)
    
    # Propagate sigma points through non-linearity
    if D == 1
        
        # y = Vector{Real}(undef, 2N+1)
        y = zeros(eltype(m), 2N+1)
        for i in 1:(2N+1)
            y[i] = g(sp[:,i])
        end

        # Compute moments of approximated distribution
        μ = y'*Wm
        Σ = Wc[1]*(y[1] - μ)*(y[1] - μ)'
        C = Wc[1]*(sp[:,1] - m)*(y[1] - μ)'
        for i = 2:2N+1
            Σ += Wc[i]*(y[i] - μ)*(y[i] - μ)'
            C += Wc[i]*(sp[:,i] - m)*(y[i] - μ)'
        end
        
        # # Compute moments of approximated distribution
        # μ = dot(y,Wm)
        # Σ = sum([Wc[i]*(y[i] .- μ)*(y[i] .- μ)' for i in 1:(2N+1)])
        # C = sum([Wc[i]*(sp[:,i] .- m)*(y[i] .- μ)' for i in 1:(2N+1)])
    else
        
        y = Matrix(undef, D,2N+1)
        for i in 1:(2N+1)
            y[:,i] = g(sp[:,i])
        end
        
        # Compute moments of approximated distribution
        μ = y*Wm
        Σ = zeros(eltype(m), D,D)
        C = zeros(eltype(m), N,D)
        for i = 1:2N+1
            Σ += Wc[i]*(y[:,i] - μ)*(y[:,i] - μ)'
            C += Wc[i]*(sp[:,i] - m)*(y[:,i] - μ)'
        end
    end
    
    if Q !== nothing; Σ += Q; end
    return μ,Σ,C
end