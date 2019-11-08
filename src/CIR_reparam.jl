#Reparametrisation using mean, scale, etc.
# First param: dXt = (δσ^2-2γXt)dt + 2σ√(Xt)dBt
# New param dXt = a(b-Xt)dt + σ'√(Xt)dBt (https://en.wikipedia.org/wiki/Cox%E2%80%93Ingersoll%E2%80%93Ross_model)

function reparam_CIR(δ, γ, σ)
    a = 2*γ
    b = δ*σ^2/(2*γ)
    σ_prime = 2*σ
    return a, b, σ_prime
end

function inverse_reparam_CIR(a, b, σ_prime)
    γ = a/2
    δ = 4*a*b/σ_prime^2
    σ = σ_prime/2
    return δ, γ, σ
end
