using SpecialFunctions


"""
    log_int_prod_2_Gammas(α_1, β_1, α_2, β_2)

Compute the log of ``\\int_{\\mathbb{R}^+} Gamma(α_1, β_1)Gammas(α_2, β_2)`` exactly. This formula is only valid for ``α_1 + α_2 > 1``.

# Examples
```julia-repl
julia> log_int_prod_2_Gammas(1, 1, 1, 1) #== -log(2)
-0.6931471805599453
julia> log_int_prod_2_Gammas(2, 0.5, 1, 0.5) #==  log(0.5^2*0.5 )
-0.6931471805599453
```
"""
function log_int_prod_2_Gammas(α_1, β_1, α_2, β_2)
    @assert α_1 + α_2 > 1 "The formula for the L2 distance is only valid for α_1 + α_2 > 1"
    return α_1 * log(β_1) + α_2 * log(β_2) + lgamma(α_1 + α_2 - 1) - lgamma(α_1) - lgamma(α_2) - (α_1 + α_2 - 1) * log(β_1 + β_2)
end
