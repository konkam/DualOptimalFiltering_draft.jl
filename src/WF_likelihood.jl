function logμπh_another_param(α::AbstractArray{T, 1}, m::AbstractArray{U, 1}, y::AbstractArray{U, 1}) where {T <: Real, U <: Integer}
    ## Needs to be written for multiple observations too
    sy = sum(y)
    sα = sum(α)
    return lfactorial(sy) - sum(lfactorial.(y)) + sum(log_pochammer.(α .+ m, y)) - log_pochammer(sα + sum(m), sy)
end
