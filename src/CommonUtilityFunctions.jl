"Normalises a vector"
function normalise(x)
    return x/sum(x)
end

function log_binomial_safe_but_slow(n::Int64, k::Int64)
    assert(n>=0)
    assert(k>=0)
    assert(k<=n)
    if k == 0 || k == n
        return 0
    elseif k == 1 || k == n-1
        return log(n)
    else
        return sum(log(i) for i in (n-k+1):n) - sum(log(i) for i in 2:k)
    end
end

function descending_fact_no0(x::Real, n::Int64)
    return prod(x-i for i in 0:(n-1))
end

function descending_fact0(x::Real, n::Int64)
    if(n==0)
        return 1
    else
        return descending_fact_no0(x, n)
    end
end

function log_descending_fact_no0(x::Real, n::Int64)
    return sum(log(x-i) for i in 0:(n-1))
end

function log_descending_fact(x::Real, n::Int64)
    if(n==0)
        return 0
    else
        return log_descending_fact_no0(x, n)
    end
end
