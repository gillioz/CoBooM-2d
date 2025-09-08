using SpecialFunctions


gamma_inverse(x::Real) = x <= 0 && isinteger(x) ? 0.0 : 1.0 / gamma(x)

function compute_truncated_sums(hϕ::Real, h::Vector{<:Real}, n_max::Int, k_max::Int)
    twohϕ = 2 * hϕ
    # recursion in k
    a_k0 = 0.5 .* h .* (twohϕ - 1 .+ h) ./ (twohϕ .+ h)    # a_{1, 0}
    summands = [a_k0 .^ 0, copy(a_k0)]
    for k = 1:(k_max-1)
        a_k0 .*=
            (h .+ k) .^ 2 .* (twohϕ - 1 + k .+ h) ./ (k + 1) ./ (2 .* h .+ k) ./
            (twohϕ + k .+ h)    # a_{k+1, 0}
        push!(summands, copy(a_k0))
    end
    a_kn = reduce(hcat, summands)
    truncated_sums = Dict(0 => vec(sum(a_kn, dims = 2)))
    # recursion in n
    h_matrix = repeat(h, 1, k_max + 1)
    k_matrix = repeat(0:k_max, 1, length(h))'
    for n = 0:(n_max-1)
        a_kn .*=
            (twohϕ + n .+ h_matrix) .* (h_matrix .+ k_matrix .- (twohϕ + n)) ./
            (twohϕ + n .+ k_matrix .+ h_matrix)
        truncated_sums[n+1] = vec(sum(a_kn, dims = 2))
    end
    return truncated_sums
end


function compute_truncation_remainders(hϕ::Real, h::Vector{<:Real}, n_max::Int, k_max::Int)
    r =
        polygamma(1, k_max + 1) / 2 .* gamma.(1 .+ 2 .* h) ./ gamma.(1 .+ h) .*
        gamma_inverse.(h) .* (2*hϕ - 1 .+ h)
    remainders = Dict(0 => copy(r))
    for n = 0:(n_max-1)
        r .*= (2*hϕ + n .+ h)
        remainders[n+1] = copy(r)
    end
    return remainders
end


function compute_T(
    hϕ::Real,
    h::Vector{<:Real},
    n_max::Int;
    k_max::Int = 0,
    include_remainder::Bool = true,
    special_normalization::Bool = false,
)
    h_len = length(h)
    h_max = maximum(h)
    if k_max <= 0
        k_max = round(Int, 100 * h_max^2)
    end
    result = compute_truncated_sums(hϕ, h, n_max, k_max)
    if include_remainder
        remainders = compute_truncation_remainders(hϕ, h, n_max, k_max)
        for n = 0:n_max
            result[n] .+= remainders[n]
        end
    end
    c = gamma_inverse.(2 * hϕ .- h)
    if special_normalization
        c .*= gamma.(2*hϕ .+ h) .^ 2 ./ gamma.(1 .+ 2 .* h)
    end
    for n = 0:n_max
        result[n] .*= c ./ gamma.(2 * hϕ + n .+ h)
    end
    return result
end


function compute_S(
    hϕ::Real,
    h::Vector{<:Real},
    n_max::Int;
    k_max::Int = 0,
    include_remainder::Bool = true,
    special_normalization::Bool = false,
)
    S1 = Dict(
        0 =>
            special_normalization ?
            0.5 * gamma.(2*hϕ .+ h) ./ gamma.(1 .+ h) .* gamma_inverse.(h) :
            0.5 * gamma.(1 .+ 2*h) ./ gamma.(1 .+ h) .* gamma_inverse.(h) .*
            gamma_inverse.(2*hϕ .+ h),
    )
    for n = 0:(n_max-1)
        S1[n+1] = (2*hϕ + n .- h) .* (2*hϕ + n - 1 .+ h) ./ (2*hϕ + n .+ h) .* S1[n]
    end
    S2 = Dict(n_max => gamma_inverse.(2*hϕ + n_max + 1 .- h))
    for n = n_max:-1:1
        S2[n-1] = (2*hϕ + n .- h) .* S2[n]
    end
    return Dict(n => S1[n] .* S2[n] for n = 0:n_max)
end
