module JuliaSteps

export AbstractStep, GaussStep, PoissonStep, AppendSteps!, plot
using DataStructures


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#    AbstractStep
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Step must have functions:
# - `ErrorSum(this, ξ₀, ξ₁)` that calculate error in data[ξ₀:ξ₁].
# - `AppendSteps!(this)` that conduct step detection.
#

abstract type AbstractStep end

function InitStep!(this::AbstractStep, data, p)
    this.data = Vector{Float64}(data)
    this.len = length(data)
    if 0 < p < 1
        this.penalty = log(p/(1-p))
    else
        this.penalty = -0.5log(this.len)
    end
    this.steplist = Vector{Int}([1, this.len + 1])
    this.fit = zeros(Float64, this.len)
    return this
end

function Finalize(this::AbstractStep)
    this.μlist = zeros(Float64, length(this.steplist)-1)

    for i in 1:length(this.steplist)-1
        ξ₀ = this.steplist[i]
        ξ₁ = this.steplist[i+1]
        μ = sum(@views(this.data[ξ₀:ξ₁-1]))/(ξ₁-ξ₀)
        this.fit[ξ₀:ξ₁-1] .= μ
        this.μlist[i] = μ
    end
    this.Δμlist = abs.(diff(this.μlist))
    return
end

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#    GaussStep
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

struct GaussMoment
    fw1::Vector{Float64}
    fw2::Vector{Float64}
    bw1::Vector{Float64}
    bw2::Vector{Float64}
    total1::Float64
    total2::Float64
    # empty struct
    GaussMoment() = new([], [], [], [], -1., -1.)
    # only forward
    GaussMoment(fw1::Vector{Float64}, fw2::Vector{Float64}, nothing, t1::Float64, t2::Float64) = new(fw1, fw2, Vector{Float64}(), Vector{Float64}(), t1, t2)
    # only backward
    GaussMoment(nothing, bw1::Vector{Float64}, bw2::Vector{Float64}, t1::Float64, t2::Float64) = new(Vector{Float64}(), Vector{Float64}(), bw1, bw2, t1, t2)
    function GaussMoment(data::Vector{Float64})
        fw1 = cumsum(data[1:end-1])
        fw2 = cumsum(data[1:end-1].^2)
        t1 = fw1[end] + data[end]
        t2 = fw2[end] + data[end]^2
        bw1 = t1 .- fw1
        bw2 = t2 .- fw2
        new(fw1, fw2, bw1, bw2, t1, t2)
    end
end

function GetOptimalSplitter(m::GaussMoment)
    N = length(m.fw1)
    χ²all = Getχ²(m)
    χ² = m.fw2 - m.fw1.^2 ./ vec(1:N) + m.bw2 - m.bw1.^2 ./ vec(N:-1:1)
    ξ = argmin(χ²)
    return χ²[ξ] - χ²all, ξ+1
end

Getχ²(m::GaussMoment) = m.total2 - m.total1^2/(length(m.fw1)+1)

function complement!(m::GaussMoment)
    if isempty(m.fw1)
        append!(m.fw1, m.total1 .- m.bw1)
        append!(m.fw2, m.total2 .- m.bw2)
    elseif isempty(m.bw1)
        append!(m.bw1, m.total1 .- m.fw1)
        append!(m.bw2, m.total2 .- m.fw2)
    end
end

function split(m::GaussMoment, i::Int)
    g₁ = GaussMoment(m.fw1[1:i-2], m.fw2[1:i-2], nothing, m.fw1[i-1], m.fw2[i-1])
    g₂ = GaussMoment(nothing, m.bw1[i:end], m.bw2[i:end], m.bw1[i-1], m.bw2[i-1])
    complement!(g₁)
    complement!(g₂)
    return g₁, g₂
end

mutable struct GaussStep <: AbstractStep
    data::Vector{Float64}
    len::Int
    penalty::Float64
    steplist::Vector{Int}
    fit::Vector{Float64}
    μlist::Vector{Float64}
    Δμlist::Vector{Float64}

    function GaussStep(data, p=-1.)
        this = new()
        InitStep!(this, data, p)
        return this
    end
end

function AppendSteps!(this::GaussStep)
    g = GaussMoment(this.data)
    χ² = Getχ²(g)
    heap = BinaryMinHeap{Tuple{Float64, Int, Int, GaussMoment}}()
    push!(heap, (GetOptimalSplitter(g)..., 1, g))

    while true
        Δχ², Δξ, ξ₀, g = pop!(heap)
        ΔlogL = this.penalty - this.len*log(1 + Δχ²/χ²) / 2
        if ΔlogL > 0
            ξ = ξ₀ + Δξ - 1
            g₁, g₂ = split(g, Δξ)
            length(g₁.fw1) > 1 && push!(heap, (GetOptimalSplitter(g₁)..., ξ₀, g₁))
            length(g₂.fw1) > 1 && push!(heap, (GetOptimalSplitter(g₂)..., ξ, g₂))
            push!(this.steplist, ξ)
            χ² += Δχ²
        else
            break
        end
    end
    sort!(this.steplist)
    Finalize(this)
    return
end


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#    PoissonStep
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

struct PoissonMoment
    fw1::Vector{Float64}
    bw1::Vector{Float64}
    total1::Float64
    # empty struct
    PoissonMoment() = new([], [], -1.)
    # only forward
    PoissonMoment(fw1::Vector{Float64}, nothing, t1::Float64) = new(fw1, Vector{Float64}(), t1)
    # only backward
    PoissonMoment(nothing, bw1::Vector{Float64}, t1::Float64) = new(Vector{Float64}(), bw1, t1)
    function PoissonMoment(data::Vector{Float64})
        fw1 = cumsum(data[1:end-1])
        t1 = fw1[end] + data[end]
        bw1 = t1 .- fw1
        new(fw1, bw1, t1)
    end
end

GetSlogμ(m::PoissonMoment) = m.total1 * log(m.total1 / (length(m.fw1)+1))

function GetOptimalSplitter(m::PoissonMoment)
    N = length(m.fw1)
    Slogμall = GetSlogμ(m)
    Slogμ = m.fw1 .* log.((m.fw1 .+ 1e-12) ./ vec(1:N)) + m.bw1 .* log.((m.bw1 .+ 1e-12) ./ vec(N:-1:1))
    ξ = argmax(Slogμ)
    return Slogμ[ξ] - Slogμall, ξ+1
end

function complement!(m::PoissonMoment)
    if isempty(m.fw1)
        append!(m.fw1, m.total1 .- m.bw1)
    elseif isempty(m.bw1)
        append!(m.bw1, m.total1 .- m.fw1)
    end
end

function split(m::PoissonMoment, i::Int)
    p₁ = PoissonMoment(m.fw1[1:i-2], nothing, m.fw1[i-1])
    p₂ = PoissonMoment(nothing, m.bw1[i:end], m.bw1[i-1])
    complement!(p₁)
    complement!(p₂)
    return p₁, p₂
end


mutable struct PoissonStep <: AbstractStep
    data::Vector{Float64}
    len::Int
    penalty::Float64
    steplist::Vector{Int}
    fit::Vector{Float64}
    μlist::Vector{Float64}
    Δμlist::Vector{Float64}

    function PoissonStep(data, p=-1)
        this = new()
        InitStep!(this, data, p)
        return this
    end

end

function AppendOneStep!(this::PoissonStep, m::PoissonMoment, start=1)
    if length(m.fw1) < 2
        return
    end
    Δslogμ, Δξ = GetOptimalSplitter(m)
    ΔlogL = this.penalty + Δslogμ
    if ΔlogL > 0
        ξ = start + Δξ - 1
        m1, m2 = split(m, Δξ)
        append!(this.steplist, ξ)
        AppendOneStep!(this, m1, start)
        AppendOneStep!(this, m2, ξ)
    end
    return 
end

function AppendSteps!(this::PoissonStep)
    m = PoissonMoment(this.data)
    AppendOneStep!(this, m)
    sort!(this.steplist)
    Finalize(this)
    return
end

end