# This file contains code from:
# Copyright 2021 Simon Le Cleac'h
# Repository link: https://github.com/simon-lc/AlgamesDriving.jl

################################################################################
# VehicleState
################################################################################

mutable struct VehicleState{T}
    x::T # x-position
    y::T # y-position
    θ::T # heading
    v::T # velocity
end

function VehicleState()
    x = 0.0
    y = 0.0
    θ = 0.0
    v = 0.0
    return VehicleState{typeof(x)}(x,y,θ,v)
end

################################################################################
# Starting Area
################################################################################

mutable struct StartingArea{T}
    # This is a rectangular starting area on state space
    x_nom::VehicleState{T} # Nominal state
    x_min::VehicleState{T} # Bound on the starting state
    x_max::VehicleState{T} # Bound on the starting state
end

function StartingArea()
    return StartingArea(VehicleState(), VehicleState(), VehicleState())
end

function StartingArea(x_nom::VehicleState{T}) where {T}
    return StartingArea{T}(copy(x_nom), copy(x_nom), copy(x_nom))
end

function StartingArea(x_min::VehicleState{T}, x_max::VehicleState{T}) where {T}
    return StartingArea{T}(0.5*(x_min+x_max), x_min, x_max)
end

function randstate(start::StartingArea{T}) where {T}
    x_min = start.x_min
    x_max = start.x_max
    return VehicleState{T}(
        x_min.x + rand(T)*(x_max.x - x_min.x),
        x_min.y + rand(T)*(x_max.y - x_min.y),
        x_min.θ + rand(T)*(x_max.θ - x_min.θ),
        x_min.v + rand(T)*(x_max.v - x_min.v),
        )
end

################################################################################
# Wall
################################################################################

abstract type AbstractWall end

mutable struct Wall <: AbstractWall
	p1::AbstractVector # initial point of the boundary
	p2::AbstractVector # final point of the boundary
	v::AbstractVector  # vector orthogonal to (p2 - p1) and indicating the forbiden halfspace
end

################################################################################
# CircularWall
################################################################################

mutable struct CircularWall{T}
    x::T
    y::T
    r::T
end

function CircularWall()
    x = 0.0
    y = 0.0
    r = 0.0
    return CircularWall{typeof(x)}(x,y,r)
end


################################################################################
# Lane
################################################################################

mutable struct Lane{T}
    id::Int
    name::Symbol
    wall::Vector{Wall}
    circle::Vector{CircularWall{T}}
    start::StartingArea{T}
    centerline::Function
end

function Lane()
    name = :lane_0
    w = Wall([0., 0.], [0., 0.], [0., 0.])
    c = CircularWall()
    start = StartingArea()
    return Lane{typeof(c.x)}(0, name, [w], [c], start, x -> x)
end

function Lane(id::Int, name::Symbol, wall::Vector{Wall}, circle::Vector{CircularWall}, start::StartingArea{T}, centerline) where {T}
    return Lane{T}(id, name, wall, circle, start, centerline)
end

################################################################################
# Roadway
################################################################################

abstract type RoadwayOptions
end

mutable struct Roadway{T}
    l::Int
    lane::Vector{Lane{T}}
	opts::RoadwayOptions
end

function Roadway(lanes::Vector{Lane{T}}, opts::RoadwayOptions) where {T}
    l = length(lanes)
    sort!(lanes, by = x -> x.id)
    if [lanes[i].id for i=1:l] != (1:l)
        # @show "Reindexing the lanes"
        for i = 1:l
            lanes[i].id = i
        end
    end
    return Roadway{T}(l,lanes,opts)
end



