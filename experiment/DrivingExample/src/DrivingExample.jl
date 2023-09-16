module DrivingExample

using MCPGameSolver: MCPGameSolver, MCPCoupledOptimizationSolver, WarmStartRecedingHorizonStrategy, NNParamsPredictor
using PATHSolver: PATHSolver
using TrajectoryGamesBase: TrajectoryGamesBase, TrajectoryGame, rollout, 
                GeneralSumCostStructure, ProductDynamics, TrajectoryGame, TrajectoryGameCost,
                state_dim, control_dim, num_players, get_constraints, state_bounds, control_bounds
using LiftedTrajectoryGames: LiftedTrajectoryStrategy
using TrajectoryGamesExamples: TrajectoryGamesExamples, animate_sim_steps, planar_double_integrator, UnicycleDynamics,
                BicycleDynamics, create_environment_axis
using DifferentiableTrajectoryOptimization: DifferentiableTrajectoryOptimization, ParametricTrajectoryOptimizationProblem, 
                MCPSolver, Optimizer, get_constraints_from_box_bounds
using ProgressMeter: ProgressMeter
using ParametricMCPs: ParametricMCPs
using BlockArrays: BlockArrays, Block, BlockVector, mortar, blocksizes, blocksize, blocks
using LazySets: LazySets
using Zygote: Zygote
using ForwardDiff: ForwardDiff
using Random: Random, shuffle!
using Distributions: Distributions, Uniform, MixtureModel
using Flux: Flux, gradient, Optimise.update!, params
using LinearAlgebra: LinearAlgebra, norm_sqr, norm
using Statistics: mean, std
using StatsBase: sem
using MCPGameSolver.ExampleProblems: n_player_collision_avoidance, two_player_guidance_game,
                two_player_guidance_game_with_collision_avoidance, shared_collision_avoidance_coupling_constraints
using Parameters
using GeometryBasics
using Plots: Plots, plot, scatter
using JLD2
using GLMakie: GLMakie
using Makie: Makie, Axis, lines!, band!, lines, band, Legend, violin!
using Colors: @colorant_str
using CSV: CSV
using HypothesisTests: HypothesisTests
# using CairoMakie

include("game/games.jl")
include("inverse_game/ramp_merging_inference.jl")
include("baseline/mpc.jl")
include("struct/infrastructure.jl")
include("struct/highway_roadway.jl")
include("struct/merging_roadway.jl")
include("struct/intersection_roadway.jl")
include("utils.jl")

end # module
