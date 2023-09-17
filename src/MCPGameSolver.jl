module MCPGameSolver

using DifferentiableTrajectoryOptimization:
    ParametricTrajectoryOptimizationProblem, get_constraints_from_box_bounds, _coo_from_sparse!
using TrajectoryGamesExamples:
    TrajectoryGamesExamples,
    PolygonEnvironment,
    two_player_meta_tag,
    animate_sim_steps,
    planar_double_integrator,
    UnicycleDynamics,
    create_environment_axis
    # BicycleDynamics
using TrajectoryGamesBase:
    TrajectoryGamesBase,
    TrajectoryGame,
    get_constraints,
    num_players,
    state_dim,
    control_dim,
    horizon,
    state_bounds,
    control_bounds,
    solve_trajectory_game!,
    JointStrategy,
    RecedingHorizonStrategy,
    rollout,
    ProductDynamics
using LiftedTrajectoryGames: LiftedTrajectoryStrategy
using BlockArrays: Block, BlockVector, mortar, blocksizes
using SparseArrays: sparse, blockdiag, findnz, spzeros
using PATHSolver: PATHSolver
using LinearAlgebra: I, norm_sqr, pinv, ColumnNorm, qr, norm
using Random: Random
using ProgressMeter: ProgressMeter
using GLMakie: GLMakie
using Symbolics: Symbolics, @variables, scalarize
using ChainRulesCore: ChainRulesCore
using ForwardDiff: ForwardDiff
using Zygote: Zygote
using Flux: Flux, glorot_uniform, Dense, Optimise.Adam, NNlib.relu, Chain
using ParametricMCPs: ParametricMCPs
using Makie: Makie
using Colors: @colorant_str
using JLD2: JLD2

include("utils/ExampleProblems.jl")
using .ExampleProblems: n_player_collision_avoidance, two_player_guidance_game

include("utils/utils.jl")
include("problem_formulation.jl")
include("solve.jl")
include("baseline/inverse_MCP_solver.jl")

function main(; 
    initial_state = mortar([
        [-1, 2.5, 0.1, -0.2],
        [1, 2.8, 0.0, 0.0],
        [-2.8, 1, 0.2, 0.1],
        [-2.8, -1, -0.27, 0.1],
    ]),
    goal = mortar([[0.0, -2.7], [2, -2.8], [2.7, 1], [2.7, -1.1]])
)
    """
    An example of the MCP game solver
    """
    environment = PolygonEnvironment(6, 8)
    game = n_player_collision_avoidance(4; environment, min_distance = 1.2)
    horizon = 20

    turn_length = 2
    solver = MCPCoupledOptimizationSolver(game, horizon, blocksizes(goal, 1))
    sim_steps = let
        n_sim_steps = 150
        progress = ProgressMeter.Progress(n_sim_steps)
        receding_horizon_strategy =
            WarmStartRecedingHorizonStrategy(; solver, game, turn_length, context_state = goal)
        rollout(
            game.dynamics,
            receding_horizon_strategy,
            initial_state,
            n_sim_steps;
            get_info = (γ, x, t) ->
                (ProgressMeter.next!(progress); γ.receding_horizon_strategy),
        )
    end
    animate_sim_steps(game, sim_steps; live = false, framerate = 20, show_turn = true)
    (; sim_steps, game)
end

end
