using Test: @testset, @inferred, @test_broken, @test
using MCPGameSolver
# using NoNNExperiments
using TrajectoryGamesBase: TrajectoryGamesBase, TrajectoryGame, rollout
using TrajectoryGamesExamples: TrajectoryGamesExamples, PolygonEnvironment, animate_sim_steps
using ProgressMeter: ProgressMeter
using GLMakie: GLMakie
using BlockArrays: Block, BlockVector, mortar, blocksizes
using Zygote: Zygote
using FiniteDiff: FiniteDiff
using Random: Random

include("utils.jl")
using BlockArrays: Block, blocksize
using TrajectoryGamesBase:
    GeneralSumCostStructure, ProductDynamics, TrajectoryGame, TrajectoryGameCost
using TrajectoryGamesExamples: planar_double_integrator
using LinearAlgebra: norm_sqr
using Statistics: mean
using MCPGameSolver.ExampleProblems: n_player_collision_avoidance

# @testset "two-player tracking game without and with collision avoidance" begin
#     @test begin
#         NoNNExperiments.experiment_with_stranger()
#         true
#     end
    
#     @test begin
#         NoNNExperiments.experiment_guidance_with_collision_avoidance()
#         true
#     end
# end

@testset "Open-loop Nash Condition Check for Player1" begin
    local sim_steps
    # test for player 1
    @test begin
        environment = PolygonEnvironment(5, 10000)
        num_player = 2
        game = n_player_collision_avoidance(num_player; environment, min_distance = 1.5)
        #     dynamics = planar_double_integrator(;
        #     state_bounds = (; lb = [-Inf, -Inf, -100, -100], ub = [Inf, Inf, 100, 100]),
        #     control_bounds = (; lb = [-100, -100], ub = [100, 100]),
        # ))
        horizon = 20
        initial_state = mortar([[-2.9, 3.1, 0.1, -0.2], [2.8, -2.8, 0.0, 0.0]])
        goal = mortar([[3.0, -3.0], [-3.0, 3.0]])
        turn_length = 1
        solver = MCPGameSolver.MCPCoupledOptimizationSolver(game, horizon, blocksizes(goal, 1))

        n_sim_steps = 100
        sim_steps = let
            progress = ProgressMeter.Progress(n_sim_steps)
            receding_horizon_strategy = MCPGameSolver.WarmStartRecedingHorizonStrategy(;
                solver,
                game,
                turn_length,
                context_state = goal,
            )
            rollout(
                game.dynamics,
                receding_horizon_strategy,
                initial_state,
                n_sim_steps;
                get_info = (γ, x, t) -> γ.receding_horizon_strategy,
            )
        end

        total_test_results = map(1:length(sim_steps.xs)) do ii
            solution = sim_steps.infos[ii]
            initial_state = sim_steps.xs[ii]
            (; joint_xs, joint_us, calibrated_cost) =
                solution2joint_trajectory_cost(game, solution, initial_state, goal, horizon)

            test_results = nash_test(
                joint_us,
                num_player,
                game,
                initial_state,
                horizon,
                goal,
                calibrated_cost,
            )
            #println("number of non-decreasing cost after perturbation: ", sum(test_results), "/20")
            sum(test_results)
        end
        println("total non-decreasing cost after perturbation: ", sum(total_test_results), "/2000")
        sum(total_test_results) == horizon * n_sim_steps
    end
    # test for player 2
    @test begin
        environment = PolygonEnvironment(5, 10000)
        num_player = 2
        game = n_player_collision_avoidance(num_player; environment, min_distance = 1.5)
        #     dynamics = planar_double_integrator(;
        #     state_bounds = (; lb = [-Inf, -Inf, -100, -100], ub = [Inf, Inf, 100, 100]),
        #     control_bounds = (; lb = [-100, -100], ub = [100, 100]),
        # ))
        horizon = 20
        initial_state = mortar([[-2.9, 3.1, 0.1, -0.2], [2.8, -2.8, 0.0, 0.0]])
        goal = mortar([[3.0, -3.0], [-3.0, 3.0]])
        turn_length = 1
        solver = MCPGameSolver.MCPCoupledOptimizationSolver(game, horizon, blocksizes(goal, 1))

        n_sim_steps = 100

        total_test_results = map(1:length(sim_steps.xs)) do ii
            solution = sim_steps.infos[ii]
            initial_state = sim_steps.xs[ii]
            (; joint_xs, joint_us, calibrated_cost) =
                solution2joint_trajectory_cost(game, solution, initial_state, goal, horizon)

            test_results = nash_test_player2(
                joint_us,
                num_player,
                game,
                initial_state,
                horizon,
                goal,
                calibrated_cost,
            )
            #println("number of non-decreasing cost after perturbation: ", sum(test_results), "/20")
            sum(test_results)
        end
        println("total non-decreasing cost after perturbation: ", sum(total_test_results), "/2000")
        sum(total_test_results) == horizon * n_sim_steps
    end
end

function rand_goal(rng; radius = 1.0)
    (rand(rng, 2) .- 0.5) * 2 * radius
end

@testset "MCP gradient checks" begin
    rng = Random.MersenneTwister(1)
    environment = PolygonEnvironment(5, 5)
    game = n_player_collision_avoidance(2; environment)
    horizon = 20
    initial_state = mortar([[-1.0, 0.0, 0.1, -0.2], [1.0, 0.0, 0.0, 0.0]])
    goal_dataset = map(1:100) do _
        mortar([rand_goal(rng), rand_goal(rng)])
    end

    turn_length = 1

    mcp_game = MCPGameSolver.MCPGame(game, horizon, blocksizes(goal_dataset[1], 1))

    function dummy_pipeline(goal)
        solution =
            MCPGameSolver.solve_mcp_game(mcp_game, initial_state, goal;)
        sum(solution.variables .^ 2)
    end

    @testset "Descent direction" begin
        for goal in goal_dataset
            original_cost = dummy_pipeline(goal)
            gradient = only(Zygote.gradient(dummy_pipeline, goal))
            new_goal = goal - 1e-5 * gradient
            new_cost = dummy_pipeline(new_goal)
            @test new_cost <= original_cost
        end
    end

    @testset "Finite differencing" begin
        for goal in goal_dataset
            ∇_autodiff = only(Zygote.gradient(dummy_pipeline, goal))
            ∇_finite_diff = FiniteDiff.finite_difference_gradient(dummy_pipeline, goal)
            @test isapprox(∇_autodiff, ∇_finite_diff; rtol = 50)
        end
    end
end
