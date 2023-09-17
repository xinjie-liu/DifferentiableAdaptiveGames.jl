"""
This file contains solving part of the MCP game solver code.
A more optimized implementation of this solver is available at: 
https://github.com/JuliaGameTheoreticPlanning/MCPTrajectoryGameSolver.jl
"""

function solve_mcp_game(
    mcp_game::MCPGame,
    x0,
    context_state;
    initial_guess = nothing,
    verbose = false,
)
    (; game, parametric_mcp, index_sets, horizon) = mcp_game
    (; dynamics) = game

    z = ChainRulesCore.ignore_derivatives() do
        #initial guess
        if !isnothing(initial_guess)
            z = initial_guess.variables
            verbose && @info "Warm-started with the previous solution."
        else
            x0_value = ForwardDiff.value.(x0)
            z = zeros(length(parametric_mcp.lower_bounds))
            control_block_dimensions =
                [control_dim(dynamics.subsystems[ii]) for ii in 1:num_players(game)]
            state_dimension = state_dim(dynamics)
            dummy_strategy =
                (x, t) ->
                    BlockVector(zeros(sum(control_block_dimensions)), control_block_dimensions)
            xs = rollout(dynamics, dummy_strategy, x0_value, horizon + 1).xs[2:end]
            xs = reduce(vcat, xs)
            z[1:(state_dimension * horizon)] = xs
            # z[1:(state_dimension * horizon)] = repeat(x0_value, horizon)
        end
        z
    end

    θ = [x0; context_state]

    variables, status, info = ParametricMCPs.solve(
        parametric_mcp,
        θ;
        initial_guess = z,
        verbose,
        cumulative_iteration_limit = 100_000,
        proximal_perturbation = 1e-2,
        use_basics = true,
        use_start = true,
        # convergence_tolerance = 1e-4,
    )

    primals = map(1:num_players(game)) do ii
        variables[index_sets.τ_idx_set[ii]]
    end

    (; primals, variables, status, info)
end

function TrajectoryGamesBase.solve_trajectory_game!(
    solver::MCPCoupledOptimizationSolver,
    game::TrajectoryGame{<:ProductDynamics},
    initial_state,
    strategy;
    verbose = false,
    solving_info = nothing
)
    problem = solver.mcp_game
    if !isnothing(strategy.last_solution) && strategy.last_solution.status == PATHSolver.MCP_Solved
        solution = solve_mcp_game(
            solver.mcp_game,
            initial_state,
            strategy.context_state;
            initial_guess = strategy.last_solution,
            verbose
        )
    else
        solution = solve_mcp_game(solver.mcp_game, initial_state, strategy.context_state; verbose)
    end
    if !isnothing(solving_info)
        push!(solving_info, solution.info)
    end
    # warm-start only when the last solution is valid
    if solution.status == PATHSolver.MCP_Solved
        strategy.last_solution = solution
    end
    strategy.solution_status = solution.status

    rng = Random.MersenneTwister(1)

    horizon = solver.mcp_game.horizon
    num_player = num_players(game)
    state_block_dimensions = [state_dim(game.dynamics.subsystems[ii]) for ii in 1:num_player]
    control_block_dimensions = [control_dim(game.dynamics.subsystems[ii]) for ii in 1:num_player]

    substrategies = let
        map(1:num_player) do ii
            xs = [
                [initial_state[Block(ii)]]
                collect.(
                    eachcol(
                        reshape(
                            solution.primals[ii][1:(horizon * state_block_dimensions[ii])],
                            state_block_dimensions[ii],
                            :,
                        ),
                    )
                )
            ]
            us =
                collect.(
                    eachcol(
                        reshape(
                            solution.primals[ii][(horizon * state_block_dimensions[ii] + 1):end],
                            control_block_dimensions[ii],
                            :,
                        ),
                    )
                )

            LiftedTrajectoryStrategy(ii, [(; xs, us)], [1], nothing, rng, Ref(0))
        end
    end
    JointStrategy(substrategies)
end
