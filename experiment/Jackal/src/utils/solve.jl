function backward_forward_solve!(mcp_problem, x0_inference, information_vector, system_state, initial_estimation, last_solution,
    receding_horizon_strategy_ego; index_set=nothing, vector_size=nothing, state_dimension=nothing, boundary_info=nothing)

    function interactive_inference_by_backprop(mcp_problem, initial_state, τs_observed,
        initial_estimation; max_grad_steps=150, lr=1e-3, last_solution)
        """
        back-propagation of the differentiable MCP solver
        """
        function likelihood_cost(τs_observed, goal_estimation, initial_state)
            solution = MCPGameSolver.solve_mcp_game(mcp_problem, initial_state,
                goal_estimation; initial_guess=last_solution)
            if solution.status != PATHSolver.MCP_Solved
                @info "Inner solve did not converge properly, re-initializing..."
                solution = MCPGameSolver.solve_mcp_game(mcp_problem, initial_state,
                    goal_estimation; initial_guess=nothing)
            end
            last_solution = solution.status == PATHSolver.MCP_Solved ? (; primals=ForwardDiff.value.(solution.primals),
                variables=ForwardDiff.value.(solution.variables), status=solution.status) : nothing
            τs_solution = solution.variables[index_set][1:(vector_size*state_dimension)]
            norm_sqr(τs_observed - τs_solution) + 0.04 * norm_sqr(goal_estimation)
        end
        goal_estimation = [clamp.(initial_estimation[1], boundary_info.x_min, boundary_info.x_max),
            clamp.(initial_estimation[2], boundary_info.y_min, boundary_info.y_max)]
        losses = []
        for i in 1:max_grad_steps
            # forward diff
            gradient = Zygote.gradient(τs_observed, goal_estimation, initial_state) do τs_observed, goal_estimation, initial_state
                Zygote.forwarddiff([goal_estimation; initial_state]; chunk_threshold=length(goal_estimation) + length(initial_state)) do θ
                    goal_estimation = BlockVector(θ[1:length(goal_estimation)], blocksizes(goal_estimation)[1])
                    initial_state = BlockVector(θ[(length(goal_estimation)+1):end], blocksizes(initial_state)[1])
                    likelihood_cost(τs_observed, goal_estimation, initial_state)
                end
            end

            objective_grad = gradient[2]
            x0_grad = gradient[3]
            clamp!(objective_grad, -50, 50)
            clamp!(x0_grad, -10, 10)
            objective_update = lr * objective_grad
            if norm(objective_update) < 1e-3
                @info "Inner iteration terminates at iteration: " * string(i)
                break
            end
            goal_estimation -= objective_update
            x0_update = 1e-3 * x0_grad
            if norm(x0_update) > 5e-4
                initial_state -= x0_update
            end
        end
        (; goal_estimation, last_solution)
    end

    index_set = mcp_problem.index_sets.τ_idx_set[1]
    time_exec = @elapsed goal_estimation, last_solution = @time interactive_inference_by_backprop(mcp_problem, x0_inference,
        information_vector, initial_estimation; max_grad_steps=10, lr=1.1e-2, last_solution=last_solution)

    receding_horizon_strategy_ego.context_state = goal_estimation

    strategy_ego = MCPGameSolver.solve_trajectory_game!(receding_horizon_strategy_ego.solver, mcp_problem.game, system_state,
        receding_horizon_strategy_ego; receding_horizon_strategy_ego.solve_kwargs...)

    if receding_horizon_strategy_ego.solution_status != PATHSolver.MCP_Solved
        @info "Ego solve failed, re-initializing..."
        receding_horizon_strategy_ego.last_solution = nothing
        receding_horizon_strategy_ego.solution_status = nothing
        strategy_ego = MCPGameSolver.solve_trajectory_game!(receding_horizon_strategy_ego.solver, mcp_problem.game, system_state,
            receding_horizon_strategy_ego; receding_horizon_strategy_ego.solve_kwargs...)
    end

    (; time_exec, goal_estimation, last_solution, strategy_ego)
end

function ground_truth_solve!(time_exec, goal, strategy)
    (; time_exec, goal_estimation=goal, last_solution=nothing, strategy_ego=strategy)
end
