function ramp_merging_inference(; number_trials = 1, solver = nothing)

    function interactive_inference_by_backprop(mcp_game, initial_state, τs_observed, 
        initial_estimation, ego_goal; max_grad_steps = 150, lr = 1e-3, last_solution = nothing)
        """
        back-propagation of the differentiable MCP solver
        """
        function likelihood_cost(τs_observed, goal_estimation, initial_state)
            solution = MCPGameSolver.solve_mcp_game(mcp_game, initial_state, 
                goal_estimation; initial_guess = last_solution)
            if solution.status != PATHSolver.MCP_Solved
                @info "Inner solve did not converge properly, re-initializing..."
                solution = MCPGameSolver.solve_mcp_game(mcp_game, initial_state, 
                    goal_estimation; initial_guess = nothing)
            end
            push!(solving_info, solution.info)
            last_solution = solution.status == PATHSolver.MCP_Solved ? (; primals = ForwardDiff.value.(solution.primals),
            variables = ForwardDiff.value.(solution.variables), status = solution.status) : nothing
            τs_solution = solution.variables[observation_opponents_idx_set]
            
            if solution.status == PATHSolver.MCP_Solved
                infeasible_counter = 0
            else
                infeasible_counter += 1
            end

            norm_sqr(τs_observed - τs_solution) #+ 0.01 * norm_sqr(goal_estimation)
        end
        infeasible_counter = 0
        solving_info = []
        goal_estimation = initial_estimation
        i_ = 0
        time_exec = 0
        for i in 1:max_grad_steps
            i_ = i
            # clip the estimation by the lower and upper bounds
            for ii in 1:num_player
                if ii != ego_agent_id
                    goal_estimation[Block(ii)] = clamp.(goal_estimation[Block(ii)], [-0.2, 0], [0.2, 1])
                end
            end
            goal_estimation[Block(ego_agent_id)] = ego_goal

            # REVERSE diff
            # original_cost = likelihood_cost(τs_observed, goal_estimation)
            # gradient = Zygote.gradient(likelihood_cost, τs_observed, goal_estimation)
            
            # FORWARD diff
            grad_step_time = @elapsed gradient = Zygote.gradient(τs_observed, goal_estimation, initial_state) do τs_observed, goal_estimation, initial_state
                Zygote.forwarddiff([goal_estimation; initial_state]; chunk_threshold = length(goal_estimation) + length(initial_state)) do θ
                    goal_estimation = BlockVector(θ[1:length(goal_estimation)], blocksizes(goal_estimation)[1])
                    initial_state = BlockVector(θ[(length(goal_estimation) + 1):end], blocksizes(initial_state)[1])
                    likelihood_cost(τs_observed, goal_estimation, initial_state)
                end
            end
            time_exec += grad_step_time
            objective_grad = gradient[2]
            x0_grad = gradient[3]
            x0_grad[ego_state_idx] .= 0 # cannot modify the ego state
            clamp!(objective_grad, -50, 50)
            clamp!(x0_grad, -10, 10)
            objective_update = lr * objective_grad
            x0_update = 1e-3 * x0_grad
            if norm(objective_update) < 1e-4 && norm(x0_update) < 1e-4
                @info "Inner iteration terminates at iteration: "*string(i)
                break
            elseif infeasible_counter >= 4
                @info "Inner iteration reached the maximal infeasible steps"
                break
            end
            goal_estimation -= objective_update
            initial_state -= x0_update
        end
        (; goal_estimation, last_solution, i_, solving_info, time_exec)
    end

    #=================================#
    # beginning of the experiment loop
    #=================================#

    num_player = 4
    ego_agent_id = 1
    opponents_id = deleteat!(Vector(1:num_player), ego_agent_id)

	ll = 4.0
	lw = 0.24
	mp = 1.2
	θ  = pi / 12
    
	x1  = [ ll,  0.]
	x2  = [ ll,  lw]
	x3  = [ 0.,  lw]
	x4  = [ 0.,  0.]
	x5  = [ 0., -lw]
	x6  = [ ll, -lw]

	x7  = [ mp,  0.]
	x8  = [ mp+lw*tan(θ), -lw]
	x9  = [ mp+lw/tan(θ),  lw]
	x10 = [ mp+lw*tan(θ)+lw/tan(θ), 0.]
	x11 = [ mp-mp*cos(θ), -mp*sin(θ)]
	x12 = [ mp+lw*tan(θ)-mp*cos(θ), -lw-mp*sin(θ)]

    vertices = Vector([x3, x2, x6, x8, x12, x11, x5])
    roadway_opts = MergingRoadwayOptions(lane_length = ll, lane_width = lw, merging_point = mp, merging_angle = θ)
    roadway = build_roadway(roadway_opts)
    collision_radius = 0.08
    player_lane_id = 3 * ones(Int, num_player)
    player_lane_id[ego_agent_id] = 4
    environment = RoadwayEnvironment(vertices, roadway, player_lane_id, collision_radius)
    
    max_velocity = 0.5
    max_acceleration = 1.0
    max_ϕ = π/4
    collision_avoidance_coefficient = 400 #0.12
    hard_constraints = true
    game = highway_game(num_player; min_distance = 2 * collision_radius, hard_constraints,
    collision_avoidance_coefficient,
    environment,
    dynamics = BicycleDynamics(; 
        l = 0.1,
        state_bounds = (; lb = [-Inf, -Inf, -max_velocity, -Inf], ub = [Inf, Inf, max_velocity, Inf]),
        control_bounds = (; lb = [-max_acceleration, -max_ϕ], ub = [max_acceleration, max_ϕ]),
        integration_scheme = :reverse_euler)
    )
    # for peters2021rss, a game without hard collision avoidance constraints
    game_soft = highway_game(num_player; min_distance = 2 * collision_radius, hard_constraints = false,
    collision_avoidance_coefficient,
    environment,
    dynamics = BicycleDynamics(;
        l = 0.1,
        state_bounds = (; lb = [-Inf, -Inf, -max_velocity, -Inf], ub = [Inf, Inf, max_velocity, Inf]),
        control_bounds = (; lb = [-max_acceleration, -max_ϕ], ub = [max_acceleration, max_ϕ]),
        integration_scheme = :reverse_euler)
    )
    rng = Random.MersenneTwister(1)
    horizon = 10
    n_sim_steps = 120

    initial_state_set, goal_dataset = highway_sampling(game, horizon, rng, num_player, ego_agent_id,
        collision_radius, number_trials; x0_range = 0.75, merging_scenario = true, vertices)

    initial_state = initial_state_set[1]
    system_state = initial_state

    vector_size = 10 # number of state-action pairs that the ego keeps as history information
    state_dimension = state_dim(game.dynamics.subsystems[1]) # assume each player has the same dimensions for simplification
    control_dimension = control_dim(game.dynamics.subsystems[1])
    turn_length = 1
    ego_state_idx = let
        offset = ego_agent_id != 1 ? sum([blocksizes(initial_state)[1][ii] for ii in 1:(ego_agent_id - 1)]) : 0
        Vector((offset + 1):(offset + blocksizes(initial_state)[1][ego_agent_id]))
    end
    solver_string_lst = ["backprop"] # solvers to compare, options: ground_truth, backprop (ours), inverseMCP (peters2021rss), mpc, heuristic_estimation (use initial states as goal estimation)

    #====================# # initialization of three solvers
    # 1. differentiable mcp
    solver = @something(solver, MCPCoupledOptimizationSolver(game, horizon, blocksizes(goal_dataset[1], 1))) # public solver for the uncontrolled agents
    mcp_game = solver.mcp_game

    observation_opponents_idx_set = mapreduce(vcat, 1:num_player) do ii
        if ii != ego_agent_id
            index = []
            for jj in 1:vector_size
                offset = state_dimension * (jj - 1)
                # partial observation
                index = vcat(index, mcp_game.index_sets.τ_idx_set[ii][[offset + 1, offset + 2, offset + 4]])
            end     
        else
            index = []
        end
        # # full observation
        # index = ii != ego_agent_id ? mcp_game.index_sets.τ_idx_set[ii][1:(vector_size * state_dimension)] : []
        index
    end
    sort!(observation_opponents_idx_set)
    block_sizes_params = blocksizes(goal_dataset[1]) |> only

    if "inverseMCP" in solver_string_lst
        # 2. inverse mcp
        dim_params = length(goal_dataset[1]) - length(goal_dataset[1][Block(ego_agent_id)])
        inverse_problem = MCPGameSolver.InverseMCPProblem(game_soft, 
            horizon; observation_index = observation_opponents_idx_set, 
            dim_params,
            params_processing_fn = create_params_processing_fn(block_sizes_params, ego_agent_id))
    end
    if "mpc" in solver_string_lst
        # 3. constant-velocity mpc
        opponent_block_sizes = let
            full_block_sizes = blocksizes(initial_state) |> only
            deleteat!(full_block_sizes, ego_agent_id)
        end
        mpc_baseline_optimizer = mpc_highway(game, horizon, ego_agent_id, 
            opponents_id, opponent_block_sizes; min_distance = hard_constraints ? collision_radius * 2 : nothing,
            collision_avoidance_coefficient)
    end

    #====================#
    # strategy of the ego agent
    receding_horizon_strategy_ego =
        WarmStartRecedingHorizonStrategy(; solver, game, turn_length, context_state = nothing)
    # a dummy strategy of constant-velocity rollout
    dummy_substrategy, _ = create_dummy_strategy(game, system_state,
        control_dim(game.dynamics.subsystems[ego_agent_id]), horizon, ego_agent_id, rng)

    for solver_string in solver_string_lst
        for trial in 1:length(goal_dataset)
            println("#########################\n New Iteration: ", trial, "/", length(goal_dataset),
                "\n#########################")
            goal = goal_dataset[trial]
            initial_state = initial_state_set[trial]           
            println("initial state: ", initial_state)
            println("goal: ", goal)
            system_state = initial_state

            goal_estimation = nothing
            # strategy of the opponet
            receding_horizon_strategy =
                WarmStartRecedingHorizonStrategy(; solver, game, turn_length, context_state = goal)
            # initial solve for plotting
            strategy = MCPGameSolver.solve_trajectory_game!(receding_horizon_strategy.solver, game, system_state, 
                receding_horizon_strategy; receding_horizon_strategy.solve_kwargs...)

            strategy.substrategies[ego_agent_id] = dummy_substrategy
            figure = Makie.Figure(resolution = (1200, 900))
            visualization = visualize!(
                figure,
                game,
                system_state,
                strategy;
                targets = nothing,
                obstacle_radius = collision_radius,
                ego_agent_id,
                opponents_id
            )
            Makie.xlims!(visualization.environment_axis, -0.2, 4)
            Makie.ylims!(visualization.environment_axis, -1.5, 1.5) 
            display(figure)
            predicted_strategy_visualization = visualize_prediction(strategy, visualization, ego_agent_id)

            xs_observation = Array{Float64}[]
            xs_pre = BlockArrays.BlockVector{Float64}[] # keep track of initial states for each inverse game solving
            last_solution = nothing # for warm-starting
            # clean up the last solution
            receding_horizon_strategy.last_solution = nothing
            receding_horizon_strategy.solution_status = nothing
            receding_horizon_strategy_ego.last_solution = nothing
            receding_horizon_strategy_ego.solution_status = nothing
            for t in 1:n_sim_steps
            """
            Start of the simulation loop
            """
            # Makie.record(figure, solver_string*"sim_steps.mp4", 1:n_sim_steps; framerate = 15) do t
                # strategy for opponents
                record_data = true
                time_exec_opponents = @elapsed strategy = MCPGameSolver.solve_trajectory_game!(receding_horizon_strategy.solver, 
                    game, system_state, receding_horizon_strategy; receding_horizon_strategy.solve_kwargs...)
                if receding_horizon_strategy.solution_status != PATHSolver.MCP_Solved
                    if solver_string == "ground_truth"
                        record_data = false
                        @info "Skipping the episode that ground truth fails..."
                        @goto end_of_episode
                    end
                    @info "Opponent solve failed, re-initializing..."
                    receding_horizon_strategy.last_solution = nothing
                    receding_horizon_strategy.solution_status = nothing
                    time_exec_opponents = @elapsed strategy = MCPGameSolver.solve_trajectory_game!(receding_horizon_strategy.solver, 
                        game, system_state, receding_horizon_strategy; receding_horizon_strategy.solve_kwargs...)
                end
                #===========================================================#
                # player 2 infers player 1's objective and plan its motion
                if length(xs_observation) < vector_size
                    dummy_substrategy, _ = create_dummy_strategy(game, system_state, 
                        control_dim(game.dynamics.subsystems[ego_agent_id]), horizon, ego_agent_id, rng)
                    strategy.substrategies[ego_agent_id] = dummy_substrategy
                    solving_status = PATHSolver.MCP_Solved
                else
                    information_vector = reduce(vcat, xs_observation)
                    if solver_string == "backprop"
                        #=================================# # backprop solver
                        random_goal = mortar([system_state[Block(ii)][2:3] for ii in 1:num_player])
                        random_goal[Block(ego_agent_id)] = goal[Block(ego_agent_id)] # ego goal known

                        initial_estimation = !isnothing(goal_estimation) ? goal_estimation : random_goal
                        goal_estimation, last_solution, i_, info_, time_exec = interactive_inference_by_backprop(mcp_game, xs_pre[1],
                            information_vector, initial_estimation, goal[Block(ego_agent_id)]; max_grad_steps = 30, lr = 2.1e-2, 
                            last_solution = last_solution)
                        receding_horizon_strategy_ego.context_state = goal_estimation

                        time_forward = @elapsed strategy_ego = MCPGameSolver.solve_trajectory_game!(receding_horizon_strategy_ego.solver, game, system_state, 
                            receding_horizon_strategy_ego; receding_horizon_strategy_ego.solve_kwargs...)
                        if receding_horizon_strategy_ego.solution_status != PATHSolver.MCP_Solved
                            @info "Ego solve failed, re-initializing..."
                            receding_horizon_strategy_ego.last_solution = nothing
                            receding_horizon_strategy_ego.solution_status = nothing
                            time_forward = @elapsed strategy_ego = MCPGameSolver.solve_trajectory_game!(receding_horizon_strategy_ego.solver, game, system_state, 
                                receding_horizon_strategy_ego; receding_horizon_strategy_ego.solve_kwargs...)
                        end
                        time_exec += time_forward
                        println(time_exec)
                        solving_status = receding_horizon_strategy_ego.solution_status
                        if solving_status == PATHSolver.MCP_Solved
                            strategy.substrategies[ego_agent_id] = strategy_ego.substrategies[ego_agent_id]
                        else
                            dummy_substrategy, _ = create_dummy_strategy(game, system_state, 
                                control_dim(game.dynamics.subsystems[ego_agent_id]), horizon, ego_agent_id, rng;
                                max_acceleration = max_acceleration, strategy_type = "max_acceleration")
                            strategy.substrategies[ego_agent_id] = dummy_substrategy
                        end
                        predicted_opponents_trajectory = strategy_ego.substrategies[opponents_id]
                        #=================================#
                    elseif solver_string == "inverseMCP"
                        #==================================# #inverse MCP solver (peters rss 2021)
                        time_exec = @elapsed @time solution = MCPGameSolver.solve_inverse_mcp_game(inverse_problem, game, information_vector, 
                                xs_pre[1]; observation_index = observation_opponents_idx_set, horizon, dim_params, initial_guess = last_solution,
                                prior_parmas = goal[Block(ego_agent_id)])

                        if solution.status != PATHSolver.MCP_Solved
                            @info "Inverse kkt solve did not converge properly, re-initializing..."
                            re_time_exec = @elapsed solution =
                            MCPGameSolver.solve_inverse_mcp_game(inverse_problem, game, information_vector, 
                                xs_pre[1]; observation_index = observation_opponents_idx_set, horizon, dim_params, initial_guess = nothing,
                                prior_parmas = goal[Block(ego_agent_id)])
                            time_exec += re_time_exec 
                        end
                        last_solution = solution.status == PATHSolver.MCP_Solved ? solution : nothing
                        goal_estimation = reproduce_goal_estimation(goal[Block(ego_agent_id)], block_sizes_params, ego_agent_id, 
                            solution.variables[1:dim_params])
                        # goal_estimation = clamp.(goal_estimation, repeat([-lw, -max_velocity], num_player), repeat([lw, max_velocity], num_player))

                        receding_horizon_strategy_ego.context_state = goal_estimation
                        # Main.Infiltrator.@infiltrate
                        time_forward = @elapsed strategy_ego = MCPGameSolver.solve_trajectory_game!(receding_horizon_strategy_ego.solver, game, system_state, 
                            receding_horizon_strategy_ego; receding_horizon_strategy_ego.solve_kwargs...)
                        if receding_horizon_strategy_ego.solution_status != PATHSolver.MCP_Solved
                            @info "Ego solve failed, re-initializing..."
                            receding_horizon_strategy_ego.last_solution = nothing
                            receding_horizon_strategy_ego.solution_status = nothing
                            time_forward = @elapsed strategy_ego = MCPGameSolver.solve_trajectory_game!(receding_horizon_strategy_ego.solver, game, system_state, 
                                receding_horizon_strategy_ego; receding_horizon_strategy_ego.solve_kwargs...)
                        end
                        solving_status = receding_horizon_strategy_ego.solution_status
                        time_exec += time_forward
                        if solving_status == PATHSolver.MCP_Solved
                            strategy.substrategies[ego_agent_id] = strategy_ego.substrategies[ego_agent_id]
                        else
                            dummy_substrategy, _ = create_dummy_strategy(game, system_state, 
                                control_dim(game.dynamics.subsystems[ego_agent_id]), horizon, ego_agent_id, rng;
                                max_acceleration = max_acceleration, strategy_type = "max_acceleration")
                            strategy.substrategies[ego_agent_id] = dummy_substrategy
                        end
                        predicted_opponents_trajectory = strategy_ego.substrategies[opponents_id]
                        #=================================#                                        
                    elseif solver_string == "mpc"
                        #==================================# # constant-velocity mpc
                        opponents_x0 = let
                            copied_state = deepcopy(system_state)
                            mortar([copied_state[Block(ii)] for ii in 1:num_player if ii != ego_agent_id])
                        end
                        mpc_params = vcat(goal[Block(ego_agent_id)], opponents_x0)
                        time_exec = @elapsed mpc_sol = @time mpc_baseline_optimizer(system_state[Block(ego_agent_id)]
                            , mpc_params; initial_guess = last_solution)
                        solving_status = mpc_sol.info.status
                        if solving_status != PATHSolver.MCP_Solved
                            @info "MPC not solved, re-solving..."
                            time_exec_re = @elapsed mpc_sol = mpc_baseline_optimizer(system_state[Block(ego_agent_id)],
                                mpc_params; initial_guess = nothing)
                            time_exec += time_exec_re
                            solving_status = mpc_sol.info.status
                        end
                        if solving_status == PATHSolver.MCP_Solved
                            last_solution = mpc_sol.info.raw_solution
                            strategy.substrategies[ego_agent_id] = LiftedTrajectoryStrategy(ego_agent_id, [(; mpc_sol.xs, mpc_sol.us)], [1], nothing, rng, Ref(0))
                        else
                            # if mpc fails to solve, use zero control input
                            last_solution = nothing
                            dummy_substrategy, _ = create_dummy_strategy(game, system_state, 
                                control_dim(game.dynamics.subsystems[ego_agent_id]), horizon, ego_agent_id, rng;
                                max_acceleration = max_acceleration, strategy_type = "max_acceleration")
                            strategy.substrategies[ego_agent_id] = dummy_substrategy
                        end
                        predicted_opponents_trajectory = constant_velocity_rollout(game, horizon, opponents_id, opponents_x0, rng)
                        goal_estimation = let
                            opponents_goal = mapreduce(vcat, 1:length(opponents_id)) do ii
                                predicted_opponents_trajectory[ii].trajectories[1].xs[end][2:3]
                            end
                            reproduce_goal_estimation(goal[Block(ego_agent_id)], block_sizes_params, ego_agent_id, opponents_goal)
                        end
                        #==================================#
                    elseif solver_string == "heuristic_estimation"
                        # use the current opponents' state as their desired state
                        time_exec = @elapsed goal_estimation = mortar([initial_state[Block(ii)][2:3] for ii in 1:num_player])
                        goal_estimation[Block(ego_agent_id)] = goal[Block(ego_agent_id)]

                        receding_horizon_strategy_ego.context_state = goal_estimation
                        time_forward = @elapsed strategy_ego = MCPGameSolver.solve_trajectory_game!(receding_horizon_strategy_ego.solver, game, system_state, 
                            receding_horizon_strategy_ego; receding_horizon_strategy_ego.solve_kwargs...)
                        if receding_horizon_strategy_ego.solution_status != PATHSolver.MCP_Solved
                            @info "Ego solve failed, re-initializing..."
                            receding_horizon_strategy_ego.last_solution = nothing
                            receding_horizon_strategy_ego.solution_status = nothing
                            time_forward = @elapsed strategy_ego = MCPGameSolver.solve_trajectory_game!(receding_horizon_strategy_ego.solver, game, system_state, 
                                receding_horizon_strategy_ego; receding_horizon_strategy_ego.solve_kwargs...)
                        end
                        time_exec += time_forward
                        solving_status = receding_horizon_strategy_ego.solution_status
                        if solving_status == PATHSolver.MCP_Solved
                            strategy.substrategies[ego_agent_id] = strategy_ego.substrategies[ego_agent_id]
                        else
                            dummy_substrategy, _ = create_dummy_strategy(game, system_state, 
                                control_dim(game.dynamics.subsystems[ego_agent_id]), horizon, ego_agent_id, rng;
                                max_acceleration = max_acceleration, strategy_type = "max_acceleration")
                            strategy.substrategies[ego_agent_id] = dummy_substrategy
                        end
                        predicted_opponents_trajectory = strategy_ego.substrategies[opponents_id]                        
                    elseif solver_string == "ground_truth"
                        #=================================# # ground truth (game-theoretic interaction in a centralized fashion)
                        time_exec = time_exec_opponents
                        goal_estimation = goal
                        predicted_opponents_trajectory = strategy.substrategies[opponents_id]
                        #=================================#
                    else
                        error("Not a valid solver name!") 
                    end
                end
                solving_status = solver_string == "ground_truth" ? receding_horizon_strategy.solution_status : solving_status
                if solving_status != PATHSolver.MCP_Solved
                    infeasible_step_counter += 1
                    solver_failure += 1
                else
                    infeasible_step_counter = 0
                end
                #===========================================================#
                if visualization.skip_button.clicked[]
                    visualization.skip_button.clicked[] = false
                    record_data = false
                    @info "Manually skipping the episode..."
                    @goto end_of_episode
                end
                while visualization.pause_button.clicked[]
                    sleep(0.1)
                    if visualization.continue_button.clicked[]
                        visualization.pause_button.clicked[] = false
                        visualization.continue_button.clicked[] = false
                    end
                end

                # visualize what the ego thinks the opponent will do
                let
                    if length(xs_observation) < vector_size
                        strategy_to_be_visualized = strategy.substrategies[opponents_id]
                    else
                        strategy_to_be_visualized = predicted_opponents_trajectory
                    end
                    map(1:length(predicted_strategy_visualization)) do ii
                        predicted_strategy_visualization[ii][] = strategy_to_be_visualized[ii]
                    end
                end

                # update state
                pointmasses_trajectory, control_sequence, _ =
                    rollout(game.dynamics, strategy, system_state, horizon)
                system_state = pointmasses_trajectory[turn_length + 1]
                previous_state = pointmasses_trajectory[turn_length]

                min_dis, collision_single_step = collision_detection(system_state, ego_agent_id, opponents_id, 2 * collision_radius - 0.0025)
                if min_dis < 2 * collision_radius - 0.0025
                    @info "collision with distance " * string(min_dis)
                end

                # compute information vector
                push!(xs_observation, reduce(vcat, [system_state[Block(ii)][[1, 2, 4]] for ii in 1:num_player if ii != ego_agent_id])) # partial observation
                # push!(xs_observation, reduce(vcat, [system_state[Block(ii)] for ii in 1:num_player if ii != ego_agent_id])) # full observation
                estimated_state = previous_state # compute_state_estimation(previous_state, system_state, num_player)
                push!(xs_pre, estimated_state)
                if length(xs_observation) > vector_size
                    popfirst!(xs_observation)
                    popfirst!(xs_pre)
                end

                # visualization
                visualization.strategy[] = strategy
                # visualization.targets[] = goal
                for (x, _) in zip(pointmasses_trajectory, 1)
                    visualization.pointmasses[] = x
                end
                sleep(0.01)
            end
            if visualization.stop_button.clicked[]
                visualization.stop_button.clicked[] = false
                break
            end
            @label end_of_episode
        end
    end
end
