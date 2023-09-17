function ramp_merging_inference(; 
    number_trials = 1, solver = nothing, num_player = 4, ego_agent_id = 1, 
    ll = 4.0, lw = 0.24, mp = 1.2, θ  = pi / 12, # roadway parameters
    collision_radius = 0.08, max_velocity = 0.5, max_acceleration = 1.0, max_ϕ = π/4,
    collision_avoidance_coefficient = 400, hard_constraints = true, # collision avoidance inequalities
    rng = Random.MersenneTwister(1), horizon = 10, n_sim_steps = 120,
    vector_size = 10, # number of position points that the ego keeps as observation
    turn_length = 1, # number of steps to take along the MPGP horizon
    max_grad_steps = 10, # max online gradient steps for our method
)

    #=================================#
    # Construction of game objects
    #=================================#
    opponents_id = deleteat!(Vector(1:num_player), ego_agent_id)
    vertices, roadway = construct_roadway(ll, lw, mp, θ)
    environment = construct_env(num_player, ego_agent_id, vertices, roadway, collision_radius)
    game = construct_game(num_player; min_distance = 2 * collision_radius, hard_constraints, 
        collision_avoidance_coefficient, environment, max_velocity, max_acceleration, max_ϕ)
    # for peters2021rss, a game without hard collision avoidance constraints
    game_soft = construct_game(num_player; min_distance = 2 * collision_radius, hard_constraints = false, 
        collision_avoidance_coefficient, environment, max_velocity, max_acceleration, max_ϕ)    

    # sample players' initial states and goals
    initial_state_set, goal_dataset = highway_sampling(game, horizon, rng, num_player, ego_agent_id,
        collision_radius, number_trials; x0_range = 0.75, merging_scenario = true, vertices)
    initial_state = initial_state_set[1]
    system_state = initial_state

    state_dimension = state_dim(game.dynamics.subsystems[1])
    control_dimension = control_dim(game.dynamics.subsystems[1])
    ego_state_idx = let
        offset = ego_agent_id != 1 ? sum([blocksizes(initial_state)[1][ii] for ii in 1:(ego_agent_id - 1)]) : 0
        Vector((offset + 1):(offset + blocksizes(initial_state)[1][ego_agent_id]))
    end
    # solvers to compare, options: ground_truth, backprop (ours), inverseMCP (peters2021rss), mpc, 
    # heuristic_estimation (use initial states as goal estimation)
    solver_string_lst = ["backprop", "ground_truth", "inverseMCP", "mpc", "heuristic_estimation"]

    #====================================#
    # Initialization of different solvers
    #====================================#
    # 1. differentiable mcp
    solver = @something(solver, MCPCoupledOptimizationSolver(game, horizon, blocksizes(goal_dataset[1], 1))) # public solver for the uncontrolled agents
    mcp_game = solver.mcp_game
    observation_opponents_idx_set = construct_observation_index_set(;
        num_player, ego_agent_id, vector_size, state_dimension, mcp_game,
    )
    block_sizes_params = blocksizes(goal_dataset[1]) |> only

    if "inverseMCP" in solver_string_lst
        # 2. inverse mcp
        dim_params = length(goal_dataset[1]) - length(goal_dataset[1][Block(ego_agent_id)])
        inverse_problem = MCPGameSolver.InverseMCPProblem(game_soft, 
            horizon; observation_index = observation_opponents_idx_set, dim_params,
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

    #==========================#
    # Begin of experiment loop
    #==========================#
    for solver_string in solver_string_lst
        for trial in 1:length(goal_dataset)
            println("#########################\n New Iteration: ", trial, "/", length(goal_dataset), "\n#########################")
            goal = goal_dataset[trial]
            initial_state = initial_state_set[trial]
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
            visualization = visualize!(figure, game, system_state, strategy; targets = nothing, obstacle_radius = collision_radius,
                ego_agent_id, opponents_id)
            Makie.xlims!(visualization.environment_axis, -0.2, 4)
            Makie.ylims!(visualization.environment_axis, -1.5, 1.5) 
            display(figure)
            predicted_strategy_visualization = visualize_prediction(strategy, visualization, ego_agent_id)

            xs_observation = Array{Float64}[]
            xs_pre = BlockArrays.BlockVector{Float64}[] # keep track of initial states for each inverse game solving
            last_solution = nothing # for warm-starting
            erase_last_solution!(receding_horizon_strategy)
            erase_last_solution!(receding_horizon_strategy_ego)

            # Start of the simulation loop
            for t in 1:n_sim_steps
                # opponents' solve
                time_exec_opponents = @elapsed strategy = solve_game_with_resolve!(receding_horizon_strategy, game, system_state)
                #===========================================================#
                # player 2 infers player 1's objective and plans her motion
                if length(xs_observation) < vector_size
                    # use a dummy strategy at the first few steps when observation is not sufficient
                    dummy_substrategy, _ = create_dummy_strategy(game, system_state, 
                        control_dim(game.dynamics.subsystems[ego_agent_id]), horizon, ego_agent_id, rng)
                    strategy.substrategies[ego_agent_id] = dummy_substrategy
                    solving_status = PATHSolver.MCP_Solved
                else
                    information_vector = reduce(vcat, xs_observation)
                    if solver_string == "backprop"
                        #=================================# # our solver
                        # very first initialization (later use the previous estimation as warm start)
                        random_goal = mortar([system_state[Block(ii)][2:3] for ii in 1:num_player])
                        random_goal[Block(ego_agent_id)] = goal[Block(ego_agent_id)] # ego goal known

                        initial_estimation = !isnothing(goal_estimation) ? goal_estimation : random_goal
                        # solve inverse game
                        goal_estimation, last_solution, i_, info_, time_exec = interactive_inference_by_backprop(mcp_game, xs_pre[1],
                            information_vector, initial_estimation, goal[Block(ego_agent_id)]; max_grad_steps, lr = 2.1e-2, 
                            last_solution = last_solution, num_player, ego_agent_id, observation_opponents_idx_set, ego_state_idx,
                        )
                        receding_horizon_strategy_ego.context_state = goal_estimation
                        # solve forward game
                        time_forward = @elapsed strategy_ego = solve_game_with_resolve!(receding_horizon_strategy_ego, game, system_state)
                        time_exec += time_forward
                        println(time_exec, "s")
                        solving_status = check_solver_status!(
                            receding_horizon_strategy_ego, strategy, strategy_ego, game, system_state, 
                            ego_agent_id, horizon, max_acceleration, rng
                        )
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
                        receding_horizon_strategy_ego.context_state = goal_estimation
                        # solve forward game
                        time_forward = @elapsed strategy_ego = solve_game_with_resolve!(receding_horizon_strategy_ego, game, system_state)
                        time_exec += time_forward
                        println(time_exec, "s")
                        solving_status = check_solver_status!(
                            receding_horizon_strategy_ego, strategy, strategy_ego, game, system_state, 
                            ego_agent_id, horizon, max_acceleration, rng
                        )
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
                            # if mpc fails to solve, use emergency strategy
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
                        # solve forward game
                        time_forward = @elapsed strategy_ego = solve_game_with_resolve!(receding_horizon_strategy_ego, game, system_state)
                        time_exec += time_forward
                        println(time_exec, "s")
                        solving_status = check_solver_status!(
                            receding_horizon_strategy_ego, strategy, strategy_ego, game, system_state, 
                            ego_agent_id, horizon, max_acceleration, rng
                        )
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
                #===========================================================#
                if visualization.skip_button.clicked[]
                    visualization.skip_button.clicked[] = false
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
            @label end_of_episode
        end
    end
end
