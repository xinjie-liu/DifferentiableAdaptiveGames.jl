function sample_target(environment; rng)
    rand(rng, [[-2.0, 1.0], [-2.0, -1.0], [2.0, -1.0], [2.0, 1.0]])
    #LazySets.sample(environment.set; rng)
end

function run(;
    optitrack_communication,
    robot_communications,
    n_players,
    figure=Makie.Figure(),
    #target_reached_tolerance = 0.3,
    rng=Random.MersenneTwister(1),
    solver_string = "ground_truth"
)
    @assert length(robot_communications) == n_players

    game, avoidance_distance = create_game(; avoidance_distance=0.8,
        hard_constraints = true, collision_avoidance_coefficient=50.0, max_velocity_component=1.0)
    state_dimension = state_dim(game.dynamics.subsystems[1])
    control_dimension = control_dim(game.dynamics.subsystems[1])
    horizon = 12
    turn_length = 1
    vector_size = 12
    ego_agent_id = 2
    opponent_id = 1
    episode_info = EpisodeInfo()
    vertices = reduce(hcat, game.env.set.vertices)
    boundary_info = (; x_min=minimum(vertices[1, :]), x_max=maximum(vertices[1, :]),
        y_min=minimum(vertices[2, :]), y_max=maximum(vertices[2, :]))

    # x, y, θ
    robots = robot_state_from_optitrack!(optitrack_communication)
    # TODO: remove this in hardware experiment
    robots[5] += 1

    # x, y, dx, dy
    pointmasses = mortar([[robot[1], robot[2], 0.0, 0.0] for robot in blocks(robots)])
    # x, y
    targets = mortar([sample_target(game.env; rng)])

    solver = MCPCoupledOptimizationSolver(game, horizon, blocksizes(targets, 1))
    mcp_problem = MCPGameSolver.MCPGame(game, horizon, blocksizes(targets, 1))
    # opponent strategy
    receding_horizon_strategy =
        WarmStartRecedingHorizonStrategy(; solver, game, turn_length, context_state=targets)
    # ego strategy
    receding_horizon_strategy_ego =
        WarmStartRecedingHorizonStrategy(; solver, game, turn_length, context_state=nothing)

    # Initial solve
    strategy = MCPGameSolver.solve_trajectory_game!(receding_horizon_strategy.solver, game, pointmasses,
        receding_horizon_strategy; receding_horizon_strategy.solve_kwargs...)
    dummy_substrategy, _ = create_dummy_strategy(game, pointmasses,
        control_dim(game.dynamics.subsystems[2]), horizon, 2, rng)
    strategy.substrategies[2] = dummy_substrategy
    pointmasses_trajectory, control_sequence, _ =
        rollout(game.dynamics, strategy, pointmasses, horizon)
    robots_trajectory = rollout_robot_dynamics(;
        initial_state=robots,
        pointmasses_trajectory,
        rollout_horizon=horizon
    )

    # Initial plot to makie
    visualization = visualize_game!(
        figure,
        game.env,
        pointmasses,
        strategy,
        targets,
        obstacle_radius=0.5 * avoidance_distance,
    )
    figure[1, 1] = visualization.axis
    goal_estimation_ = Observable(targets)
    visualize_inferred_goal!(visualization.axis, goal_estimation_)
    visualized_tracker = visualize_tracker!(visualization.axis, robots, robots_trajectory.xs)
    stop_button = visualize_button!(figure, "Stop")
    spawn_button = visualize_button!(figure, "Spawn")
    button_grid = GridLayout(tellwidth=false)
    button_grid[1, 1] = stop_button.button
    button_grid[1, 2] = spawn_button.button
    figure[2, 1] = button_grid
    display(figure)
    # also visualize what the ego thinks the opponent will do
    strategy_ = Makie.Observable(strategy.substrategies[opponent_id])

    @info "Starting solve loop"
    while !stop_button.clicked[]

        if episode_info.t == 1
            episode_info.goal_estimation = nothing
            episode_info.last_solution = nothing
            episode_info.history_trajectory = []
            receding_horizon_strategy.last_solution = nothing
            receding_horizon_strategy.solution_status = nothing
            receding_horizon_strategy_ego.last_solution = nothing
            receding_horizon_strategy_ego.solution_status = nothing
        end
        if episode_info.t == vector_size + 2
            TrajectoryGamesBase.visualize!(visualization.axis, strategy_)
        end

        sleep(0.01)

        #============== hardware experiment =================#
        # robots = robot_state_from_optitrack!(optitrack_communication)
        # # advance the pointmass consistent with its plan
        # pointmasses = find_closest_states_on_plan(robots, pointmasses_trajectory)
        #====================================================#

        # TODO: comment this out in hardware experiment
        #============== simulation test =================#
        robots = robots_trajectory.xs[turn_length+1]
        pointmasses = find_closest_states_on_plan(robots, pointmasses_trajectory)
        # pointmasses = pointmasses_trajectory[turn_length + 1]
        #================================================#

        # record the trajectory for goal inference
        load_history_information_vector!(pointmasses, episode_info.history_trajectory, vector_size)

        # ================== opponent solving ===================== #
        time_exec_opponent = @elapsed strategy = MCPGameSolver.solve_trajectory_game!(receding_horizon_strategy.solver, game, pointmasses,
            receding_horizon_strategy; receding_horizon_strategy.solve_kwargs...)
        if receding_horizon_strategy.solution_status != PATHSolver.MCP_Solved
            @info "Opponent solve failed, re-initializing..."
            receding_horizon_strategy.last_solution = nothing
            receding_horizon_strategy.solution_status = nothing
            time_exec_re = @elapsed strategy = MCPGameSolver.solve_trajectory_game!(receding_horizon_strategy.solver, game, pointmasses,
                receding_horizon_strategy; receding_horizon_strategy.solve_kwargs...)
            time_exec_opponent += time_exec_re
        end
        # ========================================================= #
        # ==================== ego solving ======================== #
        if length(episode_info.history_trajectory) < (vector_size + 1)
            # a dummy strategy while filling in the observation vector
            dummy_substrategy, _ = create_dummy_strategy(game, pointmasses,
                control_dim(game.dynamics.subsystems[2]), horizon, 2, rng)
            strategy.substrategies[2] = dummy_substrategy
        else
            information_vector = reduce(vcat, [state[Block(1)] for state in episode_info.history_trajectory[2:end]])
            x0_inference = episode_info.history_trajectory[1]
            if solver_string == "backprop"
                initial_estimation = !isnothing(episode_info.goal_estimation) ? episode_info.goal_estimation : pointmasses[Block(1)][1:2] + (rand(rng, 2) .- 0.5)
                solution_tuple = backward_forward_solve!(mcp_problem, x0_inference, information_vector, pointmasses, initial_estimation, episode_info.last_solution,
                    receding_horizon_strategy_ego; vector_size, state_dimension, boundary_info)
            elseif solver_string == "ground_truth"
                solution_tuple = ground_truth_solve!(time_exec_opponent, targets[1:end], strategy)
            end
            episode_info.last_solution = solution_tuple.last_solution
            episode_info.goal_estimation = solution_tuple.goal_estimation
            strategy.substrategies[2] = solution_tuple.strategy_ego.substrategies[2]
            predicted_opponent_trajectory = solution_tuple.strategy_ego.substrategies[1]
            # visualize what the ego thinks the opponent will do
            strategy_[] = predicted_opponent_trajectory
            goal_estimation_[] = mortar([episode_info.goal_estimation])
        end
        # ========================================================= #

        # rollout strategy
        pointmasses_trajectory, control_sequence, _ =
            rollout(game.dynamics, strategy, pointmasses, horizon)
        # compute tracking control
        robots_trajectory = rollout_robot_dynamics(;
            initial_state=robots,
            pointmasses_trajectory,
            rollout_horizon=horizon)
        us = robots_trajectory.us

        # safety stop
        # TODO: extend to n_players
        if norm(robots[Block(1)][1:2] - robots[Block(2)][1:2]) < 0.8
            stop_robots(robot_communications)
        else
            for (i, connection) in enumerate(robot_communications)
                if isnothing(connection)
                    continue
                end
                commands = map(us) do state
                    state[Block(i)][1:2]
                end
                put!(connection.channel, commands)
            end
        end

        visualization.pointmasses[] = pointmasses
        visualization.strategy[] = strategy
        visualization.targets[] = targets
        visualized_tracker.trackers[] = robots
        visualized_tracker.trackers_trajectory[] = robots_trajectory.xs
        generate_new_targets_if_reached!(
            targets,
            pointmasses;
            environment=game.env,
            rng,
            #target_reached_tolerance,
            generate_target=(environment) -> sample_target(environment; rng),
            force_generation=spawn_button.clicked[]
        )
        episode_info.t += 1
        if spawn_button.clicked[]
            clear_episode_info!(episode_info)
        end
        spawn_button.clicked[] = false
    end

    stop_robots(robot_communications)
end

function launch(;
    optitrack_ids=[nothing, nothing],
    # robot_ips = ["192.168.68.100", "192.168.68.102"],
    robot_ips=[nothing, nothing],
    #robot_ips = [nothing, nothing],
    robot_communication_port=42421,
    kwargs...
)
    @info "Connecting to OptiTrack..."
    optitrack_communication = OptiTrackCommunication(optitrack_ids)
    @info "Connecting to Robots..."
    robot_communications = [
        if isnothing(ip)
            nothing
        else
            RobotCommunication(ip, robot_communication_port)
        end for ip in robot_ips
    ]

    @info "Idling for a sec..."
    sleep(1.0)
    # RUN
    run(; optitrack_communication, robot_communications, n_players=length(robot_ips), kwargs...)

    @info "Closing down..."
    close(optitrack_communication)
    wait(optitrack_communication.task)
    for connection in robot_communications
        if isnothing(connection)
            continue
        end
        close(connection)
        wait(connection.task)
    end
    @info "Finished!"
end
