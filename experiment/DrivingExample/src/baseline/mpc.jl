function mpc_highway(game, horizon, ego_player_id, opponents_id, opponent_block_sizes; 
        min_distance = nothing, collision_avoidance_coefficient = 0, highway = true)

    cost_fn = create_cost(game, horizon, opponents_id, opponent_block_sizes,
        collision_avoidance_coefficient, min_distance; highway)

    sys_dynamics = game.dynamics.subsystems[ego_player_id]
    state_dimension = state_dim(sys_dynamics)
    control_dimension = control_dim(sys_dynamics)
    ego_goal_dimension = 2
    params_dimension = sum([state_dim(game.dynamics.subsystems[idx]) for idx in opponents_id]) + ego_goal_dimension

    inequality_constraints = let
        environment_constraints = get_constraints(game.env, ego_player_id)
        state_box_constraints = get_constraints_from_box_bounds(state_bounds(sys_dynamics))
        control_box_constraints = get_constraints_from_box_bounds(control_bounds(sys_dynamics))
        function (xs, us, params)
            ec = mapreduce(environment_constraints, vcat, xs[2:end])
            sc = mapreduce(state_box_constraints, vcat, xs[2:end])
            cc = mapreduce(control_box_constraints, vcat, us[1:end])
            coupling_constraints = isnothing(min_distance) ? [] : collision_avoidance_constraints(game, horizon, opponents_id, 
                opponent_block_sizes, min_distance, xs, us, params)
            [ec; sc; cc; coupling_constraints]
        end
    end

    problem = ParametricTrajectoryOptimizationProblem(
        cost_fn,
        sys_dynamics,
        inequality_constraints,
        state_dimension,
        control_dimension,
        params_dimension,
        horizon
    )
    backend = MCPSolver()

    optimizer = Optimizer(problem, backend)
end

function collision_avoidance_constraints(game, horizon, opponents_id, opponent_block_sizes, min_distance,
        xs, us, params)
    x0_opponents = params[3:end]
    x0_opponents = BlockArrays.BlockVector(x0_opponents, opponent_block_sizes)
    control_dimension = control_dim(game.dynamics.subsystems[opponents_id[1]]) # TODO: change this if agents' dim vary
    dummy_strategy = (x, t) -> zeros(control_dimension)
    
    xs_opponents = map(1:length(opponents_id)) do ii
        rollout(game.dynamics.subsystems[opponents_id[ii]], 
            dummy_strategy, x0_opponents[Block(ii)], horizon + 1).xs[2:end]
    end
    
    coupling_constraints = mapreduce(vcat, 1:length(opponents_id)) do ii
        single_collision_constraints = map(xs[2:end], xs_opponents[ii][2:end]) do x, x_opponent
            my_norm_sqr(x[1:2] - x_opponent[1:2]) - min_distance^2
        end
    end
end

function create_cost(game, horizon, opponents_id, opponent_block_sizes, collision_avoidance_coefficient, min_distance;
    highway = true)
    control_dimension = control_dim(game.dynamics.subsystems[opponents_id[1]]) # TODO: change this if agents' dim vary
    dummy_strategy = (x, t) -> zeros(control_dimension)
    ll = game.env.roadway.opts.lane_length
    function cost_fn(xs, us, params)
        goal = params[1:2]
        x0_opponents = params[3:end]
        x0_opponents = BlockArrays.BlockVector(x0_opponents, opponent_block_sizes)
        xs_opponents = map(1:length(opponents_id)) do ii
            rollout(game.dynamics.subsystems[opponents_id[ii]], 
                dummy_strategy, x0_opponents[Block(ii)], horizon + 1).xs[2:end]
        end
        if highway
            mean_target = mean(map(xs) do x
                # norm_sqr(x[2:3] - goal)
                px, py, v, θ = x
                1.0 * (py - goal[1])^2 + ((1.0 - tanh(4 * (px - (ll - 1.65)))) / 2) * (v * cos(θ) - goal[2])^2 + 0.2 * θ^2
            end)
        end
        control = mean(map(us) do u
            a, δ = u
            a^2 + δ^2
        end)
        safe_distance_violation = mapreduce(+, 1:length(opponents_id)) do ii
            single_collision_cost = mean(map(xs, xs_opponents[ii]) do x, x_opponent
                #1 / sqrt(norm(x - x_opponent) + 1e-5)
                max(0.0, min_distance + 0.0225 - my_norm(x[1:2] - x_opponent[1:2]))^3
            end)
        end
        1.0 * mean_target + 0.1 * control + collision_avoidance_coefficient * safe_distance_violation
    end
end

function constant_velocity_rollout(game, horizon, opponents_id, x0_opponents, rng)
    control_dimension = control_dim(game.dynamics.subsystems[opponents_id[1]])
    dummy_strategy = (x, t) -> zeros(control_dimension)

    rollouts = map(1:length(opponents_id)) do ii
        rollout(game.dynamics.subsystems[opponents_id[ii]], 
            dummy_strategy, x0_opponents[Block(ii)], horizon + 1)
    end

    predicted_opponents_trajectory = map(1:length(opponents_id)) do ii
        LiftedTrajectoryStrategy(opponents_id[ii], [(; rollouts[ii].xs, rollouts[ii].us)], 
            [1], nothing, rng, Ref(0))
    end
end