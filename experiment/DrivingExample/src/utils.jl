#======== environment =========#

struct RoadwayEnvironment{T}
    set::T
    roadway::Roadway
    lane_id_per_player::Vector
    radius::Float64
end

function RoadwayEnvironment(vertices::AbstractVector{<:AbstractVector{<:Real}}, roadway::Roadway, 
        lane_id_per_player::Vector, radius::Float64)
    RoadwayEnvironment(LazySets.VPolytope(vertices), roadway, lane_id_per_player, radius)
end

function construct_roadway(ll, lw, mp, θ)
    # Roadway points
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
    
    vertices, roadway
end 

function construct_env(num_player, ego_agent_id, vertices, roadway, collision_radius)
    player_lane_id = 3 * ones(Int, num_player)
    player_lane_id[ego_agent_id] = 4
    environment = RoadwayEnvironment(vertices, roadway, player_lane_id, collision_radius)

    environment
end

function TrajectoryGamesBase.get_constraints(env::RoadwayEnvironment, player_idx)
    lane = env.roadway.lane[env.lane_id_per_player[player_idx]]
    walls = deepcopy(lane.wall)
    ri = env.radius
    for j = 1:length(walls)
        walls[j].p1 -= ri * walls[j].v
        walls[j].p2 -= ri * walls[j].v
    end
    function (state)
        position = state[1:2]
        # wall constraints
        wall_constraints = mapreduce(vcat, walls) do wall
            left = max((position - wall.p1)' * (wall.p2 - wall.p1), 0)
            right = max((position - wall.p2)' * (wall.p1 - wall.p2), 0)
            product = (wall.p1 - position)' * wall.v
            left * right * product
        end
        # circle constraints
        if length(lane.circle) > 0
            circle_constraints = mapreduce(vcat, lane.circle) do circle
                (position[1] - circle.x)^2 + (position[2] - circle.y)^2 - (circle.r + ri)^2
            end
        else
            circle_constraints = []
        end
        vcat(wall_constraints, circle_constraints)
    end
end

function construct_game(num_player; min_distance = 2 * collision_radius, hard_constraints, 
    collision_avoidance_coefficient, environment, max_velocity, max_acceleration, max_ϕ)
    dynamics = BicycleDynamics(; 
        l = 0.1,
        state_bounds = (; lb = [-Inf, -Inf, -max_velocity, -Inf], ub = [Inf, Inf, max_velocity, Inf]),
        control_bounds = (; lb = [-max_acceleration, -max_ϕ], ub = [max_acceleration, max_ϕ]),
        integration_scheme = :reverse_euler
    )
    game = highway_game(num_player; min_distance, hard_constraints, collision_avoidance_coefficient,
        environment, dynamics)

    game
end

#======== visualization =========#

function visualize_button!(figure, label)
    button = Makie.Button(figure; label, tellwidth = false)
    clicked = Makie.Observable(false)
    Makie.on(button.clicks) do n
        clicked[] = true
    end
    (; button, clicked)
end

function visualize_prediction(strategy, visualization, ego_agent_id)
    strategy_ = deepcopy(strategy.substrategies)
    deleteat!(strategy_, ego_agent_id)
    predicted_strategy_visualization = map(1:length(strategy_)) do ii
        substrategy = Makie.Observable(strategy_[ii])
        TrajectoryGamesBase.visualize!(visualization.environment_axis, substrategy; color = colorant"rgba(238, 29, 37, 1.0)")
        substrategy
    end
end

function visualize_players!(
    axis,
    players;
    ego_agent_id,
    opponents_id,
    marker = '➤', 
)
    for player_i in 1:blocksize(players[], 1)
        player_color = player_i == ego_agent_id ? colorant"rgba(238, 29, 37, 1.0)" : colorant"rgba(46,139,87, 1.0)"
        position = Makie.@lift Makie.Point2f($players[Block(player_i)][1:2])
        rotation = Makie.@lift $players[Block(player_i)][4]
        Makie.scatter!(axis, position; rotation, color = player_color, marker, markersize = 20)
    end
end

function visualize_obstacle_bounds!(
    axis,
    players;
    obstacle_radius = 1.0,
    ego_agent_id,
    opponents_id
)
    for player_i in 1:blocksize(players[], 1)
        player_color = player_i == ego_agent_id ? colorant"rgba(238, 29, 37, 1.0)" : colorant"rgba(46,139,87, 1.0)"
        position = Makie.@lift Makie.Point2f($players[Block(player_i)][1:2])
        Makie.scatter!(
            axis,
            position;
            color = (player_color, 0.4),
            markersize = 2 * obstacle_radius,
            markerspace = :data,
        )
    end
end

function TrajectoryGamesBase.visualize!(
    canvas,
    γ::Makie.Observable{<:LiftedTrajectoryStrategy};
    color = :black,
    weight_offset = 0.0,
)
    trajectory_colors = Makie.@lift([(color, w + weight_offset) for w in $γ.weights])
    Makie.series!(canvas, γ; color = trajectory_colors, linewidth=3.5)
end

function visualize!(figure, game, pointmasses, strategy; targets = nothing, obstacle_radius = 0.0, 
    ego_agent_id = nothing, opponents_id = nothing)

    num_player = length(strategy.substrategies)
    player_colors = map(1:num_player) do ii
        color = ii == ego_agent_id ? colorant"rgba(238, 29, 37, 1.0)" : colorant"rgba(46,139,87,1.0)"
    end

    environment_axis = create_environment_axis(figure[1, 1], game.env; title = "Game")

    pointmasses = Makie.Observable(pointmasses)
    visualize_players!(environment_axis, pointmasses; ego_agent_id, opponents_id)
    if obstacle_radius > 0
        visualize_obstacle_bounds!(environment_axis, pointmasses; 
        obstacle_radius, ego_agent_id, opponents_id)
    end

    strategy = Makie.Observable(strategy)
    TrajectoryGamesBase.visualize!(environment_axis, strategy; colors = player_colors)
    if !isnothing(targets)
        targets = Makie.Observable(targets)
        visualize_targets!(environment_axis, targets)
    end

    skip_button = visualize_button!(figure, "Skip")
    pause_button = visualize_button!(figure, "Pause")
    continue_button = visualize_button!(figure, "Continue")
    button_grid = Makie.GridLayout(tellwidth = false)
    button_grid[1, 1] = skip_button.button
    button_grid[1, 2] = pause_button.button
    button_grid[1, 3] = continue_button.button
    figure[2, 1] = button_grid
    
    (; pointmasses, strategy, targets, environment_axis, skip_button, pause_button, continue_button)
end

function visualize_targets!(
    axis,
    targets;
    player_colors = range(colorant"red", colorant"blue", length = blocksize(targets[], 1)),
    marker = '+',
)
    for player_i in 1:blocksize(targets[], 1)
        color = player_colors[player_i]
        target = Makie.@lift Makie.Point2f($targets[Block(player_i)])
        Makie.scatter!(axis, target; color, marker, markersize = 30)
    end
end

function TrajectoryGamesBase.visualize!(canvas, environment::RoadwayEnvironment; color = colorant"rgba(225, 242, 251, 1.0)") #(19, 45, 82, 1.0)")
    geometry = GeometryBasics.Polygon(GeometryBasics.Point{2}.(environment.set.vertices))
    Makie.poly!(canvas, geometry; color)
	ll = environment.roadway.opts.lane_length
	lw = environment.roadway.opts.lane_width
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

    Makie.lines!(canvas, [x3[1], x2[1]], [x3[2] .+ 0.03, x2[2] .+ 0.03]; color = colorant"rgba(39, 126, 192, 1.0)", linewidth = 8)
    Makie.lines!(canvas, [x12[1], x8[1]], [x12[2] .- 0.015, x8[2] .- 0.015]; color = colorant"rgba(39, 126, 192, 1.0)", linewidth = 8)
    Makie.lines!(canvas, [x8[1], x6[1]], [x8[2] .- 0.015, x6[2] .- 0.015]; color = colorant"rgba(39, 126, 192, 1.0)", linewidth = 8)
    Makie.lines!(canvas, [x4[1], x1[1]], [x4[2], x1[2]]; color = colorant"rgba(39, 126, 192, 1.0)", linewidth = 3.5, linestyle = :dash) 
end

#================== other ================#

function interactive_inference_by_backprop(
    mcp_game, initial_state, τs_observed, 
    initial_estimation, ego_goal; 
    max_grad_steps = 150, lr = 1e-3, last_solution = nothing,
    num_player, ego_agent_id, observation_opponents_idx_set,
    ego_state_idx, 
)
    """
    solve inverse game

    gradient steps using differentiable game solver on the observation likelihood loss
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
        norm_sqr(τs_observed - τs_solution)
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

function solve_game_with_resolve!(receding_horizon_strategy, game, system_state)
    """
    solve forward game, resolve with constant velocity rollout as initialization if solve failed
    """
    strategy = MCPGameSolver.solve_trajectory_game!(receding_horizon_strategy.solver, 
        game, system_state, receding_horizon_strategy; receding_horizon_strategy.solve_kwargs...)
    if receding_horizon_strategy.solution_status != PATHSolver.MCP_Solved
        @info "Solve failed, re-initializing..."
        receding_horizon_strategy.last_solution = nothing
        receding_horizon_strategy.solution_status = nothing
        strategy = MCPGameSolver.solve_trajectory_game!(receding_horizon_strategy.solver, 
            game, system_state, receding_horizon_strategy; receding_horizon_strategy.solve_kwargs...)
    end

    strategy
end

function check_solver_status!(
    receding_horizon_strategy_ego, strategy, 
    strategy_ego, game, system_state, ego_agent_id, horizon, max_acceleration, rng
)
    """
    Check solver status, if failed, overwrite with an emergency strategy
    """
    solving_status = receding_horizon_strategy_ego.solution_status
    if solving_status == PATHSolver.MCP_Solved
        strategy.substrategies[ego_agent_id] = strategy_ego.substrategies[ego_agent_id]
    else
        dummy_substrategy, _ = create_dummy_strategy(game, system_state, 
            control_dim(game.dynamics.subsystems[ego_agent_id]), horizon, ego_agent_id, rng;
            max_acceleration = max_acceleration, strategy_type = "max_acceleration")
        strategy.substrategies[ego_agent_id] = dummy_substrategy
    end

    solving_status
end

function construct_observation_index_set(;
    num_player, ego_agent_id, vector_size, state_dimension, mcp_game,
)
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

    observation_opponents_idx_set
end

function erase_last_solution!(receding_horizon_strategy)
    # clean up the last solution
    receding_horizon_strategy.last_solution = nothing
    receding_horizon_strategy.solution_status = nothing
end

function create_dummy_strategy(game, system_state, control_dimension, horizon, player_id, rng;
    max_acceleration = nothing, strategy_type = "zero_input")
    @assert strategy_type in ["zero_input", "max_acceleration"] "Please give a valid strategy type."
    if strategy_type == "zero_input"
    dummy_strategy = (x, t) -> zeros(control_dimension)
    else
        dummy_strategy = let
            function max_acc_strategy(x, t)
                control = zeros(control_dimension)
                if x[3] >= 0
                    control[1] = -max_acceleration
                elseif x[3] < 0
                    control[1] = max_acceleration
                end
                # double integrator dynamics
                # if x[4] >= 0
                #     control[2] = -max_acceleration
                # end
                control
            end
        end
    end

    dummy_xs = rollout(game.dynamics.subsystems[player_id], 
        dummy_strategy, system_state[Block(player_id)], horizon + 1).xs
    dummy_us = rollout(game.dynamics.subsystems[player_id], 
        dummy_strategy, system_state[Block(player_id)], horizon + 1).us
    dummy_trajectory = (; xs = dummy_xs, us = dummy_us)
    
    (; dummy_substrategy = LiftedTrajectoryStrategy(player_id, [dummy_trajectory], [1], nothing, rng, Ref(0)), 
        dummy_trajectory)
end

function create_params_processing_fn(block_sizes_params, ego_agent_id)
    function params_processing_fn(context_state, ego_goal)
        context_state = reshape(context_state, block_sizes_params[1], :)
        goal_list = Vector{Any}[eachcol(context_state)...]
        insert!(goal_list, ego_agent_id, ego_goal)
        goals = reduce(vcat, goal_list)
        stack_goal = BlockVector(goals, block_sizes_params)
    end
end

function reproduce_goal_estimation(ego_goal, block_sizes_params, ego_agent_id, solution)
    solution = reshape(solution, block_sizes_params[1], :)
    goal_list = Vector{eltype(solution)}[eachcol(solution)...]
    insert!(goal_list, ego_agent_id, ego_goal)
    goals = reduce(vcat, goal_list)
    stack_goal = BlockVector(goals, block_sizes_params)
end

function collision_detection(system_state, ego_agent_id, opponents_id, min_distance)
    collision = 0
    dis = let 
        map(opponents_id) do opponent_id
            single_dis = norm(system_state[Block(ego_agent_id)][1:2] - system_state[Block(opponent_id)][1:2])
            if single_dis < min_distance
                collision += 1
            end
            single_dis
        end
    end
    minimum(dis), collision
end

function highway_sampling(game, horizon, rng, num_player, ego_agent_id, collision_radius, number_trials; 
    x0_range = 0.5, max_velocity = 0.5, merging_scenario = false, vertices = nothing)
    println("Sampling initial states and goals...")

    function sample_initial_state(ii) # generic for both unicycle and bicycle dynamics
        initial_state = [rand(rng) * x0_range, rand(rng, [-0.1, 0.1]), rand(rng) * max_velocity, 0.0]
        if merging_scenario && ii == ego_agent_id
            initial_state[1:2] = (vertices[5] + vertices[7]) / 2
            θ = game.env.roadway.opts.merging_angle
            initial_state[4] = θ
        elseif !merging_scenario && ii == ego_agent_id
            initial_state[2] = -0.1
        end
        initial_state 
    end
    
    initial_state_set = []
    goal_dataset = []
    for ii in 1:number_trials
        initial_states = Vector{Float64}[]
        goals = Vector{Float64}[]
        for ii in 1:num_player
            initial_state = sample_initial_state(ii)
            while !initial_state_feasibility(game, horizon, initial_state, initial_states, collision_radius)
                initial_state = sample_initial_state(ii)
            end
            push!(initial_states, initial_state)
            goal = [rand(rng, [-0.1, 0.1]), (rand(rng) * 0.3) + 0.2]
            if merging_scenario && ii == ego_agent_id
                goal[1] = 0.1
                goal[2] = 0.5
            elseif !merging_scenario && ii == ego_agent_id
                goal[1] = 0.1
                goal[2] = 0.5
            end
            push!(goals, goal)
        end
        initial_states = mortar(initial_states)
        goals = mortar(goals)
        push!(initial_state_set, initial_states)
        push!(goal_dataset, goals)
    end
    initial_state_set, goal_dataset
end

function initial_state_feasibility(game, horizon, initial_state, initial_states, collision_radius)

    if length(initial_states) == 0
        return true
    else
        control_dimension = control_dim(game.dynamics.subsystems[1])
        dummy_strategy = (x, t) -> zeros(control_dimension)
        collision_detection_steps = 5

        for other_state in initial_states
            other_rollout = rollout(game.dynamics.subsystems[1], 
                dummy_strategy, other_state, horizon)
            this_rollout = rollout(game.dynamics.subsystems[1], 
                dummy_strategy, initial_state, horizon)
            for tt in 1:collision_detection_steps
                if norm(this_rollout.xs[tt][1:2] - other_rollout.xs[tt][1:2]) < 2.05 * collision_radius
                    return false
                end
            end
        end
    end
    return true
end

function compute_agent_collisions(distance_dict, num_trials; collision_threshold)
    collisions = map(1:num_trials) do ii
       (distance_dict[ii] .< collision_threshold) |> sum
    end
    collisions
end

function compute_distance(system_state)
    num_player = blocksizes(system_state) |> only |> length
    distance = mapreduce(vcat, 1:(num_player - 1)) do player_i
        mapreduce(vcat, (player_i + 1):num_player) do paired_player
            norm(system_state[Block(player_i)][1:2] - system_state[Block(paired_player)][1:2])
        end
    end 
end

function prediction_errors_computation(planned_opponents_states, predicted_opponents_states, 
    system_states, horizon, opponents_id)

    length_gap = length(planned_opponents_states) - length(predicted_opponents_states)
    
    planned_opponents_states = planned_opponents_states[(length_gap + 1):end]
    openloop_prediction_errors = map(1:length(predicted_opponents_states)) do tt
        mean(map(1:length(opponents_id)) do ii
            norm(planned_opponents_states[tt][ii].trajectories[1].xs - predicted_opponents_states[tt][ii].trajectories[1].xs)
        end)
    end

    system_states = system_states[(length_gap + 1):end]
    closedloop_prediction_errors = map(1:(length(predicted_opponents_states) - horizon + 1)) do tt
        mean(map(1:length(opponents_id)) do ii
            predicted_future = predicted_opponents_states[tt][ii].trajectories[1].xs[2:end] # discard the initial state
            actual_future = [state[Block(opponents_id[ii])] for state in system_states[tt:(tt + horizon - 1)]]
            norm(predicted_future - actual_future)
        end)
    end
    (; openloop_prediction_errors, closedloop_prediction_errors)
end

function compute_state_estimation(previous_state, system_state, num_player; dt = 0.1)
    estimated_state = deepcopy(previous_state)
    for ii in 1:num_player
        estimated_velocity = ()
        estimated_state[Block(ii)][3:4] = (system_state[Block(ii)][1:2] - previous_state[Block(ii)][1:2]) / dt
    end
    estimated_state
end

# re-spawn agent when the old agent is out of the field
# TODO: also consider the information vector
# TODO: only re-warm-start the respective digits
function respawn_players!(system_state, last_solution, receding_horizon_strategy, receding_horizon_strategy_ego, 
    xs_observation, xs_pre; collision_radius = nothing, ll = nothing, ego_agent_id = 1)
    num_player = length(system_state |> blocks)
    order = sortperm(system_state |> blocks, rev = true)
    if system_state[Block(order[1])][1] >= ll
        respawn_len = let
            respawn_spot = min(system_state[Block(order[1])][1] - ll, 
                system_state[Block(order[num_player])][1] - 2 * collision_radius - 0.02)
            system_state[Block(order[1])][1] - respawn_spot
        end
        if order[2] != ego_agent_id
            system_state[Block(order[1])][1] -= respawn_len
        else
            if system_state[Block(ego_agent_id)][1] >= ll
                for ii in 1:3
                    system_state[Block(order[ii])][1] -= respawn_len
                end
            end
        end
        last_solution = nothing # for warm-starting
        # clean up the last solution
        receding_horizon_strategy.last_solution = nothing
        receding_horizon_strategy.solution_status = nothing
        receding_horizon_strategy_ego.last_solution = nothing
        receding_horizon_strategy_ego.solution_status = nothing
        xs_observation = Array{Float64}[]
        xs_pre = BlockArrays.BlockVector{Float64}[]
    end
end

Base.@kwdef mutable struct SpecialCaseCounter
    total_iter = 0
    total_elements = 0
    non_invertible_iter = 0
    weak_complementarity_iter = 0
    weak_complementarity_elements = 0     
end

function numerical_special_case_evaluation(special_case_counter, mcp_game, goal_estimation, solution)

    (; f!, jacobian_z!, lower_bounds, upper_bounds) = mcp_game.parametric_mcp

    z_star = ForwardDiff.value.(solution.variables)
    F = zeros(length(z_star))
    θ = ForwardDiff.value.(goal_estimation)
    f!(F, z_star, θ)

    special_case_counter.total_iter += 1
    special_case_counter.total_elements += length(z_star)

    active_tolerance = 1e-3

    inactive_indices = let
        lower_inactive = z_star .>= (lower_bounds .+ active_tolerance)
        upper_inactive = z_star .<= (upper_bounds .- active_tolerance)
        findall(lower_inactive .& upper_inactive)
    end

    active_indices = let
        lower_active = z_star .<= (lower_bounds .+ active_tolerance)
        upper_active = z_star .>= (upper_bounds .- active_tolerance)
        findall(lower_active .| upper_active)
    end

    ∂f_reduced∂z_reduced = let
        ∂f∂z = ParametricMCPs.get_result_buffer(jacobian_z!)
        jacobian_z!(∂f∂z, z_star, θ)
        ∂f∂z[inactive_indices, inactive_indices]
    end

    weak_complementarity_elements = (F[active_indices] .== 0) |> sum
    special_case_counter.weak_complementarity_elements += weak_complementarity_elements
    if weak_complementarity_elements > 0
        special_case_counter.weak_complementarity_iter += 1
    end

    ∂f∂z_invertible = LinearAlgebra.rank(∂f_reduced∂z_reduced) == size(∂f_reduced∂z_reduced)[2]
    if !∂f∂z_invertible
        special_case_counter.non_invertible_iter += 1
    end
end

function my_norm_sqr(x)
    x'*x
end


function my_norm(x; regularization= 1e-4)
    sqrt(sum(x' * x) + regularization)
end