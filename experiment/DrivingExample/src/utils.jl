#======== environment (dispatch of TrajectoryGamesBase) =========#

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

function plot_wall_constraint() end

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

#======== data processing =========#

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
        # for jj in 1:length(substrategy.val.trajectories[1].xs)
        #     position = Makie.@lift Point2f($(substrategy).trajectories[1].xs[jj])
        #     Makie.scatter!(visualization.environment_axis, position; color = colorant"blue", marker = "*", markersize = 15)
        # end
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
    stop_button = visualize_button!(figure, "Stop")
    pause_button = visualize_button!(figure, "Pause")
    continue_button = visualize_button!(figure, "Continue")
    button_grid = Makie.GridLayout(tellwidth = false)
    button_grid[1, 1] = skip_button.button
    button_grid[1, 2] = stop_button.button
    button_grid[1, 3] = pause_button.button
    button_grid[1, 4] = continue_button.button
    figure[2, 1] = button_grid
    
    (; pointmasses, strategy, targets, environment_axis, skip_button, stop_button, pause_button, continue_button)
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

function visualize_trajectories()
    # solver_string_lst = ["mpc", "backprop"]
    # raw_statistics = Dict()
    # for solver_string in solver_string_lst
    #     raw_statistics[solver_string] = Dict()
    #     raw_statistics[solver_string]["trajectory"] = JLD2.load("data/"*solver_string*"raw_statistics.jld2")["raw_statistic"]["system_state"]
    #     raw_statistics[solver_string]["min distance"] = JLD2.load("data/"*solver_string*"raw_statistics.jld2")["raw_statistic"]["min_distance"]
    # end
    # jldsave("data/trajectory_only.jld2"; raw_statistics)
    num_player = 7
    ego_agent_id = 1
    opponents_id = deleteat!(Vector(1:num_player), ego_agent_id)

    ll = 4.0
    lw = 0.22
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
    # vertices = Vector([x3, x2, x1, x10, x12, x11, x7, x4])
    roadway_opts = MergingRoadwayOptions(lane_length = ll, lane_width = lw, merging_point = mp, merging_angle = θ)
    roadway = build_roadway(roadway_opts)
    collision_radius = 0.08
    player_lane_id = 3 * ones(Int, 7)
    player_lane_id[ego_agent_id] = 4
    environment = RoadwayEnvironment(vertices, roadway, player_lane_id, collision_radius)
    solver_string_list = ["mpc", "backprop"]
    solver_name_map = Dict("ground_truth" => "ground truth", "inverseMCP" => "KKT-constrained", "backprop" => "ours",
    "mpc" => "mpc")
    # CairoMakie.activate!(type = "png")
    fig = Makie.Figure(resolution = (6000, 2400))
    
    for solver_idx in 1:length(solver_string_list)
        ax = Axis(fig[solver_idx, 1]; yticksvisible = false, xticksvisible = false, 
            backgroundcolor = :transparent,
            leftspinevisible = false,
            rightspinevisible = false,
            bottomspinevisible = false,
            topspinevisible = false,
            xgridcolor = :transparent,
            ygridcolor = :transparent,
            xminorticksvisible = false,
            yminorticksvisible = false,
            xautolimitmargin = (0.0,0.0),
            yautolimitmargin = (0.0,0.0))
        
        # CairoMakie.hidedecorations!(ax)
        TrajectoryGamesBase.visualize!(ax, environment)
        Makie.lines!(ax, [x3[1], x2[1]], [x3[2], x2[2]]; color = colorant"rgba(71, 131, 190, 1.0)", linewidth = 45)
        Makie.lines!(ax, [x12[1], x8[1]], [x12[2] + 0.0008, x8[2] + 0.0008]; color = colorant"rgba(71, 131, 190, 1.0)", linewidth = 45)
        Makie.lines!(ax, [x8[1], x6[1]], [x8[2], x6[2]]; color = colorant"rgba(71, 131, 190, 1.0)", linewidth = 45)
        Makie.lines!(ax, [x4[1], x1[1]], [x4[2], x1[2]]; color = colorant"rgba(227, 219, 62, 1.0)", linewidth = 24, linestyle = (:dot, 0.8))
        # Makie.lines!(ax, [x11[1], x7[1]], [x11[2], x7[2]]; color = colorant"rgba(227, 219, 62, 1.0)", linewidth = 6, linestyle = :dash)
        colorlist = [colorant"rgba(238, 29, 37, 1.0)", colorant"rgba(114, 171, 74, 1.0)", colorant"rgba(204, 121, 167, 1.0)", colorant"rgba(230, 159, 0, 1.0)",
            colorant"rgba(213, 94, 0, 1.0)", colorant"rgba(86, 180, 233, 1.0)", colorant"rgba(114, 183, 178, 1.0)"]
        trajectory_dict = JLD2.load("data/trajectory_only.jld2")["raw_statistics"]
        episode_id = 13
        trajectory = map(1:num_player) do jj
            mapreduce(hcat, 1:length(trajectory_dict[solver_string_list[solver_idx]]["trajectory"][1][1:120])) do ii
                trajectory_dict[solver_string_list[solver_idx]]["trajectory"][episode_id][ii][Block(jj)][1:2]
            end
        end
        Makie.xlims!(ax, -0.05, 4.05)
        Makie.ylims!(ax, -0.6, 0.28)

        for ii in 1:num_player
            xs = trajectory[ii][1, :]
            ys = trajectory[ii][2, :]
            initial_marker_size = 30
            end_marker_size = 70
            size_interval = (end_marker_size - initial_marker_size) / length(xs) 
            for jj in 1:length(xs)
                marker = :circle
                markersize = initial_marker_size + jj * size_interval
                min_distance = trajectory_dict[solver_string_list[solver_idx]]["min distance"][episode_id][jj]
                margin = 0.009
                if min_distance < (2 * collision_radius) - margin
                    if ii == ego_agent_id
                        marker = :star6
                        markersize = 200
                    elseif norm([xs[jj], ys[jj]] - trajectory_dict[solver_string_list[solver_idx]]["trajectory"][episode_id][jj][Block(ego_agent_id)][1:2]) < (2 * collision_radius) - margin
                        marker = :star6
                        markersize = 200
                    end
                end            
                Makie.scatter!(ax, xs[jj], ys[jj]; color = colorlist[ii], markersize, marker, rasterize = 10)
            end
        end
    end
    save("trajectories.png", fig, px_per_unit = 2)
    # Main.Infiltrator.@infiltrate
end

function compute_backprop_timing()
    num_trials = 19
    raw_statistics_iterations = JLD2.load("data/backpropraw_statistics.jld2")["raw_statistic"]["inverse solving info"][2:num_trials+1]
    raw_statistics_time = JLD2.load("data/backpropraw_statistics.jld2")["raw_statistic"]["solving time [s]"][2:num_trials+1]
    total_time = mapreduce(vcat, 1:num_trials) do ii
        len = length(raw_statistics_time[ii])
        map(1:len) do jj
            raw_statistics_time[ii][jj]
        end
    end
    major_iterations = mapreduce(vcat, 1:num_trials) do ii
        len = length(raw_statistics_iterations[ii])
        mapreduce(vcat, 1:len) do jj
            len_ = length(raw_statistics_iterations[ii][jj])
            map(1:len_) do kk
                raw_statistics_iterations[ii][jj][kk].major_iterations
            end
        end
    end
    grad_steps = mapreduce(vcat, 1:num_trials) do ii
        len = length(raw_statistics_iterations[ii])
        map(1:len) do jj
            raw_statistics_iterations[ii][jj] |> length
        end
    end
    println(mean(total_time), " ", sem(total_time))
    println(mean(major_iterations), " ", sem(major_iterations))
    println(mean(grad_steps), " ", sem(grad_steps))
end

function compute_inverseMCP_timing()
    num_trials = 20
    raw_statistics_iterations = JLD2.load("data/inverseMCPraw_statistics.jld2")["raw_statistic"]["inverse solving info"][2:num_trials+1]
    raw_statistics_time = JLD2.load("data/inverseMCPraw_statistics.jld2")["raw_statistic"]["solving time [s]"][2:num_trials+1]
    total_time = mapreduce(vcat, 1:num_trials) do ii
        len = length(raw_statistics_time[ii])
        map(1:len) do jj
            raw_statistics_time[ii][jj]
        end
    end
    major_iterations = mapreduce(vcat, 1:num_trials) do ii
        len = length(raw_statistics_iterations[ii])
        map(1:len) do jj
            raw_statistics_iterations[ii][jj].major_iterations
        end
    end
    println(mean(total_time), " ", sem(total_time))
    println(mean(major_iterations), " ", sem(major_iterations))
end

function computePValue(; compared_solvers = ["backprop", "inverseMCP"], num_trials = 100)
    ego_agent_id = JLD2.load("data/"*compared_solvers[1]*"raw_statistics.jld2")["raw_statistic"]["ego_agent_id"]
    opponents_id = JLD2.load("data/"*compared_solvers[1]*"raw_statistics.jld2")["raw_statistic"]["opponents_id"]
    num_player = length(opponents_id) + 1

    data = map(1:2) do ii
        cost_matrix = reduce(hcat, JLD2.load("data/"*compared_solvers[ii]*"raw_statistics.jld2")["raw_statistic"]["interaction cost"])[:, 1:num_trials]
        collision_record = let 
            car_collision = compute_agent_collisions(JLD2.load("data/"*compared_solvers[ii]*"raw_statistics.jld2")["raw_statistic"]["min_distance"], num_trials; collision_threshold = 0.16 - 0.0085)
            wall_collision = JLD2.load("data/"*compared_solvers[ii]*"raw_statistics.jld2")["raw_statistic"]["wall collision"][1:num_trials]
            car_collision + wall_collision    
        end
        episode_filter = sum(collision_record .== 0) == 0 ? BitVector([true for ii in 1:num_trials]) : (collision_record .== 0)

        ego_cost = cost_matrix[ego_agent_id, 1:num_trials]
        opponents_cost = cost_matrix[opponents_id, 1:num_trials]

        filtered_ego_cost = cost_matrix[ego_agent_id, episode_filter]
        filtered_opponents_cost = cost_matrix[opponents_id, episode_filter]

        prediction_error = JLD2.load("data/"*compared_solvers[ii]*"raw_statistics.jld2")["raw_statistic"]["closed-loop prediction error [m]"][1:num_trials][episode_filter]
        parameter_inference_error = JLD2.load("data/"*compared_solvers[ii]*"raw_statistics.jld2")["raw_statistic"]["parameter inference error [m]"][1:num_trials][episode_filter] 

        (; ego_cost, opponents_cost, filtered_ego_cost, filtered_opponents_cost, prediction_error, parameter_inference_error)
    end
    @info "ego cost P value: "*compared_solvers[1]*" V.S. "*compared_solvers[2]
    HypothesisTests.MannWhitneyUTest(data[1].ego_cost, data[2].ego_cost)
    @info "filtered ego cost P value: "*compared_solvers[1]*" V.S. "*compared_solvers[2]
    HypothesisTests.MannWhitneyUTest(data[1].filtered_ego_cost, data[1].filtered_ego_cost)
    @info "prediction error P value: "*compared_solvers[1]*" V.S. "*compared_solvers[2]
    HypothesisTests.MannWhitneyUTest(data[1].prediction_error, data[2].prediction_error)
    @info "parameter inference error P value: "*compared_solvers[1]*" V.S. "*compared_solvers[2]
    HypothesisTests.MannWhitneyUTest(data[1].parameter_inference_error, data[2].parameter_inference_error)
end

function compute_solver_statistics()

    solver_string_lst = ["ground_truth", "heuristic_estimation", "backprop", "inverseMCP", "mpc"]
    raw_statistics = Dict()
    for solver_string in solver_string_lst
        raw_statistics[solver_string] = JLD2.load("data/"*solver_string*"raw_statistics.jld2")["raw_statistic"]
    end

    # normalizing the interaction cost by substracting the ground truth
    if "ground_truth" in keys(raw_statistics) |> collect
        for solver_name in keys(raw_statistics)
            if solver_name != "ground_truth"
                raw_statistics[solver_name]["interaction cost"] -= raw_statistics["ground_truth"]["interaction cost"]
            end
        end
        # delete!(raw_statistics, "ground_truth")
        # delete!(solver_statistics, "ground_truth")
    end
    # Main.Infiltrator.@infiltrate
 
    results = Dict()
    raw_results = Dict()
    for solver_string in solver_string_lst
        results[solver_string] = Dict()
        raw_results[solver_string] = Dict()
        num_trials = 100
        num_player = length(raw_statistics[solver_string]["interaction cost"][1])
        ego_agent_id = raw_statistics[solver_string]["ego_agent_id"]
        opponents_id = raw_statistics[solver_string]["opponents_id"]
        cost_matrix = reduce(hcat, raw_statistics[solver_string]["interaction cost"])
        
        cost_matrix = cost_matrix[:, 1:num_trials]

        normalizing_factor = 1e3
        results[solver_string]["mean_ego_cost"] = mean(cost_matrix[ego_agent_id, 1:num_trials]) * normalizing_factor
        results[solver_string]["sem_ego_cost"] = sem(cost_matrix[ego_agent_id, 1:num_trials]) * normalizing_factor
        results[solver_string]["mean_opponents_cost"] = mean(cost_matrix[opponents_id, 1:num_trials]) * normalizing_factor
        results[solver_string]["sem_opponents_cost"] = sem(cost_matrix[opponents_id, 1:num_trials]) * normalizing_factor
        
        raw_results[solver_string]["ego_cost"] = cost_matrix[ego_agent_id, 1:num_trials]
        raw_results[solver_string]["opponents_cost"] = cost_matrix[opponents_id, 1:num_trials]

        car_collision = compute_agent_collisions(raw_statistics[solver_string]["min_distance"], num_trials; collision_threshold = 0.16 - 0.0085)
        wall_collision = raw_statistics[solver_string]["wall collision"][1:num_trials]

        collision_record = car_collision + wall_collision

        episode_filter = (collision_record .== 0)
        if sum(episode_filter) == 0
            episode_filter = BitVector([true for ii in 1:num_trials])
        end

        results[solver_string]["filtered_mean_ego_cost"] = mean(cost_matrix[ego_agent_id, episode_filter]) * normalizing_factor
        results[solver_string]["filtered_sem_ego_cost"] = sem(cost_matrix[ego_agent_id, episode_filter]) * normalizing_factor
        results[solver_string]["filtered_mean_opponents_cost"] = mean(cost_matrix[opponents_id, episode_filter]) * normalizing_factor
        results[solver_string]["filtered_sem_opponents_cost"] = sem(cost_matrix[opponents_id, episode_filter]) * normalizing_factor

        raw_results[solver_string]["filtered_ego_cost"] = cost_matrix[ego_agent_id, episode_filter]
        raw_results[solver_string]["filtered_opponents_cost"] = cost_matrix[opponents_id, episode_filter]

        results[solver_string]["total_collision"] = (collision_record .> 0) |> sum
        # results[solver_string]["mean_collision"] = collision_record |> mean
        # results[solver_string]["sem_collision"] = collision_record |> sem
        
        solver_failure_record = raw_statistics[solver_string]["solver failure"][1:num_trials] 
        results[solver_string]["total_solver_failure"] = solver_failure_record |> sum
        # results[solver_string]["sem_solver_failure"] = solver_failure_record |> sem

        results[solver_string]["mean_prediction_error"] = (raw_statistics[solver_string]["closed-loop prediction error [m]"][1:num_trials][episode_filter] .|> mean |> mean) * normalizing_factor
        results[solver_string]["sem_prediction_error"] = (raw_statistics[solver_string]["closed-loop prediction error [m]"][1:num_trials][episode_filter] .|> mean |> sem) * normalizing_factor
        results[solver_string]["mean_parameter_inference_error"] = (raw_statistics[solver_string]["parameter inference error [m]"][1:num_trials][episode_filter] .|> mean |> mean) * normalizing_factor
        results[solver_string]["sem_parameter_inference_error"] = (raw_statistics[solver_string]["parameter inference error [m]"][1:num_trials][episode_filter] .|> mean |> sem) * normalizing_factor
        
        raw_results[solver_string]["prediction_error"] = raw_statistics[solver_string]["closed-loop prediction error [m]"][1:num_trials][episode_filter]
        raw_results[solver_string]["parameter_inference_error"] = raw_statistics[solver_string]["parameter inference error [m]"][1:num_trials][episode_filter] 
        
        # Main.Infiltrator.@infiltrate
        if sum(episode_filter) >= 2
            results[solver_string]["mean_solving_time"] = raw_statistics[solver_string]["solving time [s]"][1:num_trials][episode_filter][2:end] .|> mean |> mean
        else
            results[solver_string]["mean_solving_time"] = raw_statistics[solver_string]["solving time [s]"][1:num_trials][episode_filter] .|> mean |> mean
        end
    end
    for solver in solver_string_lst
        CSV.write("data/results.csv", Dict("solver name" => solver); append = true)
        CSV.write("data/results.csv", results[solver]; append = true)
    end
    # fig_ = Makie.Figure()
    # solver_names = keys(solver_statistics) |> collect
    # properties = ["parameter inference error [m]", "open-loop prediction error [m]",
    #     "closed-loop prediction error [m]", "solving time [s]"]
    # line_colors = Dict("backprop" => "#FE5225", "inverseMCP" => "#FF8100", "mpc" => "#52BCB5")
    # band_colors = Dict("backprop" => "#FFC8C9", "inverseMCP" => "#FFDCB7", "mpc" => "#CEEBE9")
    # axes = []
    # for i in 1:length(properties)
    #     axis = Axis(fig_[i, 1]; title = properties[i], xlabel = "time step", ylabel = properties[i])
    #     push!(axes, axis)
    #     for solver in solver_names
    #         data = solver_statistics[solver][properties[i]]
    #         band!(axes[i], 1:length(data.mean), data.mean - data.std, data.mean + data.std, 
    #             label = solver, color = band_colors[solver], transparency = true, shading = false)
    #         lines!(axes[i], 1:length(data.mean), data.mean, color = line_colors[solver], transparency = true,
    #             shading = false)
    #     end
    # end
    # fig_[0, 1] = Legend(fig_, axes[2], "Solver", framevisible = false, 
    #     orientation = :horizontal, tellwidth = false, tellheight = true)

    # fig_cost = Makie.Figure()
    # violin_idx = Dict("backprop" => 1, "inverseMCP" => 2, "mpc" => 3)
    # xs = Float32[]
    # ys = Float32[]
    # for solver in solver_names
    #     cost = [cost[2] for cost in raw_statistics[solver]["interaction cost"]] # only cost for 2nd player
    #     append!(ys, cost)
    #     append!(xs, fill(violin_idx[solver], length(cost)))
    # end
    # axis = Axis(fig_cost[1, 1]; title = "interaction cost", xlabel = "solver", ylabel = "cost")
    # violin!(axis, xs, ys)
    # TODO: also include the collision times
    #save("prediction_errors.png", fig_, px_per_unit = 15.5)
    # save("interaction_costs.png", fig_cost)
end

#================== other ================#

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


function my_norm_sqr(x)
    x'*x
end


function my_norm(x; regularization= 1e-4)
    sqrt(sum(x' * x) + regularization)
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