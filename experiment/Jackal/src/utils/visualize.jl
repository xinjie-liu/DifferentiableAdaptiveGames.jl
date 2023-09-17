function color_range(n_colors; from_color = colorant"red", to_color = colorant"blue")
    range(from_color, to_color, length = n_colors)
end

function visualize_poses!(
    axis,
    poses;
    player_colors = color_range(blocksize(poses[], 1)),
    marker = '➤',
)
    for player_i in 1:blocksize(poses[], 1)
        color = player_colors[player_i]
        position = @lift Point2f($poses[Block(player_i)][1:2])
        rotation = @lift $poses[Block(player_i)][3]
        scatter!(axis, position; color, rotation, marker, markersize = 20)
    end
end

function visualize_trajectory!(
    axis,
    trajectory;
    player_colors = color_range(blocksize(trajectory[][1], 1)),
    linestyle = :dash,
)
    map(1:blocksize(trajectory[][1], 1)) do player_i
        color = player_colors[player_i]
        points = @lift [Point2f(x[Block(player_i)][1:2]) for x in $trajectory]
        lines!(axis, points; color, linestyle)
    end
end

function visualize_pointmass_sequence!(
    axis,
    states;
    player_colors = color_range(blocksize(states[][1], 1)),
    arrow_length = 0.00005,
    subsampling = 1,
)
    map(1:blocksize(states[][1], 1)) do player_i
        position =
            @lift [Point2f(state[Block(player_i)][1:2]) for state in $states[1:subsampling:end]]
        direction =
            @lift [Vec2f(state[Block(player_i)][3:4]) for state in $states[1:subsampling:end]]
        arrows!(
            axis,
            position,
            direction;
            color = player_colors[player_i],
            lengthscale = arrow_length,
        )
    end
end

function visualize_unicycle_sequence!(
    axis,
    states;
    player_colors = color_range(blocksize(states[][1], 1)),
    arrow_length = 0.00005,
    subsampling = 1,
)
    map(1:blocksize(states[][1], 1)) do player_i
        position =
            @lift [Point2f(state[Block(player_i)][1:2]) for state in $states[1:subsampling:end]]
        direction = @lift [
            Vec2f(cos(state[Block(player_i)][3]), sin(state[Block(player_i)][3])) for
            state in $states[1:subsampling:end]
        ]
        arrows!(
            axis,
            position,
            direction;
            color = player_colors[player_i],
            lengthscale = arrow_length,
        )
    end
end

windowed(ls; window_size) = [ls[i:(i + window_size)] for i in 1:(length(ls) - window_size)]

function visualize_sampled_states!(
    axis,
    sampled_states,
    n_players;
    player_colors = color_range(n_players),
)
    for player_i in 1:n_players
        positions = [Point2f(s.pointmasses[Block(player_i)][1:2]) for s in sampled_states]
        directions = [Vec2f(s.pointmasses[Block(player_i)][3:4]) for s in sampled_states]
        targets = [Point2f(s.targets[Block(player_i)][1:2]) for s in sampled_states]
        arrows!(
            axis,
            positions,
            directions;
            color = (player_colors[player_i], 0.5),
            label = "Player $(player_i)",
        )
        scatter!(axis, targets; color = (player_colors[player_i], 0.5), marker = :+)
    end
end

function visualize_game!(figure, environment, pointmasses, strategy, targets; obstacle_radius)
    axis = create_environment_axis(figure, environment; title = "Game")

    pointmasses = Observable(pointmasses)
    visualize_players!(axis, pointmasses)
    visualize_obstacle_bounds!(axis, pointmasses; obstacle_radius)
    strategy = Observable(strategy)
    TrajectoryGamesBase.visualize!(axis, strategy)
    targets = Observable(targets)
    visualize_targets!(axis, targets)

    (; axis, pointmasses, strategy, targets)
end

# function visualize_targets!(
#     axis,
#     targets;
#     player_colors = range(colorant"red", colorant"blue", length = blocksize(targets[], 1)),
#     marker = '+',
# )
#     for player_i in 1:blocksize(targets[], 1)
#         color = player_colors[player_i]
#         target = Makie.@lift Makie.Point2f($targets[Block(player_i)])
#         Makie.scatter!(axis, target; color, marker)
#     end
# end

function visualize_inferred_goal!(
    axis,
    goal_estimation;
    color = colorant"black",
    marker = '+',
)
    goal_estimation = Makie.@lift Makie.Point2f($goal_estimation)
    Makie.scatter!(axis, goal_estimation; color, marker, markersize = 30)
end 


function visualize_targets!(
    axis,
    targets;
    player_colors = colorant"red",#range(colorant"red", colorant"blue", length = blocksize(targets[], 1)),
    marker = '+',
)
    target = Makie.@lift Makie.Point2f($targets)
    Makie.scatter!(axis, target; player_colors, marker, markersize = 30)
end

function visualize_tracker!(axis, trackers, trackers_trajectory)
    trackers = Observable(trackers)
    visualize_poses!(axis, trackers)
    trackers_trajectory = Observable(trackers_trajectory)
    visualize_trajectory!(axis, trackers_trajectory)
    (; trackers, trackers_trajectory)
end

function visualize_button!(figure, label)
    button = Button(figure; label, tellwidth = false)
    clicked = Observable(false)
    on(button.clicks) do n
        clicked[] = true
    end
    (; button, clicked)
end

function visualize_strategy!(
    axis,
    γ::Makie.Observable{<:LiftedTrajectoryStrategy},
    color;
    weight_offset = 0.0,
)
    trajectory_colors = Makie.@lift([(color, w + weight_offset) for w in $γ.weights])
    Makie.series!(axis, γ; color = trajectory_colors)
end
