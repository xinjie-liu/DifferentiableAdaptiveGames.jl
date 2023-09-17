abstract type AbstractEnvironment end

# TODO add documentation
function to_sublevelset end

"""
    geometry(env::AbstractEnvironment)

Returns a geometry object that supports visualization with Makie.jl.
"""
function geometry end

function generate_new_targets_if_reached!(
    targets,
    states;
    environment,
    rng = Random.MersenneTwister(1),
    #target_reached_tolerance = 0.2,
    generate_target = (environment) -> LazySets.sample(environment.set; rng),
    force_generation = false,
)
    for (state, target) in zip(blocks(states), blocks(targets))
        # position = state[1:2]
        # distance_to_target = norm(position - target)
        # is_target_reached = distance_to_target < target_reached_tolerance
        if force_generation #|| is_target_reached
            target[1:2] = generate_target(environment)
        end
    end
end

function is_trajectory_colliding_for_player(
    player_i,
    xs,
    collision_radius,
    n_players;
    tolerance_factor = 1.0,
)
    count_collisions_on_trajectory(player_i, xs, collision_radius, n_players; tolerance_factor) > 0
end

function count_collisions_on_trajectory(
    player_i,
    xs,
    collision_radius,
    n_players;
    tolerance_factor = 1.0,
)
    sum(xs) do x
        player_position = x[Block(player_i)][1:2]
        shortest_distance_to_any_other_player = minimum([
            norm(player_position - x[Block(i)][1:2]) for
            i in [1:(player_i - 1); (player_i + 1):n_players]
        ])
        is_in_collision =
            shortest_distance_to_any_other_player < tolerance_factor * collision_radius
        is_in_collision ? 1 : 0
    end
end

function create_dummy_strategy(game, system_state, control_dimension, horizon, player_id, rng)
    dummy_strategy = (x, t) -> zeros(control_dimension)
    dummy_xs = rollout(game.dynamics.subsystems[player_id], 
        dummy_strategy, system_state[Block(player_id)], horizon + 1).xs
    dummy_us = rollout(game.dynamics.subsystems[player_id], 
        dummy_strategy, system_state[Block(player_id)], horizon + 1).us
    dummy_trajectory = (; xs = dummy_xs, us = dummy_us)
    
    (; dummy_substrategy = LiftedTrajectoryStrategy(player_id, [dummy_trajectory], [1], nothing, rng, Ref(0)), 
        dummy_trajectory)
end

function load_history_information_vector!(pointmasses, history_trajectory, vector_size)
    push!(history_trajectory, pointmasses)
    if length(history_trajectory) > (vector_size + 1)
        popfirst!(history_trajectory)
    end
end

function clear_episode_info!(episode_info)
    episode_info.t = 1
    episode_info.goal_estimation = nothing
    episode_info.last_solution = nothing
    episode_info.history_trajectory = []
end