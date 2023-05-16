module ExampleProblems

using BlockArrays: Block, blocksize
using TrajectoryGamesBase:
    GeneralSumCostStructure, ProductDynamics, TrajectoryGame, TrajectoryGameCost
using TrajectoryGamesExamples: planar_double_integrator
using LinearAlgebra: norm_sqr, norm
using Statistics: mean


export n_player_collision_avoidance

function generate_new_targets_if_reached!(
    states,
    targets;
    environment,
    rng = Random.MersenneTwister(1),
    target_reached_tolerance = 0.2,
    generate_target = (environment) -> LazySets.sample(environment.set; rng),
    force_generation = false,
)
    for (state, target) in zip(blocks(states), blocks(targets))
        position = state[1:2]
        distance_to_target = norm(position - target)
        is_target_reached = distance_to_target < target_reached_tolerance
        if force_generation || is_target_reached
            target[1:2] = generate_target(environment)
        end
    end
end

function is_trajectory_colliding_for_player(
    player_i,
    xs,
    collision_radius,
    num_players;
    tolerance_factor = 0.9,
)
    any(xs) do x
        player_position = x[Block(player_i)][1:2]
        shortest_distance_to_any_other_player = minimum([
            norm(player_position - x[Block(i)][1:2]) for
            i in [1:(player_i - 1); (player_i + 1):num_players]
        ])
        shortest_distance_to_any_other_player < tolerance_factor * collision_radius
    end
end

function my_norm_sqr(x)
    x'*x
end

function shared_collision_avoidance_coupling_constraints(num_players, min_distance)
    function coupling_constraint(xs, us)
        mapreduce(vcat, 1:(num_players - 1)) do player_i
            mapreduce(vcat, (player_i + 1):num_players) do paired_player
                map(xs) do x
                    my_norm_sqr(x[Block(player_i)][1:2] - x[Block(paired_player)][1:2]) -
                    min_distance^2
                end
            end
        end
    end
end

"""
Collision avoidance in multiplayer scenario, all trying to reach a given individual target

State layout:

point-mass states per player stacked into game state
x = [x₁ ... xᵢ ... xₙ]
where
xᵢ = [px py vx vy]
with p positional state and v spatial velocity

context state contains targets per player
c = [c₁ ... cᵢ ... cₙ]
where
cᵢ = [tx ty]
with (tx, ty) describing the position of the target

Collision avoidance bounds are modeled via n × (n-1) coupling constraints enforcing a minimal distance between all players
"""
function n_player_collision_avoidance(
    num_players;
    environment,
    min_distance = 1.0,
    collision_avoidance_coefficient = 20.0,
    dynamics = planar_double_integrator(;
        state_bounds = (; lb = [-Inf, -Inf, -0.8, -0.8], ub = [Inf, Inf, 0.8, 0.8]),
        control_bounds = (; lb = [-10, -10], ub = [10, 10]),
    ),
)
    cost = let
        function target_cost(x, context_state)
            norm_sqr(x[1:2] - context_state[1:2])
        end
        function control_cost(u)
            norm_sqr(u)
        end
        function collision_cost(x, i)
            cost = map([1:(i - 1); (i + 1):num_players]) do paired_player
                #1 / sqrt(norm(x[Block(i)][1:2] - x[Block(paired_player)][1:2]) + 1e-5) # with coefficient of 0.02
                max(0.0, min_distance + 0.2 * min_distance - norm(x[Block(i)][1:2] - x[Block(paired_player)][1:2]))^2 # with coefficient of 500
            end
            sum(cost) 
        end
        function cost_for_player(i, xs, us, context_state)
            early_target = target_cost(xs[2][Block(i)], context_state[Block(i)])
            mean_target = mean(map(xs) do x
                target_cost(x[Block(i)], context_state[Block(i)])
            end)
            minimum_target = minimum(map(xs) do x
                target_cost(x[Block(i)], context_state[Block(i)])
            end)
            control = mean(map(us) do u
                control_cost(u[Block(i)])
            end)
            safe_distance_violation = mean(map(xs) do x
                collision_cost(x, i)
            end)
            0.0 * early_target + 1.0 * mean_target + 0.0 * minimum_target + 0.1 * control + collision_avoidance_coefficient * safe_distance_violation
        end
        function cost_function(xs, us, context_state)
            num_players = blocksize(xs[1], 1)
            [cost_for_player(i, xs, us, context_state) for i in 1:num_players]
        end
        TrajectoryGameCost(cost_function, GeneralSumCostStructure())
    end
    dynamics = ProductDynamics([dynamics for _ in 1:num_players])
    TrajectoryGame(
        dynamics,
        cost,
        environment,
        # nothing,
        shared_collision_avoidance_coupling_constraints(num_players, min_distance),
    )
end

function two_player_guidance_game(;
    environment,
    dynamics = planar_double_integrator(;
        state_bounds = (; lb = [-Inf, -Inf, -0.8, -0.8], ub = [Inf, Inf, 0.8, 0.8]),
        control_bounds = (; lb = [-10, -10], ub = [10, 10]),
    ),
)
    cost = let
        function target_cost(x, context_state)
            norm_sqr(x[1:2] - context_state[1:2])
        end
        function control_cost(u)
            norm_sqr(u)
        end
        function cost_for_player1(xs, us, context_state)
            mean_target = mean(map(xs) do x
                target_cost(x[Block(1)], context_state)
            end)
            control = mean(map(us) do u
                control_cost(u[Block(1)])
            end)
            1.0 * mean_target + 0.1 * control
        end
        function cost_for_player2(xs, us)
            mean_target = mean(map(xs) do x
                target_cost(x[Block(2)], x[Block(1)])
            end)
            control = mean(map(us) do u
                control_cost(u[Block(2)])
            end)
            1.0 * mean_target + 0.1 * control
        end
        function cost_function(xs, us, context_state)
            cost1 = cost_for_player1(xs, us, context_state)
            cost2 = cost_for_player2(xs, us)
            [cost1, cost2]
        end
        TrajectoryGameCost(cost_function, GeneralSumCostStructure())
    end
    dynamics = ProductDynamics([dynamics for _ in 1:2])
    TrajectoryGame(dynamics, cost, environment, nothing)
end

function two_player_guidance_game_with_collision_avoidance(;
    environment,
    min_distance = 1.0,
    hard_constraints = true,
    collision_avoidance_coefficient = 0,
    dynamics = planar_double_integrator(;
        state_bounds = (; lb = [-Inf, -Inf, -0.8, -0.8], ub = [Inf, Inf, 0.8, 0.8]),
        control_bounds = (; lb = [-10, -10], ub = [10, 10]),
    ),
)
    cost = let
        function target_cost(x, context_state)
            norm_sqr(x[1:2] - context_state[1:2])
        end
        function control_cost(u)
            norm_sqr(u)
        end
        function cost_for_player1(xs, us, context_state)
            mean_target = mean(map(xs) do x
                target_cost(x[Block(1)], context_state)
            end)
            control = mean(map(us) do u
                control_cost(u[Block(1)])
            end)
            # safe_distance_violation = mean(map(xs) do x
            #     distance = norm(x[Block(1)][1:2] - x[Block(2)][1:2])
            #     max(0, min_distance - distance)
            # end)
            safe_distance_violation = mean(map(xs) do x
                max(0, min_distance + 0.2 - norm(x[Block(1)][1:2] - x[Block(2)][1:2]))^3
                # 1 / sqrt(norm(x[Block(1)][1:2] - x[Block(2)][1:2]) + 1e-5)
            end)
            1.0 * mean_target + 0.1 * control + collision_avoidance_coefficient * safe_distance_violation
        end
        function cost_for_player2(xs, us)
            mean_target = mean(map(xs) do x
                # direction_vector = x[Block(2)] - x[Block(1)]
                # direction_vector = 0.55 * direction_vector/norm(direction_vector)
                # target_cost(x[Block(2)], x[Block(1)] - direction_vector)
                target_cost(x[Block(2)], x[Block(1)])
            end)
            control = mean(map(us) do u
                control_cost(u[Block(2)])
            end)
            # safe_distance_violation = mean(map(xs) do x
            #     distance = norm(x[Block(1)][1:2] - x[Block(2)][1:2])
            #     max(0, min_distance - distance)
            # end)
            safe_distance_violation = mean(map(xs) do x
                # 1 / sqrt(norm(x[Block(1)][1:2] - x[Block(2)][1:2]) + 1e-5)
                max(0, min_distance + 0.2 - norm(x[Block(1)][1:2] - x[Block(2)][1:2]))^3
            end)
            1.0 * mean_target + 0.1 * control + collision_avoidance_coefficient * safe_distance_violation
        end
        function cost_function(xs, us, context_state)
            cost1 = cost_for_player1(xs, us, context_state)
            cost2 = cost_for_player2(xs, us)
            [cost1, cost2]
        end
        TrajectoryGameCost(cost_function, GeneralSumCostStructure())
    end
    dynamics = ProductDynamics([dynamics for _ in 1:2])
    coupling_constraints = hard_constraints ? 
        shared_collision_avoidance_coupling_constraints(2, min_distance) : nothing
    TrajectoryGame(dynamics, cost, environment, coupling_constraints)
end

end
