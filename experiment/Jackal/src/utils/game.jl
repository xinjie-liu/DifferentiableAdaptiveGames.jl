Base.@kwdef mutable struct EpisodeInfo
    t::Int = 1
    goal_estimation::Any = nothing
    history_trajectory::Vector{Any} = []
    last_solution::Any = nothing
end

function rectangle_points(width, height)
    [
        [-width / 2, -height / 2],
        [width / 2, -height / 2],
        [width / 2, height / 2],
        [-width / 2, height / 2],
    ]
end

function create_game(;
    hard_constraints = false,
    collision_avoidance_coefficient = 0.0,
    avoidance_distance = 1.0,
    environment_size = (; width = 6.0, height = 4.0),
    max_velocity_component = 1.0,
)
    environment_points = rectangle_points(environment_size.width, environment_size.height)
    environment = PolygonEnvironment(LazySets.VPolytope(environment_points))
    game = two_player_guidance_game_with_collision_avoidance(;
        hard_constraints,
        collision_avoidance_coefficient,
        environment,
        min_distance = avoidance_distance,
        dynamics = planar_double_integrator(; #dt = 0.8,
        state_bounds = (; lb = [-Inf, -Inf, -max_velocity_component, -max_velocity_component], 
            ub = [Inf, Inf, max_velocity_component, max_velocity_component]),
            control_bounds = (; lb = [-max_velocity_component, -max_velocity_component], 
            ub = [max_velocity_component, max_velocity_component]),
        ))
    game, avoidance_distance
end

function get_random_pointmass_states(environment, n_players; rng)
    mortar([[LazySets.sample(environment.set; rng); [0.0, 0.0]] for _ in 1:n_players])
end

function get_random_targets(environment, n_players; rng)
    mortar([LazySets.sample(environment.set; rng) for _ in 1:n_players])
end

function find_closest_state_on_plan(player_i, robots, pointmasses_trajectory)
    _, t = findmin(pointmasses_trajectory) do pointmasses
        norm(robots[Block(player_i)][1:2] - pointmasses[Block(player_i)][1:2])
    end
    pointmasses_trajectory[t][Block(player_i)]
end

function find_closest_states_on_plan(robots, pointmasses_trajectory)
    mortar([
        find_closest_state_on_plan(player_i, robots, pointmasses_trajectory) for
        player_i in 1:blocksize(robots, 1)
    ])
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
            safe_distance_violation = mean(map(xs) do x
                # 1 / sqrt(norm(x[Block(1)][1:2] - x[Block(2)][1:2]) + 1e-5)
                max(0, min_distance + 0.1 - norm(x[Block(1)][1:2] - x[Block(2)][1:2]))^3
            end)
            1.0 * mean_target + 0.1 * control + collision_avoidance_coefficient * safe_distance_violation
        end
        function cost_for_player2(xs, us)
            mean_target = mean(map(xs) do x
                target_cost(x[Block(2)], 0.85 * x[Block(1)])
            end)
            control = mean(map(us) do u
                control_cost(u[Block(2)])
            end)
            safe_distance_violation = mean(map(xs) do x
                max(0, min_distance + 0.1 - norm(x[Block(1)][1:2] - x[Block(2)][1:2]))^3
                # 1 / sqrt(norm(x[Block(1)][1:2] - x[Block(2)][1:2]) + 1e-5)
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

function shared_collision_avoidance_coupling_constraints(num_players, min_distance)
    function coupling_constraint(xs, us)
        mapreduce(vcat, 1:(num_players - 1)) do player_i
            mapreduce(vcat, (player_i + 1):num_players) do paired_player
                map(xs[3:end]) do x # loosen a bit so that the solver does not give up to solve
                    norm_sqr(x[Block(player_i)][1:2] - x[Block(paired_player)][1:2]) -
                    min_distance^2
                end
            end
        end
    end
end

function collision_avoidance_coupling_constraints(num_players, min_distance)
    map(1:num_players) do player_i
        function coupling_constraint(xs, us)
            map([1:(player_i - 1); (player_i + 1):num_players]) do paired_player
                d = minimum(xs) do x
                    norm(x[Block(player_i)][1:2] - x[Block(paired_player)][1:2])
                end
                d - min_distance
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
    min_distance = 2.0,
    max_velocity_component = 0.5,
    dynamics = planar_double_integrator(;
        state_bounds = (;
            lb = [-Inf, -Inf, -max_velocity_component, -max_velocity_component],
            ub = [Inf, Inf, max_velocity_component, max_velocity_component],
        ),
        control_bounds = (; lb = [-10, -10], ub = [10, 10]),
    )
)
    cost = let
        function target_cost(x, context_state)
            norm(x[1:2] - context_state[1:2])
        end
        function control_cost(u)
            norm_sqr(u)
        end
        # TODO: this is only a time separable stage cost now
        function cost_for_player(i, xs, us, context_state)
            mean_target = mean(map(xs) do x
                target_cost(x[Block(i)], context_state[Block(i)])
            end)
            control = mean(map(us) do u
                control_cost(u[Block(i)])
            end)
            1.0 * mean_target + 0.1 * control
        end
        function cost_function(xs, us, context_state)
            num_players = blocksize(xs[1], 1)
            [cost_for_player(i, xs, us, context_state) for i in 1:num_players]
        end
        TrajectoryGameCost(cost_function, GeneralSumCostStructure())
    end
    dynamics = ProductDynamics([dynamics for _ in 1:num_players])
    TrajectoryGame(
        cost,
        dynamics,
        environment,
        collision_avoidance_coupling_constraints(num_players, min_distance),
    )
end
