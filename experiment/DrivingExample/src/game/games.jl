function highway_game(
    num_players;
    environment,
    min_distance = 1.0,
    hard_constraints = true,
    collision_avoidance_coefficient = 0,
    dynamics = planar_double_integrator(;
        state_bounds = (; lb = [-Inf, -Inf, -0.8, -0.8], ub = [Inf, Inf, 0.8, 0.8]),
        control_bounds = (; lb = [-10, -10], ub = [10, 10]),
    ),
)
    ll = environment.roadway.opts.lane_length
    cost = let
        function target_cost(x, context_state)
            # my_norm_sqr(x[2] - context_state[1]) + my_norm_sqr(x[3] - context_state[2])
            px, py, v, θ = x
            # 1.0 * (py - context_state[1])^2 + (1.0 - 1 / (1 + exp(-(px - (ll - 1)) * 10))) * (v * cos(θ) - context_state[2])^2 + 0.2 * θ^2
            1.0 * (py - context_state[1])^2 + ((1.0 - tanh(4 * (px - (ll - 1.65)))) / 2) * (v * cos(θ) - context_state[2])^2 + 0.2 * θ^2
        end
        function control_cost(u)
            a, δ = u
            a^2 + δ^2
        end
        function collision_cost(x, i)
            cost = map([1:(i - 1); (i + 1):num_players]) do paired_player
                #1 / sqrt(norm(x[Block(i)][1:2] - x[Block(paired_player)][1:2]) + 1e-5) # with coefficient of 0.02
                max(0.0, min_distance + 0.0225 - my_norm(x[Block(i)][1:2] - x[Block(paired_player)][1:2]))^3 # with coefficient of 500
            end
            total_cost = sum(cost) 
        end
        function cost_for_player(i, xs, us, context_state)
            mean_target = mean(map(xs) do x
                target_cost(x[Block(i)], context_state[Block(i)])
            end)
            control = mean(map(us) do u
                control_cost(u[Block(i)])
            end)
            safe_distance_violation = mean(map(xs) do x
                collision_cost(x, i)
            end)
            1.0 * mean_target + 0.1 * control + collision_avoidance_coefficient * safe_distance_violation #+ 2.0 * traffic_light_cost_
        end
        function cost_function(xs, us, context_state)
            num_players = blocksize(xs[1], 1)
            [cost_for_player(i, xs, us, context_state) for i in 1:num_players]
        end
        TrajectoryGameCost(cost_function, GeneralSumCostStructure())
    end
    dynamics = ProductDynamics([dynamics for _ in 1:num_players])
    coupling_constraints = hard_constraints ? 
        shared_collision_avoidance_coupling_constraints(num_players, min_distance) : nothing
    TrajectoryGame(dynamics, cost, environment, coupling_constraints)
end