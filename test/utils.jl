function solution2joint_trajectory_cost(game, solution, initial_state, goal, horizon)
    num_player = length(solution.substrategies)

    xs_list = map(1:length(solution.substrategies)) do ii
        solution.substrategies[ii].trajectories[1].xs
    end
    us_list = map(1:length(solution.substrategies)) do ii
        solution.substrategies[ii].trajectories[1].us
    end
    joint_xs = map(1:length(xs_list[1])) do ii
        mortar([xs[ii] for xs in xs_list])
    end
    joint_us = map(1:length(us_list[1])) do ii
        mortar([us[ii] for us in us_list])
    end

    calibrated_us = deepcopy(joint_us)
    calibrated_us = push!(calibrated_us, mortar([zeros(2) for i in 1:num_player]))
    trivial_strategy = (x, t) -> calibrated_us[t]
    calibrated_trajectory = rollout(game.dynamics, trivial_strategy, initial_state, horizon + 1)
    calibrated_xs = calibrated_trajectory.xs
    calibrated_cost = game.cost(calibrated_xs, joint_us, goal)
    (; joint_xs, joint_us, calibrated_cost)
end

function nash_test(joint_us, num_player, game, initial_state, horizon, goal, calibrated_cost)
    test_results = map(1:length(joint_us)) do ii
        perturbated_us = deepcopy(joint_us)            
        perturbated_us[ii] = perturbated_us[ii] + (([rand(Float64, (2,)); 0; 0] - [0.5, 0.5, 0, 0]))./1000
        perturbated_us = push!(perturbated_us, mortar([zeros(2) for i in 1:num_player]))
        perturbated_strategy = (x, t) -> perturbated_us[t]
        perturbated_trajectory = rollout(game.dynamics, perturbated_strategy, initial_state, horizon + 1)
        perturbated_xs = perturbated_trajectory.xs
        perturbated_cost = game.cost(perturbated_xs, perturbated_us[1:(end - 1)], goal)
        if perturbated_cost[1] - calibrated_cost[1] <= -1e-5
            println("cost increase after perturbation: ", perturbated_cost[1] - calibrated_cost[1])
        end
        perturbated_cost[1] - calibrated_cost[1] >= -1e-3
    end

end

function nash_test_player2(joint_us, num_player, game, initial_state, horizon, goal, calibrated_cost)
    test_results = map(1:length(joint_us)) do ii
        perturbated_us = deepcopy(joint_us)            
        perturbated_us[ii] = perturbated_us[ii] + (([0; 0; rand(Float64, (2,))] - [0, 0, 0.5, 0.5]))./1000
        perturbated_us = push!(perturbated_us, mortar([zeros(2) for i in 1:num_player]))
        perturbated_strategy = (x, t) -> perturbated_us[t]
        perturbated_trajectory = rollout(game.dynamics, perturbated_strategy, initial_state, horizon + 1)
        perturbated_xs = perturbated_trajectory.xs
        perturbated_cost = game.cost(perturbated_xs, perturbated_us[1:(end - 1)], goal)
        #println("cost increase after perturbation: ", perturbated_cost[2] - calibrated_cost[2])
        if perturbated_cost[2] - calibrated_cost[2] <= -1e-5
            println("cost increase after perturbation: ", perturbated_cost[2] - calibrated_cost[2])
        end
        perturbated_cost[2] - calibrated_cost[2] >= -1e-3
    end

end