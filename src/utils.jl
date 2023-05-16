#== WarmStartRecedingHorizonStrategy ==#

Base.@kwdef mutable struct WarmStartRecedingHorizonStrategy{T1,T2,T3}
    solver::T1
    game::T2
    solve_kwargs::T3 = (;)
    receding_horizon_strategy::Any = nothing
    time_last_updated::Int = 0
    turn_length::Int
    last_solution::Any = nothing # for warm-starting
    context_state::Any = nothing
    solution_status::Any = nothing
end

function (strategy::WarmStartRecedingHorizonStrategy)(state, time)
    plan_exists = !isnothing(strategy.receding_horizon_strategy)
    time_along_plan = time - strategy.time_last_updated + 1
    plan_is_still_valid = 1 <= time_along_plan <= strategy.turn_length

    update_plan = !plan_exists || !plan_is_still_valid
    if update_plan
        strategy.receding_horizon_strategy =
            solve_trajectory_game!(strategy.solver, strategy.game, state, strategy; strategy.solve_kwargs...)
        strategy.time_last_updated = time
        time_along_plan = 1
    end

    strategy.receding_horizon_strategy(state, time_along_plan)
end

#== Trajectory Conversion ==#

function trajectory_from_params(dynamics, params, horizon)
    num_states = state_dim(dynamics)
    num_controls = control_dim(dynamics)
    xs = reshape(params[1:(horizon * num_states)], num_states, :) |> eachcol |> collect
    us =
        reshape(
            params[((horizon * num_states) + 1):((horizon * num_states) + (horizon * num_controls))],
            num_controls,
            :,
        ) |>
        eachcol |>
        collect

    (; xs, us)
end

function mortar_trajectories(τ1, τ2)
    xs1 = τ1.xs
    us1 = τ1.us

    xs2 = τ2.xs
    us2 = τ2.us

    xs_joint = map(xs1, xs2) do x1, x2
        mortar([x1, x2])
    end

    us_joint = map(us1, us2) do u1, u2
        mortar([u1, u2])
    end

    (; xs = xs_joint, us = us_joint)
end

function extract_subproblem_dim(MCPSubproblems)
    """
    extract the dimensionalities of the subproblems for avoiding passing the entire problem as argument
    """
    subproblems_dim = map(1:length(MCPSubproblems)) do ii
        (;
            n = MCPSubproblems[ii].n,
            num_equality = MCPSubproblems[ii].num_equality,
            num_inequality = MCPSubproblems[ii].num_inequality,
            horizon = MCPSubproblems[ii].horizon,
            state_dim = MCPSubproblems[ii].state_dim,
            control_dim = MCPSubproblems[ii].control_dim,
            parameter_dim = MCPSubproblems[ii].parameter_dim,
        )
    end
end

function is_equal_matrix(M1, M2, tol = 1e-10)
    equal_entries = sum(abs.(M1 - M2) .< tol)
    error = sum(abs.(M1 - M2))
    (; equal_entries, error)
end

function parametrize_coupled_inequality(subproblem, coupled_inequality)
    """
    Separately compute symbolic version of coupled inequality constraints and their jacobian
    """
    (; n, horizon, state_dim, control_dim, parameter_dim) = subproblem
    nx = subproblem.num_equality
    x0, z, p = let
        @variables(x0[1:state_dim], z[1:n], p[1:parameter_dim]) .|> scalarize
    end

    xs = hcat(x0, reshape(z[1:nx], state_dim, horizon)) |> eachcol |> collect
    us = reshape(z[(nx + 1):n], control_dim, horizon) |> eachcol |> collect

    coupled_constraints_val = Symbolics.Num[]
    append!(coupled_constraints_val, coupled_inequality(xs[2:end], us, p))

    expression = Val{false}

    parametric_cons = let
        con_fn! = Symbolics.build_function(coupled_constraints_val, [x0; p; z]; expression)[2]
        (cons, x0, params, primals) -> con_fn!(cons, vcat(x0, params, primals))
    end

    con_jac = Symbolics.sparsejacobian(coupled_constraints_val, z)
    (jac_rows, jac_cols, jac_vals) = findnz(con_jac)

    parametric_jac_vals = let
        jac_vals_fn! = Symbolics.build_function(jac_vals, [x0; p; z]; expression)[2]
        (vals, x0, params, primals) -> jac_vals_fn!(vals, vcat(x0, params, primals))
    end

    (;
        dim = length(coupled_constraints_val),
        parametric_cons,
        parametric_jac = (; jac_rows, jac_cols, parametric_jac_vals),
    )
end

#== Collision Avoidance Experiment ==#

function plot_trajectory!(figure, game, sim_steps; x = 1, y = 1)
    num_players = game.dynamics.subsystems |> length
    environment_axis = create_environment_axis(figure[x, y], game.env; title = "Game")
    colorlist = [colorant"rgba(238, 29, 37, 1.0)", colorant"rgba(114, 171, 74, 1.0)", colorant"rgba(204, 121, 167, 1.0)", colorant"rgba(230, 159, 0, 1.0)",
    colorant"rgba(213, 94, 0, 1.0)", colorant"rgba(86, 180, 233, 1.0)", colorant"rgba(114, 183, 178, 1.0)"]
    
    sim_len = length(sim_steps[1])
    initial_marker_size = 4
    end_marker_size = 20
    size_interval = (end_marker_size - initial_marker_size) / sim_len 
    
    for ii in 1:num_players
        xs = map(1:length(sim_steps[1])) do jj
            sim_steps[1][jj][Block(ii)][1]
        end
        ys = map(1:length(sim_steps[1])) do jj
            sim_steps[1][jj][Block(ii)][2]
        end
        for kk in 1:sim_len
            Makie.scatter!(environment_axis, xs[kk], ys[kk]; color = colorlist[ii], markersize = initial_marker_size + size_interval * kk)
        end 
    end
end

function collect_episode_trajectories()
    base_initial_state = mortar([
        [-2.6, 2.8, 0.1, -0.2],
        [2.8, 2.8, 0.0, 0.0],
        [-2.8, -2.8, 0.2, 0.1],
        [2.8, -2.8, -0.27, 0.1],
    ])
    base_goal = mortar([[3.0, -2.7], [-2.6, -2.7], [2.7, 2.5], [-2.7, 2.5]])
    base_initial_state2 = mortar([
        [-1, 2.5, 0.1, -0.2],
        [1, 2.8, 0.0, 0.0],
        [-2.8, 1, 0.2, 0.1],
        [-2.8, -1, -0.27, 0.1],
    ])
    base_goal2 = mortar([[0.0, -2.7], [2, -2.8], [3.7, 1], [3.7, -1.1]])
    sim_steps_lst = []
    for ii in 1:2
        for jj in 1:4
            if ii == 1
                sim_steps, game = main(; initial_state = base_initial_state + 0.8 * rand(length(base_initial_state)),
                    goal = base_goal + 0.8 * rand(length(base_goal)))
            elseif ii == 2
                sim_steps, game = main(; initial_state = base_initial_state2 + 1.2 * rand(length(base_initial_state2)),
                    goal = base_goal2 + 1.2 * rand(length(base_goal2)))
            end
            push!(sim_steps_lst, sim_steps)
        end
    end
    JLD2.jldsave("sim_steps_lst.jld2"; sim_steps_lst)
end

function plot_episode_trajectories()
    fig = Makie.Figure(; resolution = (2000, 1000))
    sim_steps_lst = JLD2.load("sim_steps_lst.jld2")["sim_steps_lst"]
    environment = PolygonEnvironment(6, 8)
    game = n_player_collision_avoidance(4; environment, min_distance = 1.2)

    for ii in 1:2
        for jj in 1:4
            plot_trajectory!(fig, game, sim_steps_lst[(ii - 1)*4 + jj]; x = ii, y = jj)            
        end
    end
    Makie.save("trajectories.png", fig, px_per_unit = 2)
end

