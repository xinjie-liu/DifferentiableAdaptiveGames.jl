struct InverseMCPProblem
    fill_F!::Any
    fill_J::Any
    lb::Any
    ub::Any
    horizon::Any
end

function InverseMCPProblem(
    game,
    horizon;
    observation_index = nothing,
    dim_params = 2,
    prior_parmas_dim = 2,
    params_processing_fn = nothing,
)
    num_player = num_players(game)
    lw = game.env.roadway.opts.lane_width
    state_dimension = state_dim(game.dynamics)
    control_dimension = control_dim(game.dynamics)
    trajectory_size = (state_dimension + control_dimension) * horizon
    state_block_dimensions = [state_dim(game.dynamics.subsystems[ii]) for ii in 1:num_player]
    control_block_dimensions = [control_dim(game.dynamics.subsystems[ii]) for ii in 1:num_player]

    z, context_state, x0 = let
        @variables(z[1:trajectory_size], context_state[1:dim_params], x0[1:state_dimension]) .|>
        scalarize
    end
    x0 = BlockVector(x0, state_block_dimensions)

    if !isnothing(params_processing_fn)
        prior_parmas = let
            @variables(prior_parmas[1:prior_parmas_dim]) |> only |> scalarize
        end
        processed_context_state = params_processing_fn(context_state, prior_parmas)
    else
        processed_context_state = context_state
    end

    function trajectory_from_flattened_decision_variables(flattened_z)
        states = let
            future_states = eachcol(
                reshape(flattened_z[1:(state_dimension * horizon)], state_dimension, horizon),
            )
            map(Iterators.flatten(([x0], future_states))) do x_joint
                BlockVector(collect(x_joint), state_block_dimensions)
            end
        end

        control_inputs = let # 20-element vector, each element is 3-blocked vector
            controls = eachcol(
                reshape(
                    flattened_z[(state_dimension * horizon + 1):end],
                    control_dimension,
                    horizon,
                ),
            )
            map(controls) do u_joint
                BlockVector(collect(u_joint), control_block_dimensions)
            end
        end
        (states, control_inputs)
    end

    (xs, us) = trajectory_from_flattened_decision_variables(z)
    (x_idx, u_idx) = trajectory_from_flattened_decision_variables(1:trajectory_size)
    x_idx = x_idx[2:end]

    τ_idx_set = let
        map(1:num_player) do ii
            x_idx_single_player = [idx[Block(ii)] for idx in x_idx]
            u_idx_single_player = [idx[Block(ii)] for idx in u_idx]
            [reduce(vcat, x_idx_single_player); reduce(vcat, u_idx_single_player)]
        end
    end

    cost_per_player = game.cost(xs, us, processed_context_state) .|> scalarize

    #=================================================================#
    # G: equalities of outer problem
    private_constraints = []
    num_private_constraints = []
    G = let
        @variables(G[1:trajectory_size]) |> only |> scalarize
    end
    let
        map(1:num_player) do ii
            environment_constraints = get_constraints(game.env, ii)
            subdynamics = game.dynamics.subsystems[ii]
            private_inequality_constraints = let
                state_box_constraints = get_constraints_from_box_bounds(state_bounds(subdynamics))
                control_box_constraints =
                    get_constraints_from_box_bounds(control_bounds(subdynamics))

                ec = mapreduce(x -> environment_constraints(x[Block(ii)]), vcat, xs)
                sc = mapreduce(x -> state_box_constraints(x[Block(ii)]), vcat, xs)
                cc = mapreduce(x -> control_box_constraints(x[Block(ii)]), vcat, us)
                [ec; sc; cc]
            end
            append!(private_constraints, private_inequality_constraints)
            append!(num_private_constraints, length(private_inequality_constraints))
        end
    end

    # set up the gradient of lagrangian
    private_duals = []
    # private_duals = let
    #     @variables(μ[1:sum(num_private_constraints)]) |> only |> scalarize
    # end

    for ii in 1:num_player
        # dual_offset = (ii > 1) ? sum(num_private_constraints[1:(ii - 1)]) : 0
        local_lagrangian = cost_per_player[ii] #-
        # private_duals[(dual_offset + 1):(dual_offset + num_private_constraints[ii])]' *
        # private_constraints[(dual_offset + 1):(dual_offset + num_private_constraints[ii])]
        G[τ_idx_set[ii]] = Symbolics.gradient(local_lagrangian, z[τ_idx_set[ii]])
    end

    #shared dynamics equality constraints
    shared_dynamics_equality_constraints = mapreduce(vcat, 1:horizon) do t
        game.dynamics(xs[t], us[t], t) .- xs[t + 1]
    end
    shared_dynamics_duals = let
        @variables(γ[1:length(shared_dynamics_equality_constraints)]) |> only |> scalarize
    end
    shared_dynamics_jacobian = Symbolics.sparsejacobian(shared_dynamics_equality_constraints, z)
    G -= shared_dynamics_jacobian' * shared_dynamics_duals

    # # coupled inequality constraints
    # if !isnothing(game.coupling_constraints)
    #     coupled_inequality_constraints = game.coupling_constraints(xs, us)
    #     coupled_duals = let
    #         @variables(λ[1:length(coupled_inequality_constraints)]) |> only |> scalarize
    #     end
    #     coupled_constraints_jacobian = Symbolics.sparsejacobian(coupled_inequality_constraints, z)
    #     G -= coupled_constraints_jacobian' * coupled_duals

    #     orthogonality_shared_inequality = coupled_duals' * coupled_inequality_constraints
    #     G = vcat(G, orthogonality_shared_inequality)
    #     z = vcat(context_state, z, private_duals, coupled_duals, shared_dynamics_duals)
    # else
    #     z = vcat(context_state, z, private_duals, shared_dynamics_duals) 
    # end
    z = vcat(context_state, z, shared_dynamics_duals)

    # concatenate inequalities of the inner problem as well as the corresponding decision variables
    # orthogonality_private_inequality = private_duals' * private_constraints
    # G = vcat(G, orthogonality_private_inequality)
    G = vcat(G, shared_dynamics_equality_constraints)
    inner_size = length(z)
    #============================================================#

    #============================================================#
    # F: inequalities of the outer problem    
    # orthogonality_private_inequality = -(private_duals' * private_constraints) .+ 1
    F = Symbolics.Num[]
    # bound the goal estimation range
    vel_lb = game.dynamics.subsystems[1].state_bounds.lb[3]
    vel_ub = game.dynamics.subsystems[1].state_bounds.ub[3]
    let 
        for ii in 1:num_player - 1
            offset = (ii - 1) * Int(dim_params / (num_player - 1))
            append!(F, lw + context_state[offset + 1])
            append!(F, lw - context_state[offset + 1])
            append!(F, context_state[offset + 2])
            append!(F, vel_ub - context_state[offset + 2])
        end
    end
    # append!(F, private_constraints)
    # append!(F, private_duals)
    # # append!(F, orthogonality_private_inequality)
    # # coupled inequalities
    # if !isnothing(game.coupling_constraints)
    #     append!(F, coupled_inequality_constraints)
    #     append!(F, coupled_duals)
    #     # append!(F, orthogonality_shared_inequality)
    # end
    #============================================================#

    #============================================================#
    # set up the outer problem
    observation = let
        observation_size =
            !isnothing(observation_index) ? length(observation_index) : trajectory_size
        @variables(y[1:observation_size]) |> only |> scalarize
    end
    if isnothing(observation_index)
        observation_index = Vector(1:trajectory_size)
    end

    out_cost = let
        true_observation_index = observation_index .+ length(context_state)
        norm_sqr(observation - z[true_observation_index])
            # estimation regularization
            #+ 10 * norm_sqr(max.(zeros(dim_params), z[1:dim_params] - repeat([lw, 1.0], num_player - 1)))
            #+ 10 * norm_sqr(max.(zeros(dim_params), repeat([-lw, -1.0], num_player - 1) - z[1:dim_params]))
    end
    
    outer_equality_duals = let
        @variables(α[1:length(G)]) |> only |> scalarize
    end
    outer_inequality_duals = let
        @variables(β[1:length(F)]) |> only |> scalarize
    end
    outer_lagrangian = out_cost - outer_equality_duals' * G - outer_inequality_duals' * F
    f = Symbolics.gradient(outer_lagrangian, z)
    z = vcat(z, outer_equality_duals, outer_inequality_duals)
    f = vcat(f, G, F)
    lb = vcat(fill(-Inf, (length(z) - length(F))), fill(0.0, length(F)))
    # lb = fill(-Inf, length(z))
    ub = fill(Inf, length(z))

    # F vector parametriztion
    if isnothing(params_processing_fn)
        fill_F! = let
            F! = Symbolics.build_function(f, [z; x0; observation]; expression = Val{false})[2]
            (vals, z, x0, y) -> F!(vals, vcat(z, x0, y))
        end
    else
        fill_F! = let
            F! = Symbolics.build_function(
                f,
                [z; x0; observation; prior_parmas];
                expression = Val{false},
            )[2]
            (vals, z, x0, y, prior_parmas) -> F!(vals, vcat(z, x0, y, prior_parmas))
        end
    end

    # J matrix: jacobian of F vector
    J = Symbolics.sparsejacobian(f, z)
    (J_rows, J_cols, _) = findnz(J)
    J_constant_entries = ParametricMCPs.get_constant_entries(J, z)

    if isnothing(params_processing_fn)
        fill_J! = let
            _J! = Symbolics.build_function(J, [z; x0; observation]; expression = Val{false})[2]
            ParametricMCPs.SparseFunction(
                J_rows,
                J_cols,
                size(J),
                J_constant_entries,
            ) do result, z, x0, y
                _J!(result, vcat(z, x0, y))
            end
        end
    else
        fill_J! = let
            _J! = Symbolics.build_function(
                J,
                [z; x0; observation; prior_parmas];
                expression = Val{false},
            )[2]
            ParametricMCPs.SparseFunction(
                J_rows,
                J_cols,
                size(J),
                J_constant_entries,
            ) do result, z, x0, y, prior_parmas
                _J!(result, vcat(z, x0, y, prior_parmas))
            end
        end
    end

    #============================================================#
    InverseMCPProblem(fill_F!, fill_J!, lb, ub, horizon)
end

function solve_inverse_mcp_game(
    inverse_problem::InverseMCPProblem,
    game,
    τ_observed,
    x0;
    observation_index = nothing,
    dim_params = 2,
    initial_guess = nothing,
    prior_parmas = nothing,
    horizon,
)
    (; dynamics) = game
    if !isnothing(initial_guess)
        z = initial_guess.variables
    else # start with the observation
        z = zeros(length(inverse_problem.lb))
        control_block_dimensions =
            [control_dim(dynamics.subsystems[ii]) for ii in 1:num_players(game)]
        state_dimension = state_dim(dynamics)
        dummy_strategy =
            (x, t) -> BlockVector(zeros(sum(control_block_dimensions)), control_block_dimensions)
        xs = rollout(dynamics, dummy_strategy, x0, horizon + 1).xs[2:end]
        xs = reduce(vcat, xs)
        z[(dim_params + 1):(dim_params + state_dimension * horizon)] = xs
    end

    lb = inverse_problem.lb
    ub = inverse_problem.ub

    function F(n, z, f)
        if isnothing(prior_parmas)
            inverse_problem.fill_F!(f, z, x0, τ_observed)
        else
            inverse_problem.fill_F!(f, z, x0, τ_observed, prior_parmas)
        end

        Cint(0)
    end

    function J(n, nnz, z, col, len, row, data)
        if isnothing(prior_parmas)
            inverse_problem.fill_J(inverse_problem.fill_J.result_buffer, z, x0, τ_observed)
        else
            inverse_problem.fill_J(inverse_problem.fill_J.result_buffer, z, x0, τ_observed, prior_parmas)
        end
        ParametricMCPs._coo_from_sparse!(col, len, row, data, inverse_problem.fill_J.result_buffer)

        Cint(0)
    end

    # count the non-zeros in J matrix
    nnz = length(inverse_problem.fill_J.rows)

    status, variables, info = PATHSolver.solve_mcp(
        F,
        J,
        lb,
        ub,
        z;
        silent = true,
        nnz,
        cumulative_iteration_limit = 100_000,
        use_basics = true,
        use_start = true,
        jacobian_structure_constant = true,
        jacobian_data_contiguous = true,
        jacobian_linear_elements = inverse_problem.fill_J.constant_entries,
    )

    if status != PATHSolver.MCP_Solved
        @warn "MCP not cleanly solved. Final solver status is $(status)."
    end

    (; variables, status, info)
end
