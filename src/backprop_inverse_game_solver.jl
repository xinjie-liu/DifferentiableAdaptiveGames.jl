#====================== solve inverse game without NN ======================#
function inference_by_backprop(
    mcp_game,
    initial_state,
    τs_observed,
    initial_estimation;
    grad_steps = 150,
)
    function likelihood_cost(observation, goal_estimation)
        solution = MCPGameSolver.solve_mcp_game(mcp_game, initial_state, goal_estimation;)
        τs_solution = solution.variables[1:(length(observation) - length(initial_state))]
        τs_solution = vcat(initial_state, τs_solution)
        norm_sqr(observation - τs_solution)
    end

    backprop_estimation = initial_estimation
    losses = []
    if grad_steps > 0
        for i in 1:grad_steps
            original_cost = likelihood_cost(τs_observed, backprop_estimation)
            gradient = Zygote.gradient(likelihood_cost, τs_observed, backprop_estimation)[2]

            backprop_estimation -= 1e-2 * gradient
            new_cost = likelihood_cost(τs_observed, backprop_estimation)
            append!(losses, new_cost)
        end
    end
    (; backprop_estimation, losses)
end

#====================== learning-based inverse game solver ======================#

struct NNParamsPredictor
    input_dim::Any
    output_dim::Any
    hidden_dim::Any
    optimizer::Any
    model::Any
end

function NNParamsPredictor(input_dim, output_dim, hidden_dim)
    rng = Random.MersenneTwister(1)
    model = Chain(
        Dense(input_dim, hidden_dim, relu; init = glorot_uniform(rng)),
        Dense(hidden_dim, hidden_dim, relu; init = glorot_uniform(rng)),
        Dense(hidden_dim, output_dim; init = glorot_uniform(rng)),
    )
    optimizer = Adam()

    NNParamsPredictor(input_dim, output_dim, hidden_dim, optimizer, model)
end

function (g::NNParamsPredictor)(x)
    predicted_params = g.model(x)
end
