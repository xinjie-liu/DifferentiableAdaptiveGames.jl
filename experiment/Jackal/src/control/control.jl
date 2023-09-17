function unicycle_controls(unicycle_state, pointmass_state; Kx = 2, Ky = 10, Kθ = 5)
    x_c = unicycle_state[1]
    y_c = unicycle_state[2]
    θ_c = unicycle_state[3]
    px = pointmass_state[1]
    py = pointmass_state[2]
    pvx = pointmass_state[3]
    pvy = pointmass_state[4]

    current_pose = [x_c, y_c, θ_c]
    pointmass_velocity_direction = atan(pvy, pvx)
    reference_pose = [px, py, pointmass_velocity_direction]
    vᵣ = norm([pvx, pvy])
    ωᵣ = 0.0

    Tₑ = [
        cos(θ_c) sin(θ_c) 0
        -sin(θ_c) cos(θ_c) 0
        0 0 1
    ]
    error_pose = Tₑ * (reference_pose - current_pose)
    xₑ = error_pose[1]
    yₑ = error_pose[2]
    θₑ = error_pose[3]

    v = vᵣ * cos(θₑ) + Kx * xₑ
    ω = ωᵣ + vᵣ * (Ky * yₑ + Kθ * sin(θₑ))
    [v, ω]
end

Base.@kwdef struct UnicycleDynamics <: AbstractDynamics
    dt::Float64 = 0.1
end

function TrajectoryGamesBase.horizon(sys::UnicycleDynamics)
    ∞
end

function TrajectoryGamesBase.state_dim(dynamics::UnicycleDynamics)
    3
end

function TrajectoryGamesBase.control_dim(dynamics::UnicycleDynamics)
    2
end

function (sys::UnicycleDynamics)(state, control, t)
    θ = state[3]

    A = [
        1.0 0.0 0.0
        0.0 1.0 0.0
        0.0 0.0 1.0
    ]

    B = [
        cos(θ)*sys.dt 0.0
        sin(θ)*sys.dt 0.0
        0.0 sys.dt
    ]

    new_state = A * state + B * control
    new_state[3] = wrap_pi(new_state[3])
    new_state
end

function TrajectoryGamesBase.state_bounds(dynamics::UnicycleDynamics)
    (; lb = fill(-Inf, 3), ub = fill(Inf, 3))
end

function TrajectoryGamesBase.control_bounds(dynamics::UnicycleDynamics)
    (; lb = fill(-0.2, 2), ub = fill(0.2, 2))
end

wrap_pi(x) = mod2pi(x + pi) - pi

function rollout_robot_dynamics(;
    initial_state,
    pointmasses_trajectory,
    rollout_horizon,
    look_ahead = 1,
)
    n_players = blocksize(initial_state, 1)
    strategy = JointStrategy([
        (x, t) -> let
            control = unicycle_controls(
                x[Block(player_i)],
                pointmasses_trajectory[t + look_ahead][Block(player_i)];
            )
            #clamp.(control, [-0.2, -π], [0.2, π])
        end for player_i in 1:n_players
    ])
    dynamics = ProductDynamics([UnicycleDynamics() for _ in 1:n_players])
    xs, us, _ = rollout(dynamics, strategy, initial_state, rollout_horizon - look_ahead)
    (; xs, us)
end
