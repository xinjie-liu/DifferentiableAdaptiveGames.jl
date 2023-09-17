mutable struct OptiTrackCommunication
    task::Task
    shared_state::Any
    connection::OptiTrackConnection

    function OptiTrackCommunication(ids; origin_translation::AbstractVector{Float64} = [0.0, 0.0])
        connection = OptiTrackConnection()
        shared_state = SharedState([[0.0, 0.0, 0.0] for _ in ids])
        task = errormonitor(
            Threads.@spawn listen_for_optitrack(connection, shared_state; ids, origin_translation)
        )
        communication = new(task, shared_state, connection)
        finalizer(close, communication)
        communication
    end
end

function Base.close(communication::OptiTrackCommunication)
    @async close(communication.connection)
end

function listen_for_optitrack(connection, shared_state; ids, origin_translation)
    @info "Listening for OptiTrack messages..."
    while isopen(connection)
        message = try
            OptiTrack.receive(connection)
        catch e
            @info "Failed to receive message from OptiTrack: $(e)"
            break
        end
        current_state = take!(shared_state)
        next_state = map(enumerate(ids)) do (player_index, id)
            if isnothing(id)
                return current_state[player_index]
            end

            body = OptiTrack.get_rigid_body_with_id(id, message)
            angles =
                RotXZY(
                    QuatRotation(
                        body.orientation.w,
                        body.orientation.x,
                        body.orientation.y,
                        body.orientation.z,
                    ),
                )
            yaw = angles.theta3
            [body.position.x + origin_translation[1], -body.position.z + origin_translation[2], yaw]
        end
        put!(shared_state, next_state)
    end
    @info "Finished OptiTrack task"
end

function robot_state_from_optitrack!(communication::OptiTrackCommunication)
    state = take!(communication.shared_state)
    mortar([
        begin
            x, y, θ = robot
            [x, y, θ]
        end for robot in state
    ])
end
