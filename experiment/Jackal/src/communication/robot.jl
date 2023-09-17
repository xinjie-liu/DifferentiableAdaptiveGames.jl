mutable struct RobotCommunication
    task::Task
    channel::Channel{Any}

    function RobotCommunication(address, port::Integer)
        channel = Channel(1)
        @info "Connecting to robot at $(address):$(port) ..."
        socket = Sockets.connect(address, port)
        task = errormonitor(Threads.@spawn robot_communication(channel, socket))
        communication = new(task, channel)
        finalizer(close, communication)
        communication
    end
end

function Base.close(communication::RobotCommunication)
    @async put!(communication.channel, :close)
end

function robot_communication(channel, socket)
    @info "Communicating to robot..."
    while true
        us = take!(channel)
        if us === :close
            @info "Closing robot communication"
            break
        end
        msg = """{ "us": $(us) }\n"""
        write(socket, msg)
    end
    close(socket)
    @info "Finished Robot task"
end

function stop_robots(robot_communications; rollout_horizon = 20)
    n_robots = length(robot_communications)
    us = [[0.0, 0.0] for _ in 1:rollout_horizon]
    for connection in robot_communications
        if isnothing(connection)
            continue
        end
        put!(connection.channel, us)
    end
end
