mutable struct SharedState{T}
    lock::Threads.SpinLock
    data::T
    function SharedState(data::T) where {T}
        lock = Threads.SpinLock()
        new{T}(lock, data)
    end
end

function Base.put!(shared_state::SharedState, new_data)
    lock(shared_state.lock)
    shared_state.data = new_data
    unlock(shared_state.lock)
    nothing
end

function Base.take!(shared_state::SharedState)
    lock(shared_state.lock)
    data = shared_state.data
    unlock(shared_state.lock)
    data
end
