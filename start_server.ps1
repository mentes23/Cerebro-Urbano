param (
    [int]$pid,
    [int]$port
)

# Kill the process by PID
if ($pid) {
    try {
        Stop-Process -Id $pid -Force
        Write-Output "Process $pid has been terminated."
    } catch {
        Write-Output "Error: Process $pid not found."
    }
}

# Start the server on the specified port
if ($port) {
    # Replace this with the correct command to start your server
    Write-Output "Starting server on port $port..."
    # Example: Start-Process "ollama" -ArgumentList "serve --port $port"
    # Note: Replace the above line with the actual command to start your server
}
