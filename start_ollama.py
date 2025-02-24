import os
import subprocess

def check_and_kill_process(port):
    try:
        # Find the process ID (PID) using the port
        result = subprocess.run(["netstat", "-ano"], capture_output=True, text=True)
        lines = result.stdout.splitlines()
        for line in lines:
            if f":{port}" in line:
                pid = int(line.split()[-1])
                # Kill the process
                os.system(f"taskkill /F /PID {pid}")
                print(f"Process on port {port} with PID {pid} has been terminated.")
                return
        print(f"No process found running on port {port}.")
    except Exception as e:
        print(f"An error occurred: {e}")

def start_ollama():
    port = 11434  # Specify the port you want to use
    check_and_kill_process(port)
    # Start the ollama serve command
    os.system("ollama serve")

if __name__ == "__main__":
    start_ollama()
