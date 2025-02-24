import subprocess

def kill_process_on_port(port):
    result = subprocess.run(['netstat', '-aon'], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if f":{port}" in line:
            parts = line.split()
            if len(parts) >= 5:
                pid = parts[-1]
                print(f"Encontrado processo {pid} usando a porta {port}. Encerrando...")
                subprocess.run(['taskkill', '/F', '/PID', pid])
                return pid
    return None

if __name__ == '__main__':
    port = 11434
    pid = kill_process_on_port(port)
    if pid:
        print(f"Processo {pid} encerrado na porta {port}.")
    else:
        print(f"Nenhum processo encontrado utilizando a porta {port}.")
