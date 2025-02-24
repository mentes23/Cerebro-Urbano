@echo off
FOR /F "tokens=5" %%P IN ('netstat -a -n -o ^| findstr :11434') DO TaskKill.exe /PID %%P /F
