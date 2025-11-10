@echo off
setlocal enabledelayedexpansion

REM === Ruta del proyecto (carpeta donde está este .bat) ===
set REPO=%~dp0

REM === Ruta del entorno virtual dentro del repo ===
set VENV=%REPO%\.venv

REM === Crear carpeta logs si no existe ===
set LOGDIR=%REPO%logs
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

REM === Formato de fecha (YYYY-MM-DD) ===
for /f "tokens=1-3 delims=/" %%a in ("%DATE%") do (
    set LOGFILE=%LOGDIR%\pipeline_%%c-%%a-%%b.log
)

REM === Forzar UTF-8 en Python ===
set PYTHONUTF8=1

REM === Lockfile para evitar ejecuciones simultáneas ===
set LOCK=%REPO%data\PIPELINE_TICK.lock
if exist "%LOCK%" (
  echo [%DATE% %TIME%] Pipeline ya está corriendo. Saliendo.>> "%LOGFILE%"
  exit /b 0
)
echo %DATE% %TIME% > "%LOCK%"

REM === Ejecutar usando Python del entorno virtual ===
pushd "%REPO%"
"%VENV%\Scripts\python.exe" src\pipeline_tick.py >> "%LOGFILE%" 2>&1
set ERR=%ERRORLEVEL%
del "%LOCK%" 2>nul
popd

exit /b %ERR%
