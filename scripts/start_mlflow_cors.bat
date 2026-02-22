@echo off
REM Script para iniciar MLflow con CORS y hosts permitidos configurados (Windows)
REM Uso: start_mlflow_cors.bat [puerto] [ip_publica]

SET PORT=%1
IF "%PORT%"=="" SET PORT=8050

SET PUBLIC_IP=%2
IF "%PUBLIC_IP%"=="" SET PUBLIC_IP=*

echo ==================================
echo Iniciando MLflow con CORS
echo ==================================
echo Puerto: %PORT%
echo IP Publica: %PUBLIC_IP%
echo.

REM Configurar variables de entorno
SET MLFLOW_ENABLE_CORS=true
SET MLFLOW_CORS_ALLOW_ORIGIN=*
SET MLFLOW_CORS_ALLOW_METHODS=GET,POST,PUT,DELETE,OPTIONS
SET MLFLOW_CORS_ALLOW_HEADERS=Content-Type,Authorization,X-Requested-With
SET MLFLOW_CORS_ALLOW_CREDENTIALS=true

REM Configurar hosts permitidos
IF "%PUBLIC_IP%"=="*" (
    SET MLFLOW_ALLOWED_HOSTS=*
    echo Permitiendo TODOS los hosts (modo desarrollo)
) ELSE (
    SET MLFLOW_ALLOWED_HOSTS=localhost,127.0.0.1,localhost:%PORT%,127.0.0.1:%PORT%,%PUBLIC_IP%:%PORT%,*
    echo Hosts permitidos: %MLFLOW_ALLOWED_HOSTS%
)

echo.
echo Iniciando servidor MLflow...
echo.

REM Iniciar MLflow
mlflow server ^
    --host 0.0.0.0 ^
    --port %PORT% ^
    --backend-store-uri ./outputs/hiatal_mlflow/mlruns ^
    --default-artifact-root ./outputs/hiatal_mlflow/mlartifacts ^
    --serve-artifacts

echo.
echo Servidor detenido
pause
