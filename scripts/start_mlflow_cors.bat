@echo off
REM Script para iniciar MLflow con CORS y SIN validación de host (Windows)
REM Uso: start_mlflow_cors.bat [puerto]

SET PORT=%1
IF "%PORT%"=="" SET PORT=8050

echo ==================================
echo Iniciando MLflow con CORS
echo SIN validacion de host
echo ==================================
echo Puerto: %PORT%
echo.

REM Configurar TODAS las variables de entorno necesarias
SET MLFLOW_ENABLE_CORS=true
SET MLFLOW_CORS_ALLOW_ORIGIN=*
SET MLFLOW_CORS_ALLOW_METHODS=GET,POST,PUT,DELETE,OPTIONS,PATCH,HEAD
SET MLFLOW_CORS_ALLOW_HEADERS=*
SET MLFLOW_CORS_ALLOW_CREDENTIALS=true
SET MLFLOW_CORS_EXPOSE_HEADERS=*
SET MLFLOW_CORS_MAX_AGE=3600

REM Hosts permitidos - TODOS
SET MLFLOW_ALLOWED_HOSTS=*

REM Deshabilitar validaciones (SOLO DESARROLLO)
SET MLFLOW_DISABLE_HOST_VALIDATION=true
SET MLFLOW_DISABLE_CSRF_PROTECTION=true
SET MLFLOW_TRACKING_INSECURE_TLS=true

echo Variables de entorno configuradas:
echo    MLFLOW_ENABLE_CORS=%MLFLOW_ENABLE_CORS%
echo    MLFLOW_CORS_ALLOW_ORIGIN=%MLFLOW_CORS_ALLOW_ORIGIN%
echo    MLFLOW_ALLOWED_HOSTS=%MLFLOW_ALLOWED_HOSTS%
echo    MLFLOW_DISABLE_HOST_VALIDATION=%MLFLOW_DISABLE_HOST_VALIDATION%
echo.
echo ADVERTENCIA: Validacion de host deshabilitada
echo    Solo usar en desarrollo
echo.
echo Iniciando servidor MLflow...
echo.

REM Iniciar MLflow
mlflow server ^
    --host 0.0.0.0 ^
    --port %PORT% ^
    --backend-store-uri ./outputs/hiatal_mlflow/mlruns ^
    --default-artifact-root ./outputs/hiatal_mlflow/mlartifacts ^
    --serve-artifacts ^
    --gunicorn-opts "--timeout 120 --workers 1"

echo.
echo Servidor detenido
pause
