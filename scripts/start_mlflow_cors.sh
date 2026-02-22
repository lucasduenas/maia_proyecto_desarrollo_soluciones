#!/bin/bash

# Script para iniciar MLflow con CORS y SIN validación de host
# Uso: ./start_mlflow_cors.sh [puerto]

PORT=${1:-8050}

echo "=================================="
echo "Iniciando MLflow con CORS"
echo "SIN validación de host"
echo "=================================="
echo "Puerto: $PORT"
echo ""

# Configurar TODAS las variables de entorno necesarias
export MLFLOW_ENABLE_CORS=true
export MLFLOW_CORS_ALLOW_ORIGIN="*"
export MLFLOW_CORS_ALLOW_METHODS="GET,POST,PUT,DELETE,OPTIONS,PATCH,HEAD"
export MLFLOW_CORS_ALLOW_HEADERS="*"
export MLFLOW_CORS_ALLOW_CREDENTIALS="true"
export MLFLOW_CORS_EXPOSE_HEADERS="*"
export MLFLOW_CORS_MAX_AGE="3600"

# Hosts permitidos - TODOS
export MLFLOW_ALLOWED_HOSTS="*"

# Deshabilitar validaciones (SOLO DESARROLLO)
export MLFLOW_DISABLE_HOST_VALIDATION="true"
export MLFLOW_DISABLE_CSRF_PROTECTION="true"
export MLFLOW_TRACKING_INSECURE_TLS="true"

echo "✅ Variables de entorno configuradas:"
echo "   MLFLOW_ENABLE_CORS=$MLFLOW_ENABLE_CORS"
echo "   MLFLOW_CORS_ALLOW_ORIGIN=$MLFLOW_CORS_ALLOW_ORIGIN"
echo "   MLFLOW_ALLOWED_HOSTS=$MLFLOW_ALLOWED_HOSTS"
echo "   MLFLOW_DISABLE_HOST_VALIDATION=$MLFLOW_DISABLE_HOST_VALIDATION"
echo ""
echo "⚠️  ADVERTENCIA: Validación de host deshabilitada"
echo "   Solo usar en desarrollo"
echo ""
echo "Iniciando servidor MLflow..."
echo ""

# Iniciar MLflow
mlflow server \
    --host 0.0.0.0 \
    --port $PORT \
    --backend-store-uri ./outputs/hiatal_mlflow/mlruns \
    --default-artifact-root ./outputs/hiatal_mlflow/mlartifacts \
    --serve-artifacts \
    --gunicorn-opts "--timeout 120 --workers 1"

echo ""
echo "Servidor detenido"
