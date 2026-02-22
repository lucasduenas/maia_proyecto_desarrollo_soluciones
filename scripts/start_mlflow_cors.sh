#!/bin/bash

# Script para iniciar MLflow con CORS y hosts permitidos configurados
# Uso: ./start_mlflow_cors.sh [puerto] [ip_publica]

PORT=${1:-8050}
PUBLIC_IP=${2:-"*"}

echo "=================================="
echo "Iniciando MLflow con CORS"
echo "=================================="
echo "Puerto: $PORT"
echo "IP Pública: $PUBLIC_IP"
echo ""

# Configurar variables de entorno
export MLFLOW_ENABLE_CORS=true
export MLFLOW_CORS_ALLOW_ORIGIN="*"
export MLFLOW_CORS_ALLOW_METHODS="GET,POST,PUT,DELETE,OPTIONS"
export MLFLOW_CORS_ALLOW_HEADERS="Content-Type,Authorization,X-Requested-With"
export MLFLOW_CORS_ALLOW_CREDENTIALS="true"

# Configurar hosts permitidos
if [ "$PUBLIC_IP" = "*" ]; then
    export MLFLOW_ALLOWED_HOSTS="*"
    echo "✅ Permitiendo TODOS los hosts (modo desarrollo)"
else
    export MLFLOW_ALLOWED_HOSTS="localhost,127.0.0.1,localhost:$PORT,127.0.0.1:$PORT,$PUBLIC_IP:$PORT,*"
    echo "✅ Hosts permitidos: $MLFLOW_ALLOWED_HOSTS"
fi

echo ""
echo "Iniciando servidor MLflow..."
echo ""

# Iniciar MLflow
mlflow server \
    --host 0.0.0.0 \
    --port $PORT \
    --backend-store-uri ./outputs/hiatal_mlflow/mlruns \
    --default-artifact-root ./outputs/hiatal_mlflow/mlartifacts \
    --serve-artifacts

echo ""
echo "Servidor detenido"
