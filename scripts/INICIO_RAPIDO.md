# 🚀 Inicio Rápido - MLflow con CORS

## ⚡ Solución Inmediata al Error de Host Header

Si estás viendo estos errores:
- `Invalid Host header - possible DNS rebinding attack detected`
- `Rejected request with invalid Host header: 35.173.82.156:8050`

### 🎯 Solución en 1 paso:

```bash
python scripts/start_mlflow_no_validation.py
```

**¡Eso es todo!** Este script:
- ✅ Deshabilita la validación de host
- ✅ Habilita CORS para todos los orígenes
- ✅ Permite acceso desde cualquier IP
- ✅ Funciona inmediatamente

---

## 📋 Opciones Disponibles

### Opción 1: Python (Recomendado)

```bash
# La más simple - sin validación
python scripts/start_mlflow_no_validation.py

# Con validación pero permitiendo todos los hosts
python scripts/start_mlflow_server.py
```

### Opción 2: Scripts Shell

**Linux/Mac:**
```bash
chmod +x scripts/start_mlflow_cors.sh
./scripts/start_mlflow_cors.sh
```

**Windows:**
```cmd
scripts\start_mlflow_cors.bat
```

### Opción 3: Comando directo

**Linux/Mac:**
```bash
export MLFLOW_DISABLE_HOST_VALIDATION=true
export MLFLOW_ALLOWED_HOSTS="*"
export MLFLOW_ENABLE_CORS=true
export MLFLOW_CORS_ALLOW_ORIGIN="*"

mlflow server --host 0.0.0.0 --port 8050 \
    --backend-store-uri ./outputs/hiatal_mlflow/mlruns \
    --default-artifact-root ./outputs/hiatal_mlflow/mlartifacts \
    --serve-artifacts
```

**Windows (PowerShell):**
```powershell
$env:MLFLOW_DISABLE_HOST_VALIDATION="true"
$env:MLFLOW_ALLOWED_HOSTS="*"
$env:MLFLOW_ENABLE_CORS="true"
$env:MLFLOW_CORS_ALLOW_ORIGIN="*"

mlflow server --host 0.0.0.0 --port 8050 `
    --backend-store-uri ./outputs/hiatal_mlflow/mlruns `
    --default-artifact-root ./outputs/hiatal_mlflow/mlartifacts `
    --serve-artifacts
```

---

## 🔍 Verificar que Funciona

Después de iniciar el servidor, deberías ver:

```
✅ SERVIDOR INICIADO CORRECTAMENTE
📊 Accede a la UI en: http://localhost:8050
🌐 O desde red: http://35.173.82.156:8050
```

Y NO deberías ver más estos errores:
- ❌ `Invalid Host header`
- ❌ `Rejected request with invalid Host header`

---

## 🎓 Entrenar el Modelo

Una vez que el servidor esté corriendo, en otra terminal:

```bash
python scripts/train_model_mlflow.py
```

---

## 🛑 Detener el Servidor

Presiona `Ctrl+C` en la terminal donde está corriendo el servidor.

---

## ⚠️ Nota de Seguridad

Estas configuraciones son para **DESARROLLO** solamente.

Para producción:
1. No uses `MLFLOW_DISABLE_HOST_VALIDATION=true`
2. Especifica hosts exactos en lugar de `*`
3. Implementa autenticación
4. Usa HTTPS

---

## 🆘 Si Aún Tienes Problemas

1. **Verifica que el puerto esté libre:**
   ```bash
   # Linux/Mac
   lsof -i :8050
   
   # Windows
   netstat -ano | findstr :8050
   ```

2. **Actualiza MLflow:**
   ```bash
   pip install --upgrade mlflow
   ```

3. **Reinicia completamente:**
   - Cierra todas las terminales
   - Abre una nueva terminal
   - Ejecuta el script nuevamente

4. **Verifica las variables de entorno:**
   ```bash
   # Linux/Mac
   echo $MLFLOW_DISABLE_HOST_VALIDATION
   
   # Windows PowerShell
   $env:MLFLOW_DISABLE_HOST_VALIDATION
   ```

---

## 📚 Más Información

- [FIX_HOST_HEADER_ERROR.md](FIX_HOST_HEADER_ERROR.md) - Explicación detallada
- [README_MLFLOW.md](README_MLFLOW.md) - Documentación completa

---

## 💡 Tips

- El script `start_mlflow_no_validation.py` es el más confiable
- Siempre inicia el servidor ANTES de entrenar el modelo
- Puedes cambiar el puerto con `--port 8051` si el 8050 está ocupado
- Los datos se guardan en `./outputs/hiatal_mlflow/`

---

## ✅ Checklist

- [ ] Servidor MLflow iniciado sin errores
- [ ] Puedes acceder a http://localhost:8050
- [ ] No ves errores de "Invalid Host header"
- [ ] Puedes ejecutar el entrenamiento

Si todos los puntos están marcados, ¡estás listo! 🎉
