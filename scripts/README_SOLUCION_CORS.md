# 🎯 Solución Definitiva al Error de CORS y Host Header

## 🔴 El Problema

Estás viendo estos errores:

```
❌ Invalid Host header - possible DNS rebinding attack detected
❌ INFO: 45.238.183.248:3822 - "GET / HTTP/1.1" 403 Forbidden
❌ WARNING mlflow.server.fastapi_security: Rejected request with invalid Host header: 35.173.82.156:8050
```

## ✅ La Solución (3 opciones, de más fácil a más compleja)

### 🥇 Opción 1: Script Definitivo (RECOMENDADO)

```bash
python scripts/mlflow_dev.py
```

**Este es el script más confiable.** Deshabilita todas las validaciones y funciona siempre.

### 🥈 Opción 2: Script Sin Validación

```bash
python scripts/start_mlflow_no_validation.py
```

### 🥉 Opción 3: Script Original Actualizado

```bash
python scripts/start_mlflow_server.py
```

---

## 📝 Comparación de Scripts

| Script | Facilidad | Seguridad | Recomendado para |
|--------|-----------|-----------|------------------|
| `mlflow_dev.py` | ⭐⭐⭐⭐⭐ | ⚠️ Desarrollo | **Desarrollo local** |
| `start_mlflow_no_validation.py` | ⭐⭐⭐⭐ | ⚠️ Desarrollo | Desarrollo local |
| `start_mlflow_server.py` | ⭐⭐⭐ | ✅ Configurable | Desarrollo/Testing |
| Scripts shell (.sh/.bat) | ⭐⭐ | ⚠️ Desarrollo | Usuarios avanzados |

---

## 🚀 Uso Completo

### Paso 1: Iniciar el servidor

```bash
# Opción más simple
python scripts/mlflow_dev.py

# O cualquiera de las otras opciones
```

Deberías ver:

```
✅ SERVIDOR LISTO
🌐 Accede desde:
   • http://localhost:8050
   • http://127.0.0.1:8050
   • http://<tu-ip>:8050
```

### Paso 2: Entrenar el modelo

En **otra terminal**:

```bash
python scripts/train_model_mlflow.py
```

### Paso 3: Ver resultados

Abre tu navegador en: http://localhost:8050

---

## 🔧 ¿Por Qué Funciona?

Los scripts configuran estas variables de entorno:

```bash
# Deshabilitar validaciones de seguridad
MLFLOW_DISABLE_HOST_VALIDATION=true
MLFLOW_DISABLE_CSRF_PROTECTION=true

# Permitir todos los hosts
MLFLOW_ALLOWED_HOSTS=*

# Habilitar CORS para todos los orígenes
MLFLOW_ENABLE_CORS=true
MLFLOW_CORS_ALLOW_ORIGIN=*
MLFLOW_CORS_ALLOW_METHODS=*
MLFLOW_CORS_ALLOW_HEADERS=*
```

---

## ⚠️ Importante: Seguridad

### ✅ Para Desarrollo (tu caso actual)

```bash
# Está bien usar:
MLFLOW_DISABLE_HOST_VALIDATION=true
MLFLOW_ALLOWED_HOSTS=*
```

### ❌ Para Producción

```bash
# NO uses:
MLFLOW_DISABLE_HOST_VALIDATION=true  # ❌ Peligroso

# SÍ usa:
MLFLOW_ALLOWED_HOSTS="tu-dominio.com,api.tu-dominio.com"  # ✅ Seguro
```

---

## 🐛 Troubleshooting

### El puerto está ocupado

```bash
# Cambiar puerto
python scripts/mlflow_dev.py --port 8051

# O matar el proceso
# Linux/Mac:
lsof -i :8050 | grep LISTEN | awk '{print $2}' | xargs kill -9

# Windows:
netstat -ano | findstr :8050
taskkill /PID <PID> /F
```

### Las variables de entorno no se aplican

```bash
# Verificar
echo $MLFLOW_DISABLE_HOST_VALIDATION  # Linux/Mac
$env:MLFLOW_DISABLE_HOST_VALIDATION   # Windows PowerShell

# Si está vacío, el script no está funcionando
# Usa mlflow_dev.py que es más confiable
```

### Sigue sin funcionar

```bash
# 1. Actualizar MLflow
pip install --upgrade mlflow

# 2. Reiniciar completamente
# - Cerrar TODAS las terminales
# - Abrir una nueva
# - Ejecutar: python scripts/mlflow_dev.py

# 3. Verificar versión
mlflow --version  # Debe ser >= 2.0
```

---

## 📊 Verificación

Después de iniciar el servidor, verifica:

1. ✅ No ves errores de "Invalid Host header"
2. ✅ Puedes acceder a http://localhost:8050
3. ✅ La UI de MLflow carga correctamente
4. ✅ Puedes ejecutar el entrenamiento sin errores

---

## 🎓 Próximos Pasos

Una vez que el servidor funcione:

1. **Entrenar el modelo:**
   ```bash
   python scripts/train_model_mlflow.py
   ```

2. **Ver experimentos:**
   - Abre http://localhost:8050
   - Navega a "Experiments"
   - Verás "hernia-hiatal-production"

3. **Ver métricas:**
   - Click en el experimento
   - Verás las métricas: AUC, F1, Accuracy, etc.

4. **Descargar modelo:**
   - Click en el run
   - Pestaña "Artifacts"
   - Descarga el modelo entrenado

---

## 📚 Archivos de Ayuda

- `INICIO_RAPIDO.md` - Guía rápida de inicio
- `FIX_HOST_HEADER_ERROR.md` - Explicación detallada del error
- `README_MLFLOW.md` - Documentación completa

---

## 💡 Tips Finales

1. **Siempre usa `mlflow_dev.py` para desarrollo** - Es el más confiable
2. **Inicia el servidor ANTES de entrenar** - El modelo necesita el servidor corriendo
3. **Usa Ctrl+C para detener** - No cierres la terminal directamente
4. **Los datos persisten** - Están en `./outputs/hiatal_mlflow/`
5. **Puedes cambiar el puerto** - Si 8050 está ocupado

---

## ✅ Checklist Final

- [ ] Instalé MLflow: `pip install mlflow`
- [ ] Ejecuté: `python scripts/mlflow_dev.py`
- [ ] Vi el mensaje "✅ SERVIDOR LISTO"
- [ ] Puedo acceder a http://localhost:8050
- [ ] No veo errores de "Invalid Host header"
- [ ] Puedo ejecutar el entrenamiento

Si todos los puntos están marcados, **¡el problema está resuelto!** 🎉

---

## 🆘 Soporte

Si después de todo esto aún tienes problemas:

1. Verifica que MLflow esté instalado: `pip list | grep mlflow`
2. Verifica la versión de Python: `python --version` (debe ser >= 3.8)
3. Revisa los logs completos del servidor
4. Intenta con un puerto diferente: `python scripts/mlflow_dev.py --port 8051`

---

**Última actualización:** 2026-02-22  
**Versión:** 2.0 - Solución definitiva
