"""
Script de entrenamiento de modelo de detección de hernia hiatal con MLFlow
Basado en el notebook hiatal_model_train.ipynb
"""

import sys
import json
import random
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as T
from torch import nn
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchvision.models import densenet121, DenseNet121_Weights

from PIL import Image, ImageOps

import mlflow
import mlflow.pytorch

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    accuracy_score,
    f1_score,
)


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

def configurar_experimento():
    """Configura todos los parámetros del experimento"""
    
    # Rutas
    RAIZ = Path.cwd().resolve()
    if RAIZ.name == 'scripts':
        RAIZ = RAIZ.parent
    
    config = {
        # Rutas
        'RAIZ': RAIZ,
        'RUTA_IMAGENES': RAIZ / 'data' / 'images',
        'RUTA_SALIDA': RAIZ / 'outputs' / 'hiatal_mlflow',
        
        # Hiperparámetros
        'SEMILLA': 42,
        'N_SPLITS': 5,
        'TAM_IMAGEN': 512,
        'TAM_BATCH': 8,
        'EPOCAS': 30,
        'EPOCAS_CONGELADAS': 2,
        'PACIENCIA': 8,
        
        # Arquitectura
        'USAR_PREENTRENADO': True,
        'USAR_ROI': True,
        'USAR_AUTOCONTRASTE': True,
        'DROPOUT': 0.25,
        
        # Aumentación
        'USAR_AUMENTACION_TRAIN': True,
        'FACTOR_MUESTRAS_ENTRENAMIENTO_POR_EPOCA': 2,
        'ROTACION_MAX_GRADOS': 8,
        'TRASLACION_MAX': 0.03,
        'ESCALA_MIN': 0.97,
        'ESCALA_MAX': 1.03,
        'USAR_FLIP_HORIZONTAL_TRAIN': False,
        'P_AUMENTO_INTENSIDAD': 0.35,
        'JITTER_BRILLO': 0.12,
        'JITTER_CONTRASTE': 0.15,
        
        # Evaluación
        'OPTIMIZAR_UMBRAL': True,
        'RUN_CV_COMPLETO': True,
        'MOSTRAR_GRADCAM': False,
        'USAR_ENSEMBLE_CV': True,
        
        # MLflow
        'USAR_MLFLOW': True,
        'MLFLOW_RUN_NAME': f"hiatal_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'MLFLOW_EXPERIMENTO': 'hernia-hiatal-production',
        'MLFLOW_TRACKING_URI': 'http://localhost:8050',
        'MLFLOW_REGISTRAR_MODELO': True,
        
        # MLflow Server (para iniciar automáticamente)
        'MLFLOW_AUTO_START': False,  # Cambiar a True para iniciar automáticamente
        'MLFLOW_HOST': '0.0.0.0',  # 0.0.0.0 permite acceso desde cualquier IP
        'MLFLOW_PORT': 8050,
        'MLFLOW_BACKEND_STORE': None,  # Se configurará automáticamente
        'MLFLOW_ARTIFACT_ROOT': None,  # Se configurará automáticamente
    }
    
    # Crear directorio de salida
    config['RUTA_SALIDA'].mkdir(parents=True, exist_ok=True)
    
    # Configurar rutas de MLflow
    config['MLFLOW_BACKEND_STORE'] = str(config['RUTA_SALIDA'] / 'mlruns')
    config['MLFLOW_ARTIFACT_ROOT'] = str(config['RUTA_SALIDA'] / 'mlartifacts')
    
    # Dispositivo
    config['DISPOSITIVO'] = torch.device('cuda') if torch.cuda.is_available() else (
        torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    )
    config['USAR_AMP'] = config['DISPOSITIVO'].type == 'cuda'
    
    return config


# ============================================================================
# UTILIDADES
# ============================================================================

EXTENSIONES_VALIDAS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
ETIQUETAS_NORMAL = {'0', 'normal', 'sano', 'healthy', 'neg', 'negative'}
MEDIA_IMAGENET = [0.485, 0.456, 0.406]
STD_IMAGENET = [0.229, 0.224, 0.225]


def fijar_semilla(seed: int):
    """Fija la semilla para reproducibilidad"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class Muestra:
    """Estructura de una muestra supervisada"""
    ruta: Path
    etiqueta: int
    id_paciente: str


class RecorteRetrocardiaco:
    """Recorte de región de interés"""
    def __init__(self, x1=0.2, x2=0.8, y1=0.15, y2=0.98):
        self.x1, self.x2, self.y1, self.y2 = x1, x2, y1, y2

    def __call__(self, imagen: Image.Image):
        ancho, alto = imagen.size
        return imagen.crop((
            int(self.x1 * ancho), int(self.y1 * alto),
            int(self.x2 * ancho), int(self.y2 * alto)
        ))


class ConjuntoHiatal(Dataset):
    """Dataset de PyTorch para imágenes de hernia hiatal"""
    def __init__(self, muestras, indices, transformacion, usar_autocontraste=True):
        self.muestras = [muestras[i] for i in indices]
        self.transformacion = transformacion
        self.usar_autocontraste = usar_autocontraste

    def __len__(self):
        return len(self.muestras)

    def __getitem__(self, i):
        muestra = self.muestras[i]
        
        with Image.open(muestra.ruta) as imagen:
            imagen_gris = imagen.convert('L')
        
        if self.usar_autocontraste:
            imagen_gris = ImageOps.autocontrast(imagen_gris)
        
        imagen_rgb = Image.merge('RGB', (imagen_gris, imagen_gris, imagen_gris))
        tensor_entrada = self.transformacion(imagen_rgb)
        objetivo = torch.tensor(float(muestra.etiqueta), dtype=torch.float32)
        
        return tensor_entrada, objetivo, str(muestra.ruta)


def construir_muestras_desde_carpetas(ruta_imagenes: Path):
    """Carga muestras desde estructura de carpetas"""
    pares_ruta_etiqueta = []
    
    for carpeta_clase in sorted([p for p in ruta_imagenes.iterdir() if p.is_dir()]):
        etiqueta = 0 if carpeta_clase.name.strip().lower() in ETIQUETAS_NORMAL else 1
        for ruta_imagen in sorted(carpeta_clase.rglob('*')):
            if ruta_imagen.suffix.lower() in EXTENSIONES_VALIDAS:
                pares_ruta_etiqueta.append((ruta_imagen, etiqueta))
    
    if not pares_ruta_etiqueta:
        raise RuntimeError(f'No se encontraron imágenes en {ruta_imagenes}')
    
    muestras = []
    for ruta_imagen, etiqueta in pares_ruta_etiqueta:
        id_paciente = ruta_imagen.stem.split('_')[0]
        muestras.append(Muestra(ruta=ruta_imagen, etiqueta=int(etiqueta), id_paciente=id_paciente))
    
    return muestras


def construir_transformaciones(config):
    """Construye pipelines de transformación"""
    tam_imagen = config['TAM_IMAGEN']
    usar_roi = config['USAR_ROI']
    usar_aug_train = config['USAR_AUMENTACION_TRAIN']
    
    ops_roi = [RecorteRetrocardiaco()] if usar_roi else []
    normalizacion = T.Normalize(mean=MEDIA_IMAGENET, std=STD_IMAGENET)
    
    ops_aug = []
    if usar_aug_train:
        ops_aug = [
            T.RandomApply([T.ColorJitter(
                brightness=config['JITTER_BRILLO'],
                contrast=config['JITTER_CONTRASTE']
            )], p=config['P_AUMENTO_INTENSIDAD']),
            T.RandomAffine(
                degrees=config['ROTACION_MAX_GRADOS'],
                translate=(config['TRASLACION_MAX'], config['TRASLACION_MAX']),
                scale=(config['ESCALA_MIN'], config['ESCALA_MAX']),
            ),
            T.RandomHorizontalFlip(p=0.5 if config['USAR_FLIP_HORIZONTAL_TRAIN'] else 0.0),
        ]
    
    transf_entrenamiento = T.Compose(
        ops_roi + [T.Resize((tam_imagen, tam_imagen))] + ops_aug + [T.ToTensor(), normalizacion]
    )
    transf_validacion = T.Compose(
        ops_roi + [T.Resize((tam_imagen, tam_imagen)), T.ToTensor(), normalizacion]
    )
    
    return transf_entrenamiento, transf_validacion


def construir_pliegues(muestras, n_splits=5, semilla=42):
    """Crea pliegues estratificados para validación cruzada"""
    etiquetas = np.array([m.etiqueta for m in muestras], dtype=int)
    indices = np.arange(len(muestras))
    generador = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=semilla)
    return list(generador.split(indices, etiquetas))


def crear_modelo(config):
    """Crea modelo DenseNet121"""
    pesos = DenseNet121_Weights.DEFAULT if config['USAR_PREENTRENADO'] else None
    modelo = densenet121(weights=pesos)
    in_features = modelo.classifier.in_features
    modelo.classifier = nn.Sequential(
        nn.Dropout(p=float(config['DROPOUT'])),
        nn.Linear(in_features, 1)
    )
    return modelo


def fijar_backbone_entrenable(modelo, entrenable: bool):
    """Congela o descongela el backbone"""
    for parametro in modelo.features.parameters():
        parametro.requires_grad = entrenable


def crear_escalador_amp(config):
    """Crea escalador para precisión mixta"""
    return torch.amp.GradScaler(
        'cuda',
        enabled=(config['USAR_AMP'] and config['DISPOSITIVO'].type == 'cuda')
    )


def calcular_peso_positivo(muestras, indices, dispositivo):
    """Calcula peso para balancear clases"""
    etiquetas = np.array([muestras[i].etiqueta for i in indices], dtype=np.float32)
    positivos = float(etiquetas.sum())
    negativos = float(len(etiquetas) - positivos)
    return torch.tensor([negativos / max(positivos, 1.0)], device=dispositivo)


# ============================================================================
# MLFLOW SERVER CON CORS
# ============================================================================

def iniciar_mlflow_server(config):
    """Inicia el servidor MLflow con CORS habilitado"""
    import socket
    
    # Verificar si el puerto está disponible
    def puerto_disponible(host, port):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            sock.close()
            return result != 0
        except:
            return False
    
    if not puerto_disponible('localhost', config['MLFLOW_PORT']):
        print(f"\n⚠️  El puerto {config['MLFLOW_PORT']} ya está en uso.")
        print(f"   Asumiendo que MLflow ya está corriendo...")
        return None
    
    print(f"\n🚀 Iniciando servidor MLflow con CORS habilitado...")
    print(f"   Host: {config['MLFLOW_HOST']}")
    print(f"   Puerto: {config['MLFLOW_PORT']}")
    print(f"   Backend: {config['MLFLOW_BACKEND_STORE']}")
    print(f"   Artifacts: {config['MLFLOW_ARTIFACT_ROOT']}")
    
    # Hosts permitidos
    allowed_hosts = [
        'localhost',
        '127.0.0.1',
        f'localhost:{config["MLFLOW_PORT"]}',
        f'127.0.0.1:{config["MLFLOW_PORT"]}',
        '*',  # Permite cualquier host
    ]
    allowed_hosts_str = ','.join(allowed_hosts)
    
    # Comando para iniciar MLflow con CORS
    comando = [
        'mlflow', 'server',
        '--host', config['MLFLOW_HOST'],
        '--port', str(config['MLFLOW_PORT']),
        '--backend-store-uri', config['MLFLOW_BACKEND_STORE'],
        '--default-artifact-root', config['MLFLOW_ARTIFACT_ROOT'],
        '--serve-artifacts',  # Habilita el servidor de artifacts
    ]
    
    # Variables de entorno para habilitar CORS y configurar hosts permitidos
    env = {
        **subprocess.os.environ,
        # CORS
        'MLFLOW_ENABLE_CORS': 'true',
        'MLFLOW_CORS_ALLOW_ORIGIN': '*',  # Permite todos los orígenes
        'MLFLOW_CORS_ALLOW_METHODS': 'GET,POST,PUT,DELETE,OPTIONS',
        'MLFLOW_CORS_ALLOW_HEADERS': 'Content-Type,Authorization',
        # Hosts permitidos (soluciona el error de Host header)
        'MLFLOW_ALLOWED_HOSTS': allowed_hosts_str,
    }
    
    try:
        # Iniciar el proceso en segundo plano
        proceso = subprocess.Popen(
            comando,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Esperar un momento para que el servidor inicie
        print("   Esperando a que el servidor inicie...")
        time.sleep(5)
        
        # Verificar que el servidor está corriendo
        if proceso.poll() is None:
            print(f"   ✅ Servidor MLflow iniciado correctamente")
            print(f"   📊 UI disponible en: http://localhost:{config['MLFLOW_PORT']}")
            print(f"   🌐 CORS habilitado para todos los orígenes")
            return proceso
        else:
            stdout, stderr = proceso.communicate()
            print(f"   ❌ Error al iniciar el servidor:")
            print(f"   {stderr}")
            return None
            
    except Exception as e:
        print(f"   ❌ Error al iniciar MLflow: {e}")
        return None


def detener_mlflow_server(proceso):
    """Detiene el servidor MLflow"""
    if proceso is not None:
        print("\n🛑 Deteniendo servidor MLflow...")
        proceso.terminate()
        try:
            proceso.wait(timeout=5)
            print("   ✅ Servidor detenido correctamente")
        except subprocess.TimeoutExpired:
            proceso.kill()
            print("   ⚠️  Servidor forzado a detenerse")


def configurar_mlflow_client(config):
    """Configura el cliente MLflow con las opciones necesarias"""
    import os
    
    # Configurar variables de entorno para el cliente
    os.environ['MLFLOW_TRACKING_URI'] = config['MLFLOW_TRACKING_URI']
    
    # Configurar MLflow
    mlflow.set_tracking_uri(config['MLFLOW_TRACKING_URI'])
    mlflow.set_experiment(config['MLFLOW_EXPERIMENTO'])
    
    print(f"\n📡 Cliente MLflow configurado:")
    print(f"   Tracking URI: {config['MLFLOW_TRACKING_URI']}")
    print(f"   Experimento: {config['MLFLOW_EXPERIMENTO']}")


# ============================================================================
# ENTRENAMIENTO
# ============================================================================

def ejecutar_epoca(modelo, loader, criterio, dispositivo, optimizador=None, escalador_amp=None):
    """Ejecuta una época de entrenamiento o validación"""
    es_entrenamiento = optimizador is not None
    modelo.train() if es_entrenamiento else modelo.eval()
    
    perdidas, y_todo, p_todo = [], [], []
    non_blocking = dispositivo.type == 'cuda'
    usar_autocast = (escalador_amp is not None) and (dispositivo.type == 'cuda')
    
    for entradas, objetivos, _ in loader:
        entradas = entradas.to(dispositivo, non_blocking=non_blocking)
        objetivos = objetivos.to(dispositivo, non_blocking=non_blocking).unsqueeze(1)
        
        with torch.set_grad_enabled(es_entrenamiento):
            if usar_autocast:
                with torch.amp.autocast(device_type='cuda', enabled=True):
                    logits = modelo(entradas)
                    perdida = criterio(logits, objetivos)
            else:
                logits = modelo(entradas)
                perdida = criterio(logits, objetivos)
            
            probabilidades = torch.sigmoid(logits)
            
            if es_entrenamiento:
                optimizador.zero_grad(set_to_none=True)
                if usar_autocast:
                    escalador_amp.scale(perdida).backward()
                    escalador_amp.step(optimizador)
                    escalador_amp.update()
                else:
                    perdida.backward()
                    optimizador.step()
        
        perdidas.append(perdida.detach().item() * entradas.size(0))
        y_todo.append(objetivos.detach().cpu().numpy().reshape(-1))
        p_todo.append(probabilidades.detach().cpu().numpy().reshape(-1))
    
    y_epoca = np.concatenate(y_todo)
    p_epoca = np.concatenate(p_todo)
    perdida_epoca = float(np.sum(perdidas) / len(loader.dataset))
    
    return perdida_epoca, y_epoca, p_epoca


@torch.no_grad()
def predecir_loader(modelo, loader, dispositivo):
    """Genera predicciones para un loader"""
    modelo.eval()
    non_blocking = dispositivo.type == 'cuda'
    
    y_todo, p_todo, rutas_todo = [], [], []
    
    for entradas, objetivos, rutas in loader:
        entradas = entradas.to(dispositivo, non_blocking=non_blocking)
        logits = modelo(entradas)
        probabilidades = torch.sigmoid(logits)
        
        y_todo.append(objetivos.cpu().numpy().reshape(-1))
        p_todo.append(probabilidades.cpu().numpy().reshape(-1))
        rutas_todo.extend(rutas)
    
    return np.concatenate(y_todo), np.concatenate(p_todo), rutas_todo


def calcular_metricas(y_true, y_prob, umbral=0.5):
    """Calcula métricas de evaluación"""
    y_pred = (y_prob >= umbral).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metricas = {
        'auc': roc_auc_score(y_true, y_prob),
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'sensibilidad': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        'especificidad': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'umbral': umbral,
    }
    
    return metricas


def optimizar_umbral(y_true, y_prob):
    """Encuentra el umbral óptimo maximizando F1"""
    precision, recall, umbrales = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    idx_mejor = np.argmax(f1_scores)
    return umbrales[idx_mejor] if idx_mejor < len(umbrales) else 0.5


def entrenar_fold(modelo, train_loader, val_loader, config, fold_num=0):
    """Entrena un fold completo"""
    dispositivo = config['DISPOSITIVO']
    modelo = modelo.to(dispositivo)
    
    # Configurar optimizador y criterio
    peso_positivo = calcular_peso_positivo(
        train_loader.dataset.muestras,
        range(len(train_loader.dataset)),
        dispositivo
    )
    criterio = nn.BCEWithLogitsLoss(pos_weight=peso_positivo)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=1e-4)
    escalador_amp = crear_escalador_amp(config)
    
    # Fase congelada
    fijar_backbone_entrenable(modelo, False)
    for epoca in range(config['EPOCAS_CONGELADAS']):
        perdida_train, _, _ = ejecutar_epoca(
            modelo, train_loader, criterio, dispositivo, optimizador, escalador_amp
        )
        print(f"Fold {fold_num} - Época congelada {epoca+1}/{config['EPOCAS_CONGELADAS']} - Loss: {perdida_train:.4f}")
    
    # Fase completa
    fijar_backbone_entrenable(modelo, True)
    mejor_auc = 0.0
    epocas_sin_mejora = 0
    historial = []
    
    for epoca in range(config['EPOCAS']):
        # Entrenamiento
        perdida_train, _, _ = ejecutar_epoca(
            modelo, train_loader, criterio, dispositivo, optimizador, escalador_amp
        )
        
        # Validación
        perdida_val, y_val, p_val = ejecutar_epoca(
            modelo, val_loader, criterio, dispositivo
        )
        
        # Métricas
        metricas_val = calcular_metricas(y_val, p_val)
        
        historial.append({
            'epoca': epoca + 1,
            'perdida_train': perdida_train,
            'perdida_val': perdida_val,
            **metricas_val
        })
        
        print(f"Fold {fold_num} - Época {epoca+1}/{config['EPOCAS']} - "
              f"Train Loss: {perdida_train:.4f} - Val Loss: {perdida_val:.4f} - "
              f"AUC: {metricas_val['auc']:.4f} - F1: {metricas_val['f1']:.4f}")
        
        # Early stopping
        if metricas_val['auc'] > mejor_auc:
            mejor_auc = metricas_val['auc']
            epocas_sin_mejora = 0
        else:
            epocas_sin_mejora += 1
            if epocas_sin_mejora >= config['PACIENCIA']:
                print(f"Early stopping en época {epoca+1}")
                break
    
    return modelo, historial


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal de entrenamiento"""
    print("="*80)
    print("ENTRENAMIENTO DE MODELO DE DETECCIÓN DE HERNIA HIATAL CON MLFLOW")
    print("="*80)
    
    # Configuración
    config = configurar_experimento()
    fijar_semilla(config['SEMILLA'])
    
    print(f"\nConfiguración:")
    print(f"- Dispositivo: {config['DISPOSITIVO']}")
    print(f"- Imágenes: {config['TAM_IMAGEN']}x{config['TAM_IMAGEN']}")
    print(f"- Batch size: {config['TAM_BATCH']}")
    print(f"- Épocas: {config['EPOCAS']}")
    print(f"- MLflow URI: {config['MLFLOW_TRACKING_URI']}")
    print(f"- Experimento: {config['MLFLOW_EXPERIMENTO']}")
    
    # Iniciar servidor MLflow si está configurado
    # proceso_mlflow = None
    # if config['USAR_MLFLOW'] and config['MLFLOW_AUTO_START']:
    #     proceso_mlflow = iniciar_mlflow_server(config)
    
    try:
        # Cargar datos
        print(f"\nCargando datos desde {config['RUTA_IMAGENES']}...")
        muestras = construir_muestras_desde_carpetas(config['RUTA_IMAGENES'])
        etiquetas = np.array([m.etiqueta for m in muestras], dtype=int)
        
        print(f"Total de muestras: {len(muestras)}")
        print(f"- Normal: {(etiquetas == 0).sum()}")
        print(f"- Hernia: {(etiquetas == 1).sum()}")
        
        # Configurar MLflow
        if config['USAR_MLFLOW']:
            configurar_mlflow_client(config)
        
        # Crear pliegues
        pliegues = construir_pliegues(muestras, config['N_SPLITS'], config['SEMILLA'])
        
        # Entrenar primer fold (demo)
        print(f"\n{'='*80}")
        print("ENTRENANDO FOLD 1 (DEMO)")
        print(f"{'='*80}")
        
        train_idx, val_idx = pliegues[0]
        
        # Transformaciones
        transf_train, transf_val = construir_transformaciones(config)
        
        # Datasets
        dataset_train = ConjuntoHiatal(muestras, train_idx, transf_train, config['USAR_AUTOCONTRASTE'])
        dataset_val = ConjuntoHiatal(muestras, val_idx, transf_val, config['USAR_AUTOCONTRASTE'])
        
        # DataLoaders
        train_loader = DataLoader(
            dataset_train,
            batch_size=config['TAM_BATCH'],
            shuffle=True,
            num_workers=0,
            pin_memory=config['DISPOSITIVO'].type == 'cuda'
        )
        val_loader = DataLoader(
            dataset_val,
            batch_size=config['TAM_BATCH'],
            shuffle=False,
            num_workers=0,
            pin_memory=config['DISPOSITIVO'].type == 'cuda'
        )
        
        # Crear modelo
        modelo = crear_modelo(config)
        
        # Entrenar con MLflow
        if config['USAR_MLFLOW']:
            with mlflow.start_run(run_name=config['MLFLOW_RUN_NAME']):
                # Registrar parámetros
                mlflow.log_params({
                    'tam_imagen': config['TAM_IMAGEN'],
                    'tam_batch': config['TAM_BATCH'],
                    'epocas': config['EPOCAS'],
                    'dropout': config['DROPOUT'],
                    'usar_roi': config['USAR_ROI'],
                    'usar_autocontraste': config['USAR_AUTOCONTRASTE'],
                    'usar_aumentacion': config['USAR_AUMENTACION_TRAIN'],
                    'n_splits': config['N_SPLITS'],
                    'semilla': config['SEMILLA'],
                })
                
                # Entrenar
                modelo, historial = entrenar_fold(modelo, train_loader, val_loader, config, fold_num=1)
                
                # Registrar métricas finales
                metricas_finales = historial[-1]
                mlflow.log_metrics({
                    'val_loss': metricas_finales['perdida_val'],
                    'val_auc': metricas_finales['auc'],
                    'val_accuracy': metricas_finales['accuracy'],
                    'val_f1': metricas_finales['f1'],
                    'val_sensibilidad': metricas_finales['sensibilidad'],
                    'val_especificidad': metricas_finales['especificidad'],
                })
                
                # Guardar modelo
                if config['MLFLOW_REGISTRAR_MODELO']:
                    mlflow.pytorch.log_model(modelo, "model")
                    print(f"\n✅ Modelo registrado en MLflow")
                
                print(f"\n{'='*80}")
                print("ENTRENAMIENTO COMPLETADO")
                print(f"{'='*80}")
                print(f"\nMétricas finales:")
                print(f"- AUC: {metricas_finales['auc']:.4f}")
                print(f"- Accuracy: {metricas_finales['accuracy']:.4f}")
                print(f"- F1: {metricas_finales['f1']:.4f}")
                print(f"- Sensibilidad: {metricas_finales['sensibilidad']:.4f}")
                print(f"- Especificidad: {metricas_finales['especificidad']:.4f}")
                print(f"\n📊 Puedes ver los resultados en: {config['MLFLOW_TRACKING_URI']}")
        
        else:
            # Entrenar sin MLflow
            modelo, historial = entrenar_fold(modelo, train_loader, val_loader, config, fold_num=1)
            
            print(f"\n{'='*80}")
            print("ENTRENAMIENTO COMPLETADO")
            print(f"{'='*80}")
    
    finally:
        # Detener servidor MLflow si fue iniciado por este script
        print("ENTRENAMIENTO COMPLETADO")


if __name__ == '__main__':
    main()
