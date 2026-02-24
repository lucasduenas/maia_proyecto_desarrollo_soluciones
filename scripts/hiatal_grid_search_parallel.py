from __future__ import annotations

import argparse
import json
import os
import random
import socket
import subprocess
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from urllib.parse import urlparse

import mlflow
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torchvision.models import DenseNet121_Weights, EfficientNet_B0_Weights, ResNet18_Weights, densenet121, efficientnet_b0, resnet18

EXTENSIONES_VALIDAS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
ETIQUETAS_NORMAL = {"0", "normal", "sano", "healthy", "neg", "negative"}
MEDIA_IMAGENET = [0.485, 0.456, 0.406]
STD_IMAGENET = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class Muestra:
    ruta: Path
    etiqueta: int
    id_paciente: str


class RecorteRetrocardiaco:
    def __init__(self, x1=0.2, x2=0.8, y1=0.15, y2=0.98):
        self.x1, self.x2, self.y1, self.y2 = x1, x2, y1, y2

    def __call__(self, imagen: Image.Image):
        w, h = imagen.size
        return imagen.crop((int(self.x1 * w), int(self.y1 * h), int(self.x2 * w), int(self.y2 * h)))


class ConjuntoHiatal(Dataset):
    def __init__(self, muestras: list[Muestra], indices, transformacion):
        self.muestras = [muestras[i] for i in indices]
        self.transformacion = transformacion

    def __len__(self):
        return len(self.muestras)

    def __getitem__(self, i):
        m = self.muestras[i]
        with Image.open(m.ruta) as img:
            gris = ImageOps.autocontrast(img.convert("L"))
        rgb = Image.merge("RGB", (gris, gris, gris))
        x = self.transformacion(rgb)
        y = torch.tensor(float(m.etiqueta), dtype=torch.float32)
        return x, y


def fijar_semilla(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def construir_muestras_desde_carpetas(ruta_imagenes: Path):
    pares = []
    for carpeta in sorted([p for p in ruta_imagenes.iterdir() if p.is_dir()]):
        y = 0 if carpeta.name.strip().lower() in ETIQUETAS_NORMAL else 1
        for ruta in sorted(carpeta.rglob("*")):
            if ruta.suffix.lower() in EXTENSIONES_VALIDAS:
                pares.append((ruta, y))
    return [Muestra(ruta=r, etiqueta=int(y), id_paciente=r.stem.split("_")[0]) for r, y in pares]


def construir_transformaciones(tam_imagen=320):
    roi = [RecorteRetrocardiaco()]
    norm = T.Normalize(mean=MEDIA_IMAGENET, std=STD_IMAGENET)
    aug = [
        T.RandomApply([T.ColorJitter(brightness=0.12, contrast=0.15)], p=0.35),
        T.RandomAffine(degrees=8, translate=(0.03, 0.03), scale=(0.97, 1.03)),
    ]
    train_tf = T.Compose(roi + [T.Resize((tam_imagen, tam_imagen))] + aug + [T.ToTensor(), norm])
    val_tf = T.Compose(roi + [T.Resize((tam_imagen, tam_imagen)), T.ToTensor(), norm])
    return train_tf, val_tf


def crear_modelo(nombre: str, preentrenado=True, dropout=0.25):
    nombre = nombre.lower()
    if nombre == "resnet18":
        m = resnet18(weights=ResNet18_Weights.DEFAULT if preentrenado else None)
        m.fc = nn.Sequential(nn.Dropout(float(dropout)), nn.Linear(m.fc.in_features, 1))
        return m
    if nombre == "densenet121":
        m = densenet121(weights=DenseNet121_Weights.DEFAULT if preentrenado else None)
        m.classifier = nn.Sequential(nn.Dropout(float(dropout)), nn.Linear(m.classifier.in_features, 1))
        return m
    if nombre == "efficientnetb0":
        m = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if preentrenado else None)
        in_features = m.classifier[1].in_features
        m.classifier = nn.Sequential(nn.Dropout(float(dropout), inplace=True), nn.Linear(in_features, 1))
        return m
    raise ValueError(f"Modelo no soportado: {nombre}")


def fijar_backbone_entrenable(modelo, nombre: str, entrenable: bool):
    for p in modelo.parameters():
        p.requires_grad = entrenable
    cabeza = modelo.fc.parameters() if nombre == "resnet18" else modelo.classifier.parameters()
    for p in cabeza:
        p.requires_grad = True


def calcular_peso_positivo(muestras: list[Muestra], indices, dispositivo):
    y = np.array([muestras[i].etiqueta for i in indices], dtype=np.float32)
    pos = float(y.sum())
    neg = float(len(y) - pos)
    return torch.tensor([neg / max(pos, 1.0)], device=dispositivo)


def resolver_num_workers(valor: int, es_train: bool, max_parallel_trials: int, worker_device: str):
    if valor >= 0:
        return int(valor)
    cpus = os.cpu_count() or 8
    por_trial = max(1, cpus // max(1, int(max_parallel_trials)))
    if str(worker_device).startswith("cuda"):
        if es_train:
            return int(min(12, max(4, por_trial)))
        return int(min(6, max(2, por_trial // 2)))
    if es_train:
        return int(min(8, max(2, por_trial)))
    return int(min(4, max(1, por_trial // 2)))


def resolver_torch_threads(valor: int, max_parallel_trials: int):
    if valor >= 1:
        return int(valor)
    cpus = os.cpu_count() or 8
    return int(max(1, cpus // max(1, int(max_parallel_trials))))


def kwargs_loader(dispositivo, num_workers, prefetch_factor, persistent_workers):
    workers = int(num_workers)
    kwargs = {"num_workers": workers, "pin_memory": dispositivo.type == "cuda"}
    if workers > 0:
        kwargs["prefetch_factor"] = max(2, int(prefetch_factor))
        kwargs["persistent_workers"] = bool(persistent_workers)
    return kwargs


def crear_loader_train(ds, tam_batch, dispositivo, factor, num_workers, prefetch_factor, persistent_workers):
    kwargs = kwargs_loader(dispositivo, num_workers, prefetch_factor, persistent_workers)
    if factor > 1:
        sampler = RandomSampler(ds, replacement=True, num_samples=int(len(ds) * factor))
        return DataLoader(ds, batch_size=tam_batch, shuffle=False, sampler=sampler, **kwargs)
    return DataLoader(ds, batch_size=tam_batch, shuffle=True, **kwargs)


def crear_loader_val(ds, tam_batch, dispositivo, num_workers, prefetch_factor, persistent_workers):
    kwargs = kwargs_loader(dispositivo, num_workers, prefetch_factor, persistent_workers)
    return DataLoader(ds, batch_size=tam_batch, shuffle=False, **kwargs)


def mover_x_dispositivo(x, dispositivo, channels_last):
    if dispositivo.type == "cuda" and channels_last and x.ndim == 4:
        return x.to(dispositivo, non_blocking=True, memory_format=torch.channels_last)
    return x.to(dispositivo, non_blocking=dispositivo.type == "cuda")


def ejecutar_epoca(
    modelo,
    loader,
    criterio,
    dispositivo,
    optimizador=None,
    escalador=None,
    devolver_pred=False,
    use_amp=False,
    amp_dtype=torch.float16,
    channels_last=False,
):
    train = optimizador is not None
    modelo.train() if train else modelo.eval()
    loss_total = torch.zeros((), device=dispositivo, dtype=torch.float32) if dispositivo.type == "cuda" else 0.0
    y_all, p_all = [], []
    for x, y in loader:
        x = mover_x_dispositivo(x, dispositivo, channels_last)
        y = y.to(dispositivo, non_blocking=dispositivo.type == "cuda").unsqueeze(1)
        with torch.set_grad_enabled(train):
            if use_amp and dispositivo.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
                    logits = modelo(x)
                    loss = criterio(logits, y)
            else:
                logits = modelo(x)
                loss = criterio(logits, y)
            if train:
                optimizador.zero_grad(set_to_none=True)
                if escalador is not None and escalador.is_enabled():
                    escalador.scale(loss).backward()
                    escalador.step(optimizador)
                    escalador.update()
                else:
                    loss.backward()
                    optimizador.step()
        if dispositivo.type == "cuda":
            loss_total = loss_total + loss.detach().float() * x.size(0)
        else:
            loss_total += float(loss.detach().item()) * x.size(0)
        if devolver_pred:
            p = torch.sigmoid(logits)
            y_all.append(y.detach().cpu().numpy().reshape(-1))
            p_all.append(p.detach().cpu().numpy().reshape(-1))
    if dispositivo.type == "cuda":
        loss_mean = float(loss_total.cpu().item() / len(loader.dataset))
    else:
        loss_mean = float(loss_total / len(loader.dataset))
    y_ep = np.concatenate(y_all) if devolver_pred else None
    p_ep = np.concatenate(p_all) if devolver_pred else None
    return loss_mean, y_ep, p_ep


@torch.no_grad()
def predecir_loader(modelo, loader, dispositivo, use_amp=False, amp_dtype=torch.float16, channels_last=False):
    modelo.eval()
    ys, ps = [], []
    for x, y in loader:
        x = mover_x_dispositivo(x, dispositivo, channels_last)
        if use_amp and dispositivo.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
                logits = modelo(x)
        else:
            logits = modelo(x)
        p = torch.sigmoid(logits)
        ys.append(y.numpy().astype(int).reshape(-1))
        ps.append(p.squeeze(1).detach().cpu().numpy().astype(float).reshape(-1))
    return np.concatenate(ys).astype(int), np.concatenate(ps).astype(float)


def crear_optimizador_adamw(modelo, lr, weight_decay, use_cuda: bool):
    kwargs = {"lr": float(lr), "weight_decay": float(weight_decay)}
    if use_cuda:
        try:
            return torch.optim.AdamW(modelo.parameters(), fused=True, **kwargs)
        except Exception:
            pass
    return torch.optim.AdamW(modelo.parameters(), **kwargs)


def intentar_compilar_modelo(modelo, habilitado: bool, modo: str):
    if not habilitado or not hasattr(torch, "compile"):
        return modelo
    try:
        return torch.compile(modelo, mode=modo)
    except Exception:
        return modelo


def configurar_cuda_rapida():
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cudnn, "conv") and hasattr(torch.backends.cudnn.conv, "fp32_precision"):
        torch.backends.cudnn.conv.fp32_precision = "tf32"
    else:
        try:
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
    if hasattr(torch.backends.cuda, "matmul") and hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = "tf32"
    else:
        try:
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass


def buscar_umbral_optimo(y_true, p_pred):
    y_true = np.rint(np.asarray(y_true).reshape(-1)).astype(int)
    p_pred = np.clip(np.asarray(p_pred).reshape(-1).astype(float), 0.0, 1.0)
    if len(np.unique(y_true)) < 2:
        return 0.5
    best_t, best_j = 0.5, -np.inf
    for t in np.unique(p_pred):
        yb = (p_pred >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, yb, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        esp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        j = sens + esp - 1.0
        if j > best_j:
            best_j, best_t = j, float(t)
    return best_t


def calcular_metricas(y_true, p_pred, umbral=0.5):
    y_true = np.rint(np.asarray(y_true).reshape(-1)).astype(int)
    p_pred = np.clip(np.asarray(p_pred).reshape(-1).astype(float), 0.0, 1.0)
    yb = (p_pred >= float(umbral)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, yb, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    esp = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    auc = float(roc_auc_score(y_true, p_pred)) if len(np.unique(y_true)) > 1 else np.nan
    return {"auc": auc, "accuracy": float(accuracy_score(y_true, yb)), "f1": float(f1_score(y_true, yb, zero_division=0)), "sensibilidad": float(sens), "especificidad": float(esp), "umbral": float(umbral)}


def tracking_disponible(uri: str, timeout_s: float):
    u = urlparse(uri.strip())
    if u.scheme in {"http", "https"} and u.hostname:
        port = u.port or (443 if u.scheme == "https" else 80)
        try:
            with socket.create_connection((u.hostname, port), timeout=timeout_s):
                return True
        except OSError:
            return False
    return True


def iniciar_servidor_mlflow_si_falta(repo_root: Path, output_dir: Path, tracking_uri: str, timeout_s: float):
    u = urlparse(tracking_uri)
    if (u.hostname or "") not in {"localhost", "127.0.0.1"}:
        return
    if tracking_disponible(tracking_uri, timeout_s):
        return
    log = output_dir / "mlflow_server_local.log"
    db = output_dir / "mlflow_server_backend.db"
    artifacts = output_dir / "mlflow_server_artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    cmd = [
        os.environ.get("PYTHON", "python"),
        "-m",
        "mlflow",
        "server",
        "--host",
        "127.0.0.1",
        "--port",
        "8050",
        "--workers",
        "1",
        "--backend-store-uri",
        f"sqlite:///{db.resolve()}",
        "--default-artifact-root",
        artifacts.resolve().as_uri(),
        "--allowed-hosts",
        "localhost:8050,127.0.0.1:8050",
    ]
    with open(log, "a", encoding="utf-8") as fh:
        subprocess.Popen(cmd, stdout=fh, stderr=fh, cwd=str(repo_root))
    for _ in range(30):
        time.sleep(0.5)
        if tracking_disponible(tracking_uri, timeout_s):
            return
    raise RuntimeError(f"No se pudo iniciar MLflow en {tracking_uri}.")


def run_trial(payload: dict):
    cfg, combo = payload["cfg"], payload["combo"]
    idx, trial_name = payload["trial_index"], payload["trial_name"]
    try:
        torch.set_num_threads(max(1, int(cfg["torch_threads_per_worker"])))
        try:
            torch.set_num_interop_threads(max(1, min(8, int(cfg["torch_threads_per_worker"]))))
        except Exception:
            pass
        device = torch.device(cfg["worker_device"])
        use_cuda = device.type == "cuda"
        if use_cuda and not torch.cuda.is_available():
            raise RuntimeError(f"Trial '{trial_name}' requiere CUDA pero torch.cuda.is_available()=False en worker.")
        use_amp = use_cuda and bool(cfg["use_amp"])
        amp_dtype = torch.bfloat16 if cfg["amp_dtype"] == "bfloat16" else torch.float16
        channels_last = use_cuda and bool(cfg["channels_last"])
        if use_cuda:
            torch.cuda.set_device(device.index or 0)
            configurar_cuda_rapida()

        muestras = [Muestra(Path(r), int(y), pid) for r, y, pid in payload["samples"]]
        folds = [(np.asarray(a, dtype=int), np.asarray(b, dtype=int)) for a, b in payload["folds"]]
        train_tf, val_tf = construir_transformaciones(cfg["image_size"])
        mlflow.set_tracking_uri(cfg["tracking_uri"])
        mlflow.set_experiment(cfg["experiment"])
        trial_dir = Path(cfg["output_dir"]) / "trials" / trial_name
        trial_dir.mkdir(parents=True, exist_ok=True)

        with mlflow.start_run(run_name=trial_name):
            compile_activo = bool(cfg["torch_compile"]) and int(combo["epocas_congeladas"]) == 0
            for k, v in combo.items():
                mlflow.log_param(k, v)
            mlflow.log_param("tam_batch", cfg["batch_size"])
            mlflow.log_param("num_workers_train", cfg["num_workers_train"])
            mlflow.log_param("num_workers_val", cfg["num_workers_val"])
            mlflow.log_param("worker_device", cfg["worker_device"])
            mlflow.log_param("channels_last", channels_last)
            mlflow.log_param("use_amp", use_amp)
            mlflow.log_param("amp_dtype", cfg["amp_dtype"])
            mlflow.log_param("torch_compile_activo", compile_activo)
            mlflow.log_param("log_every_epochs", int(cfg["log_every_epochs"]))
            if use_cuda:
                mlflow.log_param("cuda_device_name", torch.cuda.get_device_name(device))
            mlflow.set_tag("pipeline", "grid_search_cv_parallel")
            fold_metrics = []
            for fold_id, (tr_idx, va_idx) in enumerate(folds, start=1):
                fijar_semilla(cfg["seed"] + fold_id)
                ds_tr = ConjuntoHiatal(muestras, tr_idx, train_tf)
                ds_va = ConjuntoHiatal(muestras, va_idx, val_tf)
                ld_tr = crear_loader_train(
                    ds_tr,
                    cfg["batch_size"],
                    device,
                    cfg["factor_samples"],
                    cfg["num_workers_train"],
                    cfg["prefetch_factor"],
                    cfg["persistent_workers"],
                )
                ld_va = crear_loader_val(
                    ds_va,
                    cfg["batch_size"],
                    device,
                    cfg["num_workers_val"],
                    cfg["prefetch_factor"],
                    cfg["persistent_workers"],
                )
                model = crear_modelo(combo["arquitectura"], preentrenado=cfg["use_pretrained"], dropout=combo["dropout"]).to(device)
                if channels_last:
                    model = model.to(memory_format=torch.channels_last)
                if combo["epocas_congeladas"] > 0:
                    fijar_backbone_entrenable(model, combo["arquitectura"], False)
                model = intentar_compilar_modelo(model, compile_activo, cfg["torch_compile_mode"])
                opt = crear_optimizador_adamw(model, combo["lr"], combo["weight_decay"], use_cuda)
                sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, combo["epocas"]))
                crit = nn.BCEWithLogitsLoss(pos_weight=calcular_peso_positivo(muestras, tr_idx, device))
                scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

                best, no_imp = None, 0
                for ep in range(1, combo["epocas"] + 1):
                    if combo["epocas_congeladas"] > 0 and ep == combo["epocas_congeladas"] + 1:
                        fijar_backbone_entrenable(model, combo["arquitectura"], True)
                    loss_tr, _, _ = ejecutar_epoca(
                        model,
                        ld_tr,
                        crit,
                        device,
                        optimizador=opt,
                        escalador=scaler,
                        use_amp=use_amp,
                        amp_dtype=amp_dtype,
                        channels_last=channels_last,
                    )
                    loss_va, y_va, p_va = ejecutar_epoca(
                        model,
                        ld_va,
                        crit,
                        device,
                        devolver_pred=True,
                        use_amp=use_amp,
                        amp_dtype=amp_dtype,
                        channels_last=channels_last,
                    )
                    thr = buscar_umbral_optimo(y_va, p_va)
                    met = calcular_metricas(y_va, p_va, thr)
                    score = met["auc"] if np.isfinite(met["auc"]) else met["f1"]
                    if int(cfg["log_every_epochs"]) > 0 and (ep % int(cfg["log_every_epochs"]) == 0):
                        mlflow.log_metric(f"fold_{fold_id}.epoch.loss_train", float(loss_tr), step=ep)
                        mlflow.log_metric(f"fold_{fold_id}.epoch.loss_val", float(loss_va), step=ep)
                        mlflow.log_metric(f"fold_{fold_id}.epoch.score", float(score), step=ep)
                    if (best is None) or (score > best["score"] + cfg["min_score_improve"]):
                        best = {
                            "score": float(score),
                            "umbral": float(thr),
                            "state_dict": {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()},
                        }
                        no_imp = 0
                    else:
                        no_imp += 1
                    sch.step()
                    if no_imp >= cfg["patience"]:
                        break

                model.load_state_dict(best["state_dict"])
                y_va, p_va = predecir_loader(model, ld_va, device, use_amp=use_amp, amp_dtype=amp_dtype, channels_last=channels_last)
                met_fold = calcular_metricas(y_va, p_va, best["umbral"])
                met_fold["fold"] = fold_id
                fold_metrics.append(met_fold)

            resumen = {}
            for m in ["auc", "accuracy", "f1", "sensibilidad", "especificidad"]:
                vals = [float(x[m]) for x in fold_metrics]
                resumen[f"{m}_mean"] = float(np.nanmean(vals))
                resumen[f"{m}_std"] = float(np.nanstd(vals))
                if np.isfinite(resumen[f"{m}_mean"]):
                    mlflow.log_metric(f"cv.summary.{m}_mean", resumen[f"{m}_mean"])

            df_fold = pd.DataFrame(fold_metrics)
            folds_csv = trial_dir / "metricas_folds.csv"
            trial_json = trial_dir / "resumen_trial.json"
            df_fold.to_csv(folds_csv, index=False)
            trial_json.write_text(json.dumps({"trial_name": trial_name, "configuracion": combo, "resumen_cv": resumen}, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(folds_csv), artifact_path="trial")
            mlflow.log_artifact(str(trial_json), artifact_path="trial")

        row = dict(combo)
        row.update(resumen)
        row["trial_name"] = trial_name
        return {"ok": True, "trial_index": idx, "trial_name": trial_name, "row": row}
    except Exception as e:
        return {"ok": False, "trial_index": idx, "trial_name": trial_name, "error": str(e), "traceback": traceback.format_exc()}


def parse_float_list(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_int_list(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def build_parser():
    p = argparse.ArgumentParser(description="Grid search hiatal (script Python) con trials paralelos.")
    p.add_argument("--data-dir", default="data/images")
    p.add_argument("--output-dir", default="outputs/hiatal_grid_search_py")
    p.add_argument("--tracking-uri", default="http://localhost:8050")
    p.add_argument("--experiment", default="hernia-hiatal-grid-search")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image-size", type=int, default=320)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-splits", type=int, default=2)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--factor-samples", type=int, default=1)
    p.add_argument("--use-pretrained", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--grid-architectures", default="resnet18,densenet121,efficientnetb0")
    p.add_argument("--grid-lr", default="1e-3,3e-4,1e-4,5e-5")
    p.add_argument("--grid-dropout", default="0.2,0.3,0.4")
    p.add_argument("--grid-weight-decay", default="1e-3,1e-4")
    p.add_argument("--grid-epochs", default="6")
    p.add_argument("--grid-frozen-epochs", default="2")
    p.add_argument("--max-parallel-trials", type=int, default=0, help="0=auto")
    p.add_argument("--limit-trials", type=int, default=0)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--num-workers-train", type=int, default=-1, help="-1=auto")
    p.add_argument("--num-workers-val", type=int, default=-1, help="-1=auto")
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--amp-dtype", choices=["float16", "bfloat16"], default="float16")
    p.add_argument("--torch-compile", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--torch-compile-mode", choices=["default", "reduce-overhead", "max-autotune"], default="reduce-overhead")
    p.add_argument("--allow-gpu-oversubscription", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--log-every-epochs", type=int, default=0)
    p.add_argument("--min-score-improve", type=float, default=0.002)
    p.add_argument("--torch-threads-per-worker", type=int, default=-1, help="-1=auto")
    p.add_argument("--auto-start-mlflow-server", action=argparse.BooleanOptionalAction, default=True)
    return p


def main():
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = Path(args.data_dir) if Path(args.data_dir).is_absolute() else (repo_root / args.data_dir)
    output_dir = Path(args.output_dir) if Path(args.output_dir).is_absolute() else (repo_root / args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT"] = "true"

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if args.device == "auto":
        worker_device = "cuda" if gpu_count > 0 else "cpu"
    else:
        worker_device = args.device
    if worker_device == "cuda" and gpu_count == 0:
        raise RuntimeError("CUDA no disponible.")
    if int(args.max_parallel_trials) <= 0:
        if worker_device == "cuda":
            requested_parallel_trials = int(max(1, gpu_count))
        else:
            requested_parallel_trials = int(max(1, min(4, (os.cpu_count() or 8) // 2)))
    else:
        requested_parallel_trials = int(args.max_parallel_trials)
    if worker_device == "cuda":
        max_recomendado = max(1, gpu_count)
        if args.allow_gpu_oversubscription:
            effective_parallel_trials = max(1, requested_parallel_trials)
        else:
            effective_parallel_trials = max(1, min(requested_parallel_trials, max_recomendado))
            if requested_parallel_trials > effective_parallel_trials:
                print(
                    f"max_parallel_trials={requested_parallel_trials} excede GPUs disponibles ({gpu_count}); "
                    f"se usara {effective_parallel_trials} para evitar contencion."
                )
        if gpu_count == 1 and effective_parallel_trials > 1:
            print(
                "Aviso: 1 GPU con multiples trials en paralelo suele reducir utilizacion efectiva por contencion de contexto."
            )
        configurar_cuda_rapida()
    else:
        effective_parallel_trials = max(1, requested_parallel_trials)

    torch_threads_per_worker = resolver_torch_threads(args.torch_threads_per_worker, effective_parallel_trials)
    num_workers_train = resolver_num_workers(args.num_workers_train, True, effective_parallel_trials, worker_device)
    num_workers_val = resolver_num_workers(args.num_workers_val, False, effective_parallel_trials, worker_device)

    if args.auto_start_mlflow_server:
        iniciar_servidor_mlflow_si_falta(repo_root, output_dir, args.tracking_uri, timeout_s=1.5)

    muestras = construir_muestras_desde_carpetas(data_dir)
    y = np.array([m.etiqueta for m in muestras], dtype=int)
    grupos = np.array([m.id_paciente for m in muestras])
    indices = np.arange(len(muestras))
    folds = list(StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed).split(indices, y, groups=grupos))
    print(f"Muestras: {len(muestras)} | Pacientes unicos: {pd.Series(grupos).nunique()} | Pliegues: {len(folds)}")

    combos = [
        {"arquitectura": a, "lr": float(lr), "dropout": float(do), "weight_decay": float(wd), "epocas": int(ep), "epocas_congeladas": int(fe)}
        for (a, lr, do, wd, ep, fe) in product(
            [x.strip() for x in args.grid_architectures.split(",") if x.strip()],
            parse_float_list(args.grid_lr),
            parse_float_list(args.grid_dropout),
            parse_float_list(args.grid_weight_decay),
            parse_int_list(args.grid_epochs),
            parse_int_list(args.grid_frozen_epochs),
        )
    ]
    if args.limit_trials > 0:
        combos = combos[: args.limit_trials]
    print(
        f"Trials totales: {len(combos)} | Paralelo solicitado: {requested_parallel_trials} | "
        f"Paralelo efectivo: {effective_parallel_trials} | Device: {worker_device} | "
        f"workers(train/val): {num_workers_train}/{num_workers_val} | "
        f"torch_threads_per_worker: {torch_threads_per_worker}"
    )

    common_cfg = {
        "seed": args.seed,
        "output_dir": str(output_dir),
        "tracking_uri": args.tracking_uri,
        "experiment": args.experiment,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "factor_samples": args.factor_samples,
        "n_splits": args.n_splits,
        "use_pretrained": args.use_pretrained,
        "num_workers_train": num_workers_train,
        "num_workers_val": num_workers_val,
        "prefetch_factor": args.prefetch_factor,
        "persistent_workers": args.persistent_workers,
        "channels_last": args.channels_last,
        "use_amp": args.use_amp,
        "amp_dtype": args.amp_dtype,
        "torch_compile": args.torch_compile,
        "torch_compile_mode": args.torch_compile_mode,
        "log_every_epochs": args.log_every_epochs,
        "min_score_improve": args.min_score_improve,
        "torch_threads_per_worker": torch_threads_per_worker,
    }

    sample_payload = [(str(m.ruta), int(m.etiqueta), str(m.id_paciente)) for m in muestras]
    fold_payload = [(a.tolist(), b.tolist()) for a, b in folds]
    run_prefix = f"grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def mk_payload(i: int, combo: dict):
        name = f"{run_prefix}_trial{i:03d}_{combo['arquitectura']}_lr{combo['lr']}_do{combo['dropout']}_wd{combo['weight_decay']}_e{combo['epocas']}"
        cfg = dict(common_cfg)
        if worker_device == "cuda":
            cfg["worker_device"] = f"cuda:{(i - 1) % max(1, gpu_count)}"
        else:
            cfg["worker_device"] = "cpu"
        return {"cfg": cfg, "trial_index": i, "trial_name": name, "combo": combo, "samples": sample_payload, "folds": fold_payload}

    rows, errors = [], []
    if effective_parallel_trials <= 1:
        for i, combo in enumerate(combos, start=1):
            print(f"Trial {i}/{len(combos)}")
            res = run_trial(mk_payload(i, combo))
            if res["ok"]:
                rows.append(res["row"])
            else:
                errors.append(res)
                print(f"ERROR {res['trial_index']}: {res['trial_name']} -> {res['error']}")
    else:
        with ProcessPoolExecutor(max_workers=effective_parallel_trials) as ex:
            fut_to_i = {ex.submit(run_trial, mk_payload(i, combo)): i for i, combo in enumerate(combos, start=1)}
            for fut in as_completed(fut_to_i):
                res = fut.result()
                if res["ok"]:
                    rows.append(res["row"])
                    print(f"OK {res['trial_index']}: {res['trial_name']}")
                else:
                    errors.append(res)
                    print(f"ERROR {res['trial_index']}: {res['trial_name']} -> {res['error']}")

    if not rows:
        if errors:
            (output_dir / "grid_trial_errors.json").write_text(json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8")
        raise RuntimeError("No hubo trials exitosos.")

    df = pd.DataFrame(rows).sort_values(["auc_mean", "f1_mean"], ascending=False).reset_index(drop=True)
    ruta_resultados = output_dir / "grid_results.csv"
    df.to_csv(ruta_resultados, index=False)
    print(df.head(20).to_string(index=False))
    print(f"Resultados: {ruta_resultados}")

    resumen = {
        "fecha": datetime.now().isoformat(),
        "n_trials_ok": int(len(df)),
        "n_trials_error": int(len(errors)),
        "mejor_trial": df.iloc[0].to_dict(),
        "worker_device": worker_device,
        "max_parallel_trials_requested": int(requested_parallel_trials),
        "max_parallel_trials_effective": int(effective_parallel_trials),
    }
    ruta_resumen = output_dir / "grid_summary.json"
    ruta_resumen.write_text(json.dumps(resumen, indent=2), encoding="utf-8")
    if errors:
        (output_dir / "grid_trial_errors.json").write_text(json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8")

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=f"{run_prefix}_resumen_archivos"):
        mlflow.log_param("n_trials_ok", int(len(df)))
        mlflow.log_param("n_trials_error", int(len(errors)))
        mlflow.log_param("max_parallel_trials_requested", int(requested_parallel_trials))
        mlflow.log_param("max_parallel_trials_effective", int(effective_parallel_trials))
        mlflow.log_param("worker_device", worker_device)
        mlflow.log_param("num_workers_train", int(num_workers_train))
        mlflow.log_param("num_workers_val", int(num_workers_val))
        mlflow.log_param("torch_threads_per_worker", int(torch_threads_per_worker))
        mlflow.log_param("log_every_epochs", int(args.log_every_epochs))
        for k, v in df.iloc[0].to_dict().items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                vv = float(v)
                if np.isfinite(vv):
                    mlflow.log_metric(f"mejor.{k}", vv)
        mlflow.log_artifact(str(ruta_resultados), artifact_path="salidas")
        mlflow.log_artifact(str(ruta_resumen), artifact_path="salidas")


if __name__ == "__main__":
    main()
