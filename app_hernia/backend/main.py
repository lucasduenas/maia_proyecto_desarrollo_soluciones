from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import torch
import torchvision.transforms as T
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from torch import nn
from torchvision.models import densenet121
from PIL import Image, ImageOps


DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]
DEFAULT_TAM_IMAGEN = 512
DEFAULT_DROPOUT = 0.25

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE_PATH = PROJECT_ROOT / "app_hernia" / "model" / "production_bundle.pt"


class RecorteRetrocardiacoInferencia:
    def __init__(self, x1: float = 0.2, x2: float = 0.8, y1: float = 0.15, y2: float = 0.98):
        self.x1, self.x2, self.y1, self.y2 = x1, x2, y1, y2

    def __call__(self, imagen: Image.Image):
        ancho, alto = imagen.size
        return imagen.crop((int(self.x1 * ancho), int(self.y1 * alto), int(self.x2 * ancho), int(self.y2 * alto)))


def crear_modelo_inferencia(dropout: float = DEFAULT_DROPOUT) -> nn.Module:
    modelo = densenet121(weights=None)
    in_features = modelo.classifier.in_features
    modelo.classifier = nn.Sequential(nn.Dropout(p=float(dropout)), nn.Linear(in_features, 1))
    return modelo


def construir_modelos_desde_estados(
    estados_modelo: list[dict[str, torch.Tensor]], dispositivo: torch.device, dropout: float
) -> list[nn.Module]:
    modelos = []
    for estado in estados_modelo:
        modelo = crear_modelo_inferencia(dropout=dropout).to(dispositivo)
        modelo.load_state_dict(estado, strict=True)
        modelo.eval()
        modelos.append(modelo)
    return modelos


@torch.no_grad()
def predecir_tensor_ensemble(modelos: list[nn.Module], tensor: torch.Tensor) -> float:
    if len(modelos) == 0:
        raise RuntimeError("No hay modelos cargados para inferencia.")
    probs = [torch.sigmoid(modelo(tensor))[0, 0] for modelo in modelos]
    return float(torch.stack(probs).mean().item())


class ServicioInferenciaHiatal:
    def __init__(self, ruta_bundle: str | Path, dispositivo: Optional[torch.device] = None):
        self.ruta_bundle = Path(ruta_bundle)
        self.dispositivo = dispositivo or (
            torch.device("cuda")
            if torch.cuda.is_available()
            else (torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
        )

        self.modelos: list[nn.Module] = []
        self.umbral: float = 0.5
        self.tam_imagen: int = DEFAULT_TAM_IMAGEN
        self.usar_roi: bool = True
        self.usar_autocontraste: bool = True
        self.mean: list[float] = list(DEFAULT_MEAN)
        self.std: list[float] = list(DEFAULT_STD)
        self.dropout: float = DEFAULT_DROPOUT
        self.transformacion: Optional[Any] = None

        self._cargar_bundle()

    def _cargar_bundle(self):
        if not self.ruta_bundle.exists():
            raise RuntimeError(f"No existe bundle en: {self.ruta_bundle}")

        bundle = torch.load(self.ruta_bundle, map_location=self.dispositivo, weights_only=False)
        if not isinstance(bundle, dict):
            raise RuntimeError("Bundle invalido: estructura no es dict.")

        estados = bundle.get("model_state_dicts", [])
        if not isinstance(estados, list) or len(estados) == 0:
            raise RuntimeError("Bundle invalido: model_state_dicts vacio o ausente.")

        cfg_raw = bundle.get("config", {})
        cfg = cfg_raw if isinstance(cfg_raw, dict) else {}
        norm_raw = cfg.get("normalizacion", {})
        cfg_norm = norm_raw if isinstance(norm_raw, dict) else {}

        self.umbral = float(bundle.get("global_threshold", bundle.get("threshold", 0.5)))
        self.tam_imagen = int(cfg.get("tam_imagen", DEFAULT_TAM_IMAGEN))
        self.usar_roi = bool(cfg.get("roi_retrocardiaco", cfg.get("usar_roi", True)))
        self.usar_autocontraste = bool(cfg.get("autocontraste", cfg.get("usar_autocontraste", True)))
        self.dropout = float(cfg.get("dropout", DEFAULT_DROPOUT))

        mean = cfg_norm.get("mean", DEFAULT_MEAN)
        std = cfg_norm.get("std", DEFAULT_STD)
        self.mean = list(mean) if isinstance(mean, (list, tuple)) and len(mean) == 3 else list(DEFAULT_MEAN)
        self.std = list(std) if isinstance(std, (list, tuple)) and len(std) == 3 else list(DEFAULT_STD)

        self.modelos = construir_modelos_desde_estados(estados, self.dispositivo, self.dropout)
        self.transformacion = self._construir_transformacion()

    def _construir_transformacion(self):
        ops = []
        if self.usar_roi:
            ops.append(RecorteRetrocardiacoInferencia())
        ops.extend(
            [
                T.Resize((self.tam_imagen, self.tam_imagen)),
                T.ToTensor(),
                T.Normalize(mean=self.mean, std=self.std),
            ]
        )
        return T.Compose(ops)

    @torch.no_grad()
    def inferir_por_ruta(self, image_path: str | Path) -> dict[str, Any]:
        if self.transformacion is None:
            raise RuntimeError("Servicio no inicializado correctamente.")

        ruta_imagen = Path(image_path)
        if not ruta_imagen.exists() or not ruta_imagen.is_file():
            raise FileNotFoundError(f"Ruta de imagen invalida: {ruta_imagen}")

        with Image.open(ruta_imagen) as imagen:
            gris = imagen.convert("L")
            if self.usar_autocontraste:
                gris = ImageOps.autocontrast(gris)
            rgb = Image.merge("RGB", (gris, gris, gris))

        tensor = self.transformacion(rgb).unsqueeze(0).to(self.dispositivo)
        prob = predecir_tensor_ensemble(self.modelos, tensor)
        pred = int(prob >= self.umbral)

        return {
            "image_path": str(ruta_imagen),
            "prob_hernia": float(prob),
            "threshold": float(self.umbral),
            "pred": pred,
            "pred_texto": "Hernia" if pred == 1 else "Normal",
            "n_modelos_ensemble": int(len(self.modelos)),
            "tam_imagen": int(self.tam_imagen),
            "dispositivo": str(self.dispositivo),
        }


class InferRequest(BaseModel):
    image_path: str = Field(..., description="Ruta absoluta o relativa a la imagen en disco.")


class InferResult(BaseModel):
    image_path: str
    prob_hernia: float
    threshold: float
    pred: int
    pred_texto: str
    n_modelos_ensemble: int
    tam_imagen: int
    dispositivo: str


class InferResponse(BaseModel):
    ok: bool
    result: Optional[InferResult] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    ok: bool
    model_loaded: bool
    bundle_path: str
    detail: str


app = FastAPI(title="Hiatal Hernia Inference API", version="1.0.0")
SERVICIO_INFERENCIA: Optional[ServicioInferenciaHiatal] = None
SERVICIO_ERROR: Optional[str] = None


def _resolver_ruta_bundle() -> Path:
    env_path = os.getenv("HIATAL_BUNDLE_PATH", "").strip()
    if env_path:
        return Path(env_path)

    candidatos = [
        DEFAULT_BUNDLE_PATH,
    ]
    for ruta in candidatos:
        if ruta.exists():
            return ruta

    return DEFAULT_BUNDLE_PATH


def _resolver_dispositivo() -> torch.device:
    env_device = os.getenv("HIATAL_DEVICE", "").strip().lower()
    if env_device in {"cpu", "cuda", "mps"}:
        return torch.device(env_device)
    return (
        torch.device("cuda")
        if torch.cuda.is_available()
        else (torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
    )


@app.on_event("startup")
def startup_event():
    global SERVICIO_INFERENCIA, SERVICIO_ERROR

    ruta_bundle = _resolver_ruta_bundle()
    dispositivo = _resolver_dispositivo()
    try:
        SERVICIO_INFERENCIA = ServicioInferenciaHiatal(ruta_bundle=ruta_bundle, dispositivo=dispositivo)
        SERVICIO_ERROR = None
    except Exception as exc:
        SERVICIO_INFERENCIA = None
        SERVICIO_ERROR = str(exc)


def _obtener_servicio() -> ServicioInferenciaHiatal:
    if SERVICIO_INFERENCIA is None:
        detalle = SERVICIO_ERROR or "Servicio no inicializado."
        raise HTTPException(status_code=503, detail=f"Modelo no disponible: {detalle}")
    return SERVICIO_INFERENCIA


@app.get("/health", response_model=HealthResponse)
def health():
    ruta_bundle = _resolver_ruta_bundle()
    if SERVICIO_INFERENCIA is None:
        return HealthResponse(
            ok=False,
            model_loaded=False,
            bundle_path=str(ruta_bundle),
            detail=SERVICIO_ERROR or "Modelo no cargado.",
        )
    return HealthResponse(
        ok=True,
        model_loaded=True,
        bundle_path=str(ruta_bundle),
        detail=f"Servicio listo con {len(SERVICIO_INFERENCIA.modelos)} modelos.",
    )


@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest):
    servicio = _obtener_servicio()
    try:
        resultado = servicio.inferir_por_ruta(req.image_path)
        return InferResponse(ok=True, result=InferResult(**resultado), error=None)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error de inferencia: {exc}") from exc


def manejar_solicitud_inferencia(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"ok": False, "error": "payload debe ser un dict con la llave image_path"}

    image_path = payload.get("image_path")
    if not isinstance(image_path, str) or not image_path.strip():
        return {"ok": False, "error": "image_path es obligatorio y debe ser string"}

    try:
        resultado = _obtener_servicio().inferir_por_ruta(image_path.strip())
        return {"ok": True, "result": resultado}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app_hernia.main:app", host="0.0.0.0", port=8000, reload=False)
