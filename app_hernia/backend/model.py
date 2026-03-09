import torch
import torchvision.transforms as T
from pathlib import Path
from typing import Any, Optional
from torch import nn
from torchvision.models import densenet121
from PIL import Image, ImageOps

DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]
DEFAULT_TAM_IMAGEN = 512
DEFAULT_DROPOUT = 0.25


class RecorteRetrocardiacoInferencia:
    def __init__(self, x1=0.2, x2=0.8, y1=0.15, y2=0.98):
        self.x1, self.x2, self.y1, self.y2 = x1, x2, y1, y2

    def __call__(self, imagen):
        ancho, alto = imagen.size
        return imagen.crop(
            (int(self.x1 * ancho), int(self.y1 * alto), int(self.x2 * ancho), int(self.y2 * alto))
        )


def crear_modelo_inferencia(dropout=DEFAULT_DROPOUT):
    modelo = densenet121(weights=None)
    in_features = modelo.classifier.in_features
    modelo.classifier = nn.Sequential(
        nn.Dropout(p=float(dropout)),
        nn.Linear(in_features, 1),
    )
    return modelo


def construir_modelos_desde_estados(estados_modelo, dispositivo, dropout):
    modelos = []

    for estado in estados_modelo:
        modelo = crear_modelo_inferencia(dropout=dropout).to(dispositivo)
        modelo.load_state_dict(estado)
        modelo.eval()
        modelos.append(modelo)

    return modelos


@torch.no_grad()
def predecir_tensor_ensemble(modelos, tensor):
    probs = [torch.sigmoid(modelo(tensor))[0, 0] for modelo in modelos]
    return float(torch.stack(probs).mean().item())


class ServicioInferenciaHiatal:
    def __init__(self, ruta_bundle, dispositivo=None):

        self.ruta_bundle = Path(ruta_bundle)

        self.dispositivo = dispositivo or (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        self.modelos = []
        self.umbral = 0.5
        self.tam_imagen = DEFAULT_TAM_IMAGEN

        self._cargar_bundle()

    def _cargar_bundle(self):

        bundle = torch.load(self.ruta_bundle, map_location=self.dispositivo)

        estados = bundle["model_state_dicts"]

        self.modelos = construir_modelos_desde_estados(
            estados,
            self.dispositivo,
            DEFAULT_DROPOUT,
        )

        self.transform = T.Compose([
            RecorteRetrocardiacoInferencia(),
            T.Resize((self.tam_imagen, self.tam_imagen)),
            T.ToTensor(),
            T.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD)
        ])

    @torch.no_grad()
    def inferir_por_ruta(self, image_path):

        with Image.open(image_path) as img:

            gris = img.convert("L")
            gris = ImageOps.autocontrast(gris)

            rgb = Image.merge("RGB", (gris, gris, gris))

        tensor = self.transform(rgb).unsqueeze(0).to(self.dispositivo)

        prob = predecir_tensor_ensemble(self.modelos, tensor)

        pred = int(prob >= self.umbral)

        return {
            "prob_hernia": float(prob),
            "threshold": self.umbral,
            "pred": pred,
            "pred_texto": "Hernia" if pred == 1 else "Normal"
        }