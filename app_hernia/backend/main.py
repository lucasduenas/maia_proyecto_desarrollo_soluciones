from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from model import ServicioInferenciaHiatal

app = FastAPI(title="HernIA Inference API")

MODEL_PATH = Path("model/production_bundle.pt")

servicio = None


class InferRequest(BaseModel):
    image_path: str


@app.on_event("startup")
def cargar_modelo():
    global servicio

    try:
        servicio = ServicioInferenciaHiatal(MODEL_PATH)
    except Exception as e:
        servicio = None
        print("Error cargando modelo:", e)


@app.get("/health")
def health():

    return {
        "ok": servicio is not None
    }


@app.post("/infer")
def infer(req: InferRequest):

    if servicio is None:
        raise HTTPException(503, "Modelo no disponible")

    try:

        result = servicio.inferir_por_ruta(req.image_path)

        return {
            "ok": True,
            "result": result
        }

    except Exception as e:

        raise HTTPException(500, str(e))