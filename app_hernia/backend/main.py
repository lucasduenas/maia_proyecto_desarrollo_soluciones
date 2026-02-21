from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from model import predict_hernia

app = FastAPI(title="Hernia Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Flask will call this
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = predict_hernia(image_bytes)
    return result