from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from pydantic import BaseModel

app = FastAPI()

# Cargar modelos
modelo_temp = joblib.load("modelos/modelo_temperatura.pkl")
modelo_pm10 = joblib.load("modelos/modelo_pm10.pkl")
modelo_lluvia = joblib.load("modelos/modelo_lluvia_cluster.pkl")

# Configurar CORS para permitir todas las solicitudes (puedes restringir el origen si quieres)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cambia "*" por la URL de tu app si quieres restringir
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos HTTP (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "API funcionando"}

class DatosEntrada(BaseModel):
    HORA: float
    LLUVIA_CIUDAD: float
    NO2_CIUDAD: float
    O3_CIUDAD: float
    TEMPERATURA_CIUDAD: float
    PM10_CIUDAD: float

@app.post("/predict/")
def predecir(datos: DatosEntrada):
    # Corrigiendo el nombre del campo
    entrada_temp = np.array([[datos.HORA, datos.LLUVIA_CIUDAD, datos.NO2_CIUDAD, datos.O3_CIUDAD]])
    temp = modelo_temp.predict(entrada_temp)

    entrada_pm10 = np.array([[datos.HORA, datos.NO2_CIUDAD, datos.O3_CIUDAD, datos.TEMPERATURA_CIUDAD]])
    pm10 = modelo_pm10.predict(entrada_pm10)

    entrada_lluvia = np.array([[datos.HORA, datos.TEMPERATURA_CIUDAD, datos.NO2_CIUDAD, datos.O3_CIUDAD, datos.PM10_CIUDAD]])
    lluvia = modelo_lluvia.predict(entrada_lluvia)

    return {
        "prediccion_temperatura": float(temp[0]),
        "prediccion_pm10": float(pm10[0]),
        "cluster_lluvia": int(lluvia[0])
    }
