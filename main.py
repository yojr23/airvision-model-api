from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

app = FastAPI()

modelo_temp = joblib.load("modelos/modelo_temperatura.pkl")
modelo_pm10 = joblib.load("modelos/modelo_pm10.pkl")
modelo_lluvia = joblib.load("modelos/modelo_lluvia_cluster.pkl")

class DatosEntrada(BaseModel):
    HORA: float
    LLUVIA_CIUDAD: float
    NO2_CIUDAD: float
    O3_CIUDAD: float
    TEMPERATURA_CIUDAD: float
    PM10_CIUDAD: float

@app.post("/predict/")
def predecir(datos: DatosEntrada):
    entrada_temp = np.array([[datos.HORA, datos.LLUVIACIUAD, datos.NO2_CIUDAD, datos.O3_CIUDAD]])
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
