from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# ✅ CARGA SOLO UNA VEZ
modelo = joblib.load("modelo.pkl")
preprocesador = joblib.load("preprocesador.pkl")


@app.post("/predict")
def predict(data: dict):
    try:
        df = pd.DataFrame([data])

        df["fecha_entrada"] = pd.to_datetime(df["fecha_entrada"])
        df["mes"] = df["fecha_entrada"].dt.month
        df["dia_semana"] = df["fecha_entrada"].dt.dayofweek

        df["provincia"] = preprocesador["provincia"].transform(df["provincia"])
        df["perito"] = preprocesador["perito"].transform(df["perito"])
        df["compania"] = preprocesador["compania"].transform(df["compania"])
        df["videoperitacion"] = preprocesador["videoperitacion"].transform(df["videoperitacion"])

        X = df[[
            "compania",
            "videoperitacion",
            "perito",
            "provincia",
            "mes",
            "dia_semana"
        ]]

        pred = modelo.predict(X)

        return {"prediccion_dias": float(pred[0])}

    except Exception as e:
        return {"error": str(e)}