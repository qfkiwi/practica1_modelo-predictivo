from fastapi import FastAPI
import joblib
import pandas as pd
import os
import uvicorn

app = FastAPI()

# ✅ CARGA SOLO UNA VEZ
modelo = joblib.load("modelo.pkl")
preprocesador = joblib.load("preprocesador.pkl")


@app.post("/predict")
def predict(data: dict):
    try:
        df = pd.DataFrame([data])

        # --- FECHA ---
        df["fecha_entrada"] = pd.to_datetime(df["fecha_entrada"])
        df["mes"] = df["fecha_entrada"].dt.month
        df["dia_semana"] = df["fecha_entrada"].dt.dayofweek

        # --- ASEGURAR NUMÉRICAS ---
        df["honorario_gabinete"] = data.get("honorario_gabinete", 0)
        df["honorario_perito"] = data.get("honorario_perito", 0)
        df["tasacion_origen"] = data.get("tasacion_origen", 0)

        # --- TRANSFORMAR CATEGÓRICAS ---
        df["provincia"] = preprocesador["provincia"].transform(df["provincia"])
        df["perito"] = preprocesador["perito"].transform(df["perito"])
        df["compania"] = preprocesador["compania"].transform(df["compania"])

        # ⚠️ SI USASTE ESTO
        df["videoperitacion"] = preprocesador["videoperitacion"].transform(df["videoperitacion"])

        # --- ORDEN EXACTO ---
        df = df[[
            "provincia",
            "perito",
            "compania",
            "videoperitacion",
            "mes",
            "dia_semana",
            "honorario_gabinete",
            "honorario_perito",
            "tasacion_origen"
        ]]

        pred = modelo.predict(df)

        return {"prediccion": float(pred[0])}

    except Exception as e:
        return {"error": str(e)}
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)