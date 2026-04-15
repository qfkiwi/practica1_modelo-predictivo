from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import joblib
import pandas as pd
import os
import uvicorn

MAE_MODELO = 0.8742

app = FastAPI()

# ✅ CARGA SOLO UNA VEZ
modelo = joblib.load("modelo.pkl")
preprocesador = joblib.load("preprocesador.pkl")


@app.get("/", response_class=HTMLResponse)
def formulario():
    return """
    <html>
        <body>
            <h2>Predicción de duración de peritación</h2>
            <form action="/predict" method="post">
                Compañía: <input name="compania"><br>
                Videoperitación: <input name="videoperitacion"><br>
                Fecha entrada: <input name="fecha_entrada"><br>
                Perito: <input name="perito"><br>
                Provincia: <input name="provincia"><br>
                Honorario gabinete: <input name="honorario_gabinete"><br>
                Honorario perito: <input name="honorario_perito"><br>
                Tasación origen: <input name="tasacion_origen"><br><br>

                <input type="submit" value="Predecir">
            </form>
        </body>
    </html>
    """
@app.post("/predict", response_class=HTMLResponse)
def predict(
    compania: str = Form(...),
    videoperitacion: str = Form(...),
    fecha_entrada: str = Form(...),
    perito: str = Form(...),
    provincia: str = Form(...),
    honorario_gabinete: float = Form(...),
    honorario_perito: float = Form(...),
    tasacion_origen: float = Form(...)
):
    try:
        df = pd.DataFrame([{
            "compania": compania,
            "videoperitacion": videoperitacion,
            "fecha_entrada": fecha_entrada,
            "perito": perito,
            "provincia": provincia,
            "honorario_gabinete": honorario_gabinete,
            "honorario_perito": honorario_perito,
            "tasacion_origen": tasacion_origen
        }])

        df["fecha_entrada"] = pd.to_datetime(df["fecha_entrada"])
        df["mes"] = df["fecha_entrada"].dt.month
        df["dia_semana"] = df["fecha_entrada"].dt.dayofweek

        df["provincia"] = preprocesador["provincia"].transform(df["provincia"])
        df["perito"] = preprocesador["perito"].transform(df["perito"])
        df["compania"] = preprocesador["compania"].transform(df["compania"])

        pred = modelo.predict(df)[0]

        return f"""
<html>
    <body>
        <h2>Predicción de duración de peritación</h2>

        <form action="/predict" method="post">
            Compañía: <input name="compania"><br>
            Videoperitación: <input name="videoperitacion"><br>
            Fecha entrada: <input name="fecha_entrada"><br>
            Perito: <input name="perito"><br>
            Provincia: <input name="provincia"><br>
            Honorario gabinete: <input name="honorario_gabinete"><br>
            Honorario perito: <input name="honorario_perito"><br>
            Tasación origen: <input name="tasacion_origen"><br><br>

            <input type="submit" value="Predecir">
        </form>

        <h3>Resultado:</h3>
        <p><b>{round(pred, 2)} días</b></p>
        <p><i>(MAE del modelo: {MAE_MODELO} días)</i></p>
    </body>
</html>
"""

    except Exception as e:
        return f"""
        <html>
            <body>
                <h2>Error</h2>
                <p>{str(e)}</p>
                <a href="/">← Volver</a>
            </body>
        </html>
        """
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)