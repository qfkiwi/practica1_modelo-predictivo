from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import joblib
import pandas as pd
import os
import uvicorn
import unicodedata

MAE_MODELO = 0.874

app = FastAPI()

# ✅ CARGA SOLO UNA VEZ
modelo = joblib.load("modelo.pkl")
preprocesador = joblib.load("preprocesador.pkl")

def normalizar_texto(texto):
    """Normaliza el texto removiendo acentos y convirtiendo a minúsculas."""
    if not isinstance(texto, str):
        return texto
    # Normalizar a forma NFD (descomponer caracteres)
    texto_normalizado = unicodedata.normalize('NFD', texto)
    # Remover caracteres de combinación (acentos)
    texto_sin_acentos = ''.join(c for c in texto_normalizado if unicodedata.category(c) != 'Mn')
    return texto_sin_acentos.lower()


@app.get("/", response_class=HTMLResponse)
def formulario():
    return """
    <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Predictor de Peritación</title>
            <style>
                * {
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    padding: 40px 20px;
                }
                .container {
                    background: white;
                    border-radius: 10px;
                    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
                    padding: 160px 40px 40px 40px;
                    max-width: 500px;
                    width: 100%;
                }
                h2 {
                    color: #333;
                    margin-bottom: 30px;
                    text-align: center;
                    font-size: 28px;
                }
                .form-group {
                    margin-bottom: 16px;
                }
                label {
                    display: block;
                    margin-bottom: 6px;
                    color: #555;
                    font-weight: 500;
                    font-size: 14px;
                }
                input {
                    width: 100%;
                    padding: 12px;
                    border: 2px solid #e0e0e0;
                    border-radius: 5px;
                    font-size: 14px;
                    transition: border-color 0.3s;
                    font-family: inherit;
                }
                input:focus {
                    outline: none;
                    border-color: #667eea;
                    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
                }
                button {
                    width: 100%;
                    padding: 14px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    border: none;
                    border-radius: 5px;
                    font-size: 16px;
                    font-weight: 600;
                    cursor: pointer;
                    transition: transform 0.2s, box-shadow 0.2s;
                    margin-top: 10px;
                }
                button:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h2>🔍 Predictor de Peritación</h2>
                <form action="/predict" method="post">
                    <div class="form-group">
                        <label>Compañía</label>
                        <input type="text" name="compania" placeholder="Mapfre" required>
                    </div>
                    <div class="form-group">
                        <label>Videoperitación</label>
                        <input type="text" name="videoperitacion" placeholder="No" required>
                    </div>
                    <div class="form-group">
                        <label>Fecha entrada</label>
                        <input type="date" name="fecha_entrada" required>
                    </div>
                    <div class="form-group">
                        <label>Perito</label>
                        <input type="text" name="perito" placeholder="Perito B" required>
                    </div>
                    <div class="form-group">
                        <label>Provincia</label>
                        <input type="text" name="provincia" placeholder="Zaragoza" required>
                    </div>
                    <div class="form-group">
                        <label>Honorario gabinete (€)</label>
                        <input type="number" name="honorario_gabinete" placeholder="32" step="0.01" required>
                    </div>
                    <div class="form-group">
                        <label>Honorario perito (€)</label>
                        <input type="number" name="honorario_perito" placeholder="22" step="0.01" required>
                    </div>
                    <div class="form-group">
                        <label>Tasación origen (€)</label>
                        <input type="number" name="tasacion_origen" placeholder="3972" step="0.01" required>
                    </div>
                    <button type="submit">Predecir duración</button>
                </form>
            </div>
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

        # Normalizar texto para ignorar acentos y mayúsculas/minúsculas
        df["provincia"] = df["provincia"].apply(normalizar_texto)
        df["perito"] = df["perito"].apply(normalizar_texto)
        df["compania"] = df["compania"].apply(normalizar_texto)

        df["provincia"] = preprocesador["provincia"].transform(df["provincia"])
        df["perito"] = preprocesador["perito"].transform(df["perito"])
        df["compania"] = preprocesador["compania"].transform(df["compania"])

        pred = modelo.predict(df)[0]

        return f"""
<html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Resultado de Predicción</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 40px 20px;
            }}
            .container {{
                background: white;
                border-radius: 10px;
                box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
                padding: 40px;
                max-width: 600px;
                width: 100%;
            }}
            h2 {{
                color: #333;
                margin-bottom: 30px;
                text-align: center;
                font-size: 28px;
            }}
            .result-box {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                text-align: center;
                margin-bottom: 30px;
            }}
            .result-box h3 {{
                font-size: 18px;
                margin-bottom: 15px;
                opacity: 0.9;
            }}
            .result-value {{
                font-size: 48px;
                font-weight: bold;
                margin-bottom: 10px;
            }}
            .result-label {{
                font-size: 16px;
                opacity: 0.9;
            }}
            .mae-info {{
                background: #f0f4ff;
                border-left: 4px solid #667eea;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 30px;
                color: #555;
                font-size: 14px;
            }}
            .form-group {{
                margin-bottom: 14px;
            }}
            label {{
                display: block;
                margin-bottom: 5px;
                color: #777;
                font-weight: 500;
                font-size: 12px;
            }}
            input {{
                width: 100%;
                padding: 10px;
                border: 2px solid #e0e0e0;
                border-radius: 5px;
                font-size: 13px;
                transition: border-color 0.3s;
                font-family: inherit;
            }}
            input:focus {{
                outline: none;
                border-color: #667eea;
            }}
            button {{
                width: 100%;
                padding: 12px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 15px;
                font-weight: 600;
                cursor: pointer;
                transition: transform 0.2s, box-shadow 0.2s;
            }}
            button:hover {{
                transform: translateY(-2px);
                box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
            }}
            .back-link {{
                display: inline-block;
                margin-top: 20px;
                color: #667eea;
                text-decoration: none;
                font-weight: 600;
                transition: color 0.2s;
            }}
            .back-link:hover {{
                color: #764ba2;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>📊 Resultado de la Predicción</h2>
            
            <div class="result-box">
                <h3>Duración estimada:</h3>
                <div class="result-value">{round(pred, 2)}</div>
                <div class="result-label">días</div>
            </div>

            <div class="mae-info">
                ℹ️ <strong>Precisión del modelo:</strong> Error promedio de ±{MAE_MODELO} días
            </div>

            <h3 style="color: #333; margin-bottom: 20px; margin-top: 20px;">Ajusta los parámetros y prueba de nuevo:</h3>
            <form action="/predict" method="post">
                <div class="form-group">
                    <label>Compañía</label>
                    <input type="text" name="compania" value="{compania}" required>
                </div>
                <div class="form-group">
                    <label>Videoperitación</label>
                    <input type="text" name="videoperitacion" value="{videoperitacion}" required>
                </div>
                <div class="form-group">
                    <label>Fecha entrada</label>
                    <input type="text" name="fecha_entrada" value="{fecha_entrada}" required>
                </div>
                <div class="form-group">
                    <label>Perito</label>
                    <input type="text" name="perito" value="{perito}" required>
                </div>
                <div class="form-group">
                    <label>Provincia</label>
                    <input type="text" name="provincia" value="{provincia}" required>
                </div>
                <div class="form-group">
                    <label>Honorario gabinete (€)</label>
                    <input type="number" name="honorario_gabinete" value="{honorario_gabinete}" step="0.01" required>
                </div>
                <div class="form-group">
                    <label>Honorario perito (€)</label>
                    <input type="number" name="honorario_perito" value="{honorario_perito}" step="0.01" required>
                </div>
                <div class="form-group">
                    <label>Tasación origen (€)</label>
                    <input type="number" name="tasacion_origen" value="{tasacion_origen}" step="0.01" required>
                </div>
                <button type="submit">🔄 Nueva predicción</button>
            </form>
            <a href="/" class="back-link">← Volver al inicio</a>
        </div>
    </body>
</html>
"""

    except Exception as e:
        return f"""
        <html>
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Error</title>
                <style>
                    * {{
                        margin: 0;
                        padding: 0;
                        box-sizing: border-box;
                    }}
                    body {{
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                        min-height: 100vh;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        padding: 40px 20px;
                    }}
                    .container {{
                        background: white;
                        border-radius: 10px;
                        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
                        padding: 40px;
                        max-width: 500px;
                        width: 100%;
                        text-align: center;
                    }}
                    h2 {{
                        color: #f5576c;
                        margin-bottom: 20px;
                        font-size: 28px;
                    }}
                    p {{
                        color: #666;
                        margin-bottom: 25px;
                        line-height: 1.6;
                        font-size: 15px;
                    }}
                    .error-code {{
                        background: #fff3cd;
                        border-left: 4px solid #f5576c;
                        padding: 15px;
                        border-radius: 5px;
                        margin-bottom: 25px;
                        color: #856404;
                        text-align: left;
                        font-size: 13px;
                        font-family: 'Courier New', monospace;
                        overflow-x: auto;
                    }}
                    a {{
                        display: inline-block;
                        margin-top: 15px;
                        padding: 12px 30px;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        text-decoration: none;
                        border-radius: 5px;
                        font-weight: 600;
                        transition: transform 0.2s, box-shadow 0.2s;
                    }}
                    a:hover {{
                        transform: translateY(-2px);
                        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h2>⚠️ Error en la predicción</h2>
                    <p>Algo no fue bien al procesar tu solicitud. Por favor, verifica los datos e intenta nuevamente.</p>
                    <div class="error-code">
                        {str(e)}
                    </div>
                    <a href="/">← Volver al inicio</a>
                </div>
            </body>
        </html>
        """
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)