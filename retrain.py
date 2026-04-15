import pandas as pd
import unicodedata
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import numpy as np

def normalizar_texto(texto):
    """Normaliza el texto removiendo acentos y convirtiendo a minúsculas."""
    if not isinstance(texto, str):
        return texto
    # Normalizar a forma NFD (descomponer caracteres)
    texto_normalizado = unicodedata.normalize('NFD', texto)
    # Remover caracteres de combinación (acentos)
    texto_sin_acentos = ''.join(c for c in texto_normalizado if unicodedata.category(c) != 'Mn')
    return texto_sin_acentos.lower()

# Cargar datos
df = pd.read_csv('dataset_peritacion_10000.csv', encoding='latin-1')

# Normalizar columnas categóricas
df["provincia"] = df["provincia"].apply(normalizar_texto)
df["perito"] = df["perito"].apply(normalizar_texto)
df["compania"] = df["compania"].apply(normalizar_texto)

# Procesar fecha
df["fecha_entrada"] = pd.to_datetime(df["fecha_entrada"])
df["mes"] = df["fecha_entrada"].dt.month
df["dia_semana"] = df["fecha_entrada"].dt.dayofweek

# Codificar variables categóricas
le_provincia = LabelEncoder()
df["provincia"] = le_provincia.fit_transform(df["provincia"])

le_perito = LabelEncoder()
df["perito"] = le_perito.fit_transform(df["perito"])

le_compania = LabelEncoder()
df["compania"] = le_compania.fit_transform(df["compania"])

le_video = LabelEncoder()
df["videoperitacion"] = le_video.fit_transform(df["videoperitacion"])

# Preparar datos para el modelo
X = df[[ 
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

y = df["tiempo_resolucion_dias"]

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluar
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae}")

# Guardar modelo y preprocesadores
joblib.dump(model, "modelo.pkl")

preprocesador = {
    "provincia": le_provincia,
    "perito": le_perito,
    "compania": le_compania,
    "videoperitacion": le_video
}

joblib.dump(preprocesador, "preprocesador.pkl")

print("Modelo reentrenado y guardado con datos normalizados.")