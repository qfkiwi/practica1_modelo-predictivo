# API Predicción Tiempo de Resolución

## Descripción
API desarrollada con FastAPI para predecir el tiempo de resolución de expedientes de peritación.

## Uso

### Endpoint
POST /predict

### Ejemplo JSON

{
  "compania": "Allianz",
  "videoperitacion": "No",
  "fecha_entrada": "2024-09-08",
  "perito": "Perito B",
  "provincia": "Sevilla",
  "honorario_gabinete": 200,
  "honorario_perito": 150,
  "tasacion_origen": 1000
}

## Ejecución local

uvicorn api:app --reload

## La pagina se puede ejecutar en railway
