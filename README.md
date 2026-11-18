# Examen 1BIM — Recuperación de Información
### Estudiante:
- Jose Luis Andino

### Fecha de Entrega:
- miércoles, 18 de noviembre de 2026, 13:15

Implementación para el corpus **SciFact**: una unidad indexada por documento original, preprocesamiento en inglés con minúsculas, eliminación de stopwords y stemming Porter, TF-IDF con coseno y BM25. Los juicios de relevancia se cargan únicamente de la partición `test` para evaluar.

## Ejecutar localmente

```bash
uv sync
uv run uvicorn examen01_jla.api:app --host 0.0.0.0 --port 8000
```

Como alternativa, inicia el servidor local con `uv run examen01-jla`.

La primera carga descarga SciFact mediante `ir_datasets`. La API ofrece `GET /health` y `POST /search`:

```json
{"query":"Does vitamin D reduce respiratory infections?","model":"bm25","top_k":10}
```

`model` acepta `bm25` o `tfidf`; `top_k` admite valores entre 1 y 100. También se puede iniciar con `python -m uvicorn examen01_jla.api:app --host 0.0.0.0 --port 8000`.

## Notebook

Abrir `examen1bim.ipynb`, instalar las dependencias, cargar `beir/scifact/test` y ejecutar las celdas en orden. Se comparan Precision@10, Recall@10 y NDCG@10 en las consultas con qrels de esa partición. Las métricas se promedian macro por consulta.

## Publicación

El servicio puede desplegarse en una plataforma compatible con ASGI usando el comando de inicio:

```bash
uvicorn examen01_jla.api:app --host 0.0.0.0 --port $PORT
```

Configura `PORT` según la plataforma. Entrega la URL base del despliegue.
