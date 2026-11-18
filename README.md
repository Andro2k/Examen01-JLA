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

El notebook descarga SciFact con `ir_datasets` (grupo `dev` de `uv sync`). La API no descarga nada: lee el corpus empaquetado en `data/scifact_docs.jsonl.gz`, que se genera una sola vez con `uv run python -m examen01_jla.data` y se sube al repositorio. La API ofrece `GET /health` y `POST /search`:

```json
{"query":"Does vitamin D reduce respiratory infections?","model":"bm25","top_k":10}
```

`model` acepta `bm25` o `tfidf`; `top_k` admite valores entre 1 y 100. También se puede iniciar con `python -m uvicorn examen01_jla.api:app --host 0.0.0.0 --port 8000`.

## Notebook

Abrir `examen1bim.ipynb`, instalar las dependencias, cargar `beir/scifact/test` y ejecutar las celdas en orden. Se comparan Precision@10, Recall@10 y NDCG@10 en las consultas con qrels de esa partición. Las métricas se promedian macro por consulta.

## Publicación (Render)

1. Generar y subir el corpus: `uv run python -m examen01_jla.data`, luego `git add data/scifact_docs.jsonl.gz && git commit && git push`.
2. En Render: **New → Blueprint**, elegir este repositorio (lee `render.yaml`). También sirve **New → Web Service** con:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn --app-dir src examen01_jla.api:app --host 0.0.0.0 --port $PORT`
   - Health Check Path: `/health`
3. Probar con `GET https://<servicio>.onrender.com/health` y `POST https://<servicio>.onrender.com/search`.

En el plan gratuito el servicio se duerme tras 15 minutos sin tráfico; la primera petición tras ese lapso tarda en despertar. Abre `/health` antes de la revisión.
