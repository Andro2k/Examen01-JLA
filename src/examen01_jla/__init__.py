"""Sistema de recuperación de información para SciFact."""


def main() -> None:
    """Start the local web API."""
    import uvicorn

    uvicorn.run("examen01_jla.api:app", host="127.0.0.1", port=8000)

