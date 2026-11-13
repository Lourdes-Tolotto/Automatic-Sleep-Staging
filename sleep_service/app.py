"""Ejecutar: uvicorn sleep_service.app:app --host 127.0.0.1 --port 8000"""
import os
import threading
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from .inference import Predictor, MAX_BYTES

ROOT = Path(os.environ.get('SLEEP_PROJECT_ROOT', Path(__file__).resolve().parents[1]))
SEED = int(os.environ.get('SLEEP_SEED', '42'))
lock = threading.Lock()

@asynccontextmanager
async def lifespan(app):
    app.state.predictor = None
    app.state.load_error = None
    try:
        app.state.predictor = Predictor(ROOT, SEED)
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as e:
        app.state.load_error = str(e)
    yield

app = FastAPI(title='Sleep staging — CNN + BiGRU', version='0.1.0', lifespan=lifespan)

@app.get('/')
def home():
    return FileResponse(Path(__file__).with_name('index.html'))

@app.get('/health')
def health():
    ready = app.state.predictor is not None
    return JSONResponse(status_code=200 if ready else 503, content={
        'ready': ready, 'seed': SEED, 'context_size': 7,
        'detail': app.state.load_error,
        'input': 'NPZ procesado por NB2, no EDF', 'device': 'cpu'})

@app.post('/predict')
def predict(file: UploadFile):
    try:
        if app.state.predictor is None:
            raise HTTPException(503, app.state.load_error or 'Modelo no disponible.')
        if not file.filename or not file.filename.lower().endswith('.npz'):
            raise HTTPException(422, 'Cargá un archivo .npz generado por NB2.')
        file.file.seek(0, 2)
        if file.file.tell() > MAX_BYTES:
            raise HTTPException(413, 'Archivo mayor a 256 MiB.')
        file.file.seek(0)
        if not lock.acquire(blocking=False):
            raise HTTPException(429, 'Hay una inferencia en curso. Reintentá al terminar.')
        try:
            return app.state.predictor.predict(file.file)
        except (ValueError, KeyError, TypeError, zipfile.BadZipFile, EOFError) as e:
            raise HTTPException(422, str(e)) from e
        finally:
            lock.release()
    finally:
        file.file.close()
