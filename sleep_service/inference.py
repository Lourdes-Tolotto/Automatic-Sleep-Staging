"""Inferencia offline desde el NPZ trazable de NB2, sin usar etiquetas."""
import hashlib
import json
import platform
import time
from pathlib import Path
import zipfile
import numpy as np
import torch
from .models import SleepCNN, TemporalBiGRU

STAGES = ['W', 'N1', 'N2', 'N3', 'REM']
SEEDS = [42, 123, 2026, 7, 99]
MAX_BYTES = 256 * 1024 * 1024

def valid_windows(indices, onsets):
    windows = []
    for center in range(3, len(indices) - 3):
        w = np.arange(center - 3, center + 4)
        if np.all(np.diff(indices[w]) == 1) and np.allclose(np.diff(onsets[w]), 30., atol=1e-3):
            windows.append(w)
    return np.asarray(windows, dtype=np.int64).reshape(-1, 7)

def load_record(file, expected_channels):
    file.seek(0)
    with zipfile.ZipFile(file) as archive:
        if sum(x.file_size for x in archive.infolist()) > MAX_BYTES:
            raise ValueError('El NPZ descomprimido supera 256 MiB.')
    file.seek(0)
    # Se leen solamente campos necesarios: y y campos object de QC no se deserializan.
    with np.load(file, allow_pickle=False) as data:
        required = ['X', 'source_epoch_index', 'recording_onset_s', 'subject_id',
                    'channel_names', 'signal_unit', 'processed_sfreq']
        if any(k not in data for k in required):
            raise ValueError('Faltan campos del NPZ de NB2: ' + ', '.join(k for k in required if k not in data))
        x = data['X']
        idx, onset = data['source_epoch_index'], data['recording_onset_s']
        channels = data['channel_names'].tolist()
        if channels != expected_channels:
            raise ValueError('Los canales o su orden no coinciden con el entrenamiento.')
        if str(data['signal_unit'].item()) != 'uV' or not np.isclose(float(data['processed_sfreq']), 100):
            raise ValueError('Se requieren señales en uV, a 100 Hz, procesadas por NB2.')
        subject = str(data['subject_id'].item())
        if x.ndim != 3 or x.shape[1:] != (7, 3000) or not 7 <= len(x) <= 2000:
            raise ValueError('X debe tener forma (N, 7, 3000), con 7 <= N <= 2000.')
        if x.dtype.kind not in 'fi' or not np.isfinite(x).all():
            raise ValueError('X debe contener valores numéricos finitos.')
        if idx.shape != (len(x),) or onset.shape != (len(x),):
            raise ValueError('Los metadatos deben tener una entrada por intervalo.')
        if idx.dtype.kind not in 'iu' or not np.isfinite(onset).all():
            raise ValueError('Índices enteros y tiempos finitos requeridos.')
        if np.any(idx < 0) or np.any(onset < 0) or np.any(np.diff(idx) <= 0) or np.any(np.diff(onset) <= 0):
            raise ValueError('Índices y tiempos deben ser no negativos, únicos y crecientes.')
        windows = valid_windows(idx, onset)
        if not len(windows):
            raise ValueError('No hay ventanas completas de siete intervalos consecutivos.')
        return dict(x=x.astype(np.float32), idx=idx, onset=onset, subject=subject, windows=windows)

class Predictor:
    def __init__(self, root, seed=42):
        if seed not in SEEDS:
            raise ValueError('Semilla no evaluada en NB12.')
        root = Path(root)
        paths = [root/'models/dl_cnn_baseline.pt',
                 root/f'models/robustness/dl_context_7_seed_{seed}_bigru.pt',
                 root/'reports/results/dl_channel_normalization.json']
        for p in paths:
            if not p.is_file():
                raise FileNotFoundError(f'Falta {p.relative_to(root)}')
        norm = json.loads(paths[2].read_text())
        self.channels = norm['channel_names']
        self.mean = np.asarray(norm['channel_mean_uv'], dtype=np.float32).reshape(1, 7, 1)
        self.std = np.asarray(norm['channel_std_uv'], dtype=np.float32).reshape(1, 7, 1)
        if len(self.channels) != 7 or len(set(self.channels)) != 7 or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or np.any(self.std <= 0):
            raise ValueError('Estadísticas de normalización inválidas.')
        self.cnn, self.head = SleepCNN().eval(), TemporalBiGRU().eval()
        for model, p in zip([self.cnn, self.head], paths):
            model.load_state_dict(torch.load(p, map_location='cpu', weights_only=True), strict=True)
        self.seed = seed
        self.fingerprints = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

    @torch.inference_mode()
    def predict(self, file):
        t0 = time.perf_counter()
        r = load_record(file, self.channels)
        embeddings = []
        for start in range(0, len(r['x']), 64):
            x = (r['x'][start:start+64] - self.mean) / self.std
            embeddings.append(self.cnn.encode(torch.from_numpy(x)))
        embeddings = torch.cat(embeddings)
        probabilities = []
        for start in range(0, len(r['windows']), 256):
            seq = embeddings[torch.as_tensor(r['windows'][start:start+256])]
            probabilities.append(self.head(seq).softmax(dim=1).numpy())
        probabilities = np.concatenate(probabilities)
        centers = r['windows'][:, 3]
        rows = []
        for c, probs in zip(centers, probabilities):
            pred = int(probs.argmax())
            rows.append(dict(source_epoch_index=int(r['idx'][c]), recording_onset_s=float(r['onset'][c]),
                             y_pred=pred, stage=STAGES[pred], probabilities=probs.tolist()))
        excluded = sorted(set(range(len(r['x']))) - set(centers.tolist()))
        return dict(subject=r['subject'], seed=self.seed, context_size=7, stage_order=STAGES,
                    input_epochs=len(r['x']), predicted_epochs=len(rows),
                    excluded_source_indices=r['idx'][excluded].tolist(),
                    elapsed_seconds=time.perf_counter()-t0, device='cpu', server_platform=platform.platform(), torch_version=torch.__version__,
                    scope='NPZ de NB2 a predicciones; no incluye lectura EDF ni preprocesamiento',
                    artifacts_sha256=self.fingerprints, predictions=rows)
