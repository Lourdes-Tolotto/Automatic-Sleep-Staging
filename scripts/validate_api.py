"""Prueba HTTP del NPZ completo y concordancia exacta con NB12, para una semilla."""
import argparse
import json
import platform
import time
from pathlib import Path
import pandas as pd
import httpx

def main():
    p=argparse.ArgumentParser();p.add_argument('npz',type=Path);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--url',default='http://127.0.0.1:8000');p.add_argument('--repeats',type=int,default=3);a=p.parse_args()
    if a.repeats<1: p.error('--repeats debe ser positivo')
    root=a.root;rows=[];reference=None;first=None
    with httpx.Client(base_url=a.url,timeout=600) as client:
        health=client.get('/health');health.raise_for_status()
        for run in range(a.repeats):
            start=time.perf_counter()
            with a.npz.open('rb') as f: response=client.post('/predict',files={'file':(a.npz.name,f,'application/octet-stream')})
            response.raise_for_status();result=response.json();total=time.perf_counter()-start
            predictions=pd.DataFrame(result['predictions']).sort_values('source_epoch_index').reset_index(drop=True)
            if reference is None:
                path=root/'reports/results/robustness'/f"context_7_seed_{result['seed']}_predictions.parquet"
                reference=pd.read_parquet(path);reference=reference[reference.subject==result['subject']].sort_values('source_epoch_index').reset_index(drop=True)
                if reference.empty: raise ValueError('El sujeto no pertenece al test guardado de NB12.')
                first=result
            if predictions.source_epoch_index.tolist()!=reference.source_epoch_index.tolist():
                raise ValueError('Los centros API/NB12 no coinciden: revisar NPZ, discontinuidades o versión de artefactos.')
            agreement=float((predictions.y_pred.to_numpy()==reference.y_pred.to_numpy()).mean())
            rows.append(dict(run=run+1,http_seconds=total,inference_seconds=result['elapsed_seconds'],
                             prediction_agreement=agreement,n_centers=len(predictions)))
    out=root/'reports/results/chapter4';out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'api_integration_runs.csv',index=False)
    report=dict(subject=first['subject'],seed=first['seed'],scope=first['scope'],
                client_platform=platform.platform(),server_device=first['device'],server_platform=first['server_platform'],torch_version=first['torch_version'],
                artifacts_sha256=first['artifacts_sha256'],
                pass_exact_agreement=all(r['prediction_agreement']==1 for r in rows),runs=rows,
                timing_scope='HTTP incluye carga; inferencia incluye lectura NPZ y normalización. Excluye EDF y preparación NB2.',
                first_request='Modelos cargados al iniciar servicio; sin descartar la primera petición.')
    (out/'api_integration_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    (out/'api_example_response.json').write_text(json.dumps(first,indent=2))
    print(json.dumps(report,indent=2,ensure_ascii=False))
    if not report['pass_exact_agreement']: raise SystemExit('Hay diferencias frente a NB12; no declarar equivalencia hasta revisarlas.')

if __name__=='__main__':main()
