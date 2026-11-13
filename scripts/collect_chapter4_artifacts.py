"""Empaqueta resultados/pesos locales para completar el análisis sin reentrenar."""
import argparse
import zipfile
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--sample',type=Path,help='NPZ de NB2 de un sujeto de test (opcional)');p.add_argument('--out',type=Path,default=Path('chapter4_artifacts.zip'));a=p.parse_args()
root=a.root.resolve()
paths=[Path('models/dl_cnn_baseline.pt'),Path('reports/results/dl_channel_normalization.json'),Path('reports/results/dl_cnn_holdout_predictions.parquet'),Path('reports/results/dl_subject_split.json'),Path('data/processed/dl_metadata.parquet')]
for seed in [42,123,2026,7,99]:
    paths += [Path(f'reports/results/robustness/context_7_seed_{seed}_predictions.parquet'),Path(f'models/robustness/dl_context_7_seed_{seed}_bigru.pt')]
missing=[str(p) for p in paths if not (root/p).is_file()]
if missing: raise SystemExit('Faltan estos archivos:\n'+'\n'.join(missing))
if a.sample and not a.sample.is_file(): raise SystemExit('No existe el NPZ indicado.')
with zipfile.ZipFile(a.out,'w',zipfile.ZIP_DEFLATED) as z:
    for rel in paths:z.write(root/rel,str(rel))
    if a.sample:z.write(a.sample,'sample/'+a.sample.name)
print(a.out.resolve())
