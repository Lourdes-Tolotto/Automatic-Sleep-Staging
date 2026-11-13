"""Análisis de las cinco semillas de contexto 7; no entrena ni selecciona por test."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

SEEDS = [42, 123, 2026, 7, 99]
NAMES = ['W', 'N1', 'N2', 'N3', 'REM']
KEYS = ['subject', 'source_epoch_index']

def clean(df):
    cols = KEYS + ['y_true', 'y_pred']
    if any(c not in df for c in cols):
        raise ValueError(f'Faltan columnas: {cols}')
    df = df[cols].copy()
    if df.isna().any().any() or df.duplicated(KEYS).any():
        raise ValueError('Hay nulos o claves duplicadas.')
    if not df.y_true.isin(range(5)).all() or not df.y_pred.isin(range(5)).all():
        raise ValueError('Etiquetas fuera de 0..4.')
    return df.sort_values(KEYS).reset_index(drop=True)

def align(reference, other):
    if set(map(tuple, reference[KEYS].to_numpy())) != set(map(tuple, other[KEYS].to_numpy())):
        raise ValueError('Las semillas no tienen exactamente los mismos centros.')
    out = reference.merge(other, on=KEYS, suffixes=('_ref',''), validate='one_to_one')
    if not np.array_equal(out.y_true_ref.to_numpy(), out.y_true.to_numpy()):
        raise ValueError('Las etiquetas de referencia no coinciden.')
    return other

def run(root):
    root = Path(root)
    results = root/'reports/results'
    out = results/'chapter4'; out.mkdir(exist_ok=True)
    fig = root/'reports/figures/chapter4'; fig.mkdir(parents=True, exist_ok=True)
    paths = [results/'robustness'/f'context_7_seed_{s}_predictions.parquet' for s in SEEDS]
    paths += [results/'dl_cnn_holdout_predictions.parquet', results/'dl_subject_split.json',
              root/'data/processed/dl_metadata.parquet']
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError('Faltan artefactos (no hace falta reentrenar):\n'+'\n'.join(missing))
    preds = [clean(pd.read_parquet(p)) for p in paths[:5]]
    reference = preds[0]
    split = json.loads(paths[-2].read_text())
    if set(reference.subject) != set(split['test_subjects']):
        raise ValueError('Los sujetos no coinciden con el split de prueba.')
    for other in preds[1:]: align(reference, other)
    cnn = clean(pd.read_parquet(paths[5]))
    cnn = reference[KEYS].merge(cnn, on=KEYS, how='left', validate='one_to_one')
    if cnn.isna().any().any(): raise ValueError('CNN no cubre todos los centros.')
    align(reference, cnn)
    meta = pd.read_parquet(paths[-1]).sort_values(KEYS)
    if meta.duplicated(KEYS).any(): raise ValueError('Metadata duplicada.')
    # Transición según vecinos originales; nunca inferir continuidad saltando descartes.
    transitions = []
    for subject, g in meta.groupby('subject'):
        g = g.sort_values('source_epoch_index').reset_index(drop=True)
        for i in range(1, len(g)-1):
            w = g.iloc[i-1:i+2]
            if np.all(np.diff(w.source_epoch_index)==1) and np.allclose(np.diff(w.recording_onset_s),30,atol=1e-3):
                transitions.append(dict(subject=subject, source_epoch_index=int(g.iloc[i].source_epoch_index),
                                        is_transition=bool(w.label.nunique()>1)))
    flags = pd.DataFrame(transitions)
    matrices, perclass, conditions, examples = [], [], [], []
    for seed, df in zip(SEEDS, preds):
        cm = confusion_matrix(df.y_true, df.y_pred, labels=range(5))
        matrices.append(cm)
        pd.DataFrame(cm,index=NAMES,columns=NAMES).to_csv(out/f'confusion_seed_{seed}.csv')
        p,r,f,support = precision_recall_fscore_support(df.y_true,df.y_pred,labels=range(5),zero_division=0)
        for c in range(5):
            perclass.append(dict(seed=seed,stage=NAMES[c],precision=p[c],recall=r[c],f1=f[c],support=int(support[c])))
        enriched = df.merge(flags,on=KEYS,how='left',validate='one_to_one')
        if enriched.is_transition.isna().any(): raise ValueError('Centros sin vecinos trazables.')
        for transition,g in enriched.groupby('is_transition'):
            conditions.append(dict(seed=seed,is_transition=transition,n=len(g),errors=int((g.y_true!=g.y_pred).sum()),error_rate=float((g.y_true!=g.y_pred).mean())))
        errors = enriched[enriched.y_true!=enriched.y_pred].copy()
        errors['seed'] = seed
        # Primer error cronológico de cada par real/predicho: criterio fijo y explícito.
        examples.append(errors.groupby(['y_true','y_pred'],sort=True).head(1))
    counts = np.asarray(matrices)
    denom = counts.sum(axis=2,keepdims=True)
    normalized = np.divide(counts,denom,out=np.zeros_like(counts,dtype=float),where=denom!=0)
    cnn_cm = confusion_matrix(cnn.y_true,cnn.y_pred,labels=range(5))
    cnn_norm = np.divide(cnn_cm,cnn_cm.sum(axis=1,keepdims=True),out=np.zeros_like(cnn_cm,dtype=float),where=cnn_cm.sum(axis=1,keepdims=True)!=0)
    pd.DataFrame(normalized.mean(0),index=NAMES,columns=NAMES).to_csv(out/'confusion_temporal_mean_normalized.csv')
    pd.DataFrame(normalized.std(0,ddof=1),index=NAMES,columns=NAMES).to_csv(out/'confusion_temporal_std_normalized.csv')
    pd.DataFrame(cnn_cm,index=NAMES,columns=NAMES).to_csv(out/'confusion_cnn.csv')
    pc=pd.DataFrame(perclass);pc.to_csv(out/'per_stage_runs.csv',index=False)
    pc.groupby('stage')[['precision','recall','f1']].agg(['mean','std']).to_csv(out/'per_stage_summary.csv')
    pd.DataFrame(conditions).to_csv(out/'transition_errors.csv',index=False)
    pd.concat(examples).to_csv(out/'error_examples.csv',index=False)
    pairs=[]
    for a in range(5):
        for b in range(5):
            if a!=b: pairs.append(dict(real=NAMES[a],predicted=NAMES[b],mean_count=counts[:,a,b].mean(),mean_rate=normalized[:,a,b].mean(),std_rate=normalized[:,a,b].std(ddof=1)))
    pd.DataFrame(pairs).sort_values('mean_count',ascending=False).to_csv(out/'confusion_pairs.csv',index=False)
    figure,axes=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
    for ax,cm,title in zip(axes,[cnn_norm,normalized.mean(0)],['CNN fija','BiGRU contexto 7 — media de 5 semillas']):
        ax.imshow(cm,vmin=0,vmax=1,cmap='Blues')
        for a in range(5):
            for b in range(5): ax.text(b,a,f'{100*cm[a,b]:.1f}%',ha='center',va='center',color='white' if cm[a,b]>.55 else 'black')
        ax.set(xticks=range(5),yticks=range(5),xticklabels=NAMES,yticklabels=NAMES,xlabel='Predicción',ylabel='Referencia',title=title)
    figure.savefig(fig/'confusion_comparison.pdf');plt.close(figure)
    # Casos ya identificados en NB12; conservar las cinco semillas, sin elegir la mejor.
    for subject in ['SN069', 'SN081']:
        if subject not in set(reference.subject): continue
        figure, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True, layout='constrained')
        for ax, seed, df in zip(axes, SEEDS, preds):
            g = df[df.subject == subject].merge(meta[KEYS+['recording_onset_s']], on=KEYS, validate='one_to_one').sort_values('source_epoch_index')
            # Separar tramos: no dibujar líneas sobre huecos de contexto descartado.
            breaks = (g.source_epoch_index.diff().ne(1) | ~np.isclose(g.recording_onset_s.diff(),30,atol=1e-3)).cumsum()
            for j, (_, piece) in enumerate(g.groupby(breaks)):
                x=piece.recording_onset_s.to_numpy()/3600
                ax.step(x, piece.y_true, where='post', color='#58616a', alpha=.65, label='Referencia' if j==0 else None)
                ax.step(x, piece.y_pred, where='post', color='#176d91', alpha=.8, label='Predicción' if j==0 else None)
            ax.set(yticks=range(5), yticklabels=NAMES, ylabel=f'Semilla {seed}')
        axes[0].legend(loc='upper right');axes[0].set_title(f'{subject}: caso descriptivo, cinco semillas')
        axes[-1].set_xlabel('Horas desde el inicio del registro')
        figure.savefig(fig/f'hypnogram_{subject}.pdf');plt.close(figure)
    manifest=dict(seeds=SEEDS,n_subjects=int(reference.subject.nunique()),n_centers=len(reference),
                  aggregation='Media de matrices normalizadas por clase real; no ensamble ni cinco tests independientes.',
                  example_policy='Primer error por par de clases y semilla, ordenado por sujeto e índice.',
                  input_files=[str(p.relative_to(root)) for p in paths])
    (out/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    print(f'Resultados: {out}\nFiguras: {fig}')
    return pc, pd.DataFrame(conditions)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args=parser.parse_args();run(args.root)
