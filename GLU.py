import pandas as pd
from sklearn.metrics import roc_auc_score
from glob import glob

OUTPUT_DIR = "output/full"
SUFFIX = "-unified-uq-full.jsonl"

files = sorted(glob(f"{OUTPUT_DIR}/*{SUFFIX}"))

dfs = {}
for f in files:
    df = pd.read_json(f, lines=True, convert_dates=False)
    df['GLU'] = (1 + df['S_tilde']) * df['she_R_mean']
    name = f.split('/')[-1].replace(SUFFIX, '')
    dfs[name] = df

    sub = df.dropna(subset=['label', 'GLU'])
    n_drop = len(df) - len(sub)
    auroc = roc_auc_score(sub['label'], sub['GLU'])
    print(f"{name:30s}  AUROC(GLU): {auroc:.4f}  (dropped {n_drop})")
