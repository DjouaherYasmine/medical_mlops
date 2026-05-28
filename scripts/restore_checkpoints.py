"""
Restores model checkpoints by copying directly from mlruns artifact folders.
Usage: python scripts/restore_checkpoints.py
"""
import os, shutil, sqlite3
from pathlib import Path

ROOT = Path(__file__).parent.parent
DB   = ROOT / 'experiments' / 'mlflow.db'

os.makedirs(ROOT / 'models', exist_ok=True)

conn = sqlite3.connect(DB)
cur  = conn.cursor()
cur.execute("SELECT name, artifact_uri FROM runs WHERE status='FINISHED'")
rows = cur.fetchall()
conn.close()

for run_name, artifact_uri in rows:
    # Normalize: strip file://, convert forward slashes on Windows
    artifact_uri = artifact_uri.replace('file://', '')
    artifact_dir = Path(artifact_uri.replace('/', os.sep))
    
    # Make path relative if it's under ROOT
    if not artifact_dir.is_absolute():
        artifact_dir = ROOT / artifact_dir

    if not artifact_dir.exists():
        print(f'[SKIP] {run_name} — artifact dir not found: {artifact_dir}')
        continue

    for f in artifact_dir.iterdir():
        if f.suffix == '.pth':
            dest = ROOT / 'models' / f.name
            if dest.exists():
                print(f'[SKIP] {f.name} already exists')
            else:
                shutil.copy2(f, dest)
                print(f'[OK]   {f.name}  ←  {run_name}')

print('Done ✓')