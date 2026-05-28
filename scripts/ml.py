import sqlite3

DB = 'experiments/mlflow.db'  # local path
LOCAL_ROOT = r"D:\Studies\2CS\S2\MLA\Amal's\medical_mlops"

conn = sqlite3.connect(DB)
cur = conn.cursor()
cur.execute("""
    UPDATE runs
    SET artifact_uri = REPLACE(artifact_uri, '/content/drive/MyDrive/mlops-chestmnist', ?)
    WHERE artifact_uri LIKE '/content/drive/MyDrive/mlops-chestmnist%'
""", (LOCAL_ROOT,))
conn.commit()
print(f'✓ fixed {cur.rowcount} rows')
conn.close()