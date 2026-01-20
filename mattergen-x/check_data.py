
import pandas as pd
try:
    df = pd.read_parquet(r"d:\coding_files\Projects\matterGen\material dataset\train.parquet")
    print(f"Row count: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
except Exception as e:
    print(f"Error reading parquet: {e}")
