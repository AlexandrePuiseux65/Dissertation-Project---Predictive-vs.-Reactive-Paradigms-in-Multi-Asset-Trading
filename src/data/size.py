import os
import pandas as pd

for name, path in [
    ("AAPL", "data/raw/stocks/AAPL.parquet"),
    ("TLT", "data/raw/bonds/TLT.parquet"),
    ("BTC-USD", "data/raw/crypto/BTC-USD.parquet"),
]:
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"{name}: {size_mb:.2f} MB")

df = pd.read_parquet("data/raw/stocks/AAPL.parquet")
df.to_csv("/tmp/AAPL_test.csv", index=False)

df_2 = pd.read_parquet("data/raw/bonds/TLT.parquet")
df_2.to_csv("/tmp/TLT_test.csv", index=False)

df_3 = pd.read_parquet("data/raw/crypto/BTC-USD.parquet")
df_3.to_csv("/tmp/BTC-USD_test.csv", index=False)


print("Parquet vs CSV")
print(f"Parquet (AAPL): {os.path.getsize('data/raw/stocks/AAPL.parquet')/(1024*1024):.2f} MB")
print(f"CSV: {os.path.getsize('/tmp/AAPL_test.csv')/(1024*1024):.2f} MB")

print(f"Parquet (TLT): {os.path.getsize('data/raw/bonds/TLT.parquet')/(1024*1024):.2f} MB")
print(f"CSV: {os.path.getsize('/tmp/TLT_test.csv')/(1024*1024):.2f} MB")

print(f"Parquet (BTC-USD): {os.path.getsize('data/raw/crypto/BTC-USD.parquet')/(1024*1024):.2f} MB")
print(f"CSV: {os.path.getsize('/tmp/BTC-USD_test.csv')/(1024*1024):.2f} MB")