"""
Download the electricity-meter matrix from BDG2 and convert a manageable but
substantial subset into long format for WattWise AI.

Default output: 50 meter/building columns x about 17,544 hourly timestamps
(~875k rows). Set WATTWISE_METERS=100 for a larger subset.
"""
from pathlib import Path
import os
import urllib.request
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
URL = "https://media.githubusercontent.com/media/buds-lab/building-data-genome-project-2/master/data/meters/cleaned/electricity.csv"
RAW = DATA / "bdg2_electricity_wide.csv"
OUT = DATA / "energy_data.csv"
N_METERS = int(os.environ.get("WATTWISE_METERS", "50"))

if not RAW.exists() or RAW.stat().st_size < 1000:
    print("Downloading BDG2 cleaned electricity meter data. This can take a while...")
    req = urllib.request.Request(URL, headers={"User-Agent": "WattWise-AI-educational-project"})
    with urllib.request.urlopen(req, timeout=120) as response, open(RAW, "wb") as f:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            f.write(block)
    print(f"Downloaded {RAW.stat().st_size:,} bytes")

header = pd.read_csv(RAW, nrows=0)
columns = list(header.columns)
time_col = columns[0]
meter_cols = columns[1:1 + N_METERS]
if not meter_cols:
    raise RuntimeError("No meter columns found in the downloaded dataset.")
print(f"Converting {len(meter_cols)} meters to long format...")
if OUT.exists():
    OUT.unlink()
first = True
for chunk in pd.read_csv(RAW, usecols=[time_col] + meter_cols, chunksize=1000):
    long = chunk.melt(id_vars=[time_col], var_name="building_id", value_name="meter_reading")
    long = long.rename(columns={time_col: "timestamp"})
    long["meter"] = "electricity"
    long = long.dropna(subset=["meter_reading"])
    long.to_csv(OUT, index=False, mode="w" if first else "a", header=first)
    first = False
print(f"Prepared {OUT} ({OUT.stat().st_size:,} bytes)")
print("This is a subset of BDG2 electricity meters, not the full 53.6-million-reading dataset.")
