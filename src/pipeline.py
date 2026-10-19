from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

def load_dataset(file_obj=None, prefer_large=False):
    if file_obj is not None:
        df = pd.read_csv(file_obj)
        return standardize_columns(df)
    candidates = []
    if prefer_large:
        candidates += [DATA_DIR/"train.csv", DATA_DIR/"energy_data.csv", DATA_DIR/"bdg2_long.csv"]
    candidates += [DATA_DIR/"demo_energy.csv"]
    for path in candidates:
        if path.exists():
            # Limit only for interactive responsiveness, while keeping large data available.
            df = pd.read_csv(path, nrows=500_000 if prefer_large else None)
            return standardize_columns(df)
    raise FileNotFoundError("No dataset available. The demo CSV should be in data/demo_energy.csv. Run python download_dataset.py for the large dataset.")

def standardize_columns(df):
    df = df.copy()
    # Accept common names from public building-energy datasets.
    lower = {str(c).strip().lower(): c for c in df.columns}
    time_col = next((lower[k] for k in ["timestamp","datetime","date","time"] if k in lower), None)
    energy_col = next((lower[k] for k in ["meter_reading","energy_kwh","energy","consumption","reading","kwh"] if k in lower), None)
    if time_col is None:
        raise ValueError("CSV needs a timestamp/date/datetime column.")
    if energy_col is None:
        raise ValueError("CSV needs a meter_reading/energy_kwh/energy/consumption column.")
    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(df[time_col], errors="coerce", utc=True).dt.tz_convert(None)
    out["energy_kwh"] = pd.to_numeric(df[energy_col], errors="coerce")
    # Common optional columns
    for target, aliases in {
        "building_id":["building_id","building","site_id"],
        "meter":["meter","meter_type","primary_use"],
        "air_temperature":["air_temperature","temperature","temp"],
    }.items():
        col = next((lower[a] for a in aliases if a in lower), None)
        if col is not None:
            out[target] = df[col].values
    out = out.dropna(subset=["timestamp","energy_kwh"])
    out = out[out.energy_kwh >= 0].copy()
    if out.empty:
        raise ValueError("No valid rows after parsing timestamp and non-negative energy readings.")
    return out.sort_values("timestamp").reset_index(drop=True)

def prepare_data(raw):
    df = raw.copy()
    df["hour"] = df.timestamp.dt.hour
    df["day_of_week"] = df.timestamp.dt.dayofweek
    df["month"] = df.timestamp.dt.month
    df["day_of_year"] = df.timestamp.dt.dayofyear
    df["is_weekend"] = (df.day_of_week >= 5).astype(int)
    df["hour_sin"] = np.sin(2*np.pi*df.hour/24)
    df["hour_cos"] = np.cos(2*np.pi*df.hour/24)
    df["dow_sin"] = np.sin(2*np.pi*df.day_of_week/7)
    df["dow_cos"] = np.cos(2*np.pi*df.day_of_week/7)
    if "air_temperature" in df:
        df["air_temperature"] = pd.to_numeric(df.air_temperature, errors="coerce")
    # A global robust local baseline is helpful when multiple meters/buildings are included.
    df["rolling_median"] = df.energy_kwh.rolling(24, min_periods=3).median()
    df["rolling_mad"] = (df.energy_kwh - df.rolling_median).abs().rolling(24, min_periods=3).median()
    return df

def train_models(df):
    features = ["hour","day_of_week","month","day_of_year","is_weekend","hour_sin","hour_cos","dow_sin","dow_cos"]
    if "air_temperature" in df.columns:
        features.append("air_temperature")
    X = df[features].replace([np.inf,-np.inf], np.nan)
    expected = Pipeline([("imputer", SimpleImputer(strategy="median")), ("rf", RandomForestRegressor(n_estimators=60, max_depth=12, min_samples_leaf=3, n_jobs=-1, random_state=42))])
    # The regression is used for contextual expectation, not a causal savings claim.
    try:
        expected.fit(X, df.energy_kwh)
    except Exception:
        expected = None
    anomaly = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("iforest", IsolationForest(n_estimators=100, contamination=0.05, random_state=42, n_jobs=-1))])
    anomaly.fit(df[["energy_kwh"] + features].replace([np.inf,-np.inf], np.nan))
    return {"expected_model": expected, "anomaly_model": anomaly, "features": features}

def score_anomalies(df, models, contamination=0.05):
    out = df.copy()
    features = models["features"]
    X = out[["energy_kwh"] + features].replace([np.inf,-np.inf], np.nan)
    model = models["anomaly_model"]
    # Adjust threshold dynamically to the user's contamination setting.
    imputed = model.named_steps["imputer"].transform(X)
    scaled = model.named_steps["scale"].transform(imputed)
    raw_scores = -model.named_steps["iforest"].score_samples(scaled)
    threshold = np.quantile(raw_scores, 1-contamination)
    out["anomaly_score"] = raw_scores
    out["is_anomaly"] = raw_scores >= threshold
    out["status"] = np.where(out.is_anomaly, "Unusual", "Normal")
    if models["expected_model"] is not None:
        out["expected_kwh"] = np.maximum(models["expected_model"].predict(out[features]), 0)
        out["actual_vs_expected"] = out.energy_kwh - out.expected_kwh
    return out

def estimate_savings(df, reduction_fraction=0.10):
    # Scenario only: do not claim this amount is guaranteed savings.
    base = df.energy_kwh.where(df.is_anomaly, 0)
    return base * reduction_fraction
