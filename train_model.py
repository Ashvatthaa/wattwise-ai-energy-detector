"""Command-line training/evaluation helper. Saves evaluation metrics as JSON."""
import json
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.pipeline import load_dataset, prepare_data, train_models

df = prepare_data(load_dataset(None, prefer_large=True))
# Chronological split for a more realistic sanity check.
split = int(len(df) * 0.8)
train, test = df.iloc[:split], df.iloc[split:]
models = train_models(train)
model = models["expected_model"]
metrics = {"rows": int(len(df)), "train_rows": int(len(train)), "test_rows": int(len(test))}
if model is not None:
    pred = model.predict(test[models["features"]])
    metrics.update({"mae": float(mean_absolute_error(test.energy_kwh, pred)),
                    "rmse": float(mean_squared_error(test.energy_kwh, pred) ** 0.5),
                    "r2": float(r2_score(test.energy_kwh, pred))})
Path("models").mkdir(exist_ok=True)
Path("models/evaluation.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
