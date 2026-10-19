# WattWise AI — AI-Powered Energy Waste Detector

A Python + machine-learning internship project that flags unusual building energy readings and estimates **illustrative** energy-saving opportunities.

## Features
- Streamlit web dashboard with KPIs, daily consumption chart, anomaly explorer and CSV export.
- Isolation Forest anomaly detection.
- Random Forest expected-consumption baseline.
- Supports uploaded CSVs and a local copy of the Building Data Genome Project 2 (BDG2) dataset.
- Time features and optional air-temperature feature.
- Potential savings shown as a scenario, not a guaranteed reduction.

## Dataset (large public dataset)
This project is designed to work with **Building Data Genome Project 2**, a research dataset of building meter readings and associated metadata/weather. It is much larger than the small demo CSV. Dataset repository: https://github.com/buds-lab/building-data-genome-project-2

1. Create and activate a virtual environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Download the public data: `python download_dataset.py`
4. Run: `streamlit run app.py`

The download script fetches the cleaned electricity-meter matrix from the public repository and converts the first 50 meter/building columns into a long-format CSV (~875k rows when complete). You can increase the number with `WATTWISE_METERS=100` before running the script. The raw matrix and processed CSV are intentionally excluded from Git because large data files make deployments slow and repositories oversized. If the direct download URL changes, follow the upstream repository README and update the URL.

### Dataset citation
Miller, C. et al., *The Building Data Genome Project 2*, dataset and documentation in the linked research repository. Review the upstream license and citation instructions before redistributing the data.

## Expected input format
The app accepts CSVs with:
- Required time field: `timestamp`, `datetime`, `date`, or `time`
- Required energy field: `meter_reading`, `energy_kwh`, `energy`, `consumption`, `reading`, or `kwh`
- Optional fields: `building_id`, `meter`, `air_temperature`

The BDG2 source electricity file is a wide matrix; `download_dataset.py` converts a subset into long format with `timestamp`, `building_id`, `meter`, and `meter_reading`. For an uploaded custom CSV, each row should represent one reading.

## Important data note
The bundled demo data (11,520 rows) is **synthetic** and exists only to make the UI run before downloading real data. Do not present synthetic demo performance as real-world model accuracy.

## Model methodology
- **Isolation Forest** finds unusual combinations of energy and time/weather features.
- **Random Forest regression** learns an expected-consumption baseline from the available temporal/weather features.
- Metrics in `train_model.py` are a baseline sanity check, not proof of savings or causal energy reduction.
- A flagged reading may reflect weather, occupancy, production changes, a sensor issue, or a genuine inefficiency. Validate alerts with domain experts.

## Deploy to Streamlit Community Cloud
1. Push the code (not the large raw dataset) to a GitHub repository.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Select **Create app**, choose your repository and branch, and set the entrypoint to `app.py`.
4. Deploy. Streamlit installs dependencies from `requirements.txt`.
5. For a cloud demo, keep the small `data/demo_energy.csv` bundled and allow users to upload their own prepared CSV. Do not attempt to download the full research dataset at every app startup.
6. If using the full dataset in the cloud, store a prepared subset in external storage or use a data hosting solution with appropriate access and memory limits.

Official deployment docs: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app

## Local run (Windows PowerShell)
```powershell
cd AI_Energy_Waste_Detector
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Suggested report outline
1. Abstract
2. Problem statement and objectives
3. Literature review / existing system
4. Dataset and preprocessing
5. Methodology and algorithms
6. System architecture
7. Implementation and UI screenshots
8. Evaluation (chronological split, MAE/RMSE for baseline, precision/recall if labelled anomalies exist)
9. Limitations and future work
10. Conclusion and references
