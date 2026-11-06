# Local setup and execution guide

This repository is designed to run locally on Windows, Linux, or macOS. The public repository contains source code and documentation; download the CWRU dataset separately.

## 1. Clone and create an environment

    git clone https://github.com/Avdhoot-Ukirde/CNC-Predictive-Maintenance.git
    cd CNC-Predictive-Maintenance
    python -m venv .venv

Activate on Windows PowerShell: `.venv\\Scripts\\Activate.ps1`

Activate on Linux/macOS: `source .venv/bin/activate`

Install dependencies:

    python -m pip install --upgrade pip
    pip install -r requirements.txt

## 2. Add CWRU files

Download the 16 files listed in [bearing_classifier.md](bearing_classifier.md) from the official [CWRU 12 kHz Drive End dataset](https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data). Put them directly in `data/raw/cwru/`. Raw data is ignored by Git and should remain local.

## 3. Run tests

    python -m pytest -q

## 4. Inspect one recording

    python scripts/analyze_cwru.py data/raw/cwru/100.mat --signal-key X100_DE_time --sampling-rate 12000

Outputs are saved to `reports/cwru_analysis/`.

## 5. Train the bearing classifier

    python scripts/train_bearing_classifier.py

Outputs are saved to `reports/classifier/`. The training script lists missing files if the dataset is incomplete. Review `metrics.json` and the confusion matrix; do not rely on accuracy alone.

## 6. Run prediction on a recording

After training:

    python scripts/predict_bearing.py data/raw/cwru/100.mat --sampling-rate 12000

For CSV, pass `--csv-column your_column_name`. For MATLAB files with multiple channels, pass `--signal-key X100_DE_time`. Prediction output is saved to `reports/predictions/`.

## 7. Launch the dashboard

    streamlit run dashboard/app.py

Upload a `.mat` or `.csv` vibration recording, set the correct sampling frequency, and inspect the signal, spectrum, features, and (when trained model artifacts exist) window-level model predictions.

## Important interpretation notes

- The configured 16-file classifier uses the CWRU 12 kHz drive-end subset.
- Confirm the actual sampling rate and sensor channel for every recording.
- The model is a laboratory bearing-fault benchmark, not a validated CNC machine failure predictor.
- No Remaining Useful Life (RUL) estimate is implemented.
- Do not connect this research software to machine safety or motion-control circuits.

## 8. Train and use the anomaly detector

Train the normal-only baseline:

    python scripts/train_anomaly_detector.py

Apply it to a recording:

    python scripts/detect_anomalies.py data/raw/cwru/108.mat --sampling-rate 12000

The model is fitted only on normal recordings. An anomaly is a window that differs from the learned baseline; it is not a fault classification. Review anomaly scores and validate false alarms on separate data.
