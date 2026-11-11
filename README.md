# CNC Predictive Maintenance

> An offline vibration-based bearing condition-monitoring prototype using machine learning, Streamlit, FastAPI, MLflow, DVC, and Docker.

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/ML-Scikit--learn-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Deployment-Docker-2496ED.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

CNC Predictive Maintenance analyses recorded vibration signals from the Case Western Reserve University (CWRU) Bearing Data Center. It segments signals into windows, extracts statistical and frequency-domain features, and applies machine-learning models for bearing-condition classification and anomaly detection.

**Implemented components**
- CWRU MATLAB and numeric CSV signal loading, validation, and segmentation.
- Time-domain and FFT-based feature extraction.
- Four-class Random Forest bearing-condition classifier.
- Normal-baseline Isolation Forest anomaly detector.
- Streamlit dashboard for signal analysis and model predictions.
- FastAPI prediction service.
- MLflow experiment tracking.
- DVC dataset/pipeline versioning with an S3 remote.
- Docker Compose deployment for the dashboard, API, and MLflow.

**Scope:** This is an offline benchmark-data prototype. Live CNC sensor acquisition, Remaining Useful Life (RUL) estimation, industrial validation, and safety-certified machine alarms or controls are not implemented.

## Current Evaluation

The Random Forest classifier achieved **100% accuracy on 825 held-out windows** in the current CWRU benchmark evaluation. The evaluation used a recording-grouped split with `StratifiedGroupKFold`, keeping windows from the same source recording together rather than randomly distributing windows from one recording across training and test sets.

This score applies only to the selected CWRU recordings and evaluation setup. It does **not** establish 100% accuracy on industrial CNC machines. Independent data and validation are required for other machines, bearing types, operating speeds, loads, sensor locations, and acquisition systems.

| Component | Status |
|---|---|
| Signal loading, preprocessing, and feature extraction | Implemented |
| Four-class Random Forest classifier | Trained and evaluated |
| Isolation Forest anomaly detector | Implemented |
| Streamlit dashboard | Implemented and Docker-tested |
| FastAPI inference service | Implemented and Docker-tested |
| MLflow tracking service | Configured and containerized |
| DVC dataset tracking and S3 storage | Configured and pull-tested from a fresh clone |
| Docker Compose deployment | Tested with all three services running |
| Live CNC sensor integration | Not implemented |
| RUL estimation | Not implemented |

## System Architecture

### Offline analysis pipeline

```text
CWRU .mat / numeric CSV recordings
                 |
                 v
       Data loading and validation
                 |
                 v
        Signal segmentation
                 |
                 v
      Feature extraction per window
  RMS, standard deviation, kurtosis, peaks,
       crest factor and FFT features
                 |
                 v
       Trained machine-learning models
          /                  \
         v                    v
 Bearing classifier     Anomaly detector
          \                  /
           +----------------+
                   |
                   v
       Streamlit dashboard / FastAPI
```

### Docker Compose services

```text
                 Docker Compose
       +---------------+---------------+
       |               |               |
       v               v               v
  Streamlit UI      FastAPI          MLflow
    :8501            :8000            :5000
       |               |
       v               v
 Local model       Local model
 inference         inference API

reports/classifier is mounted read-only
into the dashboard and API containers.
```

The Streamlit dashboard performs inference locally using the mounted model bundle; it does **not** call FastAPI. FastAPI exposes a separate prediction endpoint for API clients. MLflow provides the experiment-tracking UI and backend store.

## Dataset

### CWRU Bearing Data Center

The [Case Western Reserve University Bearing Data Center](https://engineering.case.edu/bearingdatacenter) provides vibration recordings collected on a bearing test rig, including normal and seeded bearing-fault conditions such as inner-race, outer-race, and rolling-element faults.

The classifier workflow uses 16 documented CWRU MATLAB recordings. See [the classifier dataset and evaluation guide](docs/bearing_classifier.md) for the file manifest, labels, channel selection, and training details.

The raw dataset is managed with DVC rather than committed as ordinary Git files. Follow the dataset provider's terms when downloading or redistributing data.

### NASA C-MAPSS (possible separate RUL study)

[NASA's Prognostics Center of Excellence data repository](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository/) includes the C-MAPSS turbofan degradation dataset. It is an engine run-to-failure dataset, **not a CNC spindle or bearing dataset**. It may support a separate RUL learning exercise, but results from it must not be presented as validated CNC-bearing RUL predictions.

## Signal Processing and Feature Engineering

Signals are divided into fixed-size windows so that features can be calculated consistently over time.

### Time-domain features

| Feature | Description | Diagnostic relevance |
|---|---|---|
| RMS | Root mean square amplitude | Overall vibration energy |
| Standard deviation | Signal dispersion | Variation in signal amplitude |
| Kurtosis | Fourth standardized moment | Impulsive or impact-like behaviour |
| Peak amplitude | Maximum absolute amplitude | Large transient events |
| Peak-to-peak | Maximum minus minimum | Signal amplitude range |
| Crest factor | Peak amplitude divided by RMS | Peakiness relative to signal energy |

### Frequency-domain features

The Fast Fourier Transform (FFT) converts a sampled time signal into frequency components. The implementation extracts frequency-related indicators such as dominant frequency and spectral centroid, and the dashboard displays the frequency spectrum.

Sampling rate and window length must be correct for the recording being analysed. Do not assume every CWRU file uses the same sampling frequency or channel.

## Machine-Learning Models

| Task | Model | Purpose |
|---|---|---|
| Bearing condition classification | Random Forest | Classify a window into one of the trained bearing-condition classes |
| Anomaly detection | Isolation Forest | Flag feature patterns that differ from the normal baseline |

The anomaly detector is trained on normal baseline recordings. It indicates unusual patterns; it does not determine the physical cause or identify a specific fault class.

### Train the classifier

After pulling the dataset with DVC, run:

```bash
python scripts/train_bearing_classifier.py
```

The training workflow extracts features from the documented CWRU recordings, performs recording-grouped evaluation, and writes outputs under `reports/classifier/`, including the model bundle, feature dataset, metrics, and confusion matrix.

### Train and run anomaly detection

```bash
python scripts/train_anomaly_detector.py
python scripts/detect_anomalies.py data/raw/cwru/108.mat --sampling-rate 12000
```

Check the selected recording's metadata and sampling frequency before running analysis. Anomaly outputs are saved under `reports/anomaly/`.

## Streamlit Dashboard

The dashboard supports uploaded CWRU-style MATLAB files and numeric CSV recordings. It provides:
- Signal/channel selection and sampling-rate configuration.
- Time-domain signal visualization and FFT frequency spectrum.
- Overall extracted feature summary and window-level feature trends.
- Bearing-condition predictions when the trained classifier bundle is available.
- Downloadable prediction results.

Run locally from the repository root:

```bash
python -m pip install -r requirements.txt
python -m streamlit run dashboard/app.py
```

## FastAPI Prediction Service

The API accepts raw vibration samples, applies the project's inference pipeline, and returns per-window bearing-condition predictions. It requires the trained model bundle at `reports/classifier/bearing_random_forest.joblib`.

Run locally:

```bash
python -m pip install -r requirements-api.txt
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

| Endpoint | Purpose |
|---|---|
| `GET /` | API root |
| `GET /health` | Service and model health |
| `POST /predict` | Predict condition from a vibration signal |
| `GET /docs` | Interactive Swagger documentation |

The `POST /predict` request accepts `signal`, `sampling_rate_hz`, and optional `window_size` and `overlap` values. Supply enough samples for at least one complete window and use the actual sampling frequency for the selected recording.

## MLOps: MLflow and DVC

### MLflow experiment tracking

The classifier training workflow logs experiment information to the `cnc_bearing_fault_detection` experiment, including model parameters, evaluation metrics, and model/report artifacts.

For local file-based tracking, the default tracking database is SQLite. Launch the local UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

When using the Docker Compose MLflow service, its UI is available at [http://localhost:5000](http://localhost:5000). Set `MLFLOW_TRACKING_URI=http://localhost:5000` in the environment of a host-based training process if you want it to log to the containerized tracking server.

### DVC dataset and pipeline versioning

The repository contains a DVC pipeline definition in `dvc.yaml` and its lock file. The CWRU dataset is tracked by DVC, with an S3-backed remote configured for this project.

For a fresh clone:

```bash
git clone -b mlops-upgrade https://github.com/Avdhoot-Ukirde/CNC-Predictive-Maintenance.git
cd CNC-Predictive-Maintenance
python -m pip install -r requirements.txt
dvc pull data/raw/cwru.dvc
dvc repro
```

The DVC remote uses an AWS CLI profile configured on the local machine. Configure valid AWS credentials and the required profile before running `dvc pull`; credentials are not stored in the repository. Never commit AWS access keys, secret keys, or other credentials.

Useful commands:

```bash
dvc status
dvc status -c
dvc repro
dvc push
```

## Docker Deployment

Docker Compose runs three services:

| Service | Port | Local URL | Description |
|---|---:|---|---|
| Streamlit dashboard | 8501 | [http://localhost:8501](http://localhost:8501) | Interactive signal analysis and local model inference |
| FastAPI | 8000 | [http://localhost:8000](http://localhost:8000) | Prediction API and health endpoint |
| MLflow | 5000 | [http://localhost:5000](http://localhost:5000) | Experiment-tracking UI |

### Prerequisites

- Docker Desktop with the Docker Engine running (Windows with WSL 2 is supported).
- The trained model bundle in `reports/classifier/`.
- The dataset/model artifacts needed by the selected workflow.

### Start the application

From the repository root:

```bash
docker compose up --build -d
docker compose ps
```

Open the dashboard at [http://localhost:8501](http://localhost:8501), the API docs at [http://localhost:8000/docs](http://localhost:8000/docs), and MLflow at [http://localhost:5000](http://localhost:5000).

### Stop the application

```bash
docker compose down
```

To stop containers and also delete the persistent MLflow volume, use `docker compose down -v`. **This removes the stored MLflow database and artifacts in that volume.**

The dashboard and API mount `reports/classifier/` read-only from the host. Ensure the trained model bundle exists there before starting the services. The MLflow service uses a named Docker volume for persistent tracking data.

## Automated Tests

Run the test suite from the repository root:

```bash
python -m pytest -q
```

The test suite has previously completed with **21 passed** and one Starlette deprecation warning. GitHub Actions also runs tests on pushes and pull requests. CI validates software behaviour; it does not independently validate performance on industrial CNC equipment.

## Technology Stack

| Area | Tools |
|---|---|
| Language | Python |
| Data processing | NumPy, Pandas |
| Signal processing | SciPy |
| Machine learning | Scikit-learn |
| Visualization | Matplotlib |
| Dashboard | Streamlit |
| API | FastAPI, Uvicorn |
| Experiment tracking | MLflow |
| Dataset and pipeline versioning | DVC |
| Remote dataset storage | AWS S3 |
| Containerization | Docker, Docker Compose |
| Testing and CI | Pytest, GitHub Actions |
| Version control | Git, GitHub |

## Repository Structure

```text
CNC-Predictive-Maintenance/
├── .dvc/
├── .github/workflows/
├── .streamlit/
├── api/
│   └── main.py
├── dashboard/
│   └── app.py
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   ├── bearing_classifier.md
│   ├── dataset_100_analysis.md
│   ├── local_run_guide.md
│   └── system_architecture.md
├── mlflow/
├── models/
├── reports/
│   └── classifier/
├── scripts/
│   ├── analyze_cwru.py
│   ├── detect_anomalies.py
│   ├── predict_bearing.py
│   ├── train_anomaly_detector.py
│   └── train_bearing_classifier.py
├── src/
├── tests/
├── .dockerignore
├── .dvcignore
├── Dockerfile
├── Dockerfile.dashboard
├── docker-compose.yml
├── dvc.yaml
├── dvc.lock
├── requirements-api.txt
├── requirements-dashboard.txt
├── requirements.txt
└── README.md
```

Raw dataset files and generated model/report artifacts are managed locally or through DVC as configured; they should not be committed as ordinary Git files unless explicitly intended.

## Getting Started

### Prerequisites

- Python 3.11 or later.
- Git.
- CWRU dataset access and configured DVC remote credentials.
- Docker Desktop (optional, for containerized deployment).

### 1. Clone the project

```bash
git clone -b mlops-upgrade https://github.com/Avdhoot-Ukirde/CNC-Predictive-Maintenance.git
cd CNC-Predictive-Maintenance
```

### 2. Create a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Pull the dataset

Configure the AWS CLI profile used by the project's DVC S3 remote, then run:

```bash
dvc pull data/raw/cwru.dvc
```

If you only want to inspect the application code, you can skip the dataset pull; signal analysis and model inference require the relevant local data and model artifacts.

### 5. Run tests

```bash
python -m pytest -q
```

### 6. Launch the dashboard

```bash
python -m streamlit run dashboard/app.py
```

### 7. Launch the API (optional)

In another terminal:

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for the interactive API documentation.

## First CWRU Recording Analysis

The normal-baseline recording `100.mat` was inspected using:

```bash
python scripts/analyze_cwru.py data/raw/cwru/100.mat --signal-key X100_DE_time --sampling-rate 12000
```

The analysis script reports recording metadata, signal statistics, extracted features, and window-level analysis outputs. Verify the sampling rate and channel for each recording rather than assuming 12 kHz applies to every file. See [the recorded analysis and metadata](docs/dataset_100_analysis.md).

## Development Roadmap

- [x] Reusable MATLAB/CSV loading, validation, segmentation, and feature extraction.
- [x] CWRU recording analysis tooling.
- [x] Labelled feature dataset generation.
- [x] Four-class Random Forest classifier training and recording-grouped evaluation.
- [x] Normal-baseline Isolation Forest workflow.
- [x] Streamlit dashboard with signal analysis and model predictions.
- [x] FastAPI inference service.
- [x] MLflow experiment tracking integration.
- [x] DVC dataset tracking, S3 remote, and pipeline definition.
- [x] Docker Compose deployment for dashboard, API, and MLflow.
- [ ] Simulated streaming of recorded vibration signals.
- [ ] Physical sensor and CNC data-acquisition integration.
- [ ] Independent validation on representative machine data.
- [ ] RUL research using a suitable run-to-failure dataset.

## Future Hardware Integration

A future prototype could investigate:
- An accelerometer mounted near the spindle or bearing housing.
- A suitable motor-current transducer.
- RPM acquisition using an encoder or tachometer.
- Data-acquisition hardware with an appropriate sampling rate and signal conditioning.
- An edge computer or PC for acquisition and inference.

Sensor mounting, electrical isolation, grounding, sampling rate, and machine safety require engineering review. Do not connect sensors directly to CNC control wiring without checking electrical specifications and using appropriate isolation. Initial software development and evaluation can be performed with public datasets.

## Limitations

- CWRU is a laboratory bearing test-rig dataset and does not represent every industrial CNC spindle or operating condition.
- The reported classifier score is specific to the selected recordings and grouped evaluation split; external validation is required.
- Model performance may change with bearing type, speed, load, sensor placement, mounting, sampling configuration, and machine domain.
- Anomaly detection flags unusual feature patterns; it does not establish a fault's physical cause.
- The current application processes recorded files; live CNC telemetry is not implemented.
- RUL estimation is not implemented and requires suitable run-to-failure data and a separate validation methodology.
- This project is an engineering and learning prototype, not a certified safety system or a replacement for manufacturer maintenance procedures.

## References

1. Case Western Reserve University, [Bearing Data Center](https://engineering.case.edu/bearingdatacenter).
2. NASA, [Prognostics Center of Excellence Data Set Repository](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository/).
3. [SciPy documentation](https://docs.scipy.org/doc/scipy/).
4. [Scikit-learn documentation](https://scikit-learn.org/stable/).
5. [Streamlit documentation](https://docs.streamlit.io/).
6. [FastAPI documentation](https://fastapi.tiangolo.com/).
7. [MLflow documentation](https://mlflow.org/docs/latest/).
8. [DVC documentation](https://dvc.org/doc).

## Author

**Avdhoot Ukirde**  
B.Tech — Robotics & Artificial Intelligence | Minor in Data Science

- GitHub: [@Avdhoot-Ukirde](https://github.com/Avdhoot-Ukirde)
- LinkedIn: [Avdhoot Ukirde](https://www.linkedin.com/in/avdhoot-ukirde/)

---

If you find this project useful, consider giving the repository a star. Suggestions and contributions are welcome.
