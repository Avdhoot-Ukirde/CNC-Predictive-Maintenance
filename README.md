# CNC Predictive Maintenance

> **Machine-learning-based condition monitoring and predictive maintenance for CNC spindle and bearing systems.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Project-In%20Development-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

Unexpected spindle or bearing failures can interrupt CNC operations, affect part quality, increase maintenance costs, and cause unplanned production downtime. This project aims to develop a predictive-maintenance pipeline that analyses machine-condition data, identifies abnormal behaviour, and provides actionable maintenance alerts.

The initial development uses the **Case Western Reserve University (CWRU) Bearing Data Center dataset** to build and validate a vibration-analysis and bearing-fault-detection workflow. The longer-term goal is to extend the pipeline to live sensor acquisition from a CNC machine and provide a real-time monitoring dashboard.

**Project status:** The initial offline signal-processing foundation is implemented: CWRU-style MATLAB/CSV loading, signal validation, fixed-window segmentation, time/frequency feature extraction, and automated tests. Dataset-specific exploration, trained models, measured accuracy, live hardware monitoring, and RUL estimates are not implemented yet.

## Problem Statement

Traditional maintenance approaches are often time-based (service after a fixed interval) or reactive (repair after a failure). Neither approach necessarily reflects the machine's actual condition.

A condition-monitoring system can use signals such as vibration and motor current to detect changes associated with developing faults. This project focuses on extracting useful signal features and applying machine-learning methods to support earlier fault identification.

## Project Objectives

- Analyse vibration signals from rotating machinery and bearing test data.
- Preprocess raw signals into consistent windows suitable for analysis.
- Extract time-domain and frequency-domain features, including RMS, kurtosis, and FFT-based indicators.
- Explore normal-versus-faulty condition classification.
- Investigate unsupervised anomaly detection for unusual operating behaviour.
- Present machine-condition information, trends, and alerts in a Streamlit dashboard.
- Design the software so that future accelerometer, current-sensor, and RPM telemetry can be integrated.
- Explore Remaining Useful Life (RUL) estimation only when a suitable run-to-failure dataset and validation method are available.

## Planned System Architecture

```text
        DATA SOURCES
  CWRU bearing dataset / later sensor stream
                  |
                  v
        DATA INGESTION
  File loading / sensor acquisition
                  |
                  v
       PREPROCESSING
  Validation, cleaning, segmentation
                  |
                  v
      FEATURE EXTRACTION
  RMS, standard deviation, kurtosis,
  peak-to-peak, crest factor, FFT features
                  |
                  v
       MACHINE LEARNING
  Fault classification / anomaly detection
                  |
                  v
       MONITORING LAYER
  Condition labels, trends, alert logic
                  |
                  v
       STREAMLIT DASHBOARD
  Signal plots, feature trends, alerts
```

The first milestone is an offline pipeline using recorded dataset files. Live acquisition and CNC integration are later milestones, not prerequisites for validating the initial models.

## Data Sources

### 1. CWRU Bearing Data Center — primary dataset

The [Case Western Reserve University (CWRU) Bearing Data Center](https://engineering.case.edu/bearingdatacenter) provides bearing vibration data collected using a motor-driven bearing test rig. The dataset includes normal and seeded bearing-fault conditions, including inner-race, outer-race, and rolling-element (ball) faults under different test conditions.

**Why it is used:** It is a practical starting point for learning vibration preprocessing, feature extraction, and supervised bearing-fault classification.

Before training, document the selected files, sensor channel, sampling frequency, load/speed condition, fault category, and label mapping. Keep the original dataset separate from generated features and model outputs. Check the dataset's usage terms and cite the source in any report.

### 2. NASA C-MAPSS — possible RUL extension

[NASA's Prognostics Center of Excellence data resources](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository/) include the C-MAPSS turbofan engine degradation dataset. It is a run-to-failure engine dataset, **not a CNC spindle or bearing dataset**.

It may be used later as a separate exercise to study degradation modelling and RUL estimation. Results trained on C-MAPSS must not be presented as validated CNC-bearing RUL predictions.

## Signal Processing and Feature Engineering

Vibration is a time-varying signal. Processing it in fixed-size windows allows the system to calculate comparable measurements over time.

### Time-domain features

| Feature | Meaning | Potential use |
|---|---|---|
| RMS | Root mean square amplitude | Overall vibration energy |
| Standard deviation | Signal dispersion | Changes in signal variability |
| Kurtosis | Tailedness / impulsiveness of the signal | Impact-like events and bearing defects |
| Peak amplitude | Maximum absolute amplitude | Large vibration events |
| Peak-to-peak | Maximum minus minimum | Signal amplitude range |
| Crest factor | Peak amplitude divided by RMS | Peaks relative to average signal energy |

For a window containing samples (x_1, x_2, \ldots, x_N):

**RMS**

`RMS = sqrt((1/N) * sum(x_i^2))`

**Kurtosis**

`Kurtosis = mean((x - mean(x))^4) / std(x)^4`

Implementation should specify the kurtosis convention (for example, Fisher excess kurtosis versus Pearson kurtosis), because library defaults can differ.

### Frequency-domain features

The Fast Fourier Transform (FFT) converts a sampled time signal into frequency components. The frequency spectrum can help reveal periodic vibration components and changes in bearing behaviour.

Planned analysis includes:

- FFT magnitude spectrum.
- Dominant frequency and its amplitude.
- Frequency-band energy.
- Comparison of spectra across normal and faulty conditions.

Sampling frequency and window length must be recorded so that FFT bins can be converted correctly into physical frequencies (Hz). Windowing and appropriate scaling should be considered when comparing spectra.

## Machine-Learning Approach

The project will evaluate models in stages rather than assuming one model is best.

| Task | Candidate approach | Purpose |
|---|---|---|
| Bearing condition classification | Random Forest, Support Vector Machine | Predict a known condition label |
| Anomaly detection | Isolation Forest | Flag feature patterns unlike the baseline |
| Baseline comparison | Logistic Regression / simple tree model | Establish a transparent reference |
| RUL research extension | Regression or sequence models | Estimate life only with suitable run-to-failure data |

### Training and evaluation principles

- Split data by recording, bearing, or experiment where possible; do not randomly split overlapping windows from the same recording across train and test sets.
- Fit scalers and preprocessing steps on training data only.
- Use class-wise precision, recall, F1-score, and a confusion matrix for classification.
- For anomaly detection, inspect false alarms and missed faults, not just an aggregate score.
- Keep a separate test set for final evaluation.
- Record dataset selection, preprocessing choices, random seeds, model parameters, and evaluation results.
- Do not claim that a model predicts a real machine failure unless it has been tested against representative machine data.

## Bearing Fault Classifier — Implemented, Awaiting Full Dataset Run

A four-class Random Forest training script is available at [`scripts/train_bearing_classifier.py`](scripts/train_bearing_classifier.py). It extracts features from the classic CWRU 12 kHz drive-end recordings and uses a recording-grouped train/test split to keep windows from the same source file together.

See [the classifier dataset and evaluation guide](docs/bearing_classifier.md) for the exact 16-file manifest and run instructions.

```bash
python scripts/train_bearing_classifier.py
```

The script requires the 16 CWRU `.mat` files listed in that guide under `data/raw/cwru/`. It writes the trained model, feature table, metrics, and confusion matrix to `reports/classifier/`. **No model accuracy is claimed yet:** the complete dataset has not been run in this environment. CWRU test-rig performance is not equivalent to validated industrial CNC spindle performance.

## Dashboard — Planned Features

A Streamlit interface is planned to make analysis accessible without opening a notebook.

- **Machine condition:** predicted class or anomaly status for the selected data window.
- **Vibration overview:** raw signal and FFT spectrum plots.
- **Feature monitoring:** RMS, kurtosis, peak amplitude, and other extracted features.
- **History:** feature values and model outputs over successive windows.
- **Alerts:** configurable warning thresholds and anomaly notifications.
- **Data upload:** load a supported CSV or dataset file for analysis.

Dashboard status indicators will be model outputs, not safety-certified machine controls. An alert should prompt inspection; it must not automatically stop or control industrial equipment.

## Technology Stack

| Area | Planned tools |
|---|---|
| Language | Python |
| Data manipulation | NumPy, Pandas |
| Signal processing | SciPy |
| Machine learning | Scikit-learn |
| Visualization | Matplotlib, Plotly |
| Dashboard | Streamlit |
| Data storage (later) | CSV / SQLite |
| Version control | Git and GitHub |

The exact dependencies will be maintained in `requirements.txt` as implementation progresses.

## Repository Structure

The current implemented foundation is shown below. Planned components are marked as future work.

```text
CNC-Predictive-Maintenance/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── README.md
│   ├── raw/
│   └── processed/
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   └── feature_extraction.py
├── tests/
│   ├── test_data_loader.py
│   └── test_signal_pipeline.py
├── models/
└── docs/
    └── system_architecture.md

Future: notebooks/, dashboard/, anomaly_detection.py, fault_classifier.py
```

## Getting Started

### Prerequisites

- Python 3.10 or later.
- Git.
- A code editor such as VS Code or JupyterLab.
- CWRU dataset files downloaded from the official source for the data-analysis milestone.

### 1. Clone the repository

```bash
git clone https://github.com/Avdhoot-Ukirde/CNC-Predictive-Maintenance.git
cd CNC-Predictive-Maintenance
```

### 2. Create and activate a virtual environment

**Windows (PowerShell)**

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

Install the project dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Add the dataset

Download the required CWRU files from the [official Bearing Data Center](https://engineering.case.edu/bearingdatacenter). Follow the dataset documentation and place local data under the appropriate `data/` subfolder.

Dataset files are intentionally not committed to this repository by default. This keeps the repository lightweight and avoids redistributing third-party data without checking its terms.

### 5. Run the automated tests

From the repository root:

```bash
python -m pytest -q
```

### 6. Use the signal-processing modules

The loader supports CWRU-style MATLAB files and CSV files. For a MATLAB file
with multiple vibration channels, inspect the available channel names and
select one explicitly:

```python
from src.data_loader import discover_mat_signals, load_cwru_signal
from src.preprocessing import segment_signal
from src.feature_extraction import extract_feature_rows

path = "data/raw/cwru/your_recording.mat"
print(discover_mat_signals(path))
signal = load_cwru_signal(path, signal_key="X097_DE_time")
windows = segment_signal(signal, window_size=2048, overlap=0.5)
features = extract_feature_rows(windows, sampling_rate=12000)
print(features[0])
```

Replace the example channel and sampling rate with the values documented for
your actual recording. Do not copy the example sampling rate blindly.

The Streamlit dashboard command will be added when dashboard/app.py is implemented.

## First CWRU Recording Analysis

The first normal-baseline recording (`100.mat`) has been inspected. A reusable command-line analysis script is available at [`scripts/analyze_cwru.py`](scripts/analyze_cwru.py). It generates a time-domain plot, FFT spectrum, overall feature summary, and window-level feature CSV.

Run it after placing the file under `data/raw/cwru/`:

```bash
python scripts/analyze_cwru.py data/raw/cwru/100.mat \
  --signal-key X100_DE_time --sampling-rate 12000
```

The sampling rate must be verified for the selected recording; do not reuse 12,000 Hz blindly for other CWRU files. See [the recorded analysis and metadata](docs/dataset_100_analysis.md) for the first file's channel, RPM, baseline condition, and measured features. The source `.mat` file is not committed to GitHub.

## Development Roadmap

- [x] **Milestone 1 foundation — Reusable data and signal modules:** implement MATLAB/CSV loading, channel discovery, validation, windowing, time/frequency features, and unit tests. Actual CWRU file selection and dataset exploration remain to be completed once the dataset is downloaded.
- [ ] **Milestone 2 — Dataset exploration and signal review:** select CWRU recordings, document labels and metadata, inspect real signals, and confirm channel sampling rates.
- [ ] **Milestone 3 — Feature dataset:** build a reproducible feature-extraction pipeline and save labelled feature tables.
- [ ] **Milestone 4 — Fault classification:** train baseline models and evaluate with recording-aware data splits.
- [ ] **Milestone 5 — Anomaly detection:** establish a normal-condition baseline and evaluate anomaly alerts.
- [ ] **Milestone 6 — Dashboard:** display signals, spectra, extracted features, model outputs, and alert history.
- [ ] **Milestone 7 — Streaming prototype:** replay recorded signals as a simulated stream and process incoming windows.
- [ ] **Milestone 8 — Hardware integration:** investigate accelerometer, current sensor, RPM acquisition, sampling, and safe CNC integration.
- [ ] **Milestone 9 — RUL research:** assess appropriate run-to-failure data and validate a separate RUL approach.

## Future Hardware Integration

A later prototype may use:

- Industrial or suitable MEMS accelerometer mounted near the spindle/bearing housing.
- Current sensor or current transducer compatible with the motor drive and electrical installation.
- RPM measurement from an encoder or suitable tachometer.
- Data-acquisition hardware with a sampling rate appropriate to the vibration frequencies of interest.
- An edge computer or PC to collect, process, and transmit telemetry.

Sensor mounting, electrical isolation, signal conditioning, grounding, sampling rate, and machine safety require careful engineering. Do not connect sensors directly to CNC control wiring without checking electrical specifications and using appropriate isolation. Initial software development can be completed entirely with public datasets.

## Expected Learning Outcomes

- Vibration analysis and digital signal processing.
- Time-domain and frequency-domain feature engineering.
- Bearing fault diagnosis and machine-learning evaluation.
- Anomaly detection and condition monitoring.
- Data pipelines and telemetry concepts.
- Streamlit dashboard development.
- Reproducible Python project structure and Git-based version control.
- Practical understanding of predictive maintenance in manufacturing.

## Limitations

- Public test-rig data does not fully represent every industrial CNC spindle, mounting arrangement, speed, load, or operating environment.
- A model trained on CWRU data may not generalize to a different machine without domain-specific data and validation.
- Vibration features can be affected by speed, load, sensor location, mounting, and sampling configuration.
- Anomaly detection identifies unusual patterns; it does not automatically identify the physical cause.
- RUL estimation requires suitable degradation/run-to-failure data and explicit validation.
- This project is a research and engineering prototype, not a certified safety system or a replacement for manufacturer maintenance procedures.

## References

1. Case Western Reserve University, **Bearing Data Center**: https://engineering.case.edu/bearingdatacenter
2. NASA, **Prognostics Center of Excellence Data Set Repository**: https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository/
3. SciPy documentation: https://docs.scipy.org/doc/scipy/
4. Scikit-learn documentation: https://scikit-learn.org/stable/
5. Streamlit documentation: https://docs.streamlit.io/

## Author

**Avdhoot Ukirde**  
B.Tech — Robotics & Artificial Intelligence | Minor in Data Science

- GitHub: [@Avdhoot-Ukirde](https://github.com/Avdhoot-Ukirde)
- LinkedIn: [Avdhoot Ukirde](https://www.linkedin.com/in/avdhoot-ukirde/)

---

If you find this project useful, consider giving the repository a star. Suggestions and contributions are welcome.
