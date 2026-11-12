# Anomaly Detection Evaluation Report

## 1. Objective

Evaluate the project's Isolation Forest novelty detector on selected Case Western Reserve University (CWRU) bearing vibration recordings. The detector is fitted using normal-operation recordings and flags windows whose extracted features are unusual relative to that baseline.

This is an offline benchmark evaluation. It is not a validation of live CNC monitoring or an industrial alarm system.

## 2. Experimental setup

| Item | Configuration |
|---|---|
| Dataset | CWRU bearing vibration dataset |
| Signal channel | Drive-end (DE) |
| Sampling rate | 12,000 Hz |
| Window size | 2,048 samples |
| Window overlap | 50% |
| Detector | Isolation Forest |
| Number of estimators | 300 |
| Contamination parameter | 0.02 |
| Random state | 42 |
| Training recordings | `97.mat`, `98.mat`, `99.mat` |
| Training windows | 1,180 |

The fitted model was saved as `reports/anomaly/normal_isolation_forest_holdout.joblib`. Its saved metadata was checked and confirmed to list only `97.mat`, `98.mat`, and `99.mat` as normal training sources.

The normal recording `100.mat` was excluded from fitting and used as a held-out normal test recording. Three fault recordings were also evaluated using the same saved model.

## 3. Results

| Recording | Reference condition | Anomalous windows | Total windows | Window anomaly rate |
|---|---|---:|---:|---:|
| `100.mat` | Normal (held out) | 18 | 473 | 3.81% |
| `105.mat` | Inner-race fault | 117 | 117 | 100.00% |
| `118.mat` | Ball fault | 118 | 118 | 100.00% |
| `130.mat` | Outer-race fault | 118 | 118 | 100.00% |

The held-out normal recording had 455 windows flagged as normal and 18 flagged as anomalous. The three selected fault recordings had every evaluated window flagged as anomalous.

### Held-out normal false-alarm estimate

For `100.mat`, the observed window-level false-alarm rate was:

```text
18 / 473 × 100 = 3.81%
```

This is an estimate for this one held-out recording under the stated preprocessing and model configuration. It should not be interpreted as a population-wide or production false-alarm rate.

## 4. Interpretation

The results are encouraging for this selected CWRU subset:

- The detector was fitted using three normal recordings, while the normal test recording was excluded from training.
- A small proportion of windows in the held-out normal recording were flagged as anomalous (3.81%).
- All windows in the selected inner-race, ball, and outer-race fault recordings were flagged as anomalous.

The detector performs **novelty detection**, not fault classification. An anomaly score or flag indicates deviation from the learned normal feature distribution; it does not identify whether the cause is an inner-race, ball, or outer-race defect. The fault names in the table are reference labels for the test recordings, not predictions made by the Isolation Forest.

## 5. Limitations

1. **Small evaluation set:** The evaluation includes one held-out normal recording and one recording for each of three fault categories. More recordings and operating conditions are needed.
2. **Window dependence:** Windows overlap by 50%, so adjacent windows are not independent observations. The window counts must not be treated as hundreds of independent machine trials.
3. **Dataset scope:** CWRU is a laboratory bearing test-rig dataset. These results do not demonstrate performance on an actual CNC router or other industrial machine.
4. **Operating-condition coverage:** Performance across different loads, speeds, bearing types, sensor positions, mounting methods, and acquisition systems has not been established.
5. **No fault diagnosis or RUL:** The model does not identify fault type, estimate severity, predict remaining useful life, or issue safety-certified machine alarms.
6. **Threshold and contamination:** The Isolation Forest contamination setting influences the decision threshold. Further validation is needed before selecting an operational threshold.

## 6. Recommended next validation steps

- Evaluate additional normal recordings that were not used for fitting.
- Evaluate multiple independent recordings for each fault category.
- Where possible, split by complete recording and operating condition rather than randomly splitting overlapping windows.
- Report recording-level as well as window-level results, and include precision/recall only when the evaluation design supports a meaningful independent test.
- Test robustness to changes in speed, load, sensor placement, and measurement noise.
- Validate on vibration data collected from the target CNC machine before making deployment or maintenance claims.

## 7. Reproducibility

The following commands reproduce the training and the four tests from the repository root, assuming the CWRU files are available under `data/raw/cwru/`.

### Train using three normal recordings

```powershell
python scripts/train_anomaly_detector.py --normal-files 97.mat 98.mat 99.mat --output reports/anomaly/normal_isolation_forest_holdout.joblib
```

### Evaluate held-out normal and fault recordings

```powershell
python scripts/detect_anomalies.py data/raw/cwru/100.mat --sampling-rate 12000 --signal-key X100_DE_time --model reports/anomaly/normal_isolation_forest_holdout.joblib --output reports/anomaly/heldout_100_predictions.csv

python scripts/detect_anomalies.py data/raw/cwru/105.mat --sampling-rate 12000 --signal-key X105_DE_time --model reports/anomaly/normal_isolation_forest_holdout.joblib --output reports/anomaly/heldout_105_predictions.csv

python scripts/detect_anomalies.py data/raw/cwru/118.mat --sampling-rate 12000 --signal-key X118_DE_time --model reports/anomaly/normal_isolation_forest_holdout.joblib --output reports/anomaly/heldout_118_predictions.csv

python scripts/detect_anomalies.py data/raw/cwru/130.mat --sampling-rate 12000 --signal-key X130_DE_time --model reports/anomaly/normal_isolation_forest_holdout.joblib --output reports/anomaly/heldout_130_predictions.csv
```

The generated CSV files contain window-level anomaly predictions. The model and CSV outputs are local experiment artifacts; they should only be committed or versioned if that is consistent with the repository's artifact and data-management policy.

---

**Conclusion:** On this limited CWRU experiment, the normal-only Isolation Forest flagged 3.81% of windows in one held-out normal recording and flagged all windows in three selected fault recordings. This is a promising benchmark result, but broader recording-level and machine-level validation is required before drawing conclusions about industrial CNC predictive maintenance.
