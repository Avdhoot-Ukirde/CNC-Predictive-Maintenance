# Initial system architecture

This milestone implements the offline signal-processing foundation:

    CWRU .mat / vibration .csv
              |
              v
          Data loader
              |
              v
        Signal validation
              |
              v
        Fixed-size windows
              |
              v
       Time + FFT features
              |
              v
         Feature table (next milestone)

## Design decisions

1. Dataset files stay local and are not committed to the repository.
2. The loader exposes channel selection rather than silently choosing between
   drive-end, fan-end, or base channels.
3. Sampling frequency is an explicit input to feature extraction; frequency
   outputs are invalid if it is guessed incorrectly.
4. Window overlap is configurable. For model training, split by original
   recording before generating overlapping windows to prevent data leakage.
5. This is offline dataset processing, not live CNC telemetry or hardware
   monitoring yet.

## Planned extensions

- Build a metadata-aware feature table with recording IDs and labels.
- Add recording-level train/test splitting and baseline fault classification.
- Add anomaly detection and an offline Streamlit dashboard.
- Later integrate timestamped sensor acquisition for accelerometer and
  spindle-current data.
