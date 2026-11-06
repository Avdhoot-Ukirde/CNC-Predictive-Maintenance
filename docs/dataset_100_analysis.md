# First CWRU recording: 100.mat

## Recording identification

The uploaded `100.mat` file contains these MATLAB variables:

- `X100_DE_time`: drive-end accelerometer signal, 485,643 samples
- `X100_FE_time`: fan-end accelerometer signal, 485,643 samples
- `X100RPM`: recorded motor speed, 1,725 RPM

The CWRU Bearing Data Center lists the normal baseline test at 3 HP as approximately 1,730 RPM. The file's own RPM variable is 1,725 RPM; retain the recorded value and document the difference rather than overwriting it. The official CWRU documentation describes 12,000 samples/second and 48,000 samples/second drive-end experiments. For this baseline recording, use the dataset's 12 kHz baseline configuration and preserve that assumption in the analysis metadata.

- Condition: normal baseline
- Load: 3 HP
- Approximate speed in CWRU baseline table: 1,730 RPM
- RPM stored inside this file: 1,725 RPM
- Channel used for the first analysis: drive end (`X100_DE_time`)
- Sampling rate used: 12,000 Hz
- Duration at 12 kHz: approximately 40.47 seconds

Sources:
- [CWRU download-data documentation](https://engineering.case.edu/bearingdatacenter/download-data-file)
- [CWRU normal baseline data table](https://engineering.case.edu/bearingdatacenter/normal-baseline-data)

## First-pass drive-end measurements

Calculated from the uploaded `X100_DE_time` channel:

| Measurement | Value |
|---|---:|
| Samples | 485,643 |
| Mean | 0.012459 |
| Standard deviation | 0.064695 |
| RMS | 0.065884 |
| Peak absolute amplitude | 0.306456 |
| Peak-to-peak | 0.590172 |
| Skewness | -0.127530 |
| Pearson kurtosis | 2.957181 |
| Crest factor | 4.651457 |

Pearson kurtosis is used here, where a Gaussian-like distribution is approximately 3. These are descriptive measurements of this recording; they are not a fault score or a machine-health threshold. A single normal recording cannot establish a reliable alarm threshold.

## Reproduce the analysis

Place the downloaded file at:

`data/raw/cwru/100.mat`

Install dependencies and run from the repository root:

```bash
python scripts/analyze_cwru.py data/raw/cwru/100.mat \
  --signal-key X100_DE_time \
  --sampling-rate 12000
```

The script writes:

- `reports/cwru_analysis/time_signal.png`
- `reports/cwru_analysis/fft_spectrum.png`
- `reports/cwru_analysis/summary.json`
- `reports/cwru_analysis/window_features.csv`

The script defaults to 2,048-sample windows with 50% overlap. Change these only with a documented reason. Do not mix windows from the same original recording across training and test splits; that would cause data leakage.

## Interpretation and next step

This recording is a normal baseline. It is useful for establishing the signal-processing pipeline and a reference distribution, but it cannot teach a classifier to distinguish fault types by itself. The next dataset milestone is to add labelled inner-race, ball, and outer-race fault recordings under documented operating conditions, then compare their window-level features against this baseline.
