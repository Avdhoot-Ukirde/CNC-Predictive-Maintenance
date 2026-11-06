"""Interactive offline dashboard for CNC vibration analysis."""
from __future__ import annotations
import sys
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.data_loader import load_signal
from src.feature_extraction import extract_features, extract_feature_rows
from src.inference import load_classifier, predict_signal
from src.preprocessing import segment_signal

MODEL_PATH = ROOT / "reports" / "classifier" / "bearing_random_forest.joblib"
st.set_page_config(page_title="CNC Predictive Maintenance", page_icon="⚙️", layout="wide")
st.title("⚙️ CNC Predictive Maintenance")
st.caption("Offline vibration analysis and bearing-condition monitoring prototype")

with st.sidebar:
    st.header("Analysis settings")
    sampling_rate = st.number_input("Sampling rate (Hz)", min_value=1, value=12000, step=100)
    window_size = st.selectbox("Window size (samples)", [1024, 2048, 4096, 8192], index=1)
    overlap_pct = st.slider("Window overlap (%)", 0, 90, 50, 5)
    st.divider()
    st.caption("Use the sampling rate documented for your recording. The default is suitable for the configured CWRU 12 kHz subset only.")

uploaded = st.file_uploader("Upload a vibration recording (.mat or .csv)", type=["mat", "csv"])
if uploaded is None:
    st.info("Upload a CWRU MATLAB file or a CSV containing a numeric vibration column to begin.")
    st.markdown("Workflow: upload a recording, select its channel, inspect the signal and features, then view model predictions after training.")
    st.code("python scripts/train_bearing_classifier.py", language="bash")
    st.stop()

suffix = Path(uploaded.name).suffix.lower()
with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
    temp.write(uploaded.getvalue())
    temp_path = Path(temp.name)

try:
    if suffix == ".mat":
        mat = loadmat(temp_path, squeeze_me=True)
        channel_keys = sorted(key for key in mat if not key.startswith("__") and key.endswith(("_DE_time", "_FE_time", "_BA_time")))
        if not channel_keys:
            st.error("No recognized CWRU vibration channels were found in this MATLAB file.")
            st.stop()
        selected_key = st.selectbox("MATLAB vibration channel", channel_keys)
        signal = load_signal(temp_path, signal_key=selected_key)
        source_channel = selected_key
    else:
        preview = pd.read_csv(temp_path, nrows=10)
        numeric_columns = preview.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_columns:
            st.error("The CSV must contain at least one numeric vibration column.")
            st.stop()
        selected_column = st.selectbox("CSV vibration column", numeric_columns)
        signal = load_signal(temp_path, csv_column=selected_column)
        source_channel = selected_column
except Exception as exc:
    st.error(f"Could not load the uploaded signal: {exc}")
    st.stop()
finally:
    temp_path.unlink(missing_ok=True)

if signal.size < window_size:
    st.error(f"The signal has {signal.size:,} samples, fewer than the selected {window_size}-sample window.")
    st.stop()

overlap = overlap_pct / 100.0
windows = segment_signal(signal, window_size, overlap)
overall = extract_features(signal, float(sampling_rate))
duration = signal.size / float(sampling_rate)

st.subheader("Recording overview")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Samples", f"{signal.size:,}")
m2.metric("Duration", f"{duration:.2f} s")
m3.metric("RMS", f"{overall['rms']:.5f}")
m4.metric("Pearson kurtosis", f"{overall['kurtosis_pearson']:.3f}")
st.caption(f"Source: {uploaded.name} · Channel: {source_channel} · {sampling_rate:,} Hz · {len(windows):,} complete windows")

left, right = st.columns(2)
with left:
    st.markdown("**Time-domain signal (first 0.1 seconds)**")
    count = min(signal.size, max(1, int(0.1 * sampling_rate)))
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(np.arange(count) / sampling_rate, signal[:count], linewidth=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.25)
    st.pyplot(fig)
    plt.close(fig)
with right:
    st.markdown("**Frequency spectrum**")
    centred = signal - np.mean(signal)
    hann = np.hanning(signal.size)
    spectrum = np.abs(np.fft.rfft(centred * hann))
    frequencies = np.fft.rfftfreq(signal.size, d=1.0 / sampling_rate)
    amplitude = 2 * spectrum / max(np.sum(hann), np.finfo(float).eps)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(frequencies, amplitude, linewidth=0.6)
    ax.set_xlim(0, sampling_rate / 2)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Approx. amplitude")
    ax.grid(alpha=0.25)
    st.pyplot(fig)
    plt.close(fig)

st.subheader("Extracted vibration features")
feature_frame = pd.DataFrame([overall]).T.rename(columns={0: "Value"})
st.dataframe(feature_frame, use_container_width=True)
st.subheader("Window-level feature trends")
window_features = pd.DataFrame(extract_feature_rows(windows, float(sampling_rate)))
window_features.index.name = "window_index"
trend_feature = st.selectbox("Feature to trend", ["rms", "kurtosis_pearson", "peak_amplitude", "crest_factor", "dominant_frequency_hz"])
st.line_chart(window_features[trend_feature])

if MODEL_PATH.is_file():
    st.subheader("Bearing condition model")
    try:
        bundle = load_classifier(MODEL_PATH)
        predictions = predict_signal(signal, float(sampling_rate), bundle, window_size, overlap)
        counts = predictions["predicted_condition"].value_counts().rename_axis("Condition").to_frame("Windows")
        st.bar_chart(counts)
        st.dataframe(predictions, use_container_width=True)
        st.download_button("Download window predictions (CSV)", predictions.to_csv(index=False).encode("utf-8"), file_name="bearing_predictions.csv", mime="text/csv")
    except Exception as exc:
        st.warning(f"Model could not be used: {exc}")
else:
    st.warning("No trained model found. Signal analysis is available now; train the classifier to enable condition predictions.")
    st.code("python scripts/train_bearing_classifier.py", language="bash")

st.caption("Research prototype only. Predictions are not safety-certified and must not be used as the sole basis for machine shutdown or maintenance decisions.")
