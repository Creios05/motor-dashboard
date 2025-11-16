# streamlit_app.py
import streamlit as st
import pandas as pd
import paho.mqtt.client as mqtt
import threading, json, time, ssl, os
import plotly.express as px
import plotly.graph_objects as go
import joblib
from datetime import datetime
from collections import deque
import numpy as np

st.set_page_config(page_title="Motor VFD Monitor", layout="wide")

# -----------------------------
# SETTINGS (from Streamlit Secrets)
# -----------------------------
MQTT_HOST = st.secrets.get("MQTT_HOST", "eaf211c6.ala.eu-central-1.emqxsl.com")
MQTT_PORT = int(st.secrets.get("MQTT_PORT", 8883))
MQTT_USER = st.secrets.get("MQTT_USER", "Creios05")
MQTT_PASS = st.secrets.get("MQTT_PASS", "Kaiser05")
MQTT_TOPIC = st.secrets.get("MQTT_TOPIC", "motor-vfd-esp32")
BUFFER_SIZE = int(st.secrets.get("BUFFER_SIZE", 800))

# Buffer for last N records
if "buffer" not in st.session_state:
    st.session_state.buffer = deque(maxlen=BUFFER_SIZE)

# -----------------------------
# Load model and thresholds
# -----------------------------
MODEL_FILE = "model_iforest.pkl"
THRESH_FILE = "thresholds.json"

model = None
thresholds = None
freq_means = {}

# Load Isolation Forest model
if os.path.exists(MODEL_FILE):
    try:
        model = joblib.load(MODEL_FILE)
        st.sidebar.success("IsolationForest model loaded")
    except Exception as e:
        st.sidebar.error(f"Could not load model: {e}")

# Load THD thresholds
if os.path.exists(THRESH_FILE):
    try:
        with open(THRESH_FILE, "r") as fh:
            thresholds = json.load(fh)
        freq_means = {int(k): float(thresholds[k]['mean']) for k in thresholds.keys()}
        st.sidebar.success("thresholds.json loaded")
    except Exception as e:
        st.sidebar.error(f"Could not load thresholds.json: {e}")
else:
    st.sidebar.warning("thresholds.json not found — frequency prediction disabled")

# -----------------------------
# Helper functions
# -----------------------------
def classify_thd_rule(thd, pred_freq):
    """Rule-based classification."""
    if thd is None or np.isnan(thd):
        return "UNKNOWN"

    thd = float(thd)

    if thd <= 15.0 and pred_freq == 60:
        return "NORMAL"
    if 15.0 < thd <= 20.0:
        return "ALERTA"
    if thd > 20.0:
        return "ALARMA"
    return "ALERTA"

def predict_freq_by_thd_simple(thd):
    """Predict frequency from learned THD means."""
    if not freq_means:
        return None, None
    diffs = {f: abs(thd - freq_means[f]) for f in freq_means}
    pred = min(diffs, key=diffs.get)
    return int(pred), diffs[pred]

def ml_anomaly_label(V, I, THD):
    """Evaluate Isolation Forest model"""
    if model is None:
        return None
    try:
        X = [[float(V), float(I), float(THD)]]
        pred = model.predict(X)[0]
        return "NORMAL" if pred == 1 else "ANOMALIA_ML"
    except:
        return None

# -----------------------------
# MQTT Handling
# -----------------------------
def on_connect(client, userdata, flags, rc):
    print("MQTT connected, rc=", rc)
    client.subscribe(MQTT_TOPIC)

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())

        rec = {
            "ts": datetime.now(),
            "V": float(data.get("V", np.nan)),
            "I": float(data.get("I", np.nan)),
            "THD": float(data.get("THD", np.nan)),
            "freq_set": int(data.get("freq_set", 0)),
        }

        # Frequency via THD model
        if rec["THD"] is not None and freq_means:
            pred_f, diff = predict_freq_by_thd_simple(rec["THD"])
            rec["pred_freq"] = pred_f
            rec["pred_diff"] = diff
        else:
            rec["pred_freq"] = None
            rec["pred_diff"] = None

        # ML anomaly
        rec["ml_label"] = ml_anomaly_label(rec["V"], rec["I"], rec["THD"])

        # Final status
        rule_label = classify_thd_rule(rec["THD"], rec["pred_freq"] if rec["pred_freq"] else -1)

        if rec["ml_label"] == "ANOMALIA_ML":
            rec["status"] = "ANOMALIA_ML"
        else:
            rec["status"] = rule_label

        st.session_state.buffer.append(rec)

    except Exception as e:
        print("Error parsing MQTT msg:", e)

# -----------------------------
# Start MQTT thread (one time)
# -----------------------------
if "mqtt_thread" not in st.session_state:
    def mqtt_runner():
        client = mqtt.Client()
        client.username_pw_set(MQTT_USER, MQTT_PASS)
        client.tls_set(cert_reqs=ssl.CERT_NONE)
        client.tls_insecure_set(True)
        client.on_connect = on_connect
        client.on_message = on_message
        client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
        client.loop_forever()

    t = threading.Thread(target=mqtt_runner, daemon=True)
    t.start()
    st.session_state.mqtt_thread = t
    time.sleep(0.5)

# -----------------------------
# UI Layout
# -----------------------------
st.title("📡 Motor + Variador — Monitor en tiempo real")

col_a, col_b, col_c = st.columns([1, 1, 1])
with col_a:
    show_last = st.number_input("Mostrar últimas N filas", min_value=10, max_value=BUFFER_SIZE, value=200)
with col_b:
    refresh_sec = st.slider("Intervalo actualización (s)", 0.5, 5.0, 1.0, step=0.5)
with col_c:
    show_table = st.checkbox("Mostrar tabla", value=True)

placeholder = st.empty()

# -----------------------------
# Real-time Loop
# -----------------------------
with placeholder:
    while True:
        buf = st.session_state.buffer
        if len(buf) == 0:
            st.info("Esperando datos MQTT…")
            time.sleep(0.5)
            continue

        df = pd.DataFrame(list(buf)).tail(show_last).copy()

        # Metrics
        latest = df.iloc[-1]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Voltaje (V)", f"{latest['V']:.2f}")
        col2.metric("Corriente (A)", f"{latest['I']:.3f}")
        col3.metric("THD (%)", f"{latest['THD']:.2f}")
        col4.metric("Frecuencia estimada", f"{latest['pred_freq']} Hz")

        # Status
        st.markdown("**Estado actual:**")
        status = latest.get("status", "—")
        if status == "NORMAL":
            st.success("🟢 NORMAL")
        elif status == "ALERTA":
            st.warning("🟡 ALERTA (15% < THD ≤ 20%)")
        elif status == "ALARMA":
            st.error("🔴 ALARMA (THD > 20%)")
        elif status == "ANOMALIA_ML":
            st.error("🟣 ANOMALÍA (Modelo ML)")
        else:
            st.info(status)

        # THD plot
        df_plot = df.copy()
        fig_thd = go.Figure()
        color_map = {"NORMAL": "green", "ALERTA": "orange", "ALARMA": "red", "ANOMALIA_ML": "purple"}
        df_plot["color"] = df_plot["status"].map(color_map).fillna("gray")

        fig_thd.add_trace(go.Scatter(
            x=df_plot["ts"], y=df_plot["THD"],
            mode="lines+markers",
            marker=dict(color=df_plot["color"], size=7),
            line=dict(color="lightgray"),
            name="THD"
        ))
        fig_thd.update_layout(title="THD (%) vs Tiempo", yaxis_title="THD (%)")
        st.plotly_chart(fig_thd, use_container_width=True)

        # I + V plot
        fig_vi = go.Figure()
        fig_vi.add_trace(go.Scatter(x=df["ts"], y=df["I"], mode="lines+markers", name="Corriente (A)"))
        fig_vi.add_trace(go.Scatter(x=df["ts"], y=df["V"], mode="lines+markers", name="Voltaje (V)"))
        fig_vi.update_layout(title="Voltaje & Corriente vs Tiempo")
        st.plotly_chart(fig_vi, use_container_width=True)

        # Frequency prediction
        fig_freq = go.Figure()
        fig_freq.add_trace(go.Scatter(x=df["ts"], y=df["pred_freq"], mode="lines+markers", name="pred_freq"))
        fig_freq.update_layout(title="Frecuencia Estimada (Hz)")
        st.plotly_chart(fig_freq, use_container_width=True)

        # Data table
        if show_table:
            st.dataframe(df.tail(50))

        time.sleep(refresh_sec)
