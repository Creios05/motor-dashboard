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
MQTT_HOST = st.secrets.get("MQTT_HOST", "TU_HOST.emqxsl.com")
MQTT_PORT = int(st.secrets.get("MQTT_PORT", 8883))
MQTT_USER = st.secrets.get("MQTT_USER", "esp32")
MQTT_PASS = st.secrets.get("MQTT_PASS", "12345678")
MQTT_TOPIC = st.secrets.get("MQTT_TOPIC", "motor-vfd-esp32")

# Buffer for last N records
BUFFER_SIZE = int(st.secrets.get("BUFFER_SIZE", 800))
if "buffer" not in st.session_state:
    st.session_state.buffer = deque(maxlen=BUFFER_SIZE)

# Try load model and thresholds if present in repo
MODEL_FILE = "model_iforest.pkl"
THRESH_FILE = "thresholds.json"

model = None
thresholds = None
freq_means = {}
if os.path.exists(MODEL_FILE):
    try:
        model = joblib.load(MODEL_FILE)
        st.sidebar.success("IsolationForest model loaded")
    except Exception as e:
        st.sidebar.error(f"Could not load model: {e}")

if os.path.exists(THRESH_FILE):
    try:
        with open(THRESH_FILE, "r") as fh:
            thresholds = json.load(fh)
        # build freq_means map
        freq_means = {int(k): float(thresholds[k]['mean']) for k in thresholds.keys()}
        st.sidebar.success("thresholds.json loaded")
    except Exception as e:
        st.sidebar.error(f"Could not load thresholds.json: {e}")
else:
    st.sidebar.warning("thresholds.json not found — freq prediction by THD will be disabled")

# -----------------------------
# Helper functions
# -----------------------------
def classify_thd_rule(thd, pred_freq):
    """Rules:
       - NORMAL only if pred_freq==60 and thd <= 15%
       - ALERT if 15% < thd <= 20%
       - ALARM if thd > 20%
       - Otherwise ALERT by default
    """
    try:
        thd = float(thd)
    except:
        return "UNKNOWN"
    if thd <= 15.0 and pred_freq == 60:
        return "NORMAL"
    if 15.0 < thd <= 20.0:
        return "ALERTA"
    if thd > 20.0:
        return "ALARMA"
    return "ALERTA"

def predict_freq_by_thd_simple(thd):
    """If thresholds.json exists, pick frequency whose mean THD is closest."""
    if not freq_means:
        return None, None
    diffs = {f: abs(thd - freq_means[f]) for f in freq_means}
    pred = min(diffs, key=diffs.get)
    return int(pred), diffs[pred]

def ml_anomaly_label(V,I,THD):
    if model is None:
        return None
    try:
        X = [[float(V), float(I), float(THD)]]
        pred = model.predict(X)[0]
        return "NORMAL" if pred==1 else "ANOMALIA_ML"
    except Exception as e:
        return None

# -----------------------------
# MQTT callbacks
# -----------------------------
def on_connect(client, userdata, flags, rc):
    print("MQTT connected, rc=", rc)
    client.subscribe(MQTT_TOPIC)

def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode()
        data = json.loads(payload)
        # Expecting at least V,I,THD,freq_set (freq_set optional)
        rec = {
            "ts": datetime.now(),
            "V": float(data.get("V", np.nan)),
            "I": float(data.get("I", np.nan)),
            "THD": float(data.get("THD", np.nan)),
            "freq_set": int(data.get("freq_set", 0)) if data.get("freq_set") is not None else None,
            "fund": float(data.get("fund", np.nan)) if data.get("fund") is not None else None
        }

        # Predict frequency (THD mean comparision) or use optional classifier if available
        if rec["THD"] is not None and freq_means:
            pred_freq, diff = predict_freq_by_thd_simple(rec["THD"])
            rec["pred_freq"] = pred_freq
            rec["pred_diff"] = diff
        else:
            rec["pred_freq"] = None
            rec["pred_diff"] = None

        # ML anomaly
        rec["ml_label"] = ml_anomaly_label(rec["V"], rec["I"], rec["THD"])

        # Final status combining ML and rules
        rule_label = classify_thd_rule(rec["THD"], rec["pred_freq"] if rec["pred_freq"] is not None else -1)
        if rec["ml_label"] == "ANOMALIA_ML":
            rec["status"] = "ANOMALIA_ML"
        else:
            rec["status"] = rule_label

        # Add to buffer thread-safely
        st.session_state.buffer.append(rec)

    except Exception as e:
        print("Error parsing MQTT msg:", e)

# -----------------------------
# Start MQTT thread once
# -----------------------------
if "mqtt_thread" not in st.session_state:
    def mqtt_runner():
        client = mqtt.Client()
        client.username_pw_set(MQTT_USER, MQTT_PASS)
        # Use TLS but do not require server cert in Streamlit (works for demo)
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
# UI
# -----------------------------
st.title("📡 Motor + Variador — Monitor en tiempo real")
st.markdown("Visualización en vivo: THD, Corriente, Voltaje, Frecuencia estimada y estado (NORMAL/ALERTA/ALARMA/ANOMALÍA_ML).")

# Controls
col_a, col_b, col_c = st.columns([1,1,1])
with col_a:
    show_last = st.number_input("Mostrar últimas N filas", min_value=10, max_value=BUFFER_SIZE, value=200)
with col_b:
    refresh_sec = st.slider("Intervalo actualización (s)", 0.5, 5.0, 1.0, step=0.5)
with col_c:
    show_table = st.checkbox("Mostrar tabla", value=True)

# Build dataframe from buffer
with st.empty():
    while True:
        if len(st.session_state.buffer) == 0:
            st.info("Esperando datos MQTT...")
            time.sleep(0.5)
            continue

        df = pd.DataFrame(list(st.session_state.buffer))
        if df.empty:
            st.info("Sin datos aún...")
            time.sleep(0.5)
            continue

        df = df.tail(show_last).copy()
        # Ensure types
        df["V"] = pd.to_numeric(df["V"], errors="coerce")
        df["I"] = pd.to_numeric(df["I"], errors="coerce")
        df["THD"] = pd.to_numeric(df["THD"], errors="coerce")
        df["pred_freq"] = pd.to_numeric(df.get("pred_freq"), errors="coerce")
        df["ts_str"] = df["ts"].astype(str)

        # Top metrics
        latest = df.iloc[-1]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Voltaje (V)", f"{latest['V']:.2f}")
        col2.metric("Corriente (A)", f"{latest['I']:.3f}")
        col3.metric("THD (%)", f"{latest['THD']:.2f}")
        pred_freq_display = int(latest['pred_freq']) if not pd.isna(latest['pred_freq']) else (int(latest['freq_set']) if latest.get('freq_set') else None)
        col4.metric("Frecuencia estimada (Hz)", f"{pred_freq_display if pred_freq_display is not None else '—'}")

        # Status badge
        st.markdown("**Estado actual:**")
        status = latest.get("status", "—")
        if status == "NORMAL":
            st.success("🟢 NORMAL")
        elif status == "ALERTA":
            st.warning("🟡 ALERTA (15% < THD ≤ 20%)")
        elif status == "ALARMA":
            st.error("🔴 ALARMA (THD > 20%)")
        elif status == "ANOMALIA_ML":
            st.error("🔴 ANOMALÍA (ML)")
        else:
            st.info(status)

        # Time series: THD colored by status
        df_plot = df.copy()
        color_map = {"NORMAL":"green","ALERTA":"orange","ALARMA":"red","ANOMALIA_ML":"purple"}
        df_plot["color"] = df_plot["status"].map(color_map).fillna("gray")
        fig_thd = go.Figure()
        fig_thd.add_trace(go.Scatter(x=df_plot["ts"], y=df_plot["THD"], mode="lines+markers",
                                     marker=dict(color=df_plot["color"], size=6),
                                     line=dict(color="lightgray"),
                                     name="THD"))
        fig_thd.update_layout(title="THD (%) vs Tiempo", xaxis_title="Tiempo", yaxis_title="THD (%)",
                              height=350)
        st.plotly_chart(fig_thd, use_container_width=True)

        # I and V plots
        fig_vi = make_vi_figure = go.Figure()
        fig_vi.add_trace(go.Scatter(x=df["ts"], y=df["I"], mode="lines+markers", name="I (A)"))
        fig_vi.add_trace(go.Scatter(x=df["ts"], y=df["V"], mode="lines+markers", name="V (V)"))
        fig_vi.update_layout(title="Corriente (A) y Voltaje (V)", xaxis_title="Tiempo", height=350)
        st.plotly_chart(fig_vi, use_container_width=True)

        # Predicted frequency plot
        if "pred_freq" in df.columns and not df["pred_freq"].isna().all():
            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(x=df["ts"], y=df["pred_freq"], mode="lines+markers", name="pred_freq"))
            if "fund" in df.columns:
                fig_f.add_trace(go.Scatter(x=df["ts"], y=df["fund"], mode="lines+markers", name="fund (if provided)"))
            fig_f.update_layout(title="Frecuencia estimada / Frecuencia fundamental (Hz)", xaxis_title="Tiempo", yaxis_title="Hz", height=300)
            st.plotly_chart(fig_f, use_container_width=True)

        # Show table
        if show_table:
            st.subheader("Últimos registros")
            st.dataframe(df[["ts_str","V","I","THD","freq_set","pred_freq","ml_label","status"]].tail(50))

        time.sleep(refresh_sec)