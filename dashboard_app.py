"""
UAV Engine Digital Twin — Live Mission Dashboard   (Layer 8 in the roadmap)
-----------------------------------------------------------------------
This is a visual front end for the exact same engine model used by
engine_simulator.py — it imports digital_twin.py, engine_physics.py,
fault_scheduler.py, and data_logger.py directly, so the dashboard can
never show something the underlying model didn't actually produce.

DESIGN NOTES (why it looks the way it does):
  This is modelled loosely on aircraft engine instrument displays
  (EICAS/glass-cockpit style) rather than a generic "SaaS dashboard",
  because that's genuinely the right visual language for engine
  telemetry: dark background so colour-coded status reads instantly,
  a monospace face for the numbers so digits don't jitter sideways as
  they update, and a restrained green/amber/red vocabulary for health
  status instead of decorative color.

HOW TO RUN:
    pip install streamlit plotly pandas
    streamlit run dashboard_app.py

Then open the "Local URL" it prints (usually http://localhost:8501).
"""

import random
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from digital_twin import DigitalTwin
from engine_physics import EngineState, tick, detect_anomaly, engine_health_index
from fault_scheduler import FaultScheduler, FaultDefinition
from data_logger import DataLogger

# ----------------------------------------------------------------------
# PAGE CONFIG (must be the first Streamlit call)
# ----------------------------------------------------------------------

st.set_page_config(page_title="UAV Engine Digital Twin", page_icon="🛩️", layout="wide")

STATUS_COLORS = {
    "NORMAL": "#4ADE80",
    "CAUTION": "#F0A83D",
    "SUSPECTED ANOMALY": "#F0A83D",
    "ANOMALY DETECTED": "#EF5B5B",
}

MALE_MODELS = {
    "MQ-9 Reaper": {
        "engine": "Honeywell TPE331-10",
        "role": "ISR / Strike",
        "endurance": "27 hours",
        "ceiling": "50,000 ft",
        "payload": "3,850 lb",
        "description": "High-endurance tactical MALE optimized for long surveillance and precision strike missions.",
        "notes": [
            "Designed for persistent loiter with high thermal load during long-duration sorties.",
            "Primary watch items are cooling efficiency, oil pressure stability, and vibration trend.",
            "Most common health drift appears during climb and late-loiter phases."
        ],
    },
    "MQ-1C Gray Eagle": {
        "engine": "Rolls-Royce M250",
        "role": "Multi-mission ISR",
        "endurance": "36 hours",
        "ceiling": "29,000 ft",
        "payload": "1,500 lb",
        "description": "Flexible armed reconnaissance MALE with strong loiter capability and broad sensor payload support.",
        "notes": [
            "Ideal for surveillance-heavy operations with moderate endurance demands.",
            "Mission health is driven strongly by fuel flow consistency and thermal balancing.",
            "Oil pressure variance is a useful early indicator of wear or contamination."
        ],
    },
    "Heron TP": {
        "engine": "2 x 1,800 hp turboprop",
        "role": "Persistent ISR",
        "endurance": "52 hours",
        "ceiling": "45,000 ft",
        "payload": "2,200 lb",
        "description": "Long-endurance MALE built for persistent surveillance missions in demanding environments.",
        "notes": [
            "Long mission time amplifies wear on cooling and lubrication subsystems.",
            "Useful to track thermal margin and oil temperature drift over time.",
            "Good for demonstrating endurance-focused health trending."
        ],
    },
    "Bayraktar Akinci": {
        "engine": "2 x AI-450 turboprop",
        "role": "Heavy-class MALE",
        "endurance": "24 hours",
        "ceiling": "40,000 ft",
        "payload": "3,000 lb",
        "description": "A heavier MALE platform designed for remote sensing and strike mission flexibility.",
        "notes": [
            "Higher payload can increase fuel and thermal demand.",
            "Monitor vibration, cooling balance, and thermal efficiency closely.",
            "Useful for demonstrating multi-parameter health logic under load variation."
        ],
    },
}

# ----------------------------------------------------------------------
# THEME / CSS
# ----------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --bg: #09111d;
    --panel: #111d2f;
    --panel-2: #15253e;
    --border: #2a3d5a;
    --track: #1b2c43;
    --text: #ebf2ff;
    --muted: #9aaac0;
    --green: #4ADE80;
    --amber: #F0A83D;
    --red: #EF5B5B;
    --blue: #5FA8D3;
}

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp {
    background:
        radial-gradient(circle at top left, rgba(95,168,211,0.22), transparent 25%),
        radial-gradient(circle at bottom right, rgba(74,222,128,0.18), transparent 25%),
        linear-gradient(180deg, #09111d 0%, #0d1729 100%);
    color: var(--text);
    animation: appGlow 8s ease-in-out infinite alternate;
}
section[data-testid="stSidebar"] { background-color: var(--panel); border-right: 1px solid var(--border); }

@keyframes appGlow {
    0% { filter: saturate(1) brightness(1); }
    100% { filter: saturate(1.25) brightness(1.06); }
}
@keyframes floatUp {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-3px); }
}
@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 0 rgba(74,222,128,0.0), 0 0 18px rgba(95,168,211,0.15); }
    50% { box-shadow: 0 0 8px rgba(74,222,128,0.22), 0 0 24px rgba(95,168,211,0.35); }
}
@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

.topbar {
    display: flex; justify-content: space-between; align-items: center;
    gap: 12px; padding: 16px 18px; background: linear-gradient(90deg, rgba(18,30,49,0.92), rgba(24,41,68,0.98), rgba(14,25,41,0.95));
    border: 1px solid rgba(95,168,211,0.38); border-radius: 10px; margin-bottom: 18px; box-shadow: 0 0 0 1px rgba(95,168,211,0.08), 0 18px 40px rgba(3,7,18,0.35), 0 0 32px rgba(95,168,211,0.12);
    animation: floatUp 6s ease-in-out infinite;
}
.topbar-title { font-size: 1.9rem; font-weight: 700; }
.topbar-subtitle { font-size: 0.9rem; color: var(--muted); }
.status-group { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.chip {
    display: inline-flex; align-items: center; gap: 8px; padding: 6px 10px; border: 1px solid var(--border);
    border-radius: 999px; background: rgba(255,255,255,0.02); color: var(--text); font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase;
}
.chip-live { background: rgba(74, 222, 128, 0.12); border-color: rgba(74, 222, 128, 0.4); color: var(--green); }
.chip-live::before { content: ""; width: 8px; height: 8px; border-radius: 50%; background: var(--green); box-shadow: 0 0 12px rgba(74,222,128,0.9); animation: pulseDot 1.6s infinite ease-in-out; }
@keyframes pulseDot {
    0%, 100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.7); opacity: 0.7; }
}

.section-card, .info-panel, .model-box {
    background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 16px 18px; margin-bottom: 16px;
}
.section-label { font-size: 0.75rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); margin-bottom: 8px; }

.phase-banner {
    display: flex; justify-content: space-between; align-items: center;
    padding: 14px 20px; border: 1px solid rgba(95,168,211,0.4); border-radius: 10px;
    background: linear-gradient(90deg, rgba(17, 29, 47, 1), rgba(22, 38, 62, 1), rgba(18, 53, 67, 1)); margin-bottom: 16px;
    box-shadow: 0 0 18px rgba(95,168,211,0.12), inset 0 0 18px rgba(95,168,211,0.04);
    animation: pulseGlow 4s ease-in-out infinite;
}
.phase-banner .phase-name { font-size: 1.3rem; font-weight: 600; }
.phase-banner .phase-tick { color: var(--muted); font-size: 0.9rem; margin-left: 10px; }
.phase-banner .status-pill {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.9rem;
    padding: 6px 16px; border-radius: 4px; font-weight: 600;
    box-shadow: 0 0 22px rgba(255,255,255,0.08);
    animation: pulseGlow 3.2s ease-in-out infinite alternate;
}

.instrument-grid { display: grid; grid-template-columns: repeat(3, minmax(180px, 1fr)); gap: 12px; }
.instrument-tile {
    border: 1px solid rgba(95,168,211,0.25); background: linear-gradient(180deg, rgba(21,37,62,0.9), rgba(14,25,40,0.96)); padding: 12px 16px; border-radius: 8px;
    box-shadow: inset 0 0 16px rgba(95,168,211,0.06), 0 0 14px rgba(95,168,211,0.04);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.instrument-tile:hover { transform: translateY(-2px); box-shadow: inset 0 0 16px rgba(95,168,211,0.1), 0 0 20px rgba(95,168,211,0.14); }
.instrument-label { font-size: 0.75rem; color: var(--muted); margin-bottom: 6px; }
.instrument-value { font-family: 'IBM Plex Mono', monospace; font-size: 1.4rem; font-weight: 500; }
.instrument-unit { font-size: 0.82rem; color: var(--muted); margin-left: 4px; }

.metric-grid { display: grid; grid-template-columns: repeat(2, minmax(160px, 1fr)); gap: 12px; margin-top: 18px; }
.metric-card {
    border: 1px solid rgba(95,168,211,0.2); background: linear-gradient(180deg, rgba(15,25,40,0.98), rgba(17,31,49,0.96)); border-radius: 8px; padding: 12px 14px;
    box-shadow: inset 0 0 10px rgba(95,168,211,0.08);
    transition: transform 0.2s ease;
}
.metric-card:hover { transform: translateY(-2px); }
.metric-card .metric-label { color: var(--muted); font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.06em; }
.metric-card .metric-value { font-family: 'IBM Plex Mono', monospace; font-size: 1.25rem; margin-top: 8px; }
.metric-card .metric-foot { font-size: 0.75rem; color: var(--muted); margin-top: 6px; }

.ehi-wrap { border: 1px solid rgba(95,168,211,0.25); background: linear-gradient(180deg, rgba(17,29,47,0.95), rgba(11,17,29,0.98)); border-radius: 8px; padding: 16px 18px; box-shadow: 0 0 18px rgba(95,168,211,0.08), inset 0 0 18px rgba(95,168,211,0.06); }
.ehi-bar-track { width: 100%; height: 12px; background: var(--track); border-radius: 6px; overflow: hidden; margin-top: 12px; }
.ehi-bar-fill {
    height: 100%; border-radius: 6px;
    background: linear-gradient(90deg, #4ade80, #5fa8d3, #f0a83d, #ef5b5b);
    background-size: 200% 100%;
    animation: shimmer 2.5s linear infinite;
    box-shadow: 0 0 20px rgba(95,168,211,0.25);
}

.fault-ticker {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; color: var(--muted);
    border: 1px dashed var(--border); padding: 10px 14px; border-radius: 6px; margin-top: 12px;
}

.alert-box {
    padding: 14px 16px; border: 1px solid var(--border); border-left: 4px solid var(--blue); border-radius: 8px; background: rgba(95, 168, 211, 0.08);
    margin-top: 12px;
}
.alert-box strong { color: var(--text); }

.model-box ul { margin: 0.5rem 0 0 1.1rem; color: var(--muted); }
.model-box li { margin-bottom: 0.45rem; }

.gauge-row { display: grid; grid-template-columns: repeat(3, minmax(180px, 1fr)); gap: 12px; margin-bottom: 18px; }
.gauge-card {
    background: linear-gradient(180deg, rgba(17,29,47,0.94), rgba(16,25,38,0.96)); border: 1px solid rgba(95,168,211,0.2); border-radius: 10px; padding: 14px 12px; text-align: center;
    box-shadow: 0 0 18px rgba(95,168,211,0.08), inset 0 0 16px rgba(95,168,211,0.04);
    animation: floatUp 5s ease-in-out infinite;
}
.gauge {
    width: 112px; height: 112px; border-radius: 50%; margin: 0 auto 10px auto; display: grid; place-items: center;
    background: conic-gradient(var(--green) 0 75%, rgba(255,255,255,0.06) 75% 100%);
    position: relative;
    box-shadow: 0 0 25px rgba(74,222,128,0.25), inset 0 0 16px rgba(255,255,255,0.08);
}
.gauge::before {
    content: ""; position: absolute; inset: 12px; background: var(--panel); border-radius: 50%; border: 1px solid var(--border);
}
.gauge-value {
    position: relative; z-index: 1; font-family: 'IBM Plex Mono', monospace; font-size: 1.2rem; font-weight: 600; color: var(--text);
}
.gauge-label { color: var(--muted); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; }

.timeline {
    background: linear-gradient(180deg, rgba(17,29,47,0.96), rgba(13,22,34,0.96)); border: 1px solid rgba(95,168,211,0.2); border-radius: 8px; padding: 16px 18px; margin-top: 16px;
    box-shadow: inset 0 0 14px rgba(95,168,211,0.04), 0 0 18px rgba(95,168,211,0.04);
}
.timeline-item {
    display: flex; align-items: center; gap: 12px; padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.04);
}
.timeline-item:last-child { border-bottom: none; }
.timeline-dot {
    width: 10px; height: 10px; border-radius: 50%; background: var(--green); box-shadow: 0 0 12px rgba(74, 222, 128, 0.7); animation: pulseDot 1.8s infinite ease-in-out;
}
.timeline-item[data-level="amber"] .timeline-dot { background: var(--amber); box-shadow: 0 0 12px rgba(240, 168, 61, 0.7); }
.timeline-item[data-level="red"] .timeline-dot { background: var(--red); box-shadow: 0 0 12px rgba(239, 91, 91, 0.7); }
.timeline-meta { flex: 1; }
.timeline-name { font-weight: 600; }
.timeline-time { color: var(--muted); font-size: 0.8rem; }

.summary-box {
    background: linear-gradient(135deg, rgba(17,29,47,0.96), rgba(18,38,58,0.94)); border: 1px solid rgba(95,168,211,0.28); border-radius: 8px; padding: 14px 16px; margin-top: 12px;
    box-shadow: 0 0 18px rgba(95,168,211,0.06);
}
.summary-box .summary-title { font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); margin-bottom: 8px; }
.summary-box .summary-main { font-size: 1.2rem; font-weight: 600; }
.summary-box .summary-sub { color: var(--muted); font-size: 0.83rem; margin-top: 4px; }

.alert-feed {
    background: linear-gradient(180deg, rgba(17,29,47,0.96), rgba(13,22,34,0.96)); border: 1px solid rgba(95,168,211,0.22); border-radius: 8px; padding: 14px 16px; margin-top: 12px;
    box-shadow: inset 0 0 12px rgba(95,168,211,0.04), 0 0 20px rgba(95,168,211,0.08);
}
.alert-badge { animation: pulseDot 1.8s infinite ease-in-out; }
.alert-item {
    display: flex; align-items: flex-start; gap: 10px; border-bottom: 1px solid rgba(255,255,255,0.05); padding: 10px 0;
}
.alert-item:last-child { border-bottom: none; }
.alert-badge {
    width: 10px; height: 10px; border-radius: 50%; margin-top: 6px; flex-shrink: 0;
}
.alert-title { font-weight: 600; }
.alert-meta { color: var(--muted); font-size: 0.74rem; margin-top: 2px; }
.alert-message { color: var(--text); font-size: 0.83rem; margin-top: 4px; }

.log-table {
    width: 100%; border-collapse: collapse; font-size: 0.8rem; color: var(--text);
}
.log-table th, .log-table td {
    text-align: left; padding: 8px 10px; border-bottom: 1px solid rgba(255,255,255,0.07);
}
.log-table th { color: var(--muted); font-weight: 600; }
.log-table tbody tr:hover { background: rgba(95,168,211,0.08); }

.stSidebar .block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# MISSION DEFINITION (same as engine_simulator.py)
# ----------------------------------------------------------------------

MISSION_PHASES = [
    ("IDLE", 2000, 0.2),
    ("TAKEOFF", 5500, 1.0),
    ("CLIMB", 4800, 0.8),
    ("CRUISE", 4000, 0.6),
    ("DESCENT", 2800, 0.3),
    ("IDLE", 2000, 0.2),
]
SECONDS_PER_PHASE = 10
TOTAL_TICKS = len(MISSION_PHASES) * SECONDS_PER_PHASE


def build_fault_schedule():
    scheduler = FaultScheduler()
    scheduler.add_fault(FaultDefinition(
        name="COOLING_DEGRADATION", parameter="cooling_efficiency",
        healthy_value=1.0, faulty_value=0.6, start_time=15, ramp_duration=20,
    ))
    scheduler.add_fault(FaultDefinition(
        name="OIL_PRESSURE_DEGRADATION", parameter="oil_pressure_factor",
        healthy_value=1.0, faulty_value=0.75, start_time=40, ramp_duration=15,
    ))
    scheduler.add_fault(FaultDefinition(
        name="VIBRATION_FAULT", parameter="vibration_offset",
        healthy_value=0.0, faulty_value=0.35, start_time=48, ramp_duration=10,
    ))
    return scheduler


# ----------------------------------------------------------------------
# SIDEBAR — MISSION CONTROLS
# ----------------------------------------------------------------------

selected_model_name = st.sidebar.selectbox("MALE platform", list(MALE_MODELS.keys()))
selected_model = MALE_MODELS[selected_model_name]

st.sidebar.markdown("### Platform overview")
st.sidebar.markdown(f"**{selected_model_name}**")
st.sidebar.caption(selected_model["description"])
st.sidebar.markdown(f"- Engine: **{selected_model['engine']}**")
st.sidebar.markdown(f"- Role: **{selected_model['role']}**")
st.sidebar.markdown(f"- Endurance: **{selected_model['endurance']}**")
st.sidebar.markdown(f"- Ceiling: **{selected_model['ceiling']}**")
st.sidebar.markdown(f"- Payload: **{selected_model['payload']}**")

st.sidebar.markdown("---")
st.sidebar.markdown("### Mission controls")

speed_label = st.sidebar.select_slider(
    "Playback speed",
    options=["Real-time (1s/tick)", "Fast (0.15s/tick)", "Instant"],
    value="Fast (0.15s/tick)",
)
inject_faults = st.sidebar.checkbox("Inject faults", value=True)
fixed_seed = st.sidebar.checkbox("Use fixed seed (reproducible)", value=True)
show_ground_truth = st.sidebar.checkbox("Show ground-truth fault label", value=True)
mission_profile = st.sidebar.selectbox("Mission profile", ["Reconnaissance", "Loiter", "Transit", "Strike"])
run_clicked = st.sidebar.button("▶  Start mission", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### Engineering notes")
for note in selected_model["notes"]:
    st.sidebar.write(f"• {note}")

st.sidebar.caption(
    "Ground truth is what the fault scheduler actually injected. "
    "The status pill above the chart is what the detector concludes "
    "from sensor data alone — compare the two to show the detector "
    "working correctly."
)

DELAY_MAP = {"Real-time (1s/tick)": 1.0, "Fast (0.15s/tick)": 0.15, "Instant": 0.0}
delay = DELAY_MAP[speed_label]

# ----------------------------------------------------------------------
# HEADER + LAYOUT SLOTS
# ----------------------------------------------------------------------

st.markdown(
    f"<div class='topbar'><div><div class='topbar-title'>UAV Engine Digital Twin</div><div class='topbar-subtitle'>{selected_model_name} · {selected_model['engine']} · {selected_model['role']}</div></div><div class='status-group'><span class='chip chip-live'>Live</span><span class='chip'>{mission_profile}</span><span class='chip'>Ops mode</span></div></div>",
    unsafe_allow_html=True,
)

overview_tab, diagnostics_tab, logs_tab = st.tabs(["Overview", "Diagnostics", "Mission Log"])

with overview_tab:
    phase_placeholder = st.empty()
    summary_cards_placeholder = st.empty()
    chart_placeholder = st.empty()
    col_left, col_right = st.columns([2, 1])
    with col_left:
        grid_placeholder = st.empty()
    with col_right:
        ehi_placeholder = st.empty()
        ticker_placeholder = st.empty()
        model_summary_placeholder = st.empty()
        alert_placeholder = st.empty()

with diagnostics_tab:
    diagnostic_overview_placeholder = st.empty()
    timeline_placeholder = st.empty()
    diag_chart_placeholder = st.empty()

with logs_tab:
    log_placeholder = st.empty()

if not run_clicked:
    with overview_tab:
        chart_placeholder.info("No mission data yet — set your options in the sidebar, then click **Start mission**.")
        model_summary_placeholder.markdown(
            f"<div class='model-box'><div class='section-label'>Platform brief</div><strong>{selected_model_name}</strong><ul><li>Engine: {selected_model['engine']}</li><li>Role: {selected_model['role']}</li><li>Endurance: {selected_model['endurance']}</li><li>Ceiling: {selected_model['ceiling']}</li></ul></div>",
            unsafe_allow_html=True,
        )
        summary_cards_placeholder.markdown(
            """
            <div class='gauge-row'>
                <div class='gauge-card'><div class='gauge' style='background: conic-gradient(#4ADE80 0 76%, rgba(255,255,255,0.06) 76% 100%);'><div class='gauge-value'>76</div></div><div class='gauge-label'>Engine health</div></div>
                <div class='gauge-card'><div class='gauge' style='background: conic-gradient(#5FA8D3 0 82%, rgba(255,255,255,0.06) 82% 100%);'><div class='gauge-value'>82</div></div><div class='gauge-label'>Cooling margin</div></div>
                <div class='gauge-card'><div class='gauge' style='background: conic-gradient(#F0A83D 0 68%, rgba(255,255,255,0.06) 68% 100%);'><div class='gauge-value'>68</div></div><div class='gauge-label'>Vibration health</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        alert_placeholder.markdown(
            """
            <div class='alert-feed'>
                <div class='section-label'>Alert feed</div>
                <div class='alert-item'><div class='alert-badge' style='background:#4ADE80;'></div><div><div class='alert-title'>System nominal</div><div class='alert-meta'>Mission ready</div><div class='alert-message'>All health indicators are within nominal thresholds.</div></div></div>
                <div class='alert-item'><div class='alert-badge' style='background:#5FA8D3;'></div><div><div class='alert-title'>Thermal margin tracking</div><div class='alert-meta'>Monitoring</div><div class='alert-message'>Cooling margin is stable and ready for launch sequence.</div></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ----------------------------------------------------------------------
# RENDER HELPERS
# ----------------------------------------------------------------------

def render_phase_banner(phase, tick_in_phase, status, mission_profile):
    color = STATUS_COLORS.get(status, "#8792A8")
    phase_placeholder.markdown(f"""
    <div class="phase-banner">
        <div>
            <span class="phase-name">Phase: {phase}</span>
            <span class="phase-tick">({tick_in_phase}/{SECONDS_PER_PHASE}s)</span>
            <span class="phase-tick">• {mission_profile}</span>
        </div>
        <div class="status-pill" style="background:{color}22; color:{color}; border:1px solid {color}66;">{status}</div>
    </div>
    """, unsafe_allow_html=True)


def clamp(value, low, high):
    return max(low, min(high, value))


def derive_health_metrics(readings, cooling_efficiency, residual, ehi):
    exhaust_temp = 48 + readings["cht"] * 0.9 + readings["rpm"] * 0.003
    compressor_ratio = 10.5 + (readings["rpm"] / 1000) * 2.4 - (1 - cooling_efficiency) * 4.2
    thermal_margin = 100 - abs(readings["cht"] - 120) * 1.4
    oil_quality = 100 - abs((readings["oil_pressure"] - 4.2) * 18)
    fuel_efficiency = clamp(100 - abs(readings["fuel_flow"] - 18) * 4, 0, 100)
    vibration_health = clamp(100 - readings["vibration"] * 150, 0, 100)

    metrics = [
        ("Thermal margin", f"{thermal_margin:.0f}", "%", "headroom before thermal stress"),
        ("Exhaust temp", f"{exhaust_temp:.1f}", "°C", "core turbine outflow estimate"),
        ("Compressor ratio", f"{compressor_ratio:.2f}", ":1", "pressure rise relative to intake"),
        ("Oil quality", f"{oil_quality:.0f}", "%", "lubrication health indicator"),
        ("Fuel efficiency", f"{fuel_efficiency:.0f}", "%", "consumption efficiency"),
        ("Vibration health", f"{vibration_health:.0f}", "%", "mechanical stability"),
    ]
    return metrics, {
        "exhaust_temp": exhaust_temp,
        "compressor_ratio": compressor_ratio,
        "thermal_margin": thermal_margin,
        "oil_quality": oil_quality,
        "fuel_efficiency": fuel_efficiency,
        "vibration_health": vibration_health,
        "ehi": ehi,
        "residual": residual,
    }


def render_instrument_grid(readings, cooling_efficiency, residual, ehi):
    metrics, _ = derive_health_metrics(readings, cooling_efficiency, residual, ehi)
    tiles = [
        ("RPM", f"{readings['rpm']:.0f}", ""),
        ("Oil temperature", f"{readings['oil_temp']:.1f}", "°C"),
        ("Oil pressure", f"{readings['oil_pressure']:.2f}", "bar"),
        ("Fuel flow", f"{readings['fuel_flow']:.2f}", "L/min"),
        ("Vibration", f"{readings['vibration']:.3f}", "g"),
        ("Cooling efficiency", f"{cooling_efficiency * 100:.0f}", "%"),
    ]
    html = '<div class="instrument-grid">'
    for label, value, unit in tiles:
        html += (
            f'<div class="instrument-tile"><div class="instrument-label">{label}</div>'
            f'<div class="instrument-value">{value}<span class="instrument-unit">{unit}</span></div></div>'
        )
    html += "</div>"
    html += '<div class="metric-grid">'
    for label, value, unit, foot in metrics:
        status_color = "var(--green)" if label not in ["Exhaust temp", "Compressor ratio"] or float(value) > 70 else "var(--amber)"
        html += (
            f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value" style="color:{status_color}">{value}<span class="instrument-unit">{unit}</span></div>'
            f'<div class="metric-foot">{foot}</div></div>'
        )
    html += '</div>'
    grid_placeholder.markdown(html, unsafe_allow_html=True)


def render_ehi(ehi, status):
    color = STATUS_COLORS.get(status, "#8792A8")
    pct = max(0, min(100, ehi))
    ehi_placeholder.markdown(f"""
    <div class="ehi-wrap">
        <div class="instrument-label">Engine health index</div>
        <div class="instrument-value" style="color:{color}">{ehi:.0f}<span class="instrument-unit">/100</span></div>
        <div class="ehi-bar-track"><div class="ehi-bar-fill" style="width:{pct}%; background:{color};"></div></div>
    </div>
    """, unsafe_allow_html=True)


def render_model_summary(selected_model_name, selected_model):
    model_summary_placeholder.markdown(
        f"""
        <div class="model-box">
            <div class="section-label">Platform brief</div>
            <div style="font-size:1.2rem; font-weight:600; margin-bottom: 8px;">{selected_model_name}</div>
            <div style="color:#9aaac0; margin-bottom: 10px;">{selected_model['description']}</div>
            <ul>
                <li>Engine: {selected_model['engine']}</li>
                <li>Role: {selected_model['role']}</li>
                <li>Endurance: {selected_model['endurance']}</li>
                <li>Ceiling: {selected_model['ceiling']}</li>
                <li>Payload: {selected_model['payload']}</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_gauge_cards(ehi, cooling_efficiency, vibration):
    gauge_list = [
        ("Engine health", ehi, "#4ADE80" if ehi > 70 else "#F0A83D" if ehi > 45 else "#EF5B5B"),
        ("Cooling margin", cooling_efficiency * 100, "#5FA8D3"),
        ("Vibration health", max(0, 100 - vibration * 140), "#F0A83D" if vibration < 0.8 else "#EF5B5B"),
    ]
    html = "<div class='gauge-row'>"
    for label, value, color in gauge_list:
        pct = max(0, min(100, value))
        html += (
            f"<div class='gauge-card'><div class='gauge' style='background: conic-gradient({color} 0 {pct}%, rgba(255,255,255,0.06) {pct}% 100%);'><div class='gauge-value'>{pct:.0f}</div></div><div class='gauge-label'>{label}</div></div>"
        )
    html += "</div>"
    summary_cards_placeholder.markdown(html, unsafe_allow_html=True)


def render_summary_box(status, ehi, residual, cooling_efficiency):
    if status == "NORMAL":
        main = "Operationally stable"
        sub = "Engine performance remains within expected tolerances across the current phase."
    elif status == "CAUTION":
        main = "Attention required"
        sub = f"Thermal residual is {residual:+.2f} °C and cooling efficiency is {cooling_efficiency * 100:.0f}% ."
    else:
        main = "Fault condition active"
        sub = f"Engine health index is {ehi:.0f}/100 and residual drift is {residual:+.2f} °C."
    model_summary_placeholder.markdown(
        f"<div class='summary-box'><div class='summary-title'>Mission summary</div><div class='summary-main'>{main}</div><div class='summary-sub'>{sub}</div></div>",
        unsafe_allow_html=True,
    )


def render_ticker(fault_label, residual, show_gt):
    parts = [f"residual: {residual:+.2f} °C"]
    if show_gt:
        parts.append(f"ground truth: {fault_label}")
    ticker_placeholder.markdown(
        f'<div class="fault-ticker">{"&nbsp;&nbsp;|&nbsp;&nbsp;".join(parts)}</div>',
        unsafe_allow_html=True,
    )


def render_alert_feed(status, phase, residual, fault_label):
    if status == "NORMAL":
        badge = "#4ADE80"
        title = "System nominal"
        message = f"{phase} phase remains within healthy tolerance bands."
    elif status == "CAUTION":
        badge = "#F0A83D"
        title = "Watch condition"
        message = f"Residual drift is {residual:+.2f} °C — thermal behavior is trending upward."
    else:
        badge = "#EF5B5B"
        title = "Fault active"
        message = f"{fault_label} detected; health margin is degraded during {phase}."

    alert_placeholder.markdown(
        f"""
        <div class='alert-feed'>
            <div class='section-label'>Alert feed</div>
            <div class='alert-item'>
                <div class='alert-badge' style='background:{badge};'></div>
                <div>
                    <div class='alert-title'>{title}</div>
                    <div class='alert-meta'>{phase} · live telemetry</div>
                    <div class='alert-message'>{message}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_fault_timeline(active_fault_label, sim_time):
    fault_events = [
        ("Cooling degradation", "15s", "red" if 15 <= sim_time <= 35 else "amber" if sim_time < 15 else "green"),
        ("Oil pressure drift", "40s", "red" if 40 <= sim_time <= 55 else "amber" if sim_time < 40 else "green"),
        ("Vibration fault", "48s", "red" if 48 <= sim_time <= 58 else "amber" if sim_time < 48 else "green"),
    ]
    html = "<div class='timeline'><div class='section-label'>Fault timeline</div>"
    for name, at_time, level in fault_events:
        html += (
            f"<div class='timeline-item' data-level='{level}'>"
            f"<div class='timeline-dot'></div>"
            f"<div class='timeline-meta'><div class='timeline-name'>{name}</div><div class='timeline-time'>Starts at {at_time}</div></div>"
            f"<div style='color: var(--muted); font-size: 0.76rem;'>{'Active' if level == 'red' else 'Monitor' if level == 'amber' else 'Standby'}</div>"
            f"</div>"
        )
    html += "</div>"
    timeline_placeholder.markdown(html, unsafe_allow_html=True)


def render_diagnostics_chart(history_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history_df["sim_time"], y=history_df["cht"], name="Actual CHT", line=dict(color="#E4E9F2", width=2)))
    fig.add_trace(go.Scatter(x=history_df["sim_time"], y=history_df["expected_cht"], name="Expected CHT", line=dict(color="#5FA8D3", width=2, dash="dash")))
    fig.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#E4E9F2", family="IBM Plex Sans"), legend=dict(orientation="h", y=1.12, x=0), xaxis=dict(gridcolor="#1E2740"), yaxis=dict(gridcolor="#1E2740"))
    diag_chart_placeholder.plotly_chart(fig, use_container_width=True)


def render_log_table(history_rows):
    if not history_rows:
        log_placeholder.markdown("<div class='summary-box'><div class='summary-title'>Mission log</div><div class='summary-main'>No data yet</div></div>", unsafe_allow_html=True)
        return

    recent = history_rows[-20:]
    rows_html = "".join(
        f"<tr><td>{item['sim_time']}</td><td>{item['phase']}</td><td>{item['status']}</td><td>{item['cht']:.1f}</td><td>{item['residual']:+.2f}</td></tr>"
        for item in recent
    )
    log_placeholder.markdown(
        f"""
        <div class='summary-box'>
            <div class='summary-title'>Mission log</div>
            <table class='log-table'>
                <thead><tr><th>Time</th><th>Phase</th><th>Status</th><th>CHT</th><th>Residual</th></tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_chart(history_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history_df["sim_time"], y=history_df["cht"],
        name="Actual CHT", line=dict(color="#E4E9F2", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=history_df["sim_time"], y=history_df["expected_cht"],
        name="Digital twin (expected)", line=dict(color="#5FA8D3", width=2, dash="dash"),
    ))
    fig.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E4E9F2", family="IBM Plex Sans"),
        legend=dict(orientation="h", y=1.15, x=0),
        xaxis=dict(title="Simulated seconds", gridcolor="#1E2740"),
        yaxis=dict(title="CHT (°C)", gridcolor="#1E2740"),
    )
    return fig


# ----------------------------------------------------------------------
# RUN MISSION
# ----------------------------------------------------------------------

if run_clicked:
    if fixed_seed:
        random.seed(42)

    engine_state = EngineState(rpm=2000, cht=120, oil_temp=70)
    twin = DigitalTwin()
    scheduler = build_fault_schedule() if inject_faults else FaultScheduler()
    logger = DataLogger("dashboard_mission_log.csv")

    history = []
    log_rows = []
    sim_time = 0

    for phase, target_rpm, load in MISSION_PHASES:
        for second_in_phase in range(SECONDS_PER_PHASE):
            sim_time += 1

            cooling_efficiency = scheduler.get_value("cooling_efficiency", sim_time, 1.0)
            oil_pressure_factor = scheduler.get_value("oil_pressure_factor", sim_time, 1.0)
            vibration_offset = scheduler.get_value("vibration_offset", sim_time, 0.0)

            readings = tick(
                engine_state, target_rpm, load,
                cooling_efficiency, oil_pressure_factor, vibration_offset,
            )

            expected_cht = twin.update(readings["rpm"], load)
            residual = readings["cht"] - expected_cht
            status = detect_anomaly(engine_state, residual)
            ehi = engine_health_index(residual)
            fault_label = scheduler.active_fault_label_string(sim_time)

            history.append({"sim_time": sim_time, "cht": readings["cht"], "expected_cht": expected_cht})
            log_rows.append({"sim_time": sim_time, "phase": phase, "status": status, "cht": readings["cht"], "residual": residual})

            logger.log(
                sim_time_s=sim_time, mission_phase=phase,
                rpm=round(readings["rpm"], 1), load=load,
                cooling_efficiency=round(cooling_efficiency, 3),
                cht=round(readings["cht"], 2), expected_cht=round(expected_cht, 2),
                residual=round(residual, 2), engine_health_index=round(ehi, 1),
                health_status=status, oil_temperature=round(readings["oil_temp"], 2),
                oil_pressure=round(readings["oil_pressure"], 2),
                fuel_flow=round(readings["fuel_flow"], 2),
                vibration=round(readings["vibration"], 3),
                fault_type=fault_label,
            )

            with overview_tab:
                render_phase_banner(phase, second_in_phase + 1, status, mission_profile)
                render_gauge_cards(ehi, cooling_efficiency, readings["vibration"])
                render_model_summary(selected_model_name, selected_model)
                render_summary_box(status, ehi, residual, cooling_efficiency)
                render_alert_feed(status, phase, residual, fault_label)
                with chart_placeholder.container():
                    st.plotly_chart(build_chart(pd.DataFrame(history)), use_container_width=True, key=f"chart_{sim_time}")
                render_instrument_grid(readings, cooling_efficiency, residual, ehi)
                render_ehi(ehi, status)
                render_ticker(fault_label, residual, show_ground_truth)

            with diagnostics_tab:
                render_fault_timeline(fault_label, sim_time)
                render_diagnostics_chart(pd.DataFrame(history))

            with logs_tab:
                render_log_table(log_rows)

            if delay > 0:
                time.sleep(delay)

    logger.close()
    st.success(f"Mission complete — {TOTAL_TICKS} seconds logged to dashboard_mission_log.csv")
