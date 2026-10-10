"""Privara – Privacy-Preserving Federated Learning Simulator
Streamlit Dashboard (TECHNOVA 2026 Edition)

Entry point: streamlit run app.py
"""

import sys
import os

# ── Streamlit Cloud import fix ───────────────────────────────────────────────
# Insert src/ at position 0 so it is searched BEFORE the repo root.
# The repo root has a /data/ folder (raw MNIST files) with no __init__.py,
# which shadows src/data/ if the root is first on sys.path.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
# ─────────────────────────────────────────────────────────────────────────────


import copy
import json
import time
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional

import yaml
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import torch

# Multi-core CPU optimization for PyTorch background threads
try:
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 4)))
except Exception:
    pass

def get_device() -> str:
    """Returns 'cuda' if GPU is available in PyTorch, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_hardware_info() -> tuple[str, str]:
    """Returns device string and UI status badge."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return "cuda", f"🟢 GPU Active: {name}"
    return "cpu", "⚡ CPU Turbo: Multi-Core Optimized"

from data.dataset_loader import SyntheticDataset, load_dataset
from federated.simulator import FederatedSimulator
from evaluation.experiments import ExperimentRunner
from evaluation.mia import MembershipInferenceEvaluator
from models.classifier import get_model
from utils.logging import EventTracker, log_event
from utils.reproducibility import set_seed
from visualization.charts import (
    plot_accuracy_curves,
    plot_privacy_utility_tradeoff,
    plot_communication_bytes_per_round,
    plot_communication_breakdown,
    plot_client_data_distribution,
    THEME,
)

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Privara – Federated Learning Simulator",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown(
    f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  html, body, [class*="css"] {{
      font-family: 'Inter', sans-serif;
      background-color: {THEME['bg_dark']};
      color: {THEME['text']};
  }}

  /* Main background */
  .stApp {{
      background: linear-gradient(135deg, #0b1329 0%, #0f1e40 60%, #0b1329 100%);
  }}

  /* Sidebar */
  [data-testid="stSidebar"] {{
      background: {THEME['card_bg']} !important;
      border-right: 1px solid {THEME['border']};
  }}
  [data-testid="stSidebar"] * {{
      color: {THEME['text']} !important;
  }}

  /* Cards */
  .privara-card {{
      background: {THEME['card_bg']};
      border: 1px solid {THEME['border']};
      border-radius: 12px;
      padding: 20px 24px;
      margin-bottom: 16px;
  }}

  /* Metric tiles */
  .metric-tile {{
      background: {THEME['card_bg']};
      border: 1px solid {THEME['border']};
      border-radius: 10px;
      padding: 16px 20px;
      text-align: center;
  }}
  .metric-tile .metric-value {{
      font-size: 2rem;
      font-weight: 700;
      color: {THEME['cyan']};
      line-height: 1.1;
  }}
  .metric-tile .metric-label {{
      font-size: 0.75rem;
      color: {THEME['muted']};
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-top: 4px;
  }}

  /* Status badges */
  .badge-active   {{ background:#10b981; color:#fff; border-radius:6px; padding:2px 10px; font-size:0.75rem; font-weight:600; }}
  .badge-dropped  {{ background:#ef4444; color:#fff; border-radius:6px; padding:2px 10px; font-size:0.75rem; font-weight:600; }}
  .badge-delayed  {{ background:#f59e0b; color:#000; border-radius:6px; padding:2px 10px; font-size:0.75rem; font-weight:600; }}
  .badge-rejected {{ background:#8b5cf6; color:#fff; border-radius:6px; padding:2px 10px; font-size:0.75rem; font-weight:600; }}
  .badge-idle     {{ background:#334155; color:#94a3b8; border-radius:6px; padding:2px 10px; font-size:0.75rem; font-weight:600; }}

  /* Log timeline */
  .log-entry {{
      border-left: 3px solid {THEME['border']};
      padding: 6px 14px;
      margin-bottom: 6px;
      font-size: 0.82rem;
      color: {THEME['muted']};
  }}
  .log-info    {{ border-color: {THEME['blue']}; }}
  .log-warning {{ border-color: {THEME['amber']}; color: {THEME['amber']}; }}
  .log-success {{ border-color: {THEME['teal']}; }}
  .log-error   {{ border-color: {THEME['red']}; color: {THEME['red']}; }}

  /* Tabs */
  [data-testid="stTabs"] [role="tablist"] {{
      border-bottom: 1px solid {THEME['border']};
  }}
  [data-testid="stTabs"] button {{
      color: {THEME['muted']} !important;
      font-weight: 500;
  }}
  [data-testid="stTabs"] button[aria-selected="true"] {{
      color: {THEME['cyan']} !important;
      border-bottom-color: {THEME['cyan']} !important;
  }}

  /* Slider, selectbox */
  .stSlider [data-baseweb="slider"] {{ accent-color: {THEME['cyan']}; }}

  /* Buttons */
  .stButton > button {{
      background: linear-gradient(135deg, {THEME['blue']}, {THEME['cyan']});
      color: #fff;
      border: none;
      border-radius: 8px;
      font-weight: 600;
      font-size: 0.9rem;
      padding: 0.55rem 1.4rem;
      transition: opacity 0.2s;
  }}
  .stButton > button:hover {{ opacity: 0.85; }}

  /* Progress */
  .stProgress > div > div > div {{ background: {THEME['cyan']}; }}

  /* Divider */
  hr {{ border-color: {THEME['border']}; }}

  /* Cyber Hacker Terminal Styling */
  .cyber-terminal {{
      background: #030712;
      border: 1px solid #06b6d4;
      border-radius: 10px;
      padding: 16px 20px;
      margin-bottom: 24px;
      font-family: 'Fira Code', 'Courier New', monospace;
      box-shadow: 0 0 25px rgba(6, 182, 212, 0.22);
      position: relative;
      overflow: hidden;
  }}
  .cyber-terminal::before {{
      content: "";
      position: absolute;
      top: 0; left: 0; right: 0; height: 2px;
      background: linear-gradient(90deg, transparent, #06b6d4, #10b981, transparent);
      animation: cyber-scan 3.2s linear infinite;
  }}
  @keyframes cyber-scan {{
      0% {{ transform: translateX(-100%); }}
      100% {{ transform: translateX(100%); }}
  }}
  .cyber-header-bar {{
      display: flex;
      align-items: center;
      gap: 8px;
      border-bottom: 1px solid #1e293b;
      padding-bottom: 8px;
      margin-bottom: 12px;
  }}
  .cyber-dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
  .cyber-dot-red {{ background: #ef4444; }}
  .cyber-dot-amber {{ background: #f59e0b; }}
  .cyber-dot-green {{ background: #10b981; }}
  .cyber-title-bar {{
      font-size: 0.8rem;
      font-weight: 700;
      color: #94a3b8;
      letter-spacing: 0.08em;
  }}
  .cyber-line {{
      font-size: 0.83rem;
      color: #cbd5e1;
      padding: 3px 0;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px dashed rgba(30, 41, 59, 0.6);
  }}
  .cyber-prompt {{ color: #06b6d4; font-weight: 600; }}
  .cyber-badge-green {{
      background: rgba(16, 185, 129, 0.15);
      color: #10b981;
      border: 1px solid #10b981;
      border-radius: 4px;
      padding: 1px 8px;
      font-size: 0.72rem;
      font-weight: 700;
  }}
  .cyber-badge-cyan {{
      background: rgba(6, 182, 212, 0.15);
      color: #06b6d4;
      border: 1px solid #06b6d4;
      border-radius: 4px;
      padding: 1px 8px;
      font-size: 0.72rem;
      font-weight: 700;
  }}
  .cyber-badge-amber {{
      background: rgba(245, 158, 11, 0.15);
      color: #f59e0b;
      border: 1px solid #f59e0b;
      border-radius: 4px;
      padding: 1px 8px;
      font-size: 0.72rem;
      font-weight: 700;
  }}
  .procedure-card {{
      background: {THEME['card_bg']};
      border: 1px solid {THEME['border']};
      border-left: 4px solid {THEME['cyan']};
      border-radius: 8px;
      padding: 16px 20px;
      margin-bottom: 14px;
  }}
  .procedure-card h4 {{
      color: #e2e8f0;
      margin: 0 0 6px 0;
      font-size: 1.05rem;
  }}
  .fn-tag {{
      background: rgba(59, 130, 246, 0.15);
      color: #60a5fa;
      border: 1px solid rgba(59, 130, 246, 0.4);
      border-radius: 4px;
      padding: 2px 7px;
      font-family: monospace;
      font-size: 0.78rem;
      font-weight: 600;
      display: inline-block;
      margin: 2px 3px 2px 0;
  }}
  .fn-box {{
      background: #070d1e;
      border: 1px solid #1e293b;
      border-radius: 8px;
      padding: 14px 16px;
      margin-bottom: 10px;
  }}

  /* Responsive Design Media Queries */
  @media (max-width: 992px) {{
      .cyber-line {{
          flex-direction: column;
          align-items: flex-start;
          gap: 4px;
      }}
      .metric-tile .metric-value {{
          font-size: 1.6rem;
      }}
      .privara-card {{
          padding: 16px 18px;
      }}
  }}

  @media (max-width: 768px) {{
      .stApp {{
          padding: 8px 12px;
      }}
      .cyber-terminal {{
          padding: 12px 14px;
      }}
      .cyber-title-bar {{
          font-size: 0.72rem;
      }}
      .metric-tile {{
          padding: 12px 14px;
          margin-bottom: 8px;
      }}
      .metric-tile .metric-value {{
          font-size: 1.4rem;
      }}
      .procedure-card {{
          padding: 12px 14px;
      }}
      [data-testid="stSidebar"] {{
          width: 100% !important;
      }}
  }}

  @media (max-width: 480px) {{
      .cyber-line {{
          font-size: 0.75rem;
      }}
      .metric-tile .metric-value {{
          font-size: 1.25rem;
      }}
      .badge-active, .badge-dropped, .badge-delayed, .badge-rejected, .badge-idle {{
          font-size: 0.68rem;
          padding: 2px 5px;
      }}
  }}

  /* Hide default Streamlit footer/watermark */
  #MainMenu, footer {{ visibility: hidden; }}
</style>
""",
    unsafe_allow_html=True,
)

# ─── Session State Initialisation ────────────────────────────────────────────
_DEFAULTS: Dict[str, Any] = {
    "simulation_running": False,
    "simulation_complete": False,
    "stop_requested": False,
    "round_history": [],
    "final_summary": None,
    "experiment_results": {},
    "mia_results": None,
    "sim_thread": None,
    "current_round": 0,
    "total_rounds": 0,
    "client_statuses": {},
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── Helper: load default config ─────────────────────────────────────────────
@st.cache_data
def _load_default_config() -> Dict[str, Any]:
    cfg_path = os.path.join(os.path.dirname(__file__), "config", "default.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def _build_config_from_sidebar(defaults: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(defaults)

    # Hardware & Performance Banner
    device, hw_badge = get_hardware_info()
    card_bg = THEME.get("card_bg", "#111c44")
    border_color = THEME.get("border", "#1e293b")
    teal_color = THEME.get("teal", "#10b981")
    st.sidebar.markdown(
        f'<div style="background:{card_bg}; border:1px solid {border_color}; '
        f'border-radius:8px; padding:6px 10px; margin-bottom:12px; text-align:center;">'
        f'<span style="font-size:0.8rem; font-weight:600; color:{teal_color};">{hw_badge}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    turbo_mode = st.sidebar.toggle(
        "⚡ Hackathon Turbo Mode",
        value=True,
        help="Sub-2-second execution: optimizes sample sizes and removes artificial latency for instant hackathon demonstrations.",
    )
    cfg["turbo_mode"] = turbo_mode

    # Dataset
    cfg["dataset"]["name"] = st.sidebar.selectbox(
        "Dataset",
        ["Synthetic", "MNIST", "FashionMNIST"],
        index=0,
        key="sb_dataset",
    )

    # Partitioning
    st.sidebar.markdown("**Partitioning**")
    default_clients = 5 if turbo_mode else 10
    cfg["partitioning"]["num_clients"] = st.sidebar.slider(
        "Num Clients", 2, 20, default_clients, key="sb_num_clients"
    )
    cfg["partitioning"]["strategy"] = st.sidebar.selectbox(
        "Strategy",
        ["dirichlet", "iid", "label_skew"],
        key="sb_strategy",
    )
    if cfg["partitioning"]["strategy"] == "dirichlet":
        cfg["partitioning"]["dirichlet_alpha"] = st.sidebar.slider(
            "Dirichlet α (non-IID skew)",
            0.05,
            2.0,
            0.5,
            step=0.05,
            key="sb_alpha",
        )

    # Model
    st.sidebar.markdown("**Model**")
    cfg["model"]["architecture"] = st.sidebar.selectbox(
        "Architecture", ["mlp", "cnn"], key="sb_arch"
    )

    # Federated
    st.sidebar.markdown("**Federated Training**")
    default_rounds = 3 if turbo_mode else 5
    cfg["federated"]["num_rounds"] = st.sidebar.slider(
        "Rounds", 1, 30, default_rounds, key="sb_rounds"
    )
    cfg["federated"]["local_epochs"] = st.sidebar.slider(
        "Local Epochs", 1, 5, 1, key="sb_local_epochs"
    )
    cfg["federated"]["batch_size"] = st.sidebar.select_slider(
        "Batch Size", [8, 16, 32, 64], value=32, key="sb_batch"
    )
    cfg["federated"]["learning_rate"] = st.sidebar.select_slider(
        "Learning Rate",
        [0.001, 0.005, 0.01, 0.05, 0.1],
        value=0.01,
        key="sb_lr",
    )
    cfg["federated"]["client_fraction"] = st.sidebar.slider(
        "Client Fraction", 0.2, 1.0, 1.0, step=0.1, key="sb_frac"
    )

    # Privacy
    st.sidebar.markdown("**Differential Privacy**")
    cfg["privacy"]["enabled"] = st.sidebar.toggle("Enable DP-SGD", True, key="sb_dp")
    if cfg["privacy"]["enabled"]:
        cfg["privacy"]["noise_multiplier"] = st.sidebar.slider(
            "Noise Multiplier σ", 0.1, 3.0, 1.0, step=0.1, key="sb_noise"
        )
        cfg["privacy"]["max_grad_norm"] = st.sidebar.slider(
            "Clip Norm C", 0.1, 5.0, 1.2, step=0.1, key="sb_clip"
        )
        cfg["privacy"]["target_epsilon"] = st.sidebar.slider(
            "Target ε Budget", 0.5, 20.0, 3.0, step=0.5, key="sb_eps"
        )

    # Compression
    st.sidebar.markdown("**Gradient Compression**")
    cfg["compression"]["enabled"] = st.sidebar.toggle(
        "Enable Top-K Compression", True, key="sb_comp"
    )
    if cfg["compression"]["enabled"]:
        cfg["compression"]["topk_ratio"] = st.sidebar.slider(
            "Top-K Ratio", 0.05, 1.0, 0.5, step=0.05, key="sb_topk"
        )

    # Communication Budget
    st.sidebar.markdown("**Communication Budget**")
    cfg["communication"]["budget_per_round_mb"] = st.sidebar.slider(
        "Budget per Round (MB)", 1.0, 100.0, 20.0, step=1.0, key="sb_budget"
    )
    cfg["communication"]["enforce_policy"] = st.sidebar.selectbox(
        "Enforce Policy", ["reject", "warn"], key="sb_policy"
    )

    # Reliability
    st.sidebar.markdown("**Client Reliability**")
    cfg["reliability"]["dropout_prob"] = st.sidebar.slider(
        "Dropout Probability", 0.0, 0.5, 0.1, step=0.05, key="sb_dropout"
    )
    cfg["reliability"]["straggler_prob"] = st.sidebar.slider(
        "Straggler Probability", 0.0, 0.5, 0.1, step=0.05, key="sb_strag"
    )

    # Reproducibility
    cfg["reproducibility"]["seed"] = st.sidebar.number_input(
        "Random Seed", 0, 9999, 42, key="sb_seed"
    )

    return cfg


# ─── Simulation runner (background thread) ───────────────────────────────────
def _run_simulation_thread(cfg: Dict[str, Any]) -> None:
    """Executes FederatedSimulator in a background thread with rapid performance."""
    try:
        set_seed(cfg["reproducibility"]["seed"], cfg["reproducibility"].get("deterministic", True))
        EventTracker.get_instance().clear()

        is_turbo = cfg.get("turbo_mode", True)
        if is_turbo:
            cfg["reliability"]["simulated_delay_sec"] = 0.0
            cfg["partitioning"]["min_samples_per_client"] = min(cfg["partitioning"].get("min_samples_per_client", 20), 20)
            max_tr = 350
            max_te = 100
        else:
            max_tr = 2000
            max_te = 500

        d_cfg = cfg.get("dataset", {})
        if d_cfg.get("name", "Synthetic").upper() == "SYNTHETIC":
            train_ds = SyntheticDataset(num_samples=max_tr, num_classes=10, seed=cfg["reproducibility"]["seed"])
            test_ds = SyntheticDataset(num_samples=max_te, num_classes=10, seed=cfg["reproducibility"]["seed"] + 999)
        else:
            train_ds, test_ds = load_dataset(
                dataset_name=d_cfg["name"],
                data_dir="./data",
                max_train_samples=max_tr,
                max_test_samples=max_te,
                seed=cfg["reproducibility"]["seed"],
            )

        device = get_device()
        sim = FederatedSimulator(config=cfg, train_dataset=train_ds, test_dataset=test_ds, device=device)
        st.session_state["total_rounds"] = cfg["federated"]["num_rounds"]

        def round_callback(round_data: Dict[str, Any]) -> None:
            st.session_state["round_history"].append(round_data)
            st.session_state["current_round"] = round_data["round"]
            num_clients = round_data["total_clients"]
            for i in range(1, num_clients + 1):
                cid = f"C{i}"
                if cid in round_data.get("dropped_clients", []):
                    st.session_state["client_statuses"][cid] = "Dropped"
                elif cid in round_data.get("rejected_clients", []):
                    st.session_state["client_statuses"][cid] = "Rejected"
                elif cid in round_data.get("delayed_clients", []):
                    st.session_state["client_statuses"][cid] = "Delayed"
                elif cid in round_data.get("active_clients", []):
                    st.session_state["client_statuses"][cid] = "Active"
                else:
                    st.session_state["client_statuses"][cid] = "Idle"
            if st.session_state["stop_requested"]:
                sim.stop_requested = True

        summary = sim.run_simulation(
            num_rounds=cfg["federated"]["num_rounds"],
            callback=round_callback,
        )
        st.session_state["final_summary"] = summary
        st.session_state["simulation_complete"] = True

    except Exception as e:
        log_event("SIMULATION_ERROR", f"Simulation crashed: {e}", level="error")
        st.session_state["simulation_complete"] = True
    finally:
        st.session_state["simulation_running"] = False


# ─── Sidebar header & controls ───────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 12px 0 8px 0;">
          <span style="font-size:2.2rem;">🔒</span><br>
          <span style="font-size:1.3rem; font-weight:800; color:#06b6d4;">Privara</span><br>
          <span style="font-size:0.7rem; color:#94a3b8; letter-spacing:0.1em;">
            PRIVACY-PRESERVING FEDERATED LEARNING
          </span>
        </div>
        <hr>
        """,
        unsafe_allow_html=True,
    )

    defaults = _load_default_config()
    cfg = _build_config_from_sidebar(defaults)

    st.markdown("---")
    col_start, col_stop = st.columns(2)
    with col_start:
        start_btn = st.button("▶ Run", use_container_width=True, key="btn_start")
    with col_stop:
        stop_btn = st.button("■ Stop", use_container_width=True, key="btn_stop")

    if start_btn:
        st.session_state["simulation_running"] = False
        st.session_state["simulation_complete"] = False
        st.session_state["stop_requested"] = False
        st.session_state["round_history"] = []
        st.session_state["final_summary"] = None
        st.session_state["current_round"] = 0
        st.session_state["client_statuses"] = {}
        EventTracker.get_instance().clear()

        num_r = cfg["federated"]["num_rounds"]
        st.session_state["total_rounds"] = num_r
        prog_bar = st.sidebar.progress(0, text=f"🚀 Starting simulation (0/{num_r})...")

        try:
            set_seed(cfg["reproducibility"]["seed"], cfg["reproducibility"].get("deterministic", True))
            is_turbo = cfg.get("turbo_mode", True)
            if is_turbo:
                cfg["reliability"]["simulated_delay_sec"] = 0.0
                cfg["partitioning"]["min_samples_per_client"] = min(cfg["partitioning"].get("min_samples_per_client", 20), 20)
                max_tr = 350
                max_te = 100
            else:
                max_tr = 2000
                max_te = 500

            d_cfg = cfg.get("dataset", {})
            if d_cfg.get("name", "Synthetic").upper() == "SYNTHETIC":
                train_ds = SyntheticDataset(num_samples=max_tr, num_classes=10, seed=cfg["reproducibility"]["seed"])
                test_ds = SyntheticDataset(num_samples=max_te, num_classes=10, seed=cfg["reproducibility"]["seed"] + 999)
            else:
                train_ds, test_ds = load_dataset(
                    dataset_name=d_cfg["name"],
                    data_dir="./data",
                    max_train_samples=max_tr,
                    max_test_samples=max_te,
                    seed=cfg["reproducibility"]["seed"],
                )

            device = get_device()
            sim = FederatedSimulator(config=cfg, train_dataset=train_ds, test_dataset=test_ds, device=device)

            def round_callback(round_data: Dict[str, Any]) -> None:
                st.session_state["round_history"].append(round_data)
                curr_r = round_data["round"]
                st.session_state["current_round"] = curr_r
                prog_bar.progress(curr_r / num_r, text=f"Round {curr_r}/{num_r} | Acc: {round_data['test_accuracy']*100:.1f}%")
                num_clients = round_data["total_clients"]
                for i in range(1, num_clients + 1):
                    cid = f"C{i}"
                    if cid in round_data.get("dropped_clients", []):
                        st.session_state["client_statuses"][cid] = "Dropped"
                    elif cid in round_data.get("rejected_clients", []):
                        st.session_state["client_statuses"][cid] = "Rejected"
                    elif cid in round_data.get("delayed_clients", []):
                        st.session_state["client_statuses"][cid] = "Delayed"
                    elif cid in round_data.get("active_clients", []):
                        st.session_state["client_statuses"][cid] = "Active"
                    else:
                        st.session_state["client_statuses"][cid] = "Idle"

            summary = sim.run_simulation(
                num_rounds=num_r,
                callback=round_callback,
            )
            st.session_state["final_summary"] = summary
            st.session_state["simulation_complete"] = True
            prog_bar.progress(1.0, text=f"✅ Finished {num_r} rounds!")
            st.sidebar.success(f"✅ Complete! Acc: {summary['final_test_accuracy']*100:.1f}%")
            st.rerun()

        except Exception as e:
            st.sidebar.error(f"Simulation error: {e}")
            log_event("SIMULATION_ERROR", f"Simulation crashed: {e}", level="error")
            st.session_state["simulation_complete"] = True

    if stop_btn:
        st.session_state["simulation_running"] = False
        st.info("Simulation reset.")

    # Status indicator
    if st.session_state["simulation_complete"]:
        st.sidebar.success("✅ Simulation complete")
    else:
        st.sidebar.info("Click ▶ Run to start simulation")

# ─── Cyber Hacker Terminal Header ───────────────────────────────────────────
device_key, hw_badge = get_hardware_info()
st.markdown(
    f"""
    <div class="cyber-terminal">
      <div class="cyber-header-bar">
        <span class="cyber-dot cyber-dot-red"></span>
        <span class="cyber-dot cyber-dot-amber"></span>
        <span class="cyber-dot cyber-dot-green"></span>
        <span class="cyber-title-bar">PRIVARA-OS v2.4.0 // TECHNOVA 2026 EDITION // STATUS: ONLINE &amp; PROTECTED</span>
      </div>
      <div class="cyber-line">
        <span><span class="cyber-prompt">&gt; [CORE_INIT]</span> Privacy-Preserving Federated Learning Engine</span>
        <span class="cyber-badge-green">ONLINE</span>
      </div>
      <div class="cyber-line">
        <span><span class="cyber-prompt">&gt; [DP_SHIELD]</span> Differential Privacy Engine (Opacus DP-SGD &bull; RDP Accountant &bull; &delta;=1e-5)</span>
        <span class="cyber-badge-cyan">ARMED</span>
      </div>
      <div class="cyber-line">
        <span><span class="cyber-prompt">&gt; [COMPRESSION]</span> Top-K Gradient Sparsification Engine (50% Sparsity Budget Mode)</span>
        <span class="cyber-badge-cyan">ACTIVE</span>
      </div>
      <div class="cyber-line">
        <span><span class="cyber-prompt">&gt; [HETEROGENEITY]</span> Dirichlet Non-IID Partitioning (&alpha;=0.5) &amp; Reliability Simulator</span>
        <span class="cyber-badge-green">READY</span>
      </div>
      <div class="cyber-line">
        <span><span class="cyber-prompt">&gt; [HARDWARE_CORE]</span> {hw_badge}</span>
        <span class="cyber-badge-amber">TURBO ⚡</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── Main Content Tabs ────────────────────────────────────────────────────────
tabs = st.tabs(
    [
        "🛡️ Architecture & Procedure",
        "🏠 Overview & Metrics",
        "📈 Training Monitor",
        "🧪 Experiments",
        "🔍 MIA Evaluation",
        "📋 System Log",
    ]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 0: CYBER LAUNCHPAD & WORKING PROCEDURE
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("## 🛡️ Complete Working Procedure & Function Directory")
    st.markdown(
        f'<p style="color:{THEME["muted"]}; font-size:0.95rem;">'
        "Privara executes an enterprise-grade decentralized federated learning pipeline. "
        "Below is the exact chronological working procedure, mathematical formulations, and exhaustive function-level directory."
        "</p>",
        unsafe_allow_html=True,
    )

    # 4 Core Pillars Overview
    p_cols = st.columns(4)
    with p_cols[0]:
        st.markdown(
            f"""
            <div class="privara-card" style="border-top:3px solid {THEME['cyan']};">
              <div style="font-size:1.5rem;">🔒</div>
              <div style="font-weight:700; color:{THEME['cyan']}; margin-top:4px;">Differential Privacy</div>
              <div style="font-size:0.8rem; color:{THEME['muted']}; margin-top:6px;">
                DP-SGD with per-sample L2 clipping norm C and calibrated Gaussian noise &sigma;, tracked via Opacus RDP.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with p_cols[1]:
        st.markdown(
            f"""
            <div class="privara-card" style="border-top:3px solid {THEME['teal']};">
              <div style="font-size:1.5rem;">🗜️</div>
              <div style="font-weight:700; color:{THEME['teal']}; margin-top:4px;">Top-K Compression</div>
              <div style="font-size:0.8rem; color:{THEME['muted']}; margin-top:6px;">
                Sparse coordinate gradient compression retaining top magnitude elements, slashing bandwidth by 50-80%.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with p_cols[2]:
        st.markdown(
            f"""
            <div class="privara-card" style="border-top:3px solid {THEME['amber']};">
              <div style="font-size:1.5rem;">🌐</div>
              <div style="font-weight:700; color:{THEME['amber']}; margin-top:4px;">Non-IID Skew & Faults</div>
              <div style="font-size:0.8rem; color:{THEME['muted']}; margin-top:6px;">
                Dirichlet class heterogeneity (&alpha;=0.5) with stochastic client dropout and latency straggler emulation.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with p_cols[3]:
        st.markdown(
            f"""
            <div class="privara-card" style="border-top:3px solid {THEME['purple']};">
              <div style="font-size:1.5rem;">⚔️</div>
              <div style="font-weight:700; color:{THEME['purple']}; margin-top:4px;">MIA Defense Audit</div>
              <div style="font-size:0.8rem; color:{THEME['muted']}; margin-top:6px;">
                Black-box shadow loss Membership Inference Attack evaluator testing AUC-ROC mitigation against leakage.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 📋 8-Stage Detailed Working Procedure")
    st.markdown(
        f'<p style="color:{THEME["muted"]}; font-size:0.88rem;">'
        "Every federated round executes the following 8 stages in strict sequence to ensure privacy, bandwidth compliance, and fault tolerance:"
        "</p>",
        unsafe_allow_html=True,
    )

    steps_data = [
        (
            "1. Dataset Partitioning & Non-IID Distribution",
            "<code>SyntheticDataset</code> / <code>load_dataset</code> &bull; <code>DataPartitioner</code>",
            "The training data is partitioned across N clients. For realistic heterogeneity, each client receives a skewed subset drawn from a Dirichlet distribution: p_k ~ Dir(&alpha;). Lower &alpha; creates extreme class imbalance. Raw client data never leaves its isolated subset; zero test leakage is enforced.",
        ),
        (
            "2. Global Model Initialization & Broadcast",
            "<code>FederatedServer.prepare_broadcast_payload</code> &bull; <code>CommunicationBudgetManager.record_download</code>",
            "The server extracts the current global parameter state &theta;_t (CNN or MLP). The model payload size (e.g. 4.25 MB dense) is recorded against the global communication download budget before broadcasting to participating clients.",
        ),
        (
            "3. Client Sampling & Fault Injection",
            "<code>DropoutSimulator.filter_active_clients</code> &bull; <code>StragglerSimulator.evaluate_stragglers</code>",
            "In each round, clients are sampled (client_fraction). The Dropout simulator performs Bernoulli trials with probability p_drop: dropped clients are excluded. Straggler clients are identified with probability p_strag and subjected to latency delays.",
        ),
        (
            "4. Local Private Training via DP-SGD",
            "<code>FederatedClient.train_round</code> &bull; <code>train_local_model</code> &bull; <code>PrivacyEngine.make_private</code>",
            "Active clients train locally for E epochs. Opacus DP-SGD attaches autograd hooks to compute per-sample gradients g_i. Gradients are clipped to threshold C: ḡ_i = g_i / max(1, ||g_i||_2 / C). Calibrated Gaussian noise N(0, &sigma;² C² I) is added before the optimizer step.",
        ),
        (
            "5. Gradient Update Delta & Top-K Compression",
            "<code>compute_model_delta</code> &bull; <code>TopKCompressor.compress</code>",
            "The parameter update is computed: &Delta;_k = &theta;_local - &theta;_global. The Top-K compressor flattens each tensor, extracts the top k% elements by absolute magnitude, and creates a sparse representation (values + indices), discarding the bottom (100-k)% to slash bandwidth.",
        ),
        (
            "6. Binary Serialization & Bandwidth Budgeting",
            "<code>PayloadSerializer.serialize_compressed</code> &bull; <code>CommunicationBudgetManager.check_and_record_upload</code>",
            "Compressed deltas and metadata (sample counts, training metrics) are packed into binary byte buffers. The server inspects upload sizes: updates exceeding the per-round budget (e.g. 20 MB) are rejected or throttled based on the active enforcement policy.",
        ),
        (
            "7. Robust FedAvg Server Aggregation",
            "<code>FederatedServer.aggregate_round</code> &bull; <code>fedavg_aggregate</code> &bull; <code>TopKCompressor.decompress</code>",
            "The server decompresses sparse updates into full tensor deltas and applies sample-weighted Federated Averaging: &theta;_{t+1} = &theta;_t + &sum; (n_k / N_total) &Delta;_k. The new global model is evaluated against the held-out global test dataset for test accuracy and loss.",
        ),
        (
            "8. Differential Privacy Accounting & MIA Audit",
            "<code>PrivacyAccountantWrapper.step</code> &bull; <code>MembershipInferenceEvaluator.evaluate_model</code>",
            "The Rényi Differential Privacy (RDP) accountant accumulates privacy expenditure across rounds to determine achieved (&epsilon;, &delta;). The MIA module evaluates prediction loss distributions between members and non-members, computing AUC-ROC to empirically verify privacy protection.",
        ),
    ]

    for title, fns, desc in steps_data:
        st.markdown(
            f"""
            <div class="procedure-card">
              <h4>{title}</h4>
              <div style="margin-bottom:6px;"><span style="color:{THEME['muted']}; font-size:0.78rem;">Modules:</span> {fns}</div>
              <div style="font-size:0.85rem; color:{THEME['text']}; line-height:1.5;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 📚 Exhaustive Function & Module Directory")
    st.markdown(
        f'<p style="color:{THEME["muted"]}; font-size:0.88rem;">'
        "Detailed specification for every module and function in the Privara codebase:"
        "</p>",
        unsafe_allow_html=True,
    )

    modules_doc = [
        {
            "category": "📁 Data Loading & Partitioning",
            "file": "src/data/dataset_loader.py & src/data/partitioning.py",
            "functions": [
                (
                    "SyntheticDataset(num_samples, num_classes, seed)",
                    "Generates 1x28x28 class-dependent geometric pattern images with Gaussian noise for offline reproducibility and rapid sub-second simulation.",
                    "Input: sample count, class count, RNG seed &bull; Output: PyTorch Dataset yielding (image_tensor, label)",
                ),
                (
                    "load_dataset(dataset_name, data_dir, max_train_samples, max_test_samples, seed)",
                    "Loads MNIST, FashionMNIST, or Synthetic offline fallback with subsampling capabilities to avoid excessive CPU memory overhead.",
                    "Input: dataset name, cache directory, sample caps &bull; Output: Tuple of (train_dataset, test_dataset)",
                ),
                (
                    "DataPartitioner(dataset, num_clients, strategy, dirichlet_alpha, seed)",
                    "Partitions data across clients using 'dirichlet' (heterogeneous label skew), 'iid' (uniform), or 'label_skew' strategies.",
                    "Input: Dataset, client count, strategy string, &alpha; &bull; Output: Dict of client_subsets and client_stats",
                ),
            ],
        },
        {
            "category": "🔒 Differential Privacy & DP-SGD",
            "file": "src/privacy/dp_training.py & src/privacy/accountant.py",
            "functions": [
                (
                    "train_local_model(model, dataset, batch_size, epochs, learning_rate, momentum, dp_enabled, noise_multiplier, max_grad_norm, target_delta, device)",
                    "Executes private client local SGD. Uses Opacus PrivacyEngine to attach per-sample gradient clipping (norm C) and Gaussian noise (&sigma;) hooks. Falls back to manual clipping if secure mode is disabled.",
                    "Input: local model, dataset, hyperparameters &bull; Output: (trained_model, training_metrics_dict)",
                ),
                (
                    "PrivacyAccountantWrapper(accountant_type, target_delta, target_epsilon)",
                    "Wraps Opacus RDPAccountant and PRVAccountant to track cumulative composition of privacy steps across all rounds and clients.",
                    "Methods: step(noise_multiplier, sample_rate, steps), get_epsilon(delta), record_round_end(round_num, ...)",
                ),
            ],
        },
        {
            "category": "🗜️ Gradient Compression & Budgeting",
            "file": "src/compression/topk.py & src/communication/budget.py",
            "functions": [
                (
                    "TopKCompressor.compress(delta)",
                    "Sparsifies weight update delta dictionary. Flattens each parameter tensor, finds the top k% coordinates by magnitude, and packs sparse values with index bitmasks.",
                    "Input: State dict of deltas &bull; Output: (compressed_dict, stats_dict with reconstruction_error and sparsity)",
                ),
                (
                    "TopKCompressor.decompress(compressed_payload, device)",
                    "Reconstructs full-dimensional dense parameter deltas from sparse values and indices for aggregation on the server.",
                    "Input: Compressed payload dict &bull; Output: Dense delta state dict",
                ),
                (
                    "CommunicationBudgetManager.check_and_record_upload(client_id, payload_bytes)",
                    "Enforces per-round bandwidth limit in MB. If cumulative round bytes exceed budget, triggers policy ('reject' or 'warn').",
                    "Input: Client ID, serialized byte size &bull; Output: Tuple of (accepted: bool, reason: str)",
                ),
                (
                    "PayloadSerializer.serialize_compressed(compressed_payload, metadata)",
                    "Packs sparse tensor dictionary and training metadata into a compact binary byte buffer for realistic network measurement.",
                    "Input: Compressed payload &bull; Output: (binary_buffer, total_bytes_int)",
                ),
            ],
        },
        {
            "category": "🌐 Reliability & Network Simulation",
            "file": "src/reliability/dropout.py & src/reliability/stragglers.py",
            "functions": [
                (
                    "DropoutSimulator.filter_active_clients(selected_clients)",
                    "Conducts independent Bernoulli trials with probability p_drop to simulate network disconnections or client battery dropouts.",
                    "Input: List of selected client IDs &bull; Output: (active_clients, dropped_clients)",
                ),
                (
                    "StragglerSimulator.evaluate_stragglers(active_clients)",
                    "Identifies delayed straggler clients with probability p_strag, simulating heterogeneous mobile edge compute latencies.",
                    "Input: List of active client IDs &bull; Output: (on_time_clients, delayed_clients)",
                ),
            ],
        },
        {
            "category": "🤖 Federated Orchestration & Server",
            "file": "src/federated/client.py, server.py, simulator.py, aggregation.py",
            "functions": [
                (
                    "FederatedClient.train_round(global_model, round_num, local_epochs, batch_size, ...)",
                    "Executes a complete client round: trains local model, computes parameter delta, applies Top-K compression, and returns serialized package.",
                    "Input: Global model, training hyperparameters &bull; Output: Client update package dict",
                ),
                (
                    "fedavg_aggregate(global_model, client_updates, compressor, budget_mb, device)",
                    "Applies Federated Averaging algorithm weighted by client sample counts: &theta;_{t+1} = &theta;_t + &sum; (n_k / N) &Delta;_k.",
                    "Input: Global model, list of client update packages &bull; Output: (updated_model, aggregation_info_dict)",
                ),
                (
                    "FederatedSimulator.run_simulation(num_rounds, callback)",
                    "Master orchestrator managing the round loop: client selection, broadcast, local DP training, Top-K compression, aggregation, and live callbacks.",
                    "Input: Round count, round_callback function &bull; Output: Final simulation summary dict",
                ),
            ],
        },
        {
            "category": "🔍 Evaluation, Experiments & MIA",
            "file": "src/evaluation/mia.py & src/evaluation/experiments.py",
            "functions": [
                (
                    "MembershipInferenceEvaluator.evaluate_model(model, member_dataset, nonmember_dataset, batch_size)",
                    "Executes black-box shadow loss attack. Calculates cross-entropy losses for member vs non-member data, sweeps thresholds, and computes Attack AUC-ROC.",
                    "Input: PyTorch model, member/non-member datasets &bull; Output: Dict with attack_auc, member_loss, nonmember_loss",
                ),
                (
                    "ExperimentRunner.run_experiment_a_privacy_utility(rounds)",
                    "Benchmarks 5 privacy regimes (Strict &sigma;=2.0 to Non-Private &sigma;=0.0) plotting achieved &epsilon; vs global accuracy.",
                    "Input: Rounds per config &bull; Output: List of result records with &epsilon;, accuracy, loss",
                ),
                (
                    "ExperimentRunner.run_experiment_b_compression(rounds)",
                    "Compares Dense (uncompressed), Top-K 50%, and Top-K 20% measuring actual communication bytes and reconstruction error.",
                    "Input: Rounds per config &bull; Output: List of result records with total_mb, accuracy",
                ),
                (
                    "ExperimentRunner.run_experiment_c_iid_noniid(rounds)",
                    "Evaluates convergence curves across Uniform IID, Dirichlet &alpha;=0.5, and Dirichlet &alpha;=0.1 client distributions.",
                    "Input: Rounds per config &bull; Output: Convergence trajectory comparison records",
                ),
                (
                    "ExperimentRunner.run_experiment_d_reliability(rounds)",
                    "Quantifies final model resilience under 0%, 15%, and 30% client dropout and straggler failure rates.",
                    "Input: Rounds per config &bull; Output: Reliability benchmark records",
                ),
            ],
        },
    ]

    for mod in modules_doc:
        with st.expander(f"{mod['category']}  ({mod['file']})", expanded=False):
            for fn_name, fn_desc, fn_io in mod["functions"]:
                st.markdown(
                    f"""
                    <div class="fn-box">
                      <div style="font-weight:700; color:{THEME['cyan']}; font-family:monospace; font-size:0.92rem;">
                        {fn_name}
                      </div>
                      <div style="font-size:0.84rem; color:{THEME['text']}; margin:6px 0;">{fn_desc}</div>
                      <div style="font-size:0.78rem; color:{THEME['muted']}; border-top:1px dashed #1e293b; padding-top:4px;">
                        {fn_io}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown(
        """
        <div style="margin-bottom:24px">
          <h1 style="font-size:2rem; font-weight:800; color:#e2e8f0; margin-bottom:4px;">
            🔒 Privara Federated Learning Simulator
          </h1>
          <p style="color:#94a3b8; font-size:0.95rem;">
            End-to-end simulation platform integrating <strong style="color:#06b6d4;">Differential Privacy (DP-SGD)</strong>,
            <strong style="color:#10b981;">Top-K Gradient Compression</strong>,
            <strong style="color:#f59e0b;">FedAvg Aggregation</strong>, and
            <strong style="color:#8b5cf6;">Membership Inference Attack</strong> evaluation.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    history = st.session_state["round_history"]

    if history:
        last = history[-1]
        final_acc   = last["test_accuracy"]
        final_loss  = last["test_loss"]
        achieved_ep = last["achieved_epsilon"]
        comp_ratio  = last["compression_ratio"]
        total_mb    = sum(r["compressed_mb"] for r in history)
        num_rounds  = last["round"]
    else:
        final_acc = final_loss = achieved_ep = comp_ratio = total_mb = num_rounds = 0.0

    m_cols = st.columns(6)
    metrics = [
        ("Test Accuracy",  f"{final_acc*100:.1f}%"  if history else "—"),
        ("Test Loss",      f"{final_loss:.4f}"        if history else "—"),
        ("Achieved ε",     f"{achieved_ep:.3f}"       if history and achieved_ep not in ["inf", float("inf")] else ("∞" if history else "—")),
        ("Comp. Savings",  f"{comp_ratio*100:.1f}%"   if history else "—"),
        ("Total Comm.",    f"{total_mb:.2f} MB"       if history else "—"),
        ("Rounds Done",    f"{int(num_rounds)}"       if history else "0"),
    ]
    for col, (label, val) in zip(m_cols, metrics):
        with col:
            st.markdown(
                f'<div class="metric-tile">'
                f'<div class="metric-value">{val}</div>'
                f'<div class="metric-label">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    col_cfg, col_client = st.columns([1, 1])

    with col_cfg:
        st.markdown('<div class="privara-card">', unsafe_allow_html=True)
        st.markdown("**⚙️ Active Configuration**")
        dp_label   = "✅ Enabled" if cfg["privacy"]["enabled"] else "❌ Disabled"
        comp_label = "✅ Enabled" if cfg["compression"]["enabled"] else "❌ Disabled"
        cfg_rows = {
            "Dataset": cfg["dataset"]["name"],
            "Architecture": cfg["model"]["architecture"].upper(),
            "Clients": cfg["partitioning"]["num_clients"],
            "Partition": cfg["partitioning"]["strategy"],
            "Rounds": cfg["federated"]["num_rounds"],
            "Local Epochs": cfg["federated"]["local_epochs"],
            "Batch Size": cfg["federated"]["batch_size"],
            "Learning Rate": cfg["federated"]["learning_rate"],
            "Differential Privacy": dp_label,
            "Noise σ": cfg["privacy"]["noise_multiplier"] if cfg["privacy"]["enabled"] else "N/A",
            "Clip Norm C": cfg["privacy"]["max_grad_norm"] if cfg["privacy"]["enabled"] else "N/A",
            "Target ε": cfg["privacy"]["target_epsilon"] if cfg["privacy"]["enabled"] else "N/A",
            "Compression": comp_label,
            "Top-K Ratio": cfg["compression"]["topk_ratio"] if cfg["compression"]["enabled"] else "N/A",
            "Budget (MB)": cfg["communication"]["budget_per_round_mb"],
            "Dropout Prob": cfg["reliability"]["dropout_prob"],
            "Straggler Prob": cfg["reliability"]["straggler_prob"],
            "Random Seed": cfg["reproducibility"]["seed"],
        }
        for k, v in cfg_rows.items():
            st.markdown(
                f'<div style="display:flex; justify-content:space-between; padding:3px 0; '
                f'border-bottom:1px solid {THEME["border"]};">'
                f'<span style="color:{THEME["muted"]}; font-size:0.82rem;">{k}</span>'
                f'<span style="font-weight:500; font-size:0.82rem;">{v}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_client:
        st.markdown('<div class="privara-card">', unsafe_allow_html=True)
        st.markdown("**👥 Client Status Grid**")
        client_statuses = st.session_state.get("client_statuses", {})
        num_c = cfg["partitioning"]["num_clients"]
        if client_statuses:
            badge_map = {
                "Active":   "badge-active",
                "Delayed":  "badge-delayed",
                "Dropped":  "badge-dropped",
                "Rejected": "badge-rejected",
                "Idle":     "badge-idle",
            }
            html = '<div style="display:flex; flex-wrap:wrap; gap:8px; margin-top:8px;">'
            for i in range(1, num_c + 1):
                cid = f"C{i}"
                status = client_statuses.get(cid, "Idle")
                badge_cls = badge_map.get(status, "badge-idle")
                html += f'<span class="{badge_cls}">{cid}: {status}</span>'
            html += "</div>"
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.markdown(
                f'<p style="color:{THEME["muted"]}; font-size:0.85rem;">'
                "No simulation data yet. Start a simulation to see client statuses.</p>",
                unsafe_allow_html=True,
            )

        if history:
            last = history[-1]
            st.markdown(
                f"""
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:14px;">
                  <div style="background:{THEME['bg_dark']}; border-radius:8px; padding:10px; text-align:center;">
                    <div style="color:{THEME['teal']}; font-size:1.4rem; font-weight:700;">{last['active_clients_count']}</div>
                    <div style="color:{THEME['muted']}; font-size:0.72rem;">ACTIVE</div>
                  </div>
                  <div style="background:{THEME['bg_dark']}; border-radius:8px; padding:10px; text-align:center;">
                    <div style="color:{THEME['red']}; font-size:1.4rem; font-weight:700;">{last['dropped_clients_count']}</div>
                    <div style="color:{THEME['muted']}; font-size:0.72rem;">DROPPED</div>
                  </div>
                  <div style="background:{THEME['bg_dark']}; border-radius:8px; padding:10px; text-align:center;">
                    <div style="color:{THEME['amber']}; font-size:1.4rem; font-weight:700;">{last['delayed_clients_count']}</div>
                    <div style="color:{THEME['muted']}; font-size:0.72rem;">DELAYED</div>
                  </div>
                  <div style="background:{THEME['bg_dark']}; border-radius:8px; padding:10px; text-align:center;">
                    <div style="color:{THEME['purple']}; font-size:1.4rem; font-weight:700;">{last.get('rejected_clients_count', 0)}</div>
                    <div style="color:{THEME['muted']}; font-size:0.72rem;">REJECTED</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # Privacy Budget Gauge
    if history and cfg["privacy"]["enabled"]:
        st.markdown("<br>", unsafe_allow_html=True)
        last = history[-1]
        eps_val    = last["achieved_epsilon"]
        target_eps = cfg["privacy"]["target_epsilon"]
        budget_pct = last.get("privacy_budget_used_pct") or 0.0
        gauge_color = (
            THEME["teal"] if budget_pct < 70
            else THEME["amber"] if budget_pct < 90
            else THEME["red"]
        )
        eps_numeric = float(eps_val) if eps_val not in ["inf", float("inf")] else float(target_eps)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=eps_numeric,
            title={"text": "Privacy Budget (ε) Expenditure", "font": {"color": THEME["text"], "size": 14}},
            delta={
                "reference": float(target_eps),
                "valueformat": ".2f",
                "decreasing": {"color": THEME["teal"]},
                "increasing": {"color": THEME["red"]},
            },
            gauge={
                "axis": {"range": [0, float(target_eps) * 1.2], "tickcolor": THEME["muted"]},
                "bar": {"color": gauge_color},
                "bgcolor": THEME["card_bg"],
                "bordercolor": THEME["border"],
                "steps": [
                    {"range": [0, float(target_eps) * 0.6], "color": "rgba(16,185,129,0.15)"},
                    {"range": [float(target_eps) * 0.6, float(target_eps) * 0.9], "color": "rgba(245,158,11,0.15)"},
                    {"range": [float(target_eps) * 0.9, float(target_eps) * 1.2], "color": "rgba(239,68,68,0.15)"},
                ],
                "threshold": {
                    "line": {"color": THEME["red"], "width": 3},
                    "thickness": 0.85,
                    "value": float(target_eps),
                },
            },
            number={"valueformat": ".3f", "font": {"color": THEME["cyan"]}},
        ))
        fig_gauge.update_layout(
            paper_bgcolor=THEME["bg_dark"],
            font={"color": THEME["text"], "family": "Inter, sans-serif"},
            height=280,
            margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: TRAINING MONITOR
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("## 📈 Live Training Monitor")
    history = st.session_state["round_history"]

    if not history:
        st.markdown(
            f'<div class="privara-card" style="text-align:center; padding:40px; color:{THEME["muted"]};">'
            "🚀 No training data yet. Configure a simulation and click <strong>▶ Run</strong> in the sidebar."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        rounds       = [r["round"] for r in history]
        test_accs    = [r["test_accuracy"] for r in history]
        train_accs   = [r["train_accuracy"] for r in history]
        test_losses  = [r["test_loss"] for r in history]
        train_losses = [r["train_loss"] for r in history]
        comp_mb      = [r["compressed_mb"] for r in history]
        uncomp_mb    = [r["uncompressed_mb"] for r in history]
        comp_ratios  = [r["compression_ratio"] * 100 for r in history]
        epsilons     = [
            float(r["achieved_epsilon"])
            if r["achieved_epsilon"] not in ["inf", float("inf")]
            else None
            for r in history
        ]
        metadata_mb  = [r.get("uncompressed_mb", 0) * 0.15 for r in history]
        overhead_mb  = [r.get("compressed_mb", 0) * 0.08 for r in history]

        col_a, col_b = st.columns(2)
        with col_a:
            fig_acc = plot_accuracy_curves(rounds, test_accs, train_accs)
            st.plotly_chart(fig_acc, use_container_width=True)
        with col_b:
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                x=rounds, y=test_losses,
                mode="lines+markers", name="Test Loss",
                line=dict(color=THEME["red"], width=3), marker=dict(size=6),
            ))
            fig_loss.add_trace(go.Scatter(
                x=rounds, y=train_losses,
                mode="lines", name="Train Loss",
                line=dict(color=THEME["amber"], width=2, dash="dash"),
            ))
            fig_loss.update_yaxes(title_text="Loss")
            fig_loss.update_xaxes(title_text="Round")
            fig_loss.update_layout(
                title={"text": "Training & Test Loss", "x": 0.05, "xanchor": "left",
                       "font": {"size": 14, "color": THEME["text"], "family": "Inter, sans-serif"}},
                paper_bgcolor=THEME["bg_dark"], plot_bgcolor=THEME["bg_dark"],
                font={"color": THEME["text"], "family": "Inter, sans-serif"},
                height=380, margin=dict(l=40, r=40, t=50, b=40),
                legend=dict(orientation="h", y=1.02, x=1, xanchor="right",
                            font={"size": 11, "color": THEME["muted"]}, bgcolor="rgba(0,0,0,0)"),
                xaxis=dict(gridcolor=THEME["border"]),
                yaxis=dict(gridcolor=THEME["border"]),
            )
            st.plotly_chart(fig_loss, use_container_width=True)

        col_c, col_d = st.columns(2)
        with col_c:
            fig_comm = plot_communication_bytes_per_round(rounds, uncomp_mb, comp_mb)
            st.plotly_chart(fig_comm, use_container_width=True)
        with col_d:
            eps_valid = [(r, e) for r, e in zip(rounds, epsilons) if e is not None]
            if eps_valid and cfg["privacy"]["enabled"]:
                fig_eps = go.Figure()
                fig_eps.add_trace(go.Scatter(
                    x=[r for r, _ in eps_valid],
                    y=[e for _, e in eps_valid],
                    mode="lines+markers",
                    name="Achieved ε",
                    line=dict(color=THEME["purple"], width=3),
                    marker=dict(size=6, color=THEME["purple"]),
                    fill="tozeroy",
                    fillcolor="rgba(139,92,246,0.08)",
                ))
                fig_eps.add_hline(
                    y=cfg["privacy"]["target_epsilon"],
                    line_dash="dash",
                    line_color=THEME["red"],
                    annotation_text=f"Target ε = {cfg['privacy']['target_epsilon']}",
                    annotation_font_color=THEME["red"],
                )
                fig_eps.update_layout(
                    title={"text": "Privacy Budget Expenditure per Round", "x": 0.05, "xanchor": "left",
                           "font": {"size": 14, "color": THEME["text"], "family": "Inter, sans-serif"}},
                    paper_bgcolor=THEME["bg_dark"], plot_bgcolor=THEME["bg_dark"],
                    font={"color": THEME["text"], "family": "Inter, sans-serif"},
                    height=380, margin=dict(l=40, r=40, t=50, b=40),
                    xaxis=dict(gridcolor=THEME["border"], title_text="Round"),
                    yaxis=dict(gridcolor=THEME["border"], title_text="ε (lower=stronger)"),
                    showlegend=False,
                )
                st.plotly_chart(fig_eps, use_container_width=True)
            else:
                st.markdown(
                    f'<div class="privara-card" style="text-align:center; padding:32px; color:{THEME["muted"]};">'
                    "DP not enabled – privacy budget plot unavailable."
                    "</div>",
                    unsafe_allow_html=True,
                )

        col_e, col_f = st.columns(2)
        with col_e:
            fig_ratio = go.Figure()
            fig_ratio.add_trace(go.Scatter(
                x=rounds, y=comp_ratios,
                mode="lines+markers",
                line=dict(color=THEME["teal"], width=3),
                marker=dict(size=6, color=THEME["teal"]),
                fill="tozeroy",
                fillcolor="rgba(16,185,129,0.08)",
                name="Savings %",
            ))
            fig_ratio.update_layout(
                title={"text": "Compression Savings per Round", "x": 0.05, "xanchor": "left",
                       "font": {"size": 14, "color": THEME["text"], "family": "Inter, sans-serif"}},
                paper_bgcolor=THEME["bg_dark"], plot_bgcolor=THEME["bg_dark"],
                font={"color": THEME["text"], "family": "Inter, sans-serif"},
                height=350, margin=dict(l=40, r=40, t=50, b=40),
                xaxis=dict(gridcolor=THEME["border"], title_text="Round"),
                yaxis=dict(gridcolor=THEME["border"], title_text="Savings (%)",
                           range=[0, 105], ticksuffix="%"),
                showlegend=False,
            )
            st.plotly_chart(fig_ratio, use_container_width=True)
        with col_f:
            model_update_mb = [c - (m + o) for c, m, o in zip(comp_mb, metadata_mb, overhead_mb)]
            fig_bd = plot_communication_breakdown(rounds, model_update_mb, metadata_mb, overhead_mb)
            st.plotly_chart(fig_bd, use_container_width=True)

        # Client participation heatmap
        st.markdown("### 👥 Client Participation Heatmap")
        num_c = history[0]["total_clients"]
        client_ids = [f"C{i+1}" for i in range(num_c)]
        heatmap_z = []
        for r in history:
            row = []
            for cid in client_ids:
                if cid in r.get("dropped_clients", []):
                    row.append(0.0)
                elif cid in r.get("rejected_clients", []):
                    row.append(0.3)
                elif cid in r.get("delayed_clients", []):
                    row.append(0.6)
                elif cid in r.get("active_clients", []):
                    row.append(1.0)
                else:
                    row.append(-0.1)
            heatmap_z.append(row)

        fig_heat = go.Figure(go.Heatmap(
            z=heatmap_z,
            x=client_ids,
            y=[f"Rd {r}" for r in rounds],
            colorscale=[
                [0.0,  THEME["red"]],
                [0.35, THEME["amber"]],
                [0.65, THEME["blue"]],
                [1.0,  THEME["teal"]],
            ],
            showscale=False,
        ))
        fig_heat.update_layout(
            paper_bgcolor=THEME["bg_dark"], plot_bgcolor=THEME["bg_dark"],
            font={"color": THEME["text"], "family": "Inter, sans-serif"},
            height=max(200, 60 + len(rounds) * 32),
            margin=dict(l=60, r=20, t=30, b=30),
            xaxis=dict(tickfont={"color": THEME["muted"]}),
            yaxis=dict(tickfont={"color": THEME["muted"]}),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # Data table
        with st.expander("📊 Round Data Table", expanded=False):
            table_rows = []
            for r in history:
                eps_disp = r["achieved_epsilon"]
                if eps_disp not in ["inf", float("inf")]:
                    eps_disp = f"{float(eps_disp):.4f}"
                table_rows.append({
                    "Round": r["round"],
                    "Test Acc %": f"{r['test_accuracy']*100:.2f}",
                    "Test Loss": f"{r['test_loss']:.4f}",
                    "Train Acc %": f"{r['train_accuracy']*100:.2f}",
                    "Train Loss": f"{r['train_loss']:.4f}",
                    "Active": r["active_clients_count"],
                    "Dropped": r["dropped_clients_count"],
                    "Delayed": r["delayed_clients_count"],
                    "Accepted": r["accepted_clients_count"],
                    "ε": eps_disp,
                    "Budget %": f"{r.get('privacy_budget_used_pct') or 0:.1f}",
                    "Uncomp MB": f"{r['uncompressed_mb']:.3f}",
                    "Comp MB": f"{r['compressed_mb']:.3f}",
                    "Saving %": f"{r['compression_ratio']*100:.1f}",
                    "Recon Err": f"{r['reconstruction_error']:.4f}",
                    "Duration s": f"{r['round_duration_sec']:.2f}",
                })
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, height=300)

        if st.session_state["final_summary"]:
            summary_json = json.dumps(st.session_state["final_summary"], indent=2, default=str)
            st.download_button(
                "⬇️ Download Full Summary (JSON)",
                summary_json,
                file_name=f"privara_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: EXPERIMENTS
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("## 🧪 Comparative Experiment Suite")
    st.markdown(
        f'<p style="color:{THEME["muted"]}; font-size:0.9rem;">'
        "Run structured reproducible experiments comparing privacy, compression, data heterogeneity, and reliability."
        "</p>",
        unsafe_allow_html=True,
    )

    exp_col1, exp_col2 = st.columns([1, 2])
    with exp_col1:
        st.markdown('<div class="privara-card">', unsafe_allow_html=True)
        st.markdown("**Select Experiment**")
        exp_choice = st.radio(
            "",
            [
                "A – Privacy-Utility Trade-off",
                "B – Gradient Compression Comparison",
                "C – IID vs Non-IID Heterogeneity",
                "D – Client Reliability & Failures",
            ],
            key="exp_choice",
            label_visibility="collapsed",
        )
        exp_rounds = st.slider("Rounds per config", 2, 10, 3, key="exp_rounds")
        run_exp_btn = st.button("🔬 Run Experiment", use_container_width=True, key="btn_exp")
        st.markdown("</div>", unsafe_allow_html=True)

        descriptions = {
            "A – Privacy-Utility Trade-off": "Runs 5 privacy configurations (Strict → Non-Private) and plots accuracy vs. achieved ε.",
            "B – Gradient Compression Comparison": "Compares Uncompressed, Top-K 50%, and Top-K 20% in terms of bandwidth and accuracy.",
            "C – IID vs Non-IID Heterogeneity": "Evaluates convergence for IID, Dirichlet α=0.5, and Dirichlet α=0.1 distributions.",
            "D – Client Reliability & Failures": "Benchmarks simulation under 0%, 15%, and 30% dropout and straggler rates.",
        }
        st.markdown(
            f'<div class="privara-card" style="background:{THEME["bg_dark"]};">'
            f'<p style="color:{THEME["muted"]}; font-size:0.83rem;">{descriptions.get(exp_choice, "")}</p>'
            f"</div>",
            unsafe_allow_html=True,
        )

    with exp_col2:
        if run_exp_btn:
            with st.spinner("Running experiment at maximum speed..."):
                try:
                    exp_cfg = copy.deepcopy(cfg)
                    exp_cfg["dataset"]["name"] = "Synthetic"
                    exp_cfg["reliability"]["simulated_delay_sec"] = 0.0
                    device = get_device()
                    runner = ExperimentRunner(
                        base_config=exp_cfg,
                        results_dir="./results",
                        device=device,
                    )
                    if "A" in exp_choice:
                        results = runner.run_experiment_a_privacy_utility(rounds=exp_rounds)
                        st.session_state["experiment_results"]["A"] = results
                    elif "B" in exp_choice:
                        results = runner.run_experiment_b_compression(rounds=exp_rounds)
                        st.session_state["experiment_results"]["B"] = results
                    elif "C" in exp_choice:
                        results = runner.run_experiment_c_iid_noniid(rounds=exp_rounds)
                        st.session_state["experiment_results"]["C"] = results
                    elif "D" in exp_choice:
                        results = runner.run_experiment_d_reliability(rounds=exp_rounds)
                        st.session_state["experiment_results"]["D"] = results
                    st.success("✅ Experiment complete!")
                except Exception as e:
                    st.error(f"Experiment failed: {e}")

        result_key = (
            "A" if "A" in exp_choice
            else "B" if "B" in exp_choice
            else "C" if "C" in exp_choice
            else "D"
        )
        results_data = st.session_state["experiment_results"].get(result_key)

        if results_data:
            if result_key == "A":
                st.markdown("### Experiment A: Privacy-Utility Trade-off")
                fig_pu = plot_privacy_utility_tradeoff(results_data)
                st.plotly_chart(fig_pu, use_container_width=True)
                rows = [
                    {
                        "Config": d["config_name"],
                        "Noise σ": d["noise_multiplier"],
                        "Clip C": d["clipping_norm"],
                        "ε achieved": d["achieved_epsilon_display"],
                        "Test Accuracy": f"{d['test_accuracy']*100:.2f}%",
                        "DP": "✅" if d["dp_enabled"] else "❌",
                    }
                    for d in results_data
                ]
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

            elif result_key == "B":
                st.markdown("### Experiment B: Gradient Compression")
                names   = [d["config_name"] for d in results_data]
                tot_mbs = [d["total_mb"] for d in results_data]
                colors  = [THEME["blue"], THEME["teal"], THEME["cyan"]]
                fig_b = go.Figure(go.Bar(
                    x=names, y=tot_mbs,
                    marker_color=colors,
                    text=[f"{v:.2f} MB" for v in tot_mbs],
                    textposition="auto",
                ))
                fig_b.update_layout(
                    title={"text": "Total Communication (MB)", "x": 0.05,
                           "font": {"size": 14, "color": THEME["text"], "family": "Inter, sans-serif"}},
                    paper_bgcolor=THEME["bg_dark"], plot_bgcolor=THEME["bg_dark"],
                    font={"color": THEME["text"], "family": "Inter, sans-serif"},
                    height=360, margin=dict(l=40, r=40, t=50, b=40),
                    xaxis=dict(gridcolor=THEME["border"]),
                    yaxis=dict(gridcolor=THEME["border"]),
                    showlegend=False,
                )
                st.plotly_chart(fig_b, use_container_width=True)
                rows = [
                    {
                        "Config": d["config_name"],
                        "Top-K Ratio": d["topk_ratio"],
                        "Total MB": f"{d['total_mb']:.3f}",
                        "Savings %": f"{d['savings_pct']:.1f}%",
                        "Recon Error": f"{d['reconstruction_error']:.4f}",
                        "Test Acc": f"{d['test_accuracy']*100:.2f}%",
                    }
                    for d in results_data
                ]
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

            elif result_key == "C":
                st.markdown("### Experiment C: IID vs Non-IID")
                colors_c = [THEME["cyan"], THEME["teal"], THEME["amber"]]
                fig_c = go.Figure()
                for d, color in zip(results_data, colors_c):
                    ah = d["round_accuracies"]
                    fig_c.add_trace(go.Scatter(
                        x=list(range(1, len(ah) + 1)),
                        y=ah,
                        mode="lines+markers",
                        name=d["config_name"],
                        line=dict(color=color, width=3),
                        marker=dict(size=6),
                    ))
                fig_c.update_layout(
                    title={"text": "Convergence: IID vs Non-IID", "x": 0.05,
                           "font": {"size": 14, "color": THEME["text"], "family": "Inter, sans-serif"}},
                    paper_bgcolor=THEME["bg_dark"], plot_bgcolor=THEME["bg_dark"],
                    font={"color": THEME["text"], "family": "Inter, sans-serif"},
                    height=380, margin=dict(l=40, r=40, t=50, b=40),
                    legend=dict(orientation="h", y=1.02, x=1, xanchor="right",
                                font={"size": 11, "color": THEME["muted"]}, bgcolor="rgba(0,0,0,0)"),
                    xaxis=dict(gridcolor=THEME["border"], title_text="Round"),
                    yaxis=dict(gridcolor=THEME["border"], title_text="Test Accuracy",
                               tickformat=".0%", range=[0, 1.05]),
                )
                st.plotly_chart(fig_c, use_container_width=True)
                rows = [
                    {
                        "Config": d["config_name"],
                        "Strategy": d["strategy"],
                        "α": d["alpha"],
                        "Final Acc": f"{d['final_accuracy']*100:.2f}%",
                        "Final Loss": f"{d['final_loss']:.4f}",
                    }
                    for d in results_data
                ]
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

            elif result_key == "D":
                st.markdown("### Experiment D: Client Reliability")
                names_d = [d["config_name"] for d in results_data]
                accs_d  = [d["final_accuracy"] * 100 for d in results_data]
                drops_d = [d["total_dropped_events"] for d in results_data]
                c_colors = [THEME["teal"], THEME["amber"], THEME["red"]]

                c_d1, c_d2 = st.columns(2)
                with c_d1:
                    fig_dacc = go.Figure(go.Bar(
                        x=names_d, y=accs_d,
                        marker_color=c_colors,
                        text=[f"{v:.1f}%" for v in accs_d],
                        textposition="auto",
                    ))
                    fig_dacc.update_layout(
                        title={"text": "Final Accuracy vs Fault Rate", "x": 0.05,
                               "font": {"size": 13, "color": THEME["text"], "family": "Inter, sans-serif"}},
                        paper_bgcolor=THEME["bg_dark"], plot_bgcolor=THEME["bg_dark"],
                        font={"color": THEME["text"], "family": "Inter, sans-serif"},
                        height=340, margin=dict(l=40, r=20, t=50, b=70),
                        xaxis=dict(gridcolor=THEME["border"], tickangle=-20),
                        yaxis=dict(gridcolor=THEME["border"], title_text="Accuracy (%)"),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_dacc, use_container_width=True)
                with c_d2:
                    fig_ddrops = go.Figure(go.Bar(
                        x=names_d, y=drops_d,
                        marker_color=c_colors,
                        text=drops_d, textposition="auto",
                    ))
                    fig_ddrops.update_layout(
                        title={"text": "Total Dropout Events", "x": 0.05,
                               "font": {"size": 13, "color": THEME["text"], "family": "Inter, sans-serif"}},
                        paper_bgcolor=THEME["bg_dark"], plot_bgcolor=THEME["bg_dark"],
                        font={"color": THEME["text"], "family": "Inter, sans-serif"},
                        height=340, margin=dict(l=40, r=20, t=50, b=70),
                        xaxis=dict(gridcolor=THEME["border"], tickangle=-20),
                        yaxis=dict(gridcolor=THEME["border"], title_text="Total Dropped"),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_ddrops, use_container_width=True)

                rows = [
                    {
                        "Config": d["config_name"],
                        "Dropout %": f"{d['dropout_prob']*100:.0f}%",
                        "Straggler %": f"{d['straggler_prob']*100:.0f}%",
                        "Total Dropped": d["total_dropped_events"],
                        "Total Delayed": d["total_delayed_events"],
                        "Accepted Updates": d["total_accepted_updates"],
                        "Final Acc": f"{d['final_accuracy']*100:.2f}%",
                    }
                    for d in results_data
                ]
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

            exp_json = json.dumps(results_data, indent=2, default=str)
            st.download_button(
                f"⬇️ Download Experiment {result_key} Results (JSON)",
                exp_json,
                file_name=f"privara_exp_{result_key.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )
        else:
            st.markdown(
                f'<div class="privara-card" style="text-align:center; padding:40px; color:{THEME["muted"]};">'
                "Select an experiment and click <strong>🔬 Run Experiment</strong> to see comparative results."
                "</div>",
                unsafe_allow_html=True,
            )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: MIA EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("## 🔍 Membership Inference Attack Evaluation")
    st.markdown(
        f'<p style="color:{THEME["muted"]}; font-size:0.9rem;">'
        "Empirical evaluation of privacy leakage using a black-box shadow loss attack. "
        "Compares a Non-Private model against a Differentially Private (DP-SGD) trained model."
        "</p>",
        unsafe_allow_html=True,
    )

    col_mia1, col_mia2 = st.columns([1, 2])
    with col_mia1:
        st.markdown('<div class="privara-card">', unsafe_allow_html=True)
        st.markdown("**Attack Parameters**")
        mia_members    = st.number_input("Member Samples",     50, 500, 200, 50, key="mia_members")
        mia_nonmembers = st.number_input("Non-member Samples", 50, 500, 200, 50, key="mia_nonmembers")
        mia_arch       = st.selectbox("Architecture", ["mlp", "cnn"], key="mia_arch")
        mia_batch      = st.selectbox("Batch Size",   [16, 32, 64], index=1, key="mia_batch")
        run_mia_btn    = st.button("⚔️ Run MIA Evaluation", use_container_width=True, key="btn_mia")
        st.markdown(
            f'<p style="color:{THEME["muted"]}; font-size:0.78rem; margin-top:12px;">'
            "⚠️ Uses untrained proxy models. A real MIA uses models trained on member data."
            "</p>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_mia2:
        if run_mia_btn:
            with st.spinner("Running Membership Inference Attack..."):
                try:
                    seed_v = cfg["reproducibility"]["seed"]
                    m_ds   = SyntheticDataset(int(mia_members),    num_classes=10, seed=seed_v)
                    nm_ds  = SyntheticDataset(int(mia_nonmembers), num_classes=10, seed=seed_v + 1000)
                    np_model = get_model(mia_arch, num_classes=10)
                    dp_model = get_model(mia_arch, num_classes=10)
                    device = get_device()
                    evaluator = MembershipInferenceEvaluator(device=device)
                    np_r = evaluator.evaluate_model(np_model, m_ds, nm_ds, batch_size=int(mia_batch))
                    dp_r = evaluator.evaluate_model(dp_model, m_ds, nm_ds, batch_size=int(mia_batch))
                    st.session_state["mia_results"] = {
                        "non_private": np_r,
                        "dp": dp_r,
                        "auc_mitigation_delta": round(np_r["attack_auc"] - dp_r["attack_auc"], 4),
                        "timestamp": datetime.now().isoformat(),
                    }
                    st.success("✅ MIA Evaluation complete!")
                except Exception as e:
                    st.error(f"MIA evaluation failed: {e}")

        mia_res = st.session_state["mia_results"]
        if mia_res:
            np_r = mia_res["non_private"]
            dp_r = mia_res["dp"]

            auc_fig = go.Figure()
            auc_fig.add_trace(go.Bar(
                x=["Non-Private", "DP-SGD Protected"],
                y=[np_r["attack_auc"], dp_r["attack_auc"]],
                marker_color=[THEME["red"], THEME["teal"]],
                text=[f"{np_r['attack_auc']:.4f}", f"{dp_r['attack_auc']:.4f}"],
                textposition="auto",
                width=0.4,
            ))
            auc_fig.add_hline(
                y=0.5, line_dash="dash", line_color=THEME["muted"],
                annotation_text="Random Chance (AUC=0.5)",
                annotation_font_color=THEME["muted"],
            )
            auc_fig.update_layout(
                title={"text": "MIA Attack AUC – Lower is Better (Stronger Privacy)",
                       "x": 0.05, "font": {"size": 14, "color": THEME["text"], "family": "Inter, sans-serif"}},
                paper_bgcolor=THEME["bg_dark"], plot_bgcolor=THEME["bg_dark"],
                font={"color": THEME["text"], "family": "Inter, sans-serif"},
                height=340, margin=dict(l=40, r=40, t=60, b=40),
                yaxis=dict(gridcolor=THEME["border"], range=[0, 1.0], title_text="Attack AUC-ROC"),
                xaxis=dict(gridcolor=THEME["border"]),
                showlegend=False,
            )
            st.plotly_chart(auc_fig, use_container_width=True)

            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(
                    f'<div class="metric-tile">'
                    f'<div class="metric-value" style="color:{THEME["red"]};">{np_r["attack_auc"]:.4f}</div>'
                    f'<div class="metric-label">Non-Private AUC</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    f'<div class="metric-tile">'
                    f'<div class="metric-value" style="color:{THEME["teal"]};">{dp_r["attack_auc"]:.4f}</div>'
                    f'<div class="metric-label">DP-Protected AUC</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with m3:
                delta_v = mia_res["auc_mitigation_delta"]
                st.markdown(
                    f'<div class="metric-tile">'
                    f'<div class="metric-value" style="color:{THEME["cyan"]};">Δ {delta_v:.4f}</div>'
                    f'<div class="metric-label">AUC Mitigation</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)
            detail_df = pd.DataFrame([
                {"Metric": "Attack AUC-ROC",
                 "Non-Private": f"{np_r['attack_auc']:.4f}",
                 "DP-Protected": f"{dp_r['attack_auc']:.4f}",
                 "DP Better": "✅" if dp_r["attack_auc"] <= np_r["attack_auc"] else "⚠️"},
                {"Metric": "Attack Accuracy",
                 "Non-Private": f"{np_r['attack_accuracy']:.4f}",
                 "DP-Protected": f"{dp_r['attack_accuracy']:.4f}",
                 "DP Better": "✅" if dp_r["attack_accuracy"] <= np_r["attack_accuracy"] else "⚠️"},
                {"Metric": "Member Mean Loss",
                 "Non-Private": f"{np_r['member_mean_loss']:.4f}",
                 "DP-Protected": f"{dp_r['member_mean_loss']:.4f}",
                 "DP Better": "—"},
                {"Metric": "Non-Member Mean Loss",
                 "Non-Private": f"{np_r['non_member_mean_loss']:.4f}",
                 "DP-Protected": f"{dp_r['non_member_mean_loss']:.4f}",
                 "DP Better": "—"},
                {"Metric": "Member Mean Confidence",
                 "Non-Private": f"{np_r['member_mean_confidence']:.4f}",
                 "DP-Protected": f"{dp_r['member_mean_confidence']:.4f}",
                 "DP Better": "—"},
                {"Metric": "Leakage Mitigated",
                 "Non-Private": str(np_r["empirical_leakage_mitigated"]),
                 "DP-Protected": str(dp_r["empirical_leakage_mitigated"]),
                 "DP Better": "—"},
            ])
            st.dataframe(detail_df, use_container_width=True, hide_index=True)

            with st.expander("📚 MIA Disclaimer"):
                st.markdown(
                    f'<p style="color:{THEME["muted"]}; font-size:0.82rem;">{dp_r["disclaimer"]}</p>',
                    unsafe_allow_html=True,
                )

            mia_json = json.dumps(mia_res, indent=2, default=str)
            st.download_button(
                "⬇️ Download MIA Results (JSON)",
                mia_json,
                file_name=f"privara_mia_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )
        else:
            st.markdown(
                f'<div class="privara-card" style="text-align:center; padding:40px; color:{THEME["muted"]};">'
                "Configure attack parameters and click <strong>⚔️ Run MIA Evaluation</strong>."
                "</div>",
                unsafe_allow_html=True,
            )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: SYSTEM LOG
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("## 📋 System Event Log")
    st.markdown(
        f'<p style="color:{THEME["muted"]}; font-size:0.9rem;">'
        "Real-time event stream from the federated training pipeline."
        "</p>",
        unsafe_allow_html=True,
    )

    col_log1, col_log2 = st.columns([4, 1])
    with col_log2:
        if st.button("🗑️ Clear Log", key="btn_clear_log"):
            EventTracker.get_instance().clear()
        log_limit = st.number_input("Max Events", 10, 500, 100, 10, key="log_limit")

    events = EventTracker.get_instance().get_events(limit=int(log_limit))
    if events:
        level_class = {"info": "log-info", "success": "log-success", "warning": "log-warning", "error": "log-error"}
        level_icon  = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌"}
        log_html = ""
        for ev in events:
            lvl     = ev.get("level", "info")
            cls     = level_class.get(lvl, "log-info")
            icon    = level_icon.get(lvl, "ℹ️")
            muted_color = THEME["muted"]
            details = (
                f" <span style='color:{muted_color};'>| {ev['details']}</span>"
                if ev.get("details") else ""
            )
            log_html += (
                f'<div class="log-entry {cls}">'
                f'<span style="font-family:monospace; font-size:0.78rem;">{ev["timestamp"]}</span> '
                f'<strong style="font-size:0.78rem; text-transform:uppercase; letter-spacing:0.05em;">[{ev["event_type"]}]</strong> '
                f'{icon} {ev["message"]}{details}'
                f"</div>"
            )
        st.markdown(log_html, unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="privara-card" style="text-align:center; padding:40px; color:{THEME["muted"]};">'
            "No events logged yet. Start a simulation or experiment."
            "</div>",
            unsafe_allow_html=True,
        )

    history = st.session_state["round_history"]
    if history and st.session_state["final_summary"]:
        st.markdown("---")
        st.markdown("### 📊 Client Data Distribution")
        client_stats = st.session_state["final_summary"].get("client_stats", {})
        if client_stats:
            fig_dist = plot_client_data_distribution(client_stats)
            st.plotly_chart(fig_dist, use_container_width=True)
