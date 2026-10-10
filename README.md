# 🔒 Privara – Privacy-Preserving Federated Learning Simulator

> **TECHNOVA 2026 Edition**
> An end-to-end federated learning simulation platform with Differential Privacy (DP-SGD), Top-K Gradient Compression, FedAvg Aggregation, and Membership Inference Attack (MIA) evaluation.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Installation](#installation)
- [Running the App](#running-the-app)
- [Running Tests](#running-tests)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Experiments](#experiments)
- [Dashboard Tabs](#dashboard-tabs)

---

## 🔍 Overview

**Privara** simulates a federated learning system where multiple clients collaboratively train a shared model — *without sharing raw data* — while applying:

- **Differential Privacy (DP-SGD)** via Opacus — calibrated Gaussian noise injection with RDP/PRV accountant
- **Top-K Gradient Compression** — sparse gradient uploads with reconstruction error tracking
- **Communication Budget Enforcement** — per-round MB budget with reject/warn policies
- **Client Dropout and Straggler Simulation** — realistic fault injection with minimum-participant safeguard
- **FedAvg Aggregation** — weighted averaging by number of local samples
- **Membership Inference Attack (MIA) Evaluation** — empirical privacy leakage measurement

---

## 🏗️ Architecture & Working Procedure

```
Privara
├── FederatedSimulator           <- Master Lifecycle Orchestration
│   ├── DataPartitioner          <- Non-IID Dirichlet / IID Partitioning
│   ├── FederatedClient x N      <- Local DP-SGD Training + Top-K Compression
│   └── FederatedServer          <- Broadcast, FedAvg, Global Evaluation
├── PrivacyAccountantWrapper     <- Opacus RDP / PRV Accounting (ε, δ)
├── CommunicationBudgetManager   <- Upload/Download MB Budget Enforcement
├── CommunicationMetricsTracker  <- Per-Round Bandwidth Analytics
├── DropoutSimulator             <- Client Dropout Fault Injection
├── StragglerSimulator           <- Delayed Client Latency Simulation
├── MembershipInferenceEvaluator <- Black-Box Shadow MIA Attack Audit
└── ExperimentRunner             <- Reproducible Benchmark Suites (A/B/C/D)
```

### 📋 8-Stage Federated Execution Pipeline

1. **Dataset Partitioning & Non-IID Distribution** (`DataPartitioner`): Divides data across $N$ client subsets using Dirichlet skew $\text{Dir}(\alpha=0.5)$ with zero leakage to the test distribution.
2. **Global Model Broadcast & Budgeting** (`FederatedServer`, `CommunicationBudgetManager`): Serializes $\theta_t$ and records download bandwidth against the per-round budget limit.
3. **Client Sampling & Fault Injection** (`DropoutSimulator`, `StragglerSimulator`): Selects active clients, simulates dropouts ($p_{\text{drop}}$), and latency stragglers ($p_{\text{strag}}$).
4. **Local Private Training via DP-SGD** (`FederatedClient`, `train_local_model`, Opacus): Computes per-sample gradients, clips to $C$, injects calibrated Gaussian noise $\mathcal{N}(0, \sigma^2 C^2 \mathbf{I})$.
5. **Update Delta & Top-K Gradient Compression** (`compute_model_delta`, `TopKCompressor`): Computes $\Delta_k = \theta_{\text{local}} - \theta_{\text{global}}$ and extracts top $k=50\%$ magnitude coordinates.
6. **Binary Serialization & Budget Enforcement** (`PayloadSerializer`, `CommunicationBudgetManager`): Compresses payload into binary bytes; rejects or warns if budget is exceeded.
7. **Robust Federated Server Aggregation** (`FederatedServer`, `fedavg_aggregate`): Decompresses updates and applies sample-weighted FedAvg: $\theta_{t+1} = \theta_t + \sum \frac{n_k}{N} \Delta_k$.
8. **Differential Privacy Accounting & MIA Audit** (`PrivacyAccountantWrapper`, `MembershipInferenceEvaluator`): Accumulates cumulative $(\varepsilon, \delta)$ and audits empirical vulnerability via shadow loss AUC-ROC.

---

## ✨ Features

| Feature | Details |
|---|---|
| Datasets | Synthetic (built-in), MNIST, FashionMNIST |
| Models | MLP, DP-compatible CNN |
| Partitioning | IID, Dirichlet Non-IID, Label Skew |
| Privacy | DP-SGD (Opacus), RDP and PRV accountants |
| Compression | Top-K sparse gradients (configurable ratio) |
| Reliability | Dropout + straggler simulation |
| Communication | Per-round MB budget (bidirectional / upload) |
| MIA | Black-box shadow loss threshold attack |
| Dashboard | Streamlit multi-tab live UI |
| Experiments | 4 structured reproducible experiment suites |

---

## 🚀 Installation

### Prerequisites
- Python 3.10 or higher
- pip

### Steps

`ash
# 1. Enter the project directory
cd gughug

# 2. (Recommended) Create a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt
`

> Windows note: If you have a GPU, install the CUDA version of PyTorch from https://pytorch.org before running pip install.

---

## ▶️ Running the App

`ash
# Option 1: Use the launch script (Windows)
run.bat

# Option 2: Direct Streamlit command
streamlit run app.py
`

The dashboard opens at http://localhost:8501

---

## 🧪 Running Tests

`ash
# Run full test suite (20 tests)
python -m pytest tests/ -v

# Run a specific module
python -m pytest tests/test_federated.py -v
`

Expected: 20 passed in ~45s

---

## 📁 Project Structure

`
gughug/
├── app.py                      <- Streamlit dashboard entry point
├── config/
│   └── default.yaml            <- Default simulation configuration
├── requirements.txt
├── pytest.ini
├── conftest.py
├── run.bat                     <- Windows launch script
├── src/
│   ├── communication/          <- Budget, metrics, serializer
│   ├── compression/            <- Top-K sparse gradient compression
│   ├── data/                   <- Dataset loaders + partitioners
│   ├── evaluation/             <- Experiments A/B/C/D, MIA, metrics
│   ├── federated/              <- Client, Server, Simulator, FedAvg
│   ├── models/                 <- MLP and CNN classifier
│   ├── privacy/                <- Opacus DP-SGD + accountant
│   ├── reliability/            <- Dropout + straggler simulators
│   ├── utils/                  <- Logging, reproducibility
│   └── visualization/          <- Plotly dark-theme charts
├── tests/                      <- 8 test files, 20 tests total
└── results/                    <- Auto-generated experiment outputs
`

---

## ⚙️ Configuration

All parameters are configurable via config/default.yaml or the sidebar sliders in the dashboard:

| Section | Parameter | Description |
|---|---|---|
| partitioning | strategy | iid, dirichlet, label_skew |
| partitioning | dirichlet_alpha | Non-IID skew (lower = more skewed) |
| privacy | noise_multiplier | Gaussian noise scale sigma |
| privacy | max_grad_norm | L2 gradient clip threshold C |
| privacy | target_epsilon | Privacy budget upper bound epsilon |
| compression | topk_ratio | Fraction of gradients to transmit |
| communication | budget_per_round_mb | Max MB per round |
| reliability | dropout_prob | Client failure probability |
| reliability | straggler_prob | Client delay probability |

---

## 🔬 Experiments

Four structured comparative experiments are available in the Experiments tab:

| Experiment | Description |
|---|---|
| A – Privacy-Utility Trade-off | 5 configs from Strict DP (sigma=2.0) to Non-Private baseline |
| B – Compression Comparison | Uncompressed vs Top-K 50% vs Top-K 20% |
| C – IID vs Non-IID | IID, Dirichlet alpha=0.5, Dirichlet alpha=0.1 convergence |
| D – Client Reliability | 0%, 15%, 30% dropout/straggler fault rates |

Results are auto-saved to ./results/ as both JSON and CSV files.

---

## 📊 Dashboard Tabs

| Tab | Content |
|---|---|
| Overview | Key metrics, config summary, client status grid, epsilon gauge |
| Training Monitor | Live accuracy/loss curves, bandwidth charts, participation heatmap |
| Experiments | Run and visualize structured A/B/C/D experiments |
| MIA Evaluation | Membership Inference Attack AUC comparison (DP vs Non-Private) |
| System Log | Real-time event stream from the training pipeline |

---

Built with PyTorch, Opacus, Streamlit, and Plotly.
