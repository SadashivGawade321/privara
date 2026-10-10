# 🔒 Privara – Privacy-Preserving Federated Learning Simulator

> **TECHNOVA 2026 Edition**  
> An enterprise-grade decentralized simulation platform integrating **Differential Privacy (DP-SGD)**, **Top-K Gradient Compression**, **Heterogeneous Dirichlet Partitioning**, **Communication Budget Enforcement**, and **Membership Inference Attack (MIA)** empirical defense auditing.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Opacus DP](https://img.shields.io/badge/Opacus-DP--SGD-brightgreen.svg)](https://opacus.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/Tests-20%2F20%20Passing-success.svg)](https://github.com/SadashivGawade321/privara)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

---

## 📋 Table of Contents
- [🔍 Overview & Motivation](#-overview--motivation)
- [🏗️ System Architecture & Data Flow](#️-system-architecture--data-flow)
- [📋 8-Stage Detailed Working Procedure](#-8-stage-detailed-working-procedure)
- [📚 Exhaustive Function & Module Directory](#-exhaustive-function--module-directory)
- [✨ Key Capabilities & Innovation Highlights](#-key-capabilities--innovation-highlights)
- [📊 Interactive Dashboard Tabs](#-interactive-dashboard-tabs)
- [🔬 Comparative Experiments Suite (A/B/C/D)](#-comparative-experiments-suite-abcd)
- [⚙️ Configuration Reference (`default.yaml`)](#️-configuration-reference-defaultyaml)
- [🚀 Installation & Quickstart](#-installation--quickstart)
- [🧪 Testing & Verification](#-testing--verification)
- [📁 Repository File Structure](#-repository-file-structure)
- [👥 Authors & Credits](#-authors--credits)

---

## 🔍 Overview & Motivation

Modern machine learning across sensitive domains (healthcare records, banking transactions, decentralized edge sensors) faces a critical tension known as the **Federated Learning Trilemma**:

1. **Privacy Risk**: Transmitting raw model parameters or gradients leaks sensitive training records via reconstruction or Membership Inference Attacks (MIA).
2. **Network Bottlenecks**: Modern neural networks contain millions of parameters, choking edge uplink bandwidth.
3. **Data Heterogeneity (Non-IID)**: Real-world client data distributions are naturally skewed, destabilizing naive convergence.

**Privara** addresses this trilemma through an integrated, privacy-first federated simulation stack:
- **Formal Differential Privacy (DP-SGD)**: Per-sample gradient clipping with L2 threshold $C$ and calibrated Gaussian noise injection $\sigma$ tracked via Rényi Differential Privacy (RDP) accounting.
- **Top-K Gradient Sparsification**: Coordinate-level gradient compression retaining only the top $k\%$ magnitude elements, slashing communication payload sizes by $50\%\text{--}80\%$.
- **Dirichlet Non-IID Partitioning**: Parametric class heterogeneity $\text{Dir}(\alpha)$ creating real-world distributed distributions.
- **Network Fault Resilience**: Real-time modeling of stochastic client dropouts and latency stragglers.
- **Empirical MIA Shadow Audit**: Black-box shadow loss decision threshold evaluations to mathematically prove privacy defense effectiveness.

---

## 🏗️ System Architecture & Data Flow

```
                                      [Central Server]
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       │ Broadcast Global Model Weights θ_t         │
                       ▼                                           ▼
             [Federated Client 1]                        [Federated Client 2]
         (Isolated Local Dataset D_1)                (Isolated Local Dataset D_2)
                       │                                           │
         [Opacus DP-SGD Local Training]              [Opacus DP-SGD Local Training]
         • Compute per-sample gradients g_i          • Compute per-sample gradients g_i
         • Clip norm: ||g_i||_2 <= C                 • Clip norm: ||g_i||_2 <= C
         • Add Gaussian noise: N(0, σ² C² I)         • Add Gaussian noise: N(0, σ² C² I)
                       │                                           │
         [Calculate Weight Delta Δ_1]                [Calculate Weight Delta Δ_2]
         • Δ_1 = θ_local - θ_global                  • Δ_2 = θ_local - θ_global
                       │                                           │
         [Top-K Gradient Compressor]                 [Top-K Gradient Compressor]
         • Retain top k% magnitude coordinates       • Retain top k% magnitude coordinates
                       │                                           │
         [Binary Serialization & Upload]             [Binary Serialization & Upload]
                       │                                           │
                       └─────────────────────┬─────────────────────┘
                                             │
                              [Communication Budget Manager]
                              • Validate upload bytes vs budget
                              • Enforce policy: reject / warn
                                             │
                                             ▼
                               [FedAvg Server Aggregation]
                               • Decompress Top-K sparse updates
                               • θ_{t+1} = θ_t + Σ (n_k / N) Δ_k
                                             │
                                             ▼
                               [Global Test Set Evaluation]
                               • Test Accuracy & Cross-Entropy Loss
                                             │
                                             ▼
                               [Cumulative Privacy Accounting]
                               • Opacus RDP Composition: (ε, δ)
                                             │
                                             ▼
                               [Shadow MIA Attack Audit]
                               • Attack AUC-ROC Mitigation Score
```

---

## 📋 8-Stage Detailed Working Procedure

Privara executes an 8-stage lifecycle in every communication round:

### Stage 1: Dataset Partitioning & Non-IID Distribution
- **Role**: Disperses training data across $N$ client devices.
- **Mechanism**: `DataPartitioner` samples from a Dirichlet distribution $\mathbf{p}_k \sim \text{Dir}(\alpha=0.5)$ across class labels. Lower $\alpha$ values produce severe label skew. Raw training data never leaves its client device; test data is strictly isolated.

### Stage 2: Global Model Initialization & Broadcast
- **Role**: Distributes current shared model parameters $\theta_t$ to participants.
- **Mechanism**: The server serializes CNN or MLP parameters. `CommunicationBudgetManager` registers download byte consumption against the per-round budget limit ($20.0\text{ MB}$).

### Stage 3: Client Sampling & Fault Injection
- **Role**: Replicates real-world unreliable mobile edge networks.
- **Mechanism**: `DropoutSimulator` applies Bernoulli trials with probability $p_{\text{drop}}=0.1$; disconnected clients are omitted. `StragglerSimulator` identifies delayed clients ($p_{\text{strag}}=0.1$) to model heterogeneous compute latency.

### Stage 4: Local Private Training via DP-SGD
- **Role**: Client local parameter updates with formal differential privacy guarantees.
- **Mechanism**: `train_local_model` utilizes Opacus autograd hooks to compute per-sample gradients $g_i$. Gradients are clipped to threshold $C$:
  $$\bar{g}_i = \frac{g_i}{\max\left(1, \frac{\|g_i\|_2}{C}\right)}$$
  Calibrated Gaussian noise is added before the optimizer step:
  $$\tilde{g} = \frac{1}{B} \left( \sum_{i=1}^B \bar{g}_i + \mathcal{N}\left(0, \sigma^2 C^2 \mathbf{I}\right) \right)$$

### Stage 5: Gradient Update Delta & Top-K Compression
- **Role**: Extracts and compresses parameter update vectors.
- **Mechanism**: The client computes the parameter delta: $\Delta_k = \theta_{\text{local}} - \theta_{\text{global}}$. `TopKCompressor` flattens the delta tensors, selects coordinates matching the top $k\%$ absolute magnitude, and creates a sparse representation (values and index bitmasks), dropping $(100-k)\%$ of values.

### Stage 6: Binary Serialization & Bandwidth Budgeting
- **Role**: Measures and regulates network transmission overhead.
- **Mechanism**: `PayloadSerializer` packs compressed tensors and training metadata into a compact binary buffer. `CommunicationBudgetManager` validates the upload against the round budget; payloads violating budget limits are rejected or flagged.

### Stage 7: Robust Federated Server Aggregation (FedAvg)
- **Role**: Consolidates decentralized updates into an updated global model.
- **Mechanism**: The server decompresses sparse coordinate dictionaries into full tensor deltas and applies sample-weighted Federated Averaging:
  $$\theta_{t+1} = \theta_t + \sum_{k \in \mathcal{S}} \frac{n_k}{N_{\text{total}}} \Delta_k$$
  The aggregated model is evaluated against the held-out test dataset to record test accuracy and cross-entropy loss.

### Stage 8: Privacy Accounting & MIA Defense Audit
- **Role**: Formally tracks privacy expenditure and evaluates leakage resistance.
- **Mechanism**: `PrivacyAccountantWrapper` executes Rényi Differential Privacy (RDP) composition across all completed steps to calculate cumulative achieved $(\varepsilon, \delta)$ bounds. `MembershipInferenceEvaluator` performs a black-box shadow loss attack, sweeping decision thresholds to verify that DP-SGD reduces attacker AUC-ROC towards random chance ($0.5$).

---

## 📚 Exhaustive Function & Module Directory

| Module | Function / Class | Inputs | Outputs | Description |
|---|---|---|---|---|
| `src.data.dataset_loader` | `SyntheticDataset` | `num_samples`, `num_classes`, `seed` | PyTorch `Dataset` | Generates $1\times 28\times 28$ geometric pattern tensors with Gaussian noise for sub-second offline testing. |
| `src.data.dataset_loader` | `load_dataset` | `dataset_name`, `data_dir`, sample limits | `(train_ds, test_ds)` | Loads MNIST, FashionMNIST, or Synthetic datasets with sample caps. |
| `src.data.partitioning` | `DataPartitioner` | `dataset`, `num_clients`, `strategy`, `alpha` | `client_subsets`, `client_stats` | Partitions datasets into client subsets via Dirichlet or uniform IID distributions. |
| `src.models.classifier` | `SimpleCNN` / `SimpleMLP` | `in_channels`, `num_classes` | `nn.Module` | Opacus-compatible neural architectures supporting per-sample gradient hooks. |
| `src.privacy.dp_training` | `train_local_model` | `model`, `dataset`, `batch_size`, `epochs`, $\sigma$, $C$ | `(trained_model, metrics)` | Executes local DP-SGD training with per-sample clipping and noise injection. |
| `src.privacy.accountant` | `PrivacyAccountantWrapper` | `accountant_type`, $\delta$, target $\varepsilon$ | Accountant Instance | Wraps Opacus RDP and PRV accountants to track cumulative privacy composition across rounds. |
| `src.privacy.accountant` | `get_epsilon` | Optional $\delta$ | `float` (Achieved $\varepsilon$) | Computes exact cumulative differential privacy expenditure bound $\varepsilon$. |
| `src.compression.topk` | `TopKCompressor.compress` | Delta state dict | `(compressed_dict, stats)` | Filters tensors to retain only top $k\%$ magnitude coordinates; generates sparse bitmasks. |
| `src.compression.topk` | `TopKCompressor.decompress`| `compressed_dict`, `device` | Dense delta state dict | Reconstructs full dense parameter update tensors from sparse representations. |
| `src.communication.serializer` | `PayloadSerializer` | State dict, metadata | `(bytes_buffer, num_bytes)` | Packs model updates and metadata into compact binary byte buffers. |
| `src.communication.budget` | `CommunicationBudgetManager`| `budget_mb`, `enforce_policy` | Budget Manager | Tracks cumulative upload/download bandwidth; enforces `reject` or `warn` policies. |
| `src.communication.metrics` | `CommunicationMetricsTracker`| Round metrics | Metrics history | Records total bytes transmitted, compression ratios, and bandwidth savings. |
| `src.reliability.dropout` | `DropoutSimulator` | `dropout_prob`, `seed` | `(active, dropped)` | Conducts Bernoulli trials to model client dropouts. |
| `src.reliability.stragglers` | `StragglerSimulator` | `straggler_prob`, `delay_sec` | `(on_time, delayed)` | Identifies stragglers and simulates network latency delays. |
| `src.federated.client` | `FederatedClient.train_round`| `global_model`, round hyperparams | Client Update Package | Coordinates local training, delta calculation, and Top-K compression. |
| `src.federated.server` | `FederatedServer.aggregate_round`| `client_updates` | Aggregation summary | Decompresses updates and computes sample-weighted FedAvg. |
| `src.federated.aggregation` | `fedavg_aggregate` | `global_model`, updates, compressor | `(updated_model, info)` | Core FedAvg aggregation logic: $\theta_{t+1} = \theta_t + \sum \frac{n_k}{N} \Delta_k$. |
| `src.federated.simulator` | `FederatedSimulator.run_simulation`| `num_rounds`, callback | Final summary dict | Coordinates multi-round federated training with live progress updates. |
| `src.evaluation.mia` | `MembershipInferenceEvaluator` | `model`, `member_ds`, `nonmember_ds` | `{attack_auc, losses}` | Evaluates prediction loss distributions to compute black-box shadow MIA Attack AUC-ROC. |
| `src.evaluation.experiments` | `ExperimentRunner` | `base_config`, `results_dir`, `device` | Runner Instance | Executes reproducible automated benchmark suites (A, B, C, D). |

---

## ✨ Key Capabilities & Innovation Highlights

- **⚡ Hackathon Turbo Mode (< 2s Execution)**: Optimized sample sizes (350 train / 100 test) and multi-threaded CPU tensor operations provide instant dashboard updates.
- **Hardware Auto-Detection**: Dynamic device detection (`cuda` if available, multi-core optimized `cpu` fallback).
- **Opacus Integration**: Certified DP-SGD implementation using per-sample gradient hooks and Rényi DP composition.
- **Real Byte-Level Measurement**: Measures serialized binary byte payloads directly, reflecting realistic network communication costs.
- **Responsive Cyber Dashboard**: Terminal-themed design with adaptive CSS supporting mobile, tablet, and desktop viewports.

---

## 📊 Interactive Dashboard Tabs

1. **🛡️ Architecture & Procedure**: Cyber terminal boot console, 8-stage visual workflow, and function directory.
2. **🏠 Overview & Metrics**: Live metric tiles (Accuracy, Loss, Achieved $\varepsilon$, Bandwidth Saved), active configuration summary, client status grid, and privacy gauge.
3. **📈 Training Monitor**: Live convergence curves, communication bytes per round, client participation heatmap, and reconstruction error tracking.
4. **🧪 Experiments**: Automated benchmarking suites with Plotly visualizations and CSV exports.
5. **🔍 MIA Evaluation**: Empirical privacy auditing comparing non-private vs. DP-protected models with shadow loss distributions and AUC-ROC curves.
6. **📋 System Log**: Real-time event log with severity filtering (`info`, `warning`, `error`, `success`).

---

## 🔬 Comparative Experiments Suite (A/B/C/D)

| Suite | Focus | Configurations | Metrics |
|---|---|---|---|
| **Experiment A** | Privacy-Utility Trade-off | 5 regimes: Strict ($\sigma=2.0$), High ($\sigma=1.2$), Moderate ($\sigma=0.8$), Relaxed ($\sigma=0.5$), Non-Private ($\sigma=0.0$) | Test Accuracy vs. Achieved $\varepsilon$ |
| **Experiment B** | Gradient Compression | Dense (Uncompressed), Top-K 50%, Top-K 20% | Total MB Transferred, Sparsity Savings, Reconstruction Error |
| **Experiment C** | Data Heterogeneity | Uniform IID, Dirichlet $\alpha=0.5$, Dirichlet $\alpha=0.1$ | Convergence Trajectory, Final Accuracy |
| **Experiment D** | Client Reliability | 0%, 15%, 30% Dropout & Straggler Rates | Dropout Resilience, Delayed Client Counts |

*All results export automatically to `./results/*.csv` and `./results/*.json`.*

---

## ⚙️ Configuration Reference (`default.yaml`)

```yaml
dataset:
  name: "Synthetic"          # "Synthetic", "MNIST", "FashionMNIST"
  num_classes: 10

partitioning:
  num_clients: 5
  strategy: "dirichlet"       # "dirichlet", "iid", "label_skew"
  dirichlet_alpha: 0.5        # Non-IID skew parameter

model:
  architecture: "cnn"         # "cnn" (DP-compatible) or "mlp"
  hidden_dim: 128

federated:
  num_rounds: 3
  local_epochs: 1
  batch_size: 32
  learning_rate: 0.01

privacy:
  enabled: true
  noise_multiplier: 1.0       # Gaussian noise scale sigma
  max_grad_norm: 1.2          # L2 gradient clipping threshold C
  target_delta: 0.00001       # delta: 1e-5
  target_epsilon: 3.0         # Privacy budget limit

compression:
  enabled: true
  method: "topk"
  topk_ratio: 0.5             # Retain top 50% coordinates

communication:
  budget_per_round_mb: 20.0   # Per-round upload/download budget
  enforce_policy: "reject"    # "reject" or "warn"

reliability:
  dropout_prob: 0.1           # Client dropout probability
  straggler_prob: 0.1         # Client latency delay probability
  simulated_delay_sec: 0.0    # Emulated latency (0.0 for Turbo Mode)
```

---

## 🚀 Installation & Quickstart

### Prerequisites
- Python 3.10, 3.11, or 3.12
- `pip` package manager

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/SadashivGawade321/privara.git
cd privara

python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Optional GPU note: For CUDA acceleration on NVIDIA GPUs, install the CUDA PyTorch build from [pytorch.org](https://pytorch.org) before installing requirements).*

### 3. Launch the Dashboard
```bash
# Windows launch script:
run.bat

# Or direct Streamlit command:
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 🧪 Testing & Verification

Privara includes a test suite covering unit, component, and end-to-end integration flows:

```bash
# Run all tests
python -m pytest tests/ -v
```

**Results**:
```text
======================= 20 passed in 47.67s =======================
tests/test_communication.py::test_budget_manager_allow PASSED
tests/test_communication.py::test_budget_manager_reject PASSED
tests/test_communication.py::test_metrics_tracker PASSED
tests/test_communication.py::test_payload_serializer PASSED
tests/test_compression.py::test_topk_compression_ratio PASSED
tests/test_compression.py::test_topk_reconstruction PASSED
tests/test_data.py::test_synthetic_dataset PASSED
tests/test_data.py::test_partitioner_iid PASSED
tests/test_data.py::test_partitioner_dirichlet PASSED
tests/test_federated.py::test_client_train_round PASSED
tests/test_federated.py::test_fedavg_aggregation PASSED
tests/test_federated.py::test_simulator_full_run PASSED
tests/test_mia.py::test_shadow_loss_attack PASSED
tests/test_models.py::test_cnn_forward PASSED
tests/test_models.py::test_mlp_forward PASSED
tests/test_privacy.py::test_dp_training_loop PASSED
tests/test_privacy.py::test_privacy_accountant_rdp PASSED
tests/test_privacy.py::test_privacy_accountant_budget_bound PASSED
tests/test_reliability.py::test_dropout_simulator PASSED
tests/test_reliability.py::test_straggler_simulator PASSED
```

---

## 📁 Repository File Structure

```
privara/
├── app.py                      <- Streamlit Cyber Dashboard & UI Architecture
├── config/
│   └── default.yaml            <- System configuration parameters
├── requirements.txt            <- Production dependencies
├── pytest.ini                  <- Test suite configuration
├── conftest.py                 <- Test fixtures & root path setup
├── run.bat                     <- One-click Windows launch script
├── .gitignore                  <- Git ignore specification
├── README.md                   <- Comprehensive documentation
├── src/
│   ├── communication/          <- Bandwidth budgets, metrics & binary serializers
│   │   ├── budget.py
│   │   ├── metrics.py
│   │   └── serializer.py
│   ├── compression/            <- Top-K gradient sparsification engine
│   │   └── topk.py
│   ├── data/                   <- Synthetic & benchmark loaders, Dirichlet partitioners
│   │   ├── dataset_loader.py
│   │   └── partitioning.py
│   ├── evaluation/             <- MIA shadow loss evaluator & reproducible benchmark runners
│   │   ├── experiments.py
│   │   ├── metrics.py
│   │   └── mia.py
│   ├── federated/              <- Decentralized nodes, coordinator server & FedAvg aggregation
│   │   ├── aggregation.py
│   │   ├── client.py
│   │   ├── server.py
│   │   └── simulator.py
│   ├── models/                 <- Opacus DP-compatible CNN and MLP architectures
│   │   └── classifier.py
│   ├── privacy/                <- DP-SGD per-sample gradient engine & RDP accountants
│   │   ├── accountant.py
│   │   └── dp_training.py
│   ├── reliability/            <- Stochastic client dropouts and latency stragglers
│   │   ├── dropout.py
│   │   └── stragglers.py
│   ├── utils/                  <- Structured event loggers and reproducibility seeds
│   │   ├── logging.py
│   │   └── reproducibility.py
│   └── visualization/          <- Plotly dark-navy cyber charts & theme palettes
│       └── charts.py
├── tests/                      <- 8 test files covering all modules (20 tests)
└── results/                    <- Benchmark output directory (CSV / JSON exports)
```

---

## 👥 Authors & Credits

**Made by:**
- **Nikhil Chopde**
- **Sadashiv Gawade**

---
*Privara – Privacy-Preserving Federated Learning Simulator • TECHNOVA 2026*
