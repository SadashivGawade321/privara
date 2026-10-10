"""Reproducible federated experiments runner for Privara.

Implements Experiments A, B, C, D and Membership Inference evaluations:
- Experiment A: Privacy-Utility Trade-off across >= 4 privacy configurations
- Experiment B: Compression Comparison (Uncompressed, Top-K 50%, Top-K 20%)
- Experiment C: IID vs Non-IID Dirichlet partitioning
- Experiment D: Client Reliability (varying dropout and straggler rates)
"""

import os
import json
import copy
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime

from data.dataset_loader import load_dataset
from federated.simulator import FederatedSimulator
from evaluation.mia import MembershipInferenceEvaluator
from utils.logging import log_event, logger


class ExperimentRunner:
    """Orchestrates structured comparative experiments."""

    def __init__(
        self,
        base_config: Dict[str, Any],
        results_dir: str = "./results",
        device: str = "cpu",
        train_dataset: Optional[Any] = None,
        test_dataset: Optional[Any] = None,
    ):
        self.base_config = copy.deepcopy(base_config)
        self.results_dir = results_dir
        self.device = device
        os.makedirs(results_dir, exist_ok=True)

        # Pre-load shared datasets for consistent evaluations (optimized for rapid execution)
        if train_dataset is not None and test_dataset is not None:
            self.train_dataset = train_dataset
            self.test_dataset = test_dataset
        else:
            d_cfg = self.base_config.get("dataset", {})
            d_name = d_cfg.get("name", "Synthetic")
            max_tr = d_cfg.get("max_train_samples", 400)
            max_te = d_cfg.get("max_test_samples", 100)
            self.train_dataset, self.test_dataset = load_dataset(
                dataset_name=d_name,
                data_dir=d_cfg.get("data_dir", "./data"),
                max_train_samples=max_tr,
                max_test_samples=max_te,
                seed=self.base_config.get("reproducibility", {}).get("seed", 42)
            )

    def run_experiment_a_privacy_utility(
        self,
        rounds: int = 5,
        save_results: bool = True
    ) -> List[Dict[str, Any]]:
        """Experiment A: Evaluate >= 4 privacy configurations + non-private baseline."""
        log_event("EXP_A_START", f"Starting Experiment A: Privacy-Utility Trade-off ({rounds} rounds per config).")
        
        configs = [
            {"id": "strict", "name": "Strict Privacy", "noise": 2.0, "clip": 1.0, "dp": True},
            {"id": "high", "name": "High Privacy", "noise": 1.2, "clip": 1.2, "dp": True},
            {"id": "moderate", "name": "Moderate Privacy", "noise": 0.8, "clip": 1.5, "dp": True},
            {"id": "relaxed", "name": "Relaxed Privacy", "noise": 0.5, "clip": 2.0, "dp": True},
            {"id": "non_private", "name": "Non-Private Baseline", "noise": 0.0, "clip": 0.0, "dp": False},
        ]

        results = []
        for c in configs:
            run_cfg = copy.deepcopy(self.base_config)
            run_cfg["federated"]["num_rounds"] = rounds
            run_cfg["privacy"]["enabled"] = c["dp"]
            run_cfg["privacy"]["noise_multiplier"] = c["noise"]
            run_cfg["privacy"]["max_grad_norm"] = c["clip"]
            run_cfg["reliability"]["simulated_delay_sec"] = 0.0

            sim = FederatedSimulator(
                config=run_cfg,
                train_dataset=self.train_dataset,
                test_dataset=self.test_dataset,
                device=self.device
            )
            summary = sim.run_simulation(num_rounds=rounds)

            achieved_eps = summary["privacy_summary"]["achieved_epsilon"]
            item = {
                "config_id": c["id"],
                "config_name": c["name"],
                "noise_multiplier": c["noise"],
                "clipping_norm": c["clip"],
                "achieved_epsilon": round(achieved_eps, 4) if achieved_eps != float("inf") else 999.0,
                "achieved_epsilon_display": f"{achieved_eps:.2f}" if achieved_eps != float("inf") else "∞ (Non-Private)",
                "delta": run_cfg["privacy"]["target_delta"],
                "test_accuracy": summary["final_test_accuracy"],
                "test_loss": summary["final_test_loss"],
                "dp_enabled": c["dp"]
            }
            results.append(item)
            log_event("EXP_A_STEP", f"Finished {c['name']}: Acc={item['test_accuracy']*100:.2f}%, ε={item['achieved_epsilon_display']}")

        if save_results:
            self._save_json("experiment_a_privacy_utility.json", results)
            pd.DataFrame(results).to_csv(os.path.join(self.results_dir, "experiment_a_privacy_utility.csv"), index=False)

        return results

    def run_experiment_b_compression(
        self,
        rounds: int = 5,
        save_results: bool = True
    ) -> List[Dict[str, Any]]:
        """Experiment B: Compare Uncompressed, Top-K 50%, Top-K 20%."""
        log_event("EXP_B_START", f"Starting Experiment B: Gradient Compression Comparison ({rounds} rounds).")

        configs = [
            {"id": "uncompressed", "name": "Uncompressed (Dense)", "enabled": False, "ratio": 1.0},
            {"id": "topk_50", "name": "Top-K 50% Sparsity", "enabled": True, "ratio": 0.5},
            {"id": "topk_20", "name": "Top-K 20% Sparsity", "enabled": True, "ratio": 0.2},
        ]

        results = []
        for c in configs:
            run_cfg = copy.deepcopy(self.base_config)
            run_cfg["federated"]["num_rounds"] = rounds
            run_cfg["compression"]["enabled"] = c["enabled"]
            run_cfg["compression"]["topk_ratio"] = c["ratio"]
            run_cfg["reliability"]["simulated_delay_sec"] = 0.0

            sim = FederatedSimulator(
                config=run_cfg,
                train_dataset=self.train_dataset,
                test_dataset=self.test_dataset,
                device=self.device
            )
            summary = sim.run_simulation(num_rounds=rounds)

            comm = summary["communication_summary"]
            # Compute average reconstruction error from rounds
            recons = [r.get("reconstruction_error", 0.0) for r in summary["round_history"]]
            avg_recon = sum(recons) / max(1, len(recons))

            item = {
                "config_id": c["id"],
                "config_name": c["name"],
                "compression_enabled": c["enabled"],
                "topk_ratio": c["ratio"],
                "total_mb": comm["total_compressed_mb"] if c["enabled"] else comm["total_uncompressed_mb"],
                "avg_round_mb": comm["avg_compressed_mb"] if c["enabled"] else comm["avg_uncompressed_mb"],
                "compression_ratio": comm["overall_compression_ratio"] if c["enabled"] else 0.0,
                "savings_pct": comm["overall_savings_pct"] if c["enabled"] else 0.0,
                "reconstruction_error": round(avg_recon, 4),
                "test_accuracy": summary["final_test_accuracy"],
                "total_duration_sec": summary["total_duration_sec"]
            }
            results.append(item)
            log_event("EXP_B_STEP", f"Finished {c['name']}: Total={item['total_mb']:.2f} MB, Acc={item['test_accuracy']*100:.2f}%")

        if save_results:
            self._save_json("experiment_b_compression.json", results)
            pd.DataFrame(results).to_csv(os.path.join(self.results_dir, "experiment_b_compression.csv"), index=False)

        return results

    def run_experiment_c_iid_noniid(
        self,
        rounds: int = 5,
        save_results: bool = True
    ) -> List[Dict[str, Any]]:
        """Experiment C: Compare IID, Dirichlet α=0.5, Dirichlet α=0.1."""
        log_event("EXP_C_START", f"Starting Experiment C: IID vs Non-IID Heterogeneity ({rounds} rounds).")

        configs = [
            {"id": "iid", "name": "IID (Uniform)", "strategy": "iid", "alpha": None},
            {"id": "noniid_0.5", "name": "Non-IID (Dirichlet α=0.5)", "strategy": "dirichlet", "alpha": 0.5},
            {"id": "noniid_0.1", "name": "Non-IID (Dirichlet α=0.1)", "strategy": "dirichlet", "alpha": 0.1},
        ]

        results = []
        for c in configs:
            run_cfg = copy.deepcopy(self.base_config)
            run_cfg["federated"]["num_rounds"] = rounds
            run_cfg["partitioning"]["strategy"] = c["strategy"]
            run_cfg["reliability"]["simulated_delay_sec"] = 0.0
            if c["alpha"] is not None:
                run_cfg["partitioning"]["dirichlet_alpha"] = c["alpha"]

            sim = FederatedSimulator(
                config=run_cfg,
                train_dataset=self.train_dataset,
                test_dataset=self.test_dataset,
                device=self.device
            )
            summary = sim.run_simulation(num_rounds=rounds)

            acc_history = [r["test_accuracy"] for r in summary["round_history"]]
            item = {
                "config_id": c["id"],
                "config_name": c["name"],
                "strategy": c["strategy"],
                "alpha": c["alpha"] if c["alpha"] else "Uniform",
                "final_accuracy": summary["final_test_accuracy"],
                "final_loss": summary["final_test_loss"],
                "round_accuracies": acc_history
            }
            results.append(item)
            log_event("EXP_C_STEP", f"Finished {c['name']}: Acc={item['final_accuracy']*100:.2f}%")

        if save_results:
            self._save_json("experiment_c_iid_noniid.json", results)
            pd.DataFrame(results).to_csv(os.path.join(self.results_dir, "experiment_c_iid_noniid.csv"), index=False)

        return results

    def run_experiment_d_reliability(
        self,
        rounds: int = 5,
        save_results: bool = True
    ) -> List[Dict[str, Any]]:
        """Experiment D: Evaluate dropout & straggler rates (0%, 15%, 30%)."""
        log_event("EXP_D_START", f"Starting Experiment D: Client Reliability & Failures ({rounds} rounds).")

        configs = [
            {"id": "reliable", "name": "Ideal (0% Drop, 0% Strag)", "p_drop": 0.0, "p_strag": 0.0},
            {"id": "moderate_faults", "name": "Moderate (15% Drop, 15% Strag)", "p_drop": 0.15, "p_strag": 0.15},
            {"id": "severe_faults", "name": "Severe (30% Drop, 30% Strag)", "p_drop": 0.30, "p_strag": 0.30},
        ]

        results = []
        for c in configs:
            run_cfg = copy.deepcopy(self.base_config)
            run_cfg["federated"]["num_rounds"] = rounds
            run_cfg["reliability"]["dropout_prob"] = c["p_drop"]
            run_cfg["reliability"]["straggler_prob"] = c["p_strag"]
            run_cfg["reliability"]["simulated_delay_sec"] = 0.0

            sim = FederatedSimulator(
                config=run_cfg,
                train_dataset=self.train_dataset,
                test_dataset=self.test_dataset,
                device=self.device
            )
            summary = sim.run_simulation(num_rounds=rounds)

            total_dropped = sum(r["dropped_clients_count"] for r in summary["round_history"])
            total_delayed = sum(r["delayed_clients_count"] for r in summary["round_history"])
            total_accepted = sum(r["accepted_clients_count"] for r in summary["round_history"])

            item = {
                "config_id": c["id"],
                "config_name": c["name"],
                "dropout_prob": c["p_drop"],
                "straggler_prob": c["p_strag"],
                "total_dropped_events": total_dropped,
                "total_delayed_events": total_delayed,
                "total_accepted_updates": total_accepted,
                "final_accuracy": summary["final_test_accuracy"],
                "final_loss": summary["final_test_loss"]
            }
            results.append(item)
            log_event("EXP_D_STEP", f"Finished {c['name']}: Acc={item['final_accuracy']*100:.2f}%, Dropped={total_dropped}")

        if save_results:
            self._save_json("experiment_d_reliability.json", results)
            pd.DataFrame(results).to_csv(os.path.join(self.results_dir, "experiment_d_reliability.csv"), index=False)

        return results

    def run_mia_evaluation(
        self,
        dp_model: Any,
        non_private_model: Any,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """Stretch Goal: Run Membership Inference Attack comparison."""
        log_event("MIA_START", "Running Membership Inference Attack evaluation (Non-Private vs DP).")
        evaluator = MembershipInferenceEvaluator(device=self.device)

        # Non-private model evaluation
        np_mia = evaluator.evaluate_model(
            model=non_private_model,
            member_dataset=self.train_dataset,
            non_member_dataset=self.test_dataset
        )

        # DP model evaluation
        dp_mia = evaluator.evaluate_model(
            model=dp_model,
            member_dataset=self.train_dataset,
            non_member_dataset=self.test_dataset
        )

        mia_result = {
            "timestamp": datetime.now().isoformat(),
            "non_private_model": np_mia,
            "dp_model": dp_mia,
            "auc_mitigation_delta": round(np_mia["attack_auc"] - dp_mia["attack_auc"], 4)
        }

        if save_results:
            self._save_json("mia_evaluation_results.json", mia_result)

        log_event("MIA_COMPLETE", f"MIA Complete: Non-Private AUC={np_mia['attack_auc']:.3f}, DP AUC={dp_mia['attack_auc']:.3f}")
        return mia_result

    def _save_json(self, filename: str, data: Any) -> None:
        """Helper to write JSON results."""
        path = os.path.join(self.results_dir, filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
