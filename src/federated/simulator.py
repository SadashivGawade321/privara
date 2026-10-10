"""End-to-end Federated Learning Simulator for Privara.

Coordinates the complete round execution lifecycle:
client sampling -> dropout/straggler handling -> local DP-SGD -> Top-K compression ->
communication budget checking -> FedAvg aggregation -> global evaluation -> metrics logging.
"""

from typing import Dict, Any, List, Optional, Callable
import time
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from models.classifier import get_model
from federated.client import FederatedClient
from federated.server import FederatedServer
from data.partitioner import DataPartitioner
from privacy.accountant import PrivacyAccountantWrapper
from communication.budget import CommunicationBudgetManager
from communication.metrics import CommunicationMetricsTracker
from reliability.dropout import DropoutSimulator
from reliability.stragglers import StragglerSimulator
from utils.logging import log_event, logger


class FederatedSimulator:
    """Orchestrates multi-client federated training simulation."""

    def __init__(
        self,
        config: Dict[str, Any],
        train_dataset: Dataset,
        test_dataset: Dataset,
        device: str = "cpu"
    ):
        self.config = config
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.device = device

        # Initialize Data Partitioner
        p_cfg = config.get("partitioning", {})
        self.partitioner = DataPartitioner(
            dataset=train_dataset,
            num_clients=p_cfg.get("num_clients", 10),
            strategy=p_cfg.get("strategy", "dirichlet"),
            dirichlet_alpha=p_cfg.get("dirichlet_alpha", 0.5),
            classes_per_client=p_cfg.get("classes_per_client", 3),
            min_samples_per_client=p_cfg.get("min_samples_per_client", 50),
            seed=config.get("reproducibility", {}).get("seed", 42)
        )

        # Initialize Clients
        self.clients: Dict[str, FederatedClient] = {}
        for cid, subset in self.partitioner.client_subsets.items():
            stats = self.partitioner.client_stats[cid]
            self.clients[cid] = FederatedClient(
                client_id=cid,
                dataset=subset,
                client_stats=stats,
                device=device
            )

        # Initialize Global Model
        m_cfg = config.get("model", {})
        self.global_model = get_model(
            architecture=m_cfg.get("architecture", "cnn"),
            num_classes=m_cfg.get("num_classes", 10)
        ).to(device)

        # Initialize Privacy Accountant
        pr_cfg = config.get("privacy", {})
        self.privacy_accountant = PrivacyAccountantWrapper(
            accountant_type=pr_cfg.get("accountant", "rdp"),
            target_delta=pr_cfg.get("target_delta", 1e-5),
            target_epsilon=pr_cfg.get("target_epsilon", 3.0)
        )

        # Initialize Communication Budget & Metrics
        c_cfg = config.get("communication", {})
        self.budget_manager = CommunicationBudgetManager(
            budget_per_round_mb=c_cfg.get("budget_per_round_mb", 20.0),
            budget_mode=c_cfg.get("budget_mode", "bidirectional"),
            enforce_policy=c_cfg.get("enforce_policy", "reject")
        )
        self.metrics_tracker = CommunicationMetricsTracker()

        # Initialize Server
        self.server = FederatedServer(
            global_model=self.global_model,
            test_dataset=test_dataset,
            privacy_accountant=self.privacy_accountant,
            budget_manager=self.budget_manager,
            metrics_tracker=self.metrics_tracker,
            device=device
        )

        # Initialize Reliability Simulators
        r_cfg = config.get("reliability", {})
        seed = config.get("reproducibility", {}).get("seed", 42)
        self.dropout_sim = DropoutSimulator(
            dropout_prob=r_cfg.get("dropout_prob", 0.1),
            min_participants=config.get("federated", {}).get("min_participants", 2),
            seed=seed
        )
        self.straggler_sim = StragglerSimulator(
            straggler_prob=r_cfg.get("straggler_prob", 0.1),
            simulated_delay_sec=r_cfg.get("simulated_delay_sec", 0.05),
            seed=seed
        )

        self.round_history: List[Dict[str, Any]] = []
        self.is_running: bool = False
        self.stop_requested: bool = False

    def run_simulation(
        self,
        num_rounds: Optional[int] = None,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """Execute full federated training simulation across rounds."""
        f_cfg = self.config.get("federated", {})
        pr_cfg = self.config.get("privacy", {})
        comp_cfg = self.config.get("compression", {})

        total_rounds = num_rounds or f_cfg.get("num_rounds", 10)
        self.is_running = True
        self.stop_requested = False

        start_time = time.time()
        log_event("TRAINING_START", f"Starting Privara Federated Simulation for {total_rounds} rounds.")

        # Baseline evaluation before round 1
        initial_acc, initial_loss = self.server.evaluate_global_model()
        log_event("INITIAL_EVAL", f"Pre-training global test accuracy: {initial_acc * 100:.2f}%, loss: {initial_loss:.4f}")

        for round_idx in range(1, total_rounds + 1):
            if self.stop_requested:
                log_event("TRAINING_STOPPED", f"Training stopped by user request at round {round_idx - 1}", level="warning")
                break

            round_data = self._execute_round(
                round_num=round_idx,
                total_rounds=total_rounds,
                f_cfg=f_cfg,
                pr_cfg=pr_cfg,
                comp_cfg=comp_cfg
            )
            self.round_history.append(round_data)

            if callback:
                callback(round_data)

        self.is_running = False
        total_duration = time.time() - start_time
        final_summary = self._create_final_summary(total_duration)
        log_event("TRAINING_COMPLETE", f"Completed training in {total_duration:.2f}s. Final Test Acc: {final_summary['final_test_accuracy']*100:.2f}%")
        return final_summary

    def _execute_round(
        self,
        round_num: int,
        total_rounds: int,
        f_cfg: Dict[str, Any],
        pr_cfg: Dict[str, Any],
        comp_cfg: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single communication round."""
        round_start_time = time.time()
        self.server.current_round = round_num
        self.budget_manager.start_round(round_num)

        all_client_ids = list(self.clients.keys())
        
        # 1. Client sampling
        client_fraction = f_cfg.get("client_fraction", 1.0)
        if client_fraction < 1.0:
            k_select = max(2, int(round(len(all_client_ids) * client_fraction)))
            rng = self.dropout_sim.rng
            selected_clients = rng.choice(all_client_ids, size=k_select, replace=False).tolist()
        else:
            selected_clients = list(all_client_ids)

        # 2. Simulate dropout
        active_clients, dropped_clients = self.dropout_sim.filter_active_clients(selected_clients)

        # 3. Simulate stragglers
        on_time_clients, delayed_clients = self.straggler_sim.evaluate_stragglers(active_clients)

        # Update client statuses
        for cid in all_client_ids:
            if cid in dropped_clients:
                self.clients[cid].status = "Dropped"
            elif cid in delayed_clients:
                self.clients[cid].status = "Delayed"
            elif cid in on_time_clients:
                self.clients[cid].status = "Active"
            else:
                self.clients[cid].status = "Idle"

        # 4. Prepare broadcast and record download bandwidth
        global_state, broadcast_bytes = self.server.prepare_broadcast_payload()
        self.budget_manager.record_download(broadcast_bytes, len(active_clients))

        # 5. Local Training on active clients
        client_updates = []
        round_uncompressed_bytes = 0
        round_compressed_bytes = 0
        total_recon_error = 0.0
        avg_train_loss = 0.0
        avg_train_acc = 0.0

        for cid in active_clients:
            client = self.clients[cid]
            is_delayed = cid in delayed_clients
            self.straggler_sim.simulate_delay(is_delayed)

            update_pkg = client.train_round(
                global_model=self.global_model,
                round_num=round_num,
                local_epochs=f_cfg.get("local_epochs", 1),
                batch_size=f_cfg.get("batch_size", 32),
                learning_rate=f_cfg.get("learning_rate", 0.01),
                momentum=f_cfg.get("momentum", 0.9),
                dp_enabled=pr_cfg.get("enabled", True),
                noise_multiplier=pr_cfg.get("noise_multiplier", 1.0),
                max_grad_norm=pr_cfg.get("max_grad_norm", 1.2),
                target_delta=pr_cfg.get("target_delta", 1e-5),
                compression_enabled=comp_cfg.get("enabled", True),
                topk_ratio=comp_cfg.get("topk_ratio", 0.5)
            )

            payload_size = (
                update_pkg["compressed_bytes"]
                if comp_cfg.get("enabled", True)
                else update_pkg["uncompressed_bytes"]
            )

            round_uncompressed_bytes += update_pkg["uncompressed_bytes"]
            round_compressed_bytes += update_pkg["compressed_bytes"]
            total_recon_error += update_pkg["reconstruction_error"]
            avg_train_loss += update_pkg["train_metrics"]["train_loss"]
            avg_train_acc += update_pkg["train_metrics"]["train_accuracy"]

            # 6. Check communication budget
            accepted, reason = self.budget_manager.check_and_record_upload(cid, payload_size)
            if accepted:
                client_updates.append(update_pkg)
            else:
                client.status = "Rejected"

        # 7. Privacy accounting step
        if pr_cfg.get("enabled", True) and active_clients:
            sample_rate = client_updates[0]["train_metrics"]["sample_rate"] if client_updates else 0.05
            steps_per_client = client_updates[0]["train_metrics"]["steps_taken"] if client_updates else 1
            self.privacy_accountant.step(
                noise_multiplier=pr_cfg.get("noise_multiplier", 1.0),
                sample_rate=sample_rate,
                steps=steps_per_client
            )
        privacy_record = self.privacy_accountant.record_round_end(
            round_num=round_num,
            noise_multiplier=pr_cfg.get("noise_multiplier", 1.0) if pr_cfg.get("enabled", True) else 0.0,
            sample_rate=sample_rate if pr_cfg.get("enabled", True) and active_clients else 0.0
        )

        # 8. FedAvg Aggregation of accepted updates
        agg_info = self.server.aggregate_round(client_updates)

        # 9. Global model evaluation
        test_acc, test_loss = self.server.evaluate_global_model(batch_size=self.config.get("evaluation", {}).get("test_batch_size", 64))

        # 10. Record communication metrics
        round_time = time.time() - round_start_time
        comp_ratio = (
            1.0 - (round_compressed_bytes / max(1, round_uncompressed_bytes))
            if round_uncompressed_bytes > 0
            else 0.0
        )
        avg_recon_error = total_recon_error / max(1, len(active_clients))
        metadata_bytes = int(round_compressed_bytes * 0.15)  # Metadata header proportion
        overhead_bytes = int(round_compressed_bytes * 0.08)

        comm_record = self.metrics_tracker.record_round(
            round_num=round_num,
            uncompressed_bytes=round_uncompressed_bytes,
            compressed_bytes=round_compressed_bytes,
            metadata_bytes=metadata_bytes,
            overhead_bytes=overhead_bytes,
            time_seconds=round_time,
            compression_ratio=comp_ratio,
            reconstruction_error=avg_recon_error
        )

        budget_record = self.budget_manager.end_round()

        round_data = {
            "round": round_num,
            "total_rounds": total_rounds,
            "test_accuracy": round(test_acc, 4),
            "test_loss": round(test_loss, 4),
            "train_loss": round(avg_train_loss / max(1, len(active_clients)), 4),
            "train_accuracy": round(avg_train_acc / max(1, len(active_clients)), 4),
            "total_clients": len(all_client_ids),
            "active_clients_count": len(active_clients),
            "delayed_clients_count": len(delayed_clients),
            "dropped_clients_count": len(dropped_clients),
            "accepted_clients_count": len(client_updates),
            "rejected_clients_count": len(self.budget_manager.rejected_clients),
            "active_clients": active_clients,
            "delayed_clients": delayed_clients,
            "dropped_clients": dropped_clients,
            "rejected_clients": self.budget_manager.rejected_clients,
            "achieved_epsilon": privacy_record["achieved_epsilon"],
            "delta": privacy_record["delta"],
            "privacy_budget_used_pct": privacy_record["budget_used_pct"],
            "uncompressed_mb": comm_record["uncompressed_mb"],
            "compressed_mb": comm_record["compressed_mb"],
            "compression_ratio": comm_record["compression_ratio"],
            "reconstruction_error": comm_record["reconstruction_error"],
            "round_duration_sec": round_time,
            "budget_mb": budget_record["budget_mb"],
            "budget_exceeded": budget_record["budget_exceeded"],
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }

        log_event(
            "ROUND_FINISH",
            f"Round {round_num}/{total_rounds}: Test Acc={test_acc*100:.2f}%, Loss={test_loss:.3f}, ε={privacy_record['achieved_epsilon']}",
            details=f"Active: {len(active_clients)}, Dropped: {len(dropped_clients)}, Delayed: {len(delayed_clients)}, Comm: {comm_record['compressed_mb']:.2f} MB"
        )

        return round_data

    def _create_final_summary(self, total_duration: float) -> Dict[str, Any]:
        """Package final simulation results."""
        final_acc, final_loss = (
            (self.round_history[-1]["test_accuracy"], self.round_history[-1]["test_loss"])
            if self.round_history
            else (0.0, 0.0)
        )
        return {
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
            "rounds_completed": len(self.round_history),
            "total_duration_sec": round(total_duration, 2),
            "final_test_accuracy": final_acc,
            "final_test_loss": final_loss,
            "privacy_summary": self.privacy_accountant.get_summary(),
            "communication_summary": self.metrics_tracker.get_summary(),
            "round_history": self.round_history,
            "client_stats": self.partitioner.get_client_stats()
        }
