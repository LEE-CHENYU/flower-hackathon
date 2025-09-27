"""
Federated Learning Server for LoRA Aggregation
"""

import flwr as fl
from flwr.common import FitRes, Parameters, Scalar
from flwr.server.strategy import FedAvg
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from pathlib import Path
import sys
import json
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))
from core.llava_lora_model import LLaVALoRAModel

class LoRAFedAvg(FedAvg):
    """Custom FedAvg strategy for LoRA weight aggregation"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.round_num = 0
        self.metrics_history = []

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate LoRA weights from clients"""

        if not results:
            return None, {}

        # Log round information
        print(f"\n{'='*50}")
        print(f"Server Round {server_round}: Aggregating LoRA weights")
        print(f"Successful clients: {len(results)}")
        print(f"Failed clients: {len(failures)}")

        # Extract metrics
        metrics = {}
        total_samples = 0
        total_loss = 0.0

        for client, fit_res in results:
            num_samples = fit_res.num_examples
            client_metrics = fit_res.metrics

            total_samples += num_samples
            if "loss" in client_metrics:
                total_loss += client_metrics["loss"] * num_samples

            print(f"  Client {client.cid}: {num_samples} samples, loss={client_metrics.get('loss', 'N/A')}")

        # Calculate weighted average loss
        if total_samples > 0:
            avg_loss = total_loss / total_samples
            metrics["avg_loss"] = avg_loss
            print(f"Average loss: {avg_loss:.4f}")

        # Store metrics
        self.metrics_history.append({
            "round": server_round,
            "avg_loss": metrics.get("avg_loss", 0),
            "num_clients": len(results),
            "total_samples": total_samples
        })

        # Standard FedAvg aggregation for LoRA weights
        aggregated = super().aggregate_fit(server_round, results, failures)

        # Save checkpoint every 5 rounds
        if server_round % 5 == 0:
            self._save_checkpoint(server_round, aggregated[0])

        return aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation metrics"""

        if not results:
            return None, {}

        # Aggregate quality scores
        total_quality = 0.0
        total_samples = 0

        for client, eval_res in results:
            quality = eval_res.loss  # We use loss field for quality score
            num_samples = eval_res.num_examples
            total_quality += quality * num_samples
            total_samples += num_samples

            print(f"  Client {client.cid}: quality={quality:.4f}")

        if total_samples > 0:
            avg_quality = total_quality / total_samples
            print(f"Average quality score: {avg_quality:.4f}")
            return avg_quality, {"avg_quality": avg_quality}

        return None, {}

    def _save_checkpoint(self, round_num: int, parameters: Optional[Parameters]):
        """Save LoRA checkpoint"""
        if parameters is None:
            return

        checkpoint_dir = Path("checkpoints/lora_adapters")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save as numpy arrays
        checkpoint_path = checkpoint_dir / f"round_{round_num}"
        checkpoint_path.mkdir(exist_ok=True)

        # Convert parameters to numpy and save
        params_array = fl.common.parameters_to_ndarrays(parameters)
        np.savez(checkpoint_path / "lora_weights.npz", *params_array)

        # Save metadata
        metadata = {
            "round": round_num,
            "timestamp": datetime.now().isoformat(),
            "metrics_history": self.metrics_history
        }
        with open(checkpoint_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Checkpoint saved to {checkpoint_path}")

class DentalFLServer:
    def __init__(
        self,
        num_rounds: int = 10,
        min_clients: int = 2,
        min_fit_clients: int = 2,
        min_eval_clients: int = 2,
        fraction_fit: float = 1.0,
        fraction_eval: float = 1.0,
        local_epochs: int = 1,
        server_address: str = "[::]:8080"
    ):
        """
        Initialize Federated Learning Server

        Args:
            num_rounds: Number of federated rounds
            min_clients: Minimum number of clients before starting
            min_fit_clients: Minimum clients for training
            min_eval_clients: Minimum clients for evaluation
            fraction_fit: Fraction of clients for training
            fraction_eval: Fraction of clients for evaluation
            local_epochs: Number of local epochs per round
            server_address: Server address
        """
        self.num_rounds = num_rounds
        self.server_address = server_address

        # Initialize dummy model for initial parameters
        print("Initializing server model for parameter initialization...")
        from core.model_configs import get_model_config

        # Try TinyLLaVA first, with automatic fallback to llava-7b-qlora
        config = get_model_config("tiny-llava")
        try:
            self.model = LLaVALoRAModel(
                model_name=config["model_name"],
                lora_config=config.get("lora_config"),
                use_quantization=config.get("use_quantization", False),
                quantization_bits=config.get("quantization_bits", 4)
            )
        except RuntimeError as e:
            print(f"Server initialization with TinyLLaVA failed: {e}")
            print("Server will wait for client parameters instead")
            # Create dummy initial parameters
            import numpy as np
            initial_ndarrays = [
                np.zeros((8, 768), dtype=np.float32),  # Dummy LoRA weights
                np.zeros((768, 8), dtype=np.float32)
            ]
            self.model = None

        if self.model is not None:
            initial_params = self.model.get_lora_weights()
            initial_ndarrays = list(initial_params.values())

        # Configure strategy
        self.strategy = LoRAFedAvg(
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_eval_clients,
            min_available_clients=min_clients,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_eval,
            on_fit_config_fn=self.fit_config,
            on_evaluate_config_fn=self.evaluate_config,
            initial_parameters=fl.common.ndarrays_to_parameters(initial_ndarrays)
        )

        # Configure server
        self.config = fl.server.ServerConfig(num_rounds=num_rounds)
        self.local_epochs = local_epochs

    def fit_config(self, server_round: int) -> Dict:
        """Configuration for client training"""
        config = {
            "local_epochs": self.local_epochs,
            "server_round": server_round,
        }
        return config

    def evaluate_config(self, server_round: int) -> Dict:
        """Configuration for client evaluation"""
        config = {
            "server_round": server_round,
        }
        return config

    def start(self):
        """Start the federated learning server"""
        print("\n" + "="*60)
        print("Starting Federated Learning Server for LoRA Fine-tuning")
        print(f"Server address: {self.server_address}")
        print(f"Number of rounds: {self.num_rounds}")
        print(f"Waiting for clients to connect...")
        print("="*60 + "\n")

        # Start server
        fl.server.start_server(
            server_address=self.server_address,
            config=self.config,
            strategy=self.strategy
        )

        # Save final model
        self._save_final_model()

    def _save_final_model(self):
        """Save the final aggregated model"""
        final_path = Path("checkpoints/lora_adapters/final")
        final_path.mkdir(parents=True, exist_ok=True)

        # Copy latest checkpoint
        checkpoints = list(Path("checkpoints/lora_adapters").glob("round_*"))
        if checkpoints:
            # Extract round number safely, ignoring invalid formats
            def get_round_number(p):
                try:
                    parts = p.name.split("_")
                    if len(parts) >= 2 and parts[1].isdigit():
                        return int(parts[1])
                except:
                    pass
                return -1

            valid_checkpoints = [cp for cp in checkpoints if get_round_number(cp) >= 0]
            if valid_checkpoints:
                latest = max(valid_checkpoints, key=get_round_number)
            else:
                latest = checkpoints[0]  # Fallback to first if no valid format
            import shutil
            shutil.copytree(latest, final_path, dirs_exist_ok=True)
            print(f"\nFinal model saved to {final_path}")

        # Save training summary
        if hasattr(self.strategy, 'metrics_history'):
            with open(final_path / "training_summary.json", 'w') as f:
                json.dump(self.strategy.metrics_history, f, indent=2)

def start_server(
    num_rounds: int = 10,
    min_clients: int = 2,
    local_epochs: int = 1,
    server_address: str = "[::]:8080"
):
    """Start the federated learning server"""
    server = DentalFLServer(
        num_rounds=num_rounds,
        min_clients=min_clients,
        local_epochs=local_epochs,
        server_address=server_address
    )
    server.start()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10, help="Number of federated rounds")
    parser.add_argument("--min-clients", type=int, default=2, help="Minimum number of clients")
    parser.add_argument("--local-epochs", type=int, default=1, help="Local training epochs")
    parser.add_argument("--address", type=str, default="[::]:8080", help="Server address")
    args = parser.parse_args()

    start_server(
        num_rounds=args.rounds,
        min_clients=args.min_clients,
        local_epochs=args.local_epochs,
        server_address=args.address
    )