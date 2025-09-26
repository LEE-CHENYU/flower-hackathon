import flwr as fl
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.strategy import FedAvg
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import torch
from privacy_model import PrivacyProtectionModel


class PrivacyAwareFedAvg(FedAvg):
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        privacy_budget_threshold: float = 1.0,
        **kwargs
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            **kwargs
        )
        self.privacy_budget_threshold = privacy_budget_threshold
        self.privacy_scores = []
        self.round_privacy_metrics = {}

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        print(f"\n=== Server Round {server_round} ===")
        print(f"Aggregating updates from {len(results)} clients")

        aggregated_parameters, metrics = super().aggregate_fit(
            server_round, results, failures
        )

        privacy_metrics = self.calculate_privacy_metrics(results)
        self.round_privacy_metrics[server_round] = privacy_metrics

        metrics.update({
            "avg_privacy_score": privacy_metrics["avg_privacy_score"],
            "min_privacy_score": privacy_metrics["min_privacy_score"],
            "max_privacy_score": privacy_metrics["max_privacy_score"],
        })

        print(f"Privacy Metrics - Avg: {privacy_metrics['avg_privacy_score']:.4f}, "
              f"Min: {privacy_metrics['min_privacy_score']:.4f}, "
              f"Max: {privacy_metrics['max_privacy_score']:.4f}")

        if privacy_metrics["avg_privacy_score"] < self.privacy_budget_threshold:
            print("⚠️  Warning: Privacy score below threshold!")

        return aggregated_parameters, metrics

    def calculate_privacy_metrics(
        self, results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]]
    ) -> Dict[str, float]:
        privacy_scores = []

        for client, fit_res in results:
            if hasattr(fit_res, 'metrics') and fit_res.metrics:
                if 'privacy_score' in fit_res.metrics:
                    privacy_scores.append(fit_res.metrics['privacy_score'])

        if not privacy_scores:
            privacy_scores = [1.0] * len(results)

        return {
            "avg_privacy_score": np.mean(privacy_scores),
            "min_privacy_score": np.min(privacy_scores),
            "max_privacy_score": np.max(privacy_scores),
            "std_privacy_score": np.std(privacy_scores),
        }

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}

        losses = []
        privacy_scores = []
        total_samples = 0

        for client, evaluate_res in results:
            losses.append(evaluate_res.loss * evaluate_res.num_examples)
            total_samples += evaluate_res.num_examples

            if hasattr(evaluate_res, 'metrics') and evaluate_res.metrics:
                if 'privacy_score' in evaluate_res.metrics:
                    privacy_scores.append(evaluate_res.metrics['privacy_score'])

        avg_loss = sum(losses) / total_samples if total_samples > 0 else 0
        avg_privacy = np.mean(privacy_scores) if privacy_scores else 0

        print(f"\n=== Evaluation Round {server_round} ===")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Average Privacy Score: {avg_privacy:.4f}")
        print(f"Total Samples Evaluated: {total_samples}")

        return avg_loss, {"privacy_score": avg_privacy}


def get_initial_parameters() -> Parameters:
    model = PrivacyProtectionModel(privacy_budget=1.0, blur_strength=31)
    weights = [val.cpu().numpy() for _, val in model.state_dict().items()]
    return ndarrays_to_parameters(weights)


def create_strategy(
    num_rounds: int = 10,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    min_fit_clients: int = 2,
    min_evaluate_clients: int = 2,
    min_available_clients: int = 2,
    privacy_budget_threshold: float = 1.0,
) -> PrivacyAwareFedAvg:
    strategy = PrivacyAwareFedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        privacy_budget_threshold=privacy_budget_threshold,
        initial_parameters=get_initial_parameters(),
        evaluate_fn=None,
        on_fit_config_fn=lambda server_round: {"epochs": 1, "server_round": server_round},
        on_evaluate_config_fn=lambda server_round: {"server_round": server_round},
    )
    return strategy


def start_server(
    num_rounds: int = 10,
    num_clients: int = 3,
    privacy_budget_threshold: float = 1.0,
    server_address: str = "0.0.0.0:8080"
):
    print("\n🚀 Starting Flower Server for Privacy-Preserving Federated Learning")
    print(f"Configuration:")
    print(f"  - Number of rounds: {num_rounds}")
    print(f"  - Expected clients: {num_clients}")
    print(f"  - Privacy budget threshold: {privacy_budget_threshold}")
    print(f"  - Server address: {server_address}")

    strategy = create_strategy(
        num_rounds=num_rounds,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        privacy_budget_threshold=privacy_budget_threshold,
    )

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    print("\n✅ Server completed all rounds")
    print("Final Privacy Metrics by Round:")
    for round_num, metrics in strategy.round_privacy_metrics.items():
        print(f"  Round {round_num}: {metrics}")


if __name__ == "__main__":
    start_server(
        num_rounds=5,
        num_clients=3,
        privacy_budget_threshold=0.8,
        server_address="0.0.0.0:8080"
    )