from typing import Any, Literal

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule

from radar_forcast.ml.models import GNNModel


class GraphForecastingGNN(LightningModule):
    def __init__(self, **model_kwargs: Any) -> None:
        self.save_hyperparameters()

        self.model = GNNModel(**model_kwargs)
        self.loss_module = nn.MSELoss()

    def forward(self, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for graph forecasting.

        Args:
            data: Graph data with node features containing 2 timesteps

        Returns:
            Predicted next timestep values
        """
        x, edge_index = data.x, data.edge_index  # TODO
        # x shape: [num_nodes, 2 * feature_dim] (contains k-1 and k timesteps)

        # Predict next timestep
        pred = self.model(x, edge_index)
        return pred

    def _compute_loss_and_metrics(
        self, batch: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """Compute loss and metrics for a batch."""
        # Forward pass
        pred = self.forward(batch)  # [num_nodes, output_dim]
        target = batch.y  # [num_nodes, output_dim] - target timestep k+1

        # Compute loss
        loss = self.loss_module(pred, target)

        # Compute additional metrics
        with torch.no_grad():
            mae = F.l1_loss(pred, target)
            mse = F.mse_loss(pred, target)
            rmse = torch.sqrt(mse)

            # Compute relative error (if target values are not zero)
            mask = target.abs() > 1e-6
            if mask.sum() > 0:
                relative_error = (pred[mask] - target[mask]).abs() / target[mask].abs()
                mape = relative_error.mean() * 100  # Mean Absolute Percentage Error
            else:
                mape = torch.tensor(0.0, device=pred.device)

        metrics = {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape}

        return loss, metrics

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss, metrics = self._compute_loss_and_metrics(batch)

        # Log metrics
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_mae", metrics["mae"])
        self.log("train_rmse", metrics["rmse"])
        self.log("train_mape", metrics["mape"])

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        loss, metrics = self._compute_loss_and_metrics(batch)

        # Log metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_mae", metrics["mae"])
        self.log("val_rmse", metrics["rmse"])
        self.log("val_mape", metrics["mape"])

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        loss, metrics = self._compute_loss_and_metrics(batch)

        # Log metrics
        self.log("test_loss", loss)
        self.log("test_mae", metrics["mae"])
        self.log("test_rmse", metrics["rmse"])
        self.log("test_mape", metrics["mape"])

    def predict_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Prediction step for inference."""
        return self.forward(batch)

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        # Optional: Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
            },
        }

    def on_train_epoch_end(self) -> None:
        """Log learning rate at the end of each epoch."""
        current_lr = self.optimizers().param_groups[0]["lr"]
        self.log("lr", current_lr)


class MultiStepGraphForecastingGNN(GraphForecastingGNN):
    """Extended version for multi-step forecasting."""

    def __init__(self, forecast_steps: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.forecast_steps = forecast_steps

        # Modify output dimension for multi-step prediction
        self.model.output_dim = self.hparams.output_dim * forecast_steps

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Predict multiple future timesteps."""
        pred = super().forward(data)
        # Reshape to [num_nodes, forecast_steps, feature_dim]
        return pred.view(-1, self.forecast_steps, self.hparams.output_dim)

    def _compute_loss_and_metrics(
        self, batch: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        pred = self.forward(batch)  # [num_nodes, forecast_steps, feature_dim]
        target = batch.y  # [num_nodes, forecast_steps, feature_dim]

        # Compute loss across all timesteps
        loss = self.loss_module(pred, target)

        # Compute metrics
        with torch.no_grad():
            mae = F.l1_loss(pred, target)
            mse = F.mse_loss(pred, target)
            rmse = torch.sqrt(mse)

            # Per-timestep metrics
            timestep_maes = []
            for t in range(self.forecast_steps):
                t_mae = F.l1_loss(pred[:, t], target[:, t])
                timestep_maes.append(t_mae)
                self.log(f"mae_step_{t+1}", t_mae)

        metrics = {"mae": mae, "mse": mse, "rmse": rmse, "timestep_maes": timestep_maes}

        return loss, metrics
