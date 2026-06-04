"""
src/evaluator.py
================
Clase Evaluator: encapsula el cálculo y visualización de métricas
para clasificación y regresión.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

logger = logging.getLogger(__name__)

UNAM_BLUE = "#003f8a"
UNAM_GOLD = "#c8a951"


class Evaluator:
    """
    Evalúa y visualiza el desempeño de un modelo.

    Parameters
    ----------
    task : str
        "classification" o "regression"
    output_dir : Path | str | None
        Directorio donde se guardan las figuras. Si es None, solo se muestran.
    """

    def __init__(
        self,
        task: str = "classification",
        output_dir: Path | str | None = None,
    ) -> None:
        if task not in ("classification", "regression"):
            raise ValueError("task debe ser 'classification' o 'regression'")
        self.task = task
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── API pública ────────────────────────────────────────────────────────
    def evaluate(self, y_true, y_pred, y_proba=None) -> dict:
        """Calcula y muestra todas las métricas según la tarea."""
        if self.task == "classification":
            return self._eval_classification(y_true, y_pred, y_proba)
        return self._eval_regression(y_true, y_pred)

    # ── Clasificación ──────────────────────────────────────────────────────
    def _eval_classification(self, y_true, y_pred, y_proba=None) -> dict:
        report = classification_report(y_true, y_pred, output_dict=True)
        logger.info("\n%s", classification_report(y_true, y_pred, digits=4))

        self._plot_confusion_matrix(y_true, y_pred)

        if y_proba is not None:
            self._plot_roc(y_true, y_proba)

        return report

    def _plot_confusion_matrix(self, y_true, y_pred) -> None:
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, ax=ax,
            cmap="Blues", colorbar=False, values_format="d",
        )
        ax.set_title("Matriz de Confusión", fontsize=13, fontweight="bold")
        plt.tight_layout()
        self._save_or_show(fig, "confusion_matrix.png")

    def _plot_roc(self, y_true, y_proba) -> None:
        fig, ax = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax, color=UNAM_BLUE)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_title("Curva ROC", fontsize=13, fontweight="bold")
        plt.tight_layout()
        self._save_or_show(fig, "roc_curve.png")

    # ── Regresión ──────────────────────────────────────────────────────────
    def _eval_regression(self, y_true, y_pred) -> dict:
        mae  = mean_absolute_error(y_true, y_pred)
        mse  = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2   = r2_score(y_true, y_pred)

        metrics = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}
        logger.info("MAE=%.4f | RMSE=%.4f | R²=%.4f", mae, rmse, r2)

        self._plot_residuals(y_true, y_pred)
        return metrics

    def _plot_residuals(self, y_true, y_pred) -> None:
        residuals = np.array(y_true) - np.array(y_pred)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].scatter(y_pred, residuals, alpha=0.4, color=UNAM_BLUE, s=15)
        axes[0].axhline(0, color="red", linewidth=1.5, linestyle="--")
        axes[0].set_xlabel("Predicho"); axes[0].set_ylabel("Residual")
        axes[0].set_title("Residuales vs. Predicho")

        axes[1].hist(residuals, bins=40, color=UNAM_BLUE, edgecolor="white")
        axes[1].set_title("Distribución de Residuales")

        plt.suptitle("Análisis de Residuales", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        self._save_or_show(fig, "residuals.png")

    # ── Utilidades ─────────────────────────────────────────────────────────
    def _save_or_show(self, fig: plt.Figure, filename: str) -> None:
        if self.output_dir:
            path = self.output_dir / filename
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info("Figura guardada: %s", path)
        else:
            plt.show()
        plt.close(fig)
