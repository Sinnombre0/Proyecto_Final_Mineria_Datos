"""
src/pipeline.py
===============
Patrón de diseño: Pipeline + Strategy
--------------------------------------
Pipeline: el procesamiento de datos ocurre en etapas secuenciales bien definidas
          (limpieza → encoding → escalado → modelo).
Strategy: permite intercambiar el algoritmo de limpieza/encoding en tiempo de ejecución.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# ESTRATEGIAS DE LIMPIEZA (Strategy)
# ══════════════════════════════════════════════════════════════════════════════

class CleaningStrategy(ABC):
    """Interfaz para estrategias de limpieza de datos."""

    @abstractmethod
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        ...


class DropMissingStrategy(CleaningStrategy):
    """Elimina filas con valores faltantes por encima de un umbral."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.dropna(thresh=int(len(df.columns) * (1 - self.threshold)))
        df = df.drop_duplicates()
        logger.info("DropMissingStrategy: %d → %d filas", before, len(df))
        return df


class ImputeMedianStrategy(CleaningStrategy):
    """Imputa la mediana en numéricas y la moda en categóricas."""

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in df.select_dtypes(include="number").columns:
            df[col].fillna(df[col].median(), inplace=True)
        for col in df.select_dtypes(include="object").columns:
            df[col].fillna(df[col].mode()[0], inplace=True)
        df.drop_duplicates(inplace=True)
        logger.info("ImputeMedianStrategy aplicada")
        return df


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

class DataPipeline:
    """
    Orquesta las etapas de procesamiento de datos.
    Cada etapa puede intercambiarse mediante estrategias (Strategy).
    """

    def __init__(
        self,
        cleaning_strategy: CleaningStrategy | None = None,
        target_col: str = "target",
    ) -> None:
        self.cleaning_strategy = cleaning_strategy or ImputeMedianStrategy()
        self.target_col = target_col

    def run(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """
        Ejecuta el pipeline completo.

        Returns
        -------
        X : pd.DataFrame  — features listos para modelado
        y : pd.Series     — variable objetivo
        """
        logger.info("=== Iniciando DataPipeline ===")

        # Etapa 1: Limpieza
        df = self.cleaning_strategy.clean(df)

        # Etapa 2: Separar X e y
        if self.target_col not in df.columns:
            raise ValueError(f"Columna objetivo '{self.target_col}' no encontrada.")
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        logger.info("DataPipeline completado: %d filas × %d features", *X.shape)
        return X, y


# ══════════════════════════════════════════════════════════════════════════════
# BUILDER DE PREPROCESADOR SKLEARN
# ══════════════════════════════════════════════════════════════════════════════

class PreprocessorBuilder:
    """Construye un ColumnTransformer de sklearn para features numéricas y categóricas."""

    def __init__(self, X: pd.DataFrame) -> None:
        self.num_features = X.select_dtypes(include="number").columns.tolist()
        self.cat_features = X.select_dtypes(include="object").columns.tolist()

    def build(self) -> ColumnTransformer:
        transformers = []
        if self.num_features:
            transformers.append(("num", StandardScaler(), self.num_features))
        if self.cat_features:
            transformers.append((
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                self.cat_features,
            ))
        return ColumnTransformer(transformers, remainder="drop")
