"""
Pipeline: el procesamiento de datos ocurre en etapas secuenciales bien definidas
          (limpieza → encoding → escalado → modelo).
Strategy: permite intercambiar el algoritmo de limpieza/encoding en tiempo de ejecución.
"""

import logging
import pandas as pd

from __future__ import annotations
from abc import ABC, abstractmethod
from sklearn.compose import ColumnTransformer
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
        target_col: str = "bien_juridico_afectado"
    ) -> None:
        # Si no se proporciona estrategia, se usa ImputeMedianStrategy por defecto
        self.cleaning_strategy = cleaning_strategy or ImputeMedianStrategy()
        self.target_col = target_col
        self.orden_meses = {
            'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5, 'Junio': 6,
            'Julio': 7, 'Agosto': 8, 'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12
        }
        self.features = ['anio', 'mes_num', 'clave_ent', 'incidencia_delictiva']

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
        df_cleaned = self.cleaning_strategy.clean(df)

        df_transformed = df_cleaned.copy()
        df_transformed['mes_num'] = df_transformed['mes'].map(self.orden_meses)
        
        # Etapa 3: Separación de X e Y
        if self.target_col not in df_transformed.columns:
            raise ValueError(f"La columna objetivo '{self.target_col}' no existe en el dataset.")
            
        X = df_transformed[self.features]
        y = df_transformed[self.target_col]

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
