"""
src/data_repository.py
======================
Patrón de diseño: Repository
-------------------------------
Abstrae el acceso a datos (CSV, Parquet, API) detrás de una interfaz uniforme.
El código cliente no necesita saber de dónde vienen los datos.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# ── Interfaz base ──────────────────────────────────────────────────────────────
class BaseRepository(ABC):
    """Interfaz que todo repositorio de datos debe implementar."""

    @abstractmethod
    def load_raw(self) -> pd.DataFrame:
        """Devuelve los datos crudos sin ningún procesamiento."""

    @abstractmethod
    def load_processed(self) -> pd.DataFrame:
        """Devuelve los datos ya limpios y listos para modelado."""

    @abstractmethod
    def save_processed(self, df: pd.DataFrame) -> None:
        """Persiste el DataFrame procesado."""


# ── Implementación CSV ────────────────────────────────────────────────────────
class CSVRepository(BaseRepository):
    """Repositorio que lee/escribe archivos CSV."""

    def __init__(
        self,
        raw_path: str | Path,
        processed_path: str | Path,
        **read_kwargs,
    ) -> None:
        self.raw_path = Path(raw_path)
        self.processed_path = Path(processed_path)
        self.read_kwargs = read_kwargs

    def load_raw(self) -> pd.DataFrame:
        logger.info("Cargando datos crudos desde %s", self.raw_path)
        return pd.read_csv(self.raw_path, **self.read_kwargs)

    def load_processed(self) -> pd.DataFrame:
        logger.info("Cargando datos procesados desde %s", self.processed_path)
        return pd.read_csv(self.processed_path)

    def save_processed(self, df: pd.DataFrame) -> None:
        self.processed_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.processed_path, index=False)
        logger.info("Datos procesados guardados en %s  (%d filas)", self.processed_path, len(df))


# ── Implementación Parquet ────────────────────────────────────────────────────
class ParquetRepository(BaseRepository):
    """Repositorio que lee/escribe archivos Parquet (más eficiente para datasets grandes)."""

    def __init__(self, raw_path: str | Path, processed_path: str | Path) -> None:
        self.raw_path = Path(raw_path)
        self.processed_path = Path(processed_path)

    def load_raw(self) -> pd.DataFrame:
        logger.info("Cargando datos crudos desde %s", self.raw_path)
        return pd.read_parquet(self.raw_path)

    def load_processed(self) -> pd.DataFrame:
        logger.info("Cargando datos procesados desde %s", self.processed_path)
        return pd.read_parquet(self.processed_path)

    def save_processed(self, df: pd.DataFrame) -> None:
        self.processed_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.processed_path, index=False)
        logger.info("Datos procesados guardados en %s  (%d filas)", self.processed_path, len(df))


# ── Factory de repositorios ───────────────────────────────────────────────────
class DataRepository:
    """
    Punto de entrada único para obtener el repositorio correcto
    según el formato del archivo.
    """

    _registry: dict[str, type[BaseRepository]] = {
        ".csv":     CSVRepository,
        ".parquet": ParquetRepository,
    }

    @classmethod
    def create(
        cls,
        raw_path: str | Path,
        processed_path: str | Path,
        **kwargs,
    ) -> BaseRepository:
        ext = Path(raw_path).suffix.lower()
        if ext not in cls._registry:
            raise ValueError(f"Formato '{ext}' no soportado. Usa: {list(cls._registry)}")
        return cls._registry[ext](raw_path, processed_path, **kwargs)
