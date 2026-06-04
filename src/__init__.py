"""
src — Módulos del Proyecto de Minería de Datos
================================================
Importaciones principales para facilitar el uso desde notebooks.
"""

from src.data_repository import DataRepository, CSVRepository, ParquetRepository
from src.pipeline import DataPipeline, ImputeMedianStrategy, DropMissingStrategy, PreprocessorBuilder
from src.model_factory import ModelRegistry
from src.evaluator import Evaluator

__all__ = [
    "DataRepository",
    "CSVRepository",
    "ParquetRepository",
    "DataPipeline",
    "ImputeMedianStrategy",
    "DropMissingStrategy",
    "PreprocessorBuilder",
    "ModelRegistry",
    "Evaluator",
]
