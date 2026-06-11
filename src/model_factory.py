"""
src/model_factory.py
====================
Patrón de diseño: Factory Method
----------------------------------
Permite crear instancias de diferentes modelos de forma intercambiable
sin cambiar el código cliente.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# ── Interfaz creadora ──────────────────────────────────────────────────────────
class ModelFactory(ABC):
    """Fábrica abstracta de modelos."""

    SEED = 42

    @abstractmethod
    def create_model(self, **kwargs) -> BaseEstimator:
        """Devuelve una instancia del modelo configurada con kwargs."""

    def create_baseline(self) -> BaseEstimator:
        """Devuelve el clasificador/regresor trivial para comparación."""
        return DummyClassifier(strategy="most_frequent", random_state=self.SEED)


# ── Fábricas concretas ────────────────────────────────────────────────────────

class RandomForestFactory(ModelFactory):
    def create_model(self, **kwargs) -> RandomForestClassifier:
        kwargs.setdefault("random_state", self.SEED)
        kwargs.setdefault("n_jobs", -1)
        return RandomForestClassifier(**kwargs)


class LogisticRegressionFactory(ModelFactory):
    def create_model(self, **kwargs) -> LogisticRegression:
        kwargs.setdefault("random_state", self.SEED)
        kwargs.setdefault("max_iter", 1000)
        return LogisticRegression(**kwargs)


class DecisionTreeFactory(ModelFactory):
    def create_model(self, **kwargs) -> DecisionTreeClassifier:
        kwargs.setdefault("random_state", self.SEED)
        return DecisionTreeClassifier(**kwargs)


class GradientBoostingFactory(ModelFactory):
    def create_model(self, **kwargs) -> GradientBoostingClassifier:
        kwargs.setdefault("random_state", self.SEED)
        return GradientBoostingClassifier(**kwargs)


class KNNFactory(ModelFactory):
    def create_model(self, **kwargs) -> KNeighborsClassifier:
        kwargs.setdefault("n_neighbors", 5)
        return KNeighborsClassifier(**kwargs)


class SVMFactory(ModelFactory):
    def create_model(self, **kwargs) -> SVC:
        kwargs.setdefault("random_state", self.SEED)
        kwargs.setdefault("probability", True)
        return SVC(**kwargs)


class RidgeRegressionFactory(ModelFactory):
    """Fábrica para regresión Ridge (usa cuando la tarea es regresión)."""
    def create_model(self, **kwargs) -> Ridge:
        return Ridge(**kwargs)

    def create_baseline(self) -> DummyRegressor:
        return DummyRegressor(strategy="mean")


# ── Registro central ──────────────────────────────────────────────────────────
class ModelRegistry:
    """
    Registro de fábricas disponibles.
    Uso:
        factory = ModelRegistry.get("random_forest")
        model   = factory.create_model(n_estimators=200)
    """

    _factories: dict[str, type[ModelFactory]] = {
        "random_forest":       RandomForestFactory,
        "logistic_regression": LogisticRegressionFactory,
        "decision_tree":       DecisionTreeFactory,
        "gradient_boosting":   GradientBoostingFactory,
        "knn":                 KNNFactory,
        "svm":                 SVMFactory,
        "ridge":               RidgeRegressionFactory,
    }

    @classmethod
    def get(cls, name: str) -> ModelFactory:
        name = name.lower().replace(" ", "_").replace("-", "_")
        if name not in cls._factories:
            raise ValueError(
                f"Modelo '{name}' no registrado. "
                f"Opciones disponibles: {sorted(cls._factories)}"
            )
        return cls._factories[name]()

    @classmethod
    def available(cls) -> list[str]:
        return sorted(cls._factories.keys())
