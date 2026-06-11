"""
predict.py
==========
Script de demostración en vivo — Proyecto Final Minería de Datos
-----------------------------------------------------------------
Carga el modelo entrenado (models/modelo_final.joblib) y ejecuta
predicciones nuevas SIN re-entrenar.

Uso:
    python predict.py
"""

import sys
from pathlib import Path
import pandas as pd
import joblib

MODEL_PATH = Path("models/modelo_final.joblib")

ORDEN_MESES = {
    'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4,
    'Mayo': 5, 'Junio': 6, 'Julio': 7, 'Agosto': 8,
    'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12
}

ENTIDADES = {
    1: 'Aguascalientes', 2: 'Baja California', 3: 'Baja California Sur',
    4: 'Campeche', 5: 'Coahuila', 6: 'Colima', 7: 'Chiapas',
    8: 'Chihuahua', 9: 'Ciudad de México', 10: 'Durango',
    11: 'Guanajuato', 12: 'Guerrero', 13: 'Hidalgo', 14: 'Jalisco',
    15: 'Estado de México', 16: 'Michoacán', 17: 'Morelos', 18: 'Nayarit',
    19: 'Nuevo León', 20: 'Oaxaca', 21: 'Puebla', 22: 'Querétaro',
    23: 'Quintana Roo', 24: 'San Luis Potosí', 25: 'Sinaloa',
    26: 'Sonora', 27: 'Tabasco', 28: 'Tamaulipas', 29: 'Tlaxcala',
    30: 'Veracruz', 31: 'Yucatán', 32: 'Zacatecas'
}


def cargar_modelo(path: Path):
    if not path.exists():
        print(f"❌ No se encontró el modelo en: {path}")
        print("   Asegúrate de haber ejecutado modelo.qmd primero.")
        sys.exit(1)
    modelo = joblib.load(path)
    print(f"✅ Modelo cargado desde: {path}\n")
    return modelo


def predecir(modelo, anio: int, mes: int, clave_ent: int, incidencia: int) -> dict:
    obs = pd.DataFrame({
        'anio':                 [anio],
        'mes_num':              [mes],
        'clave_ent':            [clave_ent],
        'incidencia_delictiva': [incidencia],
    })
    clase      = modelo.predict(obs)[0]
    probabilidades = modelo.predict_proba(obs)[0]
    confianza  = probabilidades.max()
    clases     = modelo.classes_

    return {
        "clase_predicha": clase,
        "confianza":      confianza,
        "probabilidades": dict(zip(clases, probabilidades)),
    }


def imprimir_resultado(resultado: dict, contexto: dict) -> None:
    print("─" * 55)
    print("  PREDICCIÓN EN VIVO")
    print("─" * 55)
    print(f"  Año          : {contexto['anio']}")
    print(f"  Mes          : {contexto['mes_nombre']}")
    print(f"  Estado       : {contexto['entidad']} (clave {contexto['clave_ent']})")
    print(f"  Incidencia   : {contexto['incidencia']}")
    print("─" * 55)
    print(f"  → Bien jurídico predicho : {resultado['clase_predicha']}")
    print(f"  → Confianza              : {resultado['confianza']*100:.1f}%")
    print("\n  Probabilidades por clase:")
    for clase, prob in sorted(resultado['probabilidades'].items(),
                               key=lambda x: x[1], reverse=True):
        barra = "█" * int(prob * 30)
        print(f"    {prob*100:5.1f}%  {barra:<30}  {clase}")
    print("─" * 55)


def main():
    print("=" * 55)
    print("  DEMO EN VIVO — Incidencia Delictiva México")
    print("  Proyecto Final — Almacenes y Minería de Datos")
    print("=" * 55 + "\n")

    modelo = cargar_modelo(MODEL_PATH)

    # ── Casos de demostración ──────────────────────────────────────────────
    casos = [
        {"anio": 2024, "mes": 12, "mes_nombre": "Diciembre",
         "clave_ent": 9,  "entidad": "Ciudad de México", "incidencia": 500},

        {"anio": 2023, "mes": 6,  "mes_nombre": "Junio",
         "clave_ent": 15, "entidad": "Estado de México",  "incidencia": 1200},

        {"anio": 2022, "mes": 3,  "mes_nombre": "Marzo",
         "clave_ent": 14, "entidad": "Jalisco",            "incidencia": 80},
    ]

    for i, caso in enumerate(casos, 1):
        print(f"\nCaso {i} de {len(casos)}:")
        resultado = predecir(
            modelo,
            anio=caso["anio"],
            mes=caso["mes"],
            clave_ent=caso["clave_ent"],
            incidencia=caso["incidencia"],
        )
        imprimir_resultado(resultado, caso)

    # ── Predicción interactiva ─────────────────────────────────────────────
    print("\n¿Quieres hacer una predicción personalizada? (s/n): ", end="")
    resp = input().strip().lower()
    if resp == 's':
        print("\nIngresa los datos:")
        anio      = int(input("  Año (2015-2025): "))
        mes_str   = input("  Mes (ej. Marzo): ").strip().capitalize()
        mes_num   = ORDEN_MESES.get(mes_str, 1)
        clave_ent = int(input("  Clave entidad (1-32): "))
        incidencia = int(input("  Incidencia delictiva: "))

        resultado = predecir(modelo, anio, mes_num, clave_ent, incidencia)
        contexto  = {
            "anio": anio, "mes_nombre": mes_str,
            "clave_ent": clave_ent,
            "entidad": ENTIDADES.get(clave_ent, f"Entidad {clave_ent}"),
            "incidencia": incidencia,
        }
        print()
        imprimir_resultado(resultado, contexto)

    print("\n✅ Demo finalizada.")


if __name__ == "__main__":
    main()