# Auditoría y Modelado Estadístico de la Incidencia Delictiva en México

Este proyecto aplica un ciclo completo de minería de datos para descubrir y analizar patrones delictivos a partir de los datos del **Secretariado Ejecutivo del Sistema Nacional de Seguridad Pública (SESNSP)**. El enfoque integra un riguroso análisis exploratorio, algoritmos de aprendizaje estadístico y un diseño de arquitectura de software bajo los estándares de la Programación Orientada a Objetos (POO).

##  Objetivo General
Auditar la distribución, concentración y evolución de las carpetas de investigación a nivel estatal. A través de la reducción de dimensionalidad y modelos de Machine Learning, el sistema segmenta perfiles de riesgo geográfico (Clustering) y evalúa las deficiencias estructurales o burocráticas en el registro penal (Fuga de Datos).

##  Arquitectura del Sistema
El código fuente ha sido diseñado con alta cohesión y bajo acoplamiento, implementando múltiples patrones de diseño para asegurar la reproducibilidad y modularidad:

- **`data_repository.py`**: Implementa el patrón *Repository* para encapsular el acceso a disco (archivos CSV o Parquet) detrás de una interfaz uniforme.
- **`pipeline.py`**: Combina los patrones *Pipeline* y *Strategy* para orquestar secuencialmente las etapas de procesamiento e intercambiar reglas de limpieza dinámicamente.
- **`model_factory.py`**: Utiliza el patrón *Factory Method* mediante un registro central estático que permite instanciar distintos modelos matemáticos (Random Forest, SVM, Regresión Logística) de manera intercambiable.
- **`evaluator.py`**: Aísla la lógica de las métricas (MAE, MSE, F1-Score, AUC-ROC) y la generación de gráficos.

##  Fases del Proyecto
1. **Análisis Exploratorio de Datos (EDA)**: Análisis temporal y geográfico confirmando la fuerte incidencia del delito de Robo (36.73% del total nacional) y evaluando la estacionalidad criminal.
2. **Aprendizaje No Supervisado**: Uso de PCA para retener el 64.20% de la varianza en 2 componentes y aplicación de K-Means (k=4) para descubrir perfiles estatales cohesivos (Alta Incidencia Patrimonial, Alta Violencia, etc.).
3. **Aprendizaje Supervisado**: Ensamble de *Random Forest* optimizado vía *GridSearchCV*. Este modelo sirve además para analizar el fenómeno de "Data Leakage" subyacente en la burocracia judicial al mapear tipos de delito con bienes jurídicos.

##  Estructura del Repositorio
```
Proyecto_Mineria_Datos/
├── data/
│   ├── raw/            
│   └── processed/      
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modelo_supervisado.ipynb
│   ├── 03_clustering.ipynb
│   └── 04_demo_reutilizacion.ipynb
├── src/
│   ├── __init__.py
│   ├── data_repository.py
│   ├── pipeline.py       
│   ├── model_factory.py  
│   └── evaluator.py
├── models/
│   └── modelo_final.joblib
├── reports/
│   ├── reporte_final.pdf
│   └── slides.pdf
├── diagrams/
│   └── arquitectura_uml.png
├── docs/               
├── _quarto.yml
├── styles.css
├── index.qmd
├── eda.qmd
├── modelo.qmd
├── clustering.qmd
├── instalacion.qmd
├── README.md
├── requirements.txt
└── .gitignore
```

##  Reutilización del Modelo
El modelo entrenado está completamente desacoplado y exportado mediante la librería `joblib`. Es posible cargar los binarios para inferencias instantáneas sobre nuevos vectores numéricos sintéticos sin necesidad de ejecutar las celdas de entrenamiento.

## 🛠️ Guía de Instalación y Ejecución Rápida

Para clonar el proyecto, configurar el entorno virtual y reproducir los experimentos, ejecute los siguientes comandos en su terminal:

```bash
# 1. Clonación e instalación de dependencias
git clone [https://github.com/Sinnombre0/Proyecto_Final_Mineria_Datos.git](https://github.com/Sinnombre0/Proyecto_Final_Mineria_Datos.git)
cd Proyecto_Final_Mineria_Datos
python -m venv .venv
source .venv/bin/activate  # En Windows use: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Ejecución de la demostración interactiva en vivo
python predict.py
```

## Link al sitio para mayo información y guias

https://sinnombre0.github.io/Proyecto_Final_Mineria_Datos/ 

## Link de la presentacion

https://canva.link/4w2s93cql13gujp


