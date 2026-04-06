# PtMeOH Streamlit Sizing Tool — Version 1

Aplicación modular en Streamlit para el dimensionamiento preliminar y evaluación tecnoeconómica de una planta PtMeOH basada en H2 verde y CO2 capturado.

## Ejecución
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Estructura
- `app.py`: punto de entrada Streamlit
- `application/`: orquestación del caso
- `domain/`: modelos de ingeniería, simulación, KPIs, optimización, sensibilidad
- `infrastructure/`: carga de configuraciones, descubrimiento y carga de modelos surrogate
- `presentation/`: utilidades gráficas
- `models/packages/`: carpetas de cada paquete de surrogate model

## Convención de paquetes de modelos
Cada modelo vive en:
`models/packages/<Model_Name>/`

Archivos esperados dentro de cada carpeta:
- `<Model_Name>.joblib`
- `<Model_Name>.py`
- `<Model_Name>.txt`
- `metadata.json`
- `model_parameters.xlsx`
- `consolidated_model_report.pdf`
- `training_validation_report.pdf`

La app detecta automáticamente las carpetas presentes, valida los archivos esperados y pide confirmación explícita del usuario antes de ejecutar.

## Librerías de modelos soportadas
- `variable_h2_constant_co2`
- `variable_h2_variable_co2`

## Notas
- Si falta un `.joblib`, la app mantiene la ejecución usando un surrogate determinístico de respaldo y muestra advertencias.
- La simulación V1 es secuencial, determinística y trazable; la optimización usa grid search.
