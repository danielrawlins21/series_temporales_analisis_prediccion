# Comparación de modelos de series temporales

## Información académica
- Autor: Daniel Rawlins Poveda
- Materia: Minería de datos y modelización predictiva

Este script compara **SARIMAX (ARIMA estacional)**, **Suavizado Exponencial Simple (SES)** y **Holt-Winters** sobre una serie temporal transformada con logaritmo.

El flujo principal es:
1. Cargar datos desde CSV o Excel.
2. Ordenar por fecha y fijar índice temporal.
3. Aplicar `log(y)` (solo si todos los valores son > 0).
4. Separar en entrenamiento y prueba.
5. Entrenar los 3 modelos en escala log.
6. Predecir en escala log y destransformar a escala original (`exp`).
7. Calcular métricas y guardar resultados.
8. Generar gráficos comparativos.

## Archivo
- `compare_models.py`

## Requisitos
Instala estas dependencias:

```bash
pip install numpy pandas matplotlib scikit-learn statsmodels openpyxl
```

`openpyxl` se usa para leer archivos Excel (`.xlsx`).

## Uso

```bash
python compare_models.py \
  --input "ruta/datos.csv" \
  --date_col "fecha" \
  --value_col "valor"
```

### Parámetros
- `--input` (obligatorio): ruta al archivo de entrada (`.csv`, `.xlsx`, `.xls`).
- `--date_col` (obligatorio): nombre de la columna de fechas.
- `--value_col` (obligatorio): nombre de la variable objetivo.
- `--freq` (opcional, default: `MS`): frecuencia pandas (`MS`, `M`, `D`, etc.).
- `--test_size` (opcional, default: `12`): horizonte de test.
- `--out_csv` (opcional, default: `model_comparison.csv`): CSV de resultados.
- `--fig_dir` (opcional, default: `imagenes`): carpeta de figuras.

Ejemplo completo:

```bash
python compare_models.py \
  --input "datos_serie.xlsx" \
  --date_col "Fecha" \
  --value_col "Demanda" \
  --freq "MS" \
  --test_size 12 \
  --out_csv "resultados/model_comparison.csv" \
  --fig_dir "resultados/imagenes"
```

## Salidas
El script genera:
- Un CSV con la comparación de modelos (`out_csv`).
- Dos figuras PNG en `fig_dir`:
  - `predicciones_comparacion_log.png`
  - `predicciones_comparacion_original.png`

## Métricas reportadas
Para cada modelo:
- En escala log: `MAE`, `RMSE`, `MAPE`
- En escala original: `MAE_orig`, `RMSE_orig`, `MAPE_orig`
- Criterios de información (si están disponibles): `AIC`, `BIC`

## Consideraciones importantes
- La transformación log exige **valores estrictamente positivos**. Si hay valores `<= 0`, el script falla con `ValueError`.
- El tamaño de test debe cumplir: `0 < test_size < longitud_serie`.
- El script usa por defecto una configuración fija de modelos en `ModelConfig`:
  - SARIMAX: `order=(0,1,1)`, `seasonal_order=(1,0,1,12)`
  - Holt-Winters: `seasonal="add"`, `seasonal_periods=12`

Si quieres ajustar hiperparámetros, modifica `ModelConfig` dentro de `compare_models.py`.

## Estructura esperada de datos
Ejemplo mínimo:

| fecha       | valor |
|-------------|-------|
| 2020-01-01  | 120.0 |
| 2020-02-01  | 132.5 |
| 2020-03-01  | 128.7 |

La columna de fecha debe ser parseable por `pandas.to_datetime`.

