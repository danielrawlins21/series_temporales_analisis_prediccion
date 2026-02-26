# scripts/compare_models_log_only.py

import argparse
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error

from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


# -----------------------------
# Config
# -----------------------------
@dataclass
class ModelConfig:
    # SARIMAX "ganador" (ajusta estos parámetros con tu modelo final)
    arima_order: Tuple[int, int, int] = (0, 1, 1)
    seasonal_order: Tuple[int, int, int, int] = (1, 0, 1, 12)

    # Holt-Winters "ganador"
    hw_trend: Optional[str] = None         # "add" | "mul" | None
    hw_seasonal: Optional[str] = "add"       # "add" | "mul" | None
    hw_seasonal_periods: int = 12

    # Test horizon
    test_size: int = 12

    # Frequency (optional): "MS" monthly start, "M" month end, "D", etc.
    freq: Optional[str] = None


# -----------------------------
# Utils
# -----------------------------
def safe_log(series: pd.Series) -> pd.Series:
    """Aplica logaritmo asegurando positividad."""
    if (series <= 0).any():
        raise ValueError(
            "La serie contiene valores <= 0. "
            "No se puede aplicar log sin una transformación adicional (offset). "
            "Como pediste SOLO log, corrige los datos o filtra valores."
        )
    return np.log(series)


def train_test_split_ts(y: pd.Series, test_size: int) -> Tuple[pd.Series, pd.Series]:
    if test_size <= 0 or test_size >= len(y):
        raise ValueError("test_size debe ser > 0 y < longitud de la serie.")
    return y.iloc[:-test_size], y.iloc[-test_size:]


def metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def get_aic_bic(model_fit: Any) -> Dict[str, Optional[float]]:
    return {"AIC": getattr(model_fit, "aic", None), "BIC": getattr(model_fit, "bic", None)}


def load_series(input_path: str, date_col: str, value_col: str, freq: Optional[str]) -> pd.Series:
    if input_path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(input_path)
    else:
        df = pd.read_csv(input_path)

    if date_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"Columnas no encontradas. Disponibles: {list(df.columns)}")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)

    y = df[value_col].astype(float)

    if freq:
        y = y.asfreq(freq)

    y = y.dropna()
    return y


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# -----------------------------
# Models
# -----------------------------
def fit_forecast_sarimax(y_train_log: pd.Series, steps: int, cfg: ModelConfig):
    model = SARIMAX(
        endog=y_train_log,
        order=cfg.arima_order,
        seasonal_order=cfg.seasonal_order,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    res = model.fit(disp=False)
    fc = res.get_forecast(steps=steps).predicted_mean
    return res, fc


def fit_forecast_ses(y_train_log: pd.Series, steps: int):
    res = SimpleExpSmoothing(y_train_log, initialization_method="estimated").fit(optimized=True)
    fc = res.forecast(steps)
    return res, fc


def fit_forecast_hw(y_train_log: pd.Series, steps: int, cfg: ModelConfig):
    res = ExponentialSmoothing(
        y_train_log,
        trend=cfg.hw_trend,
        seasonal=cfg.hw_seasonal,
        seasonal_periods=cfg.hw_seasonal_periods,
        initialization_method="estimated"
    ).fit(optimized=True)
    fc = res.forecast(steps)
    return res, fc


# -----------------------------
# Plotting
# -----------------------------
def plot_predictions(
    y_full: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
    pred_arima: pd.Series,
    pred_ses: pd.Series,
    pred_hw: pd.Series,
    title: str,
    out_path: str
) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(y_full.index, y_full.values, label="Serie (completa)")
    plt.plot(y_train.index, y_train.values, label="Train")
    plt.plot(y_test.index, y_test.values, label="Test (real)")

    plt.plot(pred_arima.index, pred_arima.values, label="ARIMA/SARIMAX (pred)")
    plt.plot(pred_ses.index, pred_ses.values, label="SES (pred)")
    plt.plot(pred_hw.index, pred_hw.values, label="Holt-Winters (pred)")

    plt.title(title)
    plt.xlabel("Fecha")
    plt.ylabel("Valor")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Comparación ARIMA vs SES vs Holt-Winters sobre serie log-transformada.")
    parser.add_argument("--input", required=True, help="Ruta al CSV o Excel")
    parser.add_argument("--date_col", required=True, help="Nombre de la columna fecha")
    parser.add_argument("--value_col", required=True, help="Nombre de la columna objetivo")

    parser.add_argument("--freq", default='MS', help="Frecuencia pandas (opcional): MS, M, D, etc.")
    parser.add_argument("--test_size", type=int, default=12, help="Tamaño del conjunto de test (horizonte)")

    parser.add_argument("--out_csv", default="model_comparison.csv", help="Salida CSV con resultados")
    parser.add_argument("--fig_dir", default="imagenes", help="Carpeta donde guardar figuras")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    cfg = ModelConfig(test_size=args.test_size, freq=args.freq)

    ensure_dir(os.path.dirname(args.out_csv) or ".")
    ensure_dir(args.fig_dir)

    # 1) Load original series
    y = load_series(args.input, args.date_col, args.value_col, cfg.freq)

    # 2) Log transform only
    y_log = safe_log(y)

    # 3) Split train/test
    y_train_log, y_test_log = train_test_split_ts(y_log, cfg.test_size)
    y_train, y_test = train_test_split_ts(y, cfg.test_size)
    steps = len(y_test_log)

    # --- SARIMAX / ARIMA ---
    arima_fit, arima_fc_log = fit_forecast_sarimax(y_train_log, steps, cfg)
    arima_fc_log.index = y_test_log.index
    arima_fc = np.exp(arima_fc_log)
    arima_fc.index = y_test.index

    # --- SES ---
    ses_fit, ses_fc_log = fit_forecast_ses(y_train_log, steps)
    ses_fc_log.index = y_test_log.index
    ses_fc = np.exp(ses_fc_log)
    ses_fc.index = y_test.index

    # --- Holt-Winters ---
    hw_fit, hw_fc_log = fit_forecast_hw(y_train_log, steps, cfg)
    hw_fc_log.index = y_test_log.index
    hw_fc = np.exp(hw_fc_log)
    hw_fc.index = y_test.index

    # 4) Metrics table
    results = []

    row = {"Modelo": "ARIMA/SARIMAX"}
    row.update(metrics(y_test_log, arima_fc_log))
    row.update({f"{k}_orig": v for k, v in metrics(y_test, arima_fc).items()})
    row.update(get_aic_bic(arima_fit))
    results.append(row)

    row = {"Modelo": "SES"}
    row.update(metrics(y_test_log, ses_fc_log))
    row.update({f"{k}_orig": v for k, v in metrics(y_test, ses_fc).items()})
    row.update(get_aic_bic(ses_fit))
    results.append(row)

    row = {"Modelo": "Holt-Winters"}
    row.update(metrics(y_test_log, hw_fc_log))
    row.update({f"{k}_orig": v for k, v in metrics(y_test, hw_fc).items()})
    row.update(get_aic_bic(hw_fit))
    results.append(row)

    df_res = pd.DataFrame(results)

    cols = [
        "Modelo",
        "MAE", "RMSE", "MAPE",
        "MAE_orig", "RMSE_orig", "MAPE_orig",
        "AIC", "BIC"
    ]
    df_res = df_res[[c for c in cols if c in df_res.columns]]

    print("\n=== Comparación de modelos (entrenados en log(y)) ===")
    print(df_res.sort_values(by=["RMSE_orig", "MAE_orig"], ascending=True))

    df_res.to_csv(args.out_csv, index=False)
    print(f"\nResultados guardados en: {args.out_csv}")

    # 5) Plots
    plot_predictions(
        y_full=y_log,
        y_train=y_train_log,
        y_test=y_test_log,
        pred_arima=arima_fc_log,
        pred_ses=ses_fc_log,
        pred_hw=hw_fc_log,
        title="Comparación de predicciones (escala log)",
        out_path=os.path.join(args.fig_dir, "predicciones_comparacion_log.png")
    )

    plot_predictions(
        y_full=y,
        y_train=y_train,
        y_test=y_test,
        pred_arima=arima_fc,
        pred_ses=ses_fc,
        pred_hw=hw_fc,
        title="Comparación de predicciones (escala original)",
        out_path=os.path.join(args.fig_dir, "predicciones_comparacion_original.png")
    )

    print(f"Figuras guardadas en: {args.fig_dir}/predicciones_comparacion_log.png")
    print(f"Figuras guardadas en: {args.fig_dir}/predicciones_comparacion_original.png")


if __name__ == "__main__":
    main()