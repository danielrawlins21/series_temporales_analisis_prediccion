# scripts/05_arima_ganador.py
# ------------------------------------------------------------
# ARIMA GANADOR: ARIMA(0,1,1) sobre la SERIE ORIGINAL (mensual)
# - Construye serie mensual (precio medio) desde raw_sales.csv
# - Split train/test (últimos 12 meses)
# - (Opcional) ajusta en escala log para estabilizar varianza y
#   evalúa MAE/RMSE en escala ORIGINAL (deshaciendo exp)
# - Genera ACF/PACF de la serie diferenciada (d=1)
# - Ajusta SARIMAX(0,1,1), guarda summary, diagnósticos y forecast
# ------------------------------------------------------------

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX


# =========================
# CONFIG
# =========================
CSV_PATH = "data/raw/raw_sales.csv"

# Serie objetivo (como vienes trabajando):
PROPERTY_TYPE = "house"
BEDROOMS = 3

FREQ = "MS"          # Monthly Start
S = 12               # Estacionalidad mensual (para split y lags)
TEST_SIZE = 12

USE_LOG = True       # Ajustar ARIMA sobre log(precio) y revertir a escala original

# Salidas
OUT_DIR_IMG = "imagenes"
OUT_DIR_PROC = "data/processed"

IMG_SERIE_ORIG = f"{OUT_DIR_IMG}/serie_original.png"
IMG_ACF_DIFF = f"{OUT_DIR_IMG}/acf_serie_diferenciada.png"
IMG_PACF_DIFF = f"{OUT_DIR_IMG}/pacf_serie_diferenciada.png"
IMG_FORECAST = f"{OUT_DIR_IMG}/forecast_arima_101.png"
IMG_DIAGNOSTICS = f"{OUT_DIR_IMG}/diagnosticos_residuos_arima_011.png"

TXT_SUMMARY = f"{OUT_DIR_PROC}/arima_011_summary.txt"
JSON_METRICS = f"{OUT_DIR_PROC}/arima_011_metrics.json"
CSV_PRED = f"{OUT_DIR_PROC}/arima_011_predicciones.csv"


# =========================
# HELPERS
# =========================
def ensure_dirs():
    os.makedirs(OUT_DIR_IMG, exist_ok=True)
    os.makedirs(OUT_DIR_PROC, exist_ok=True)


def load_monthly_series_from_raw(csv_path: str) -> pd.Series:
    """
    Carga raw_sales.csv y construye serie mensual del precio medio para:
      propertyType == 'house' y bedrooms == 3
    """
    df = pd.read_csv(csv_path)
    df["datesold"] = pd.to_datetime(df["datesold"], errors="coerce")

    df = df.dropna(subset=["datesold", "price", "propertyType", "bedrooms"]).copy()
    df["propertyType"] = df["propertyType"].astype(str).str.lower()

    df = df[(df["propertyType"] == PROPERTY_TYPE.lower()) & (df["bedrooms"] == BEDROOMS)].copy()
    df = df.sort_values("datesold")

    serie = (
        df.set_index("datesold")["price"]
          .resample(FREQ)
          .mean()
          .asfreq(FREQ)
    )

    # Rellenar huecos si existen (puedes cambiar a dropna() si prefieres)
    serie = serie.interpolate(limit_direction="both")

    serie.name = "price"
    return serie


def plot_and_save_series(serie: pd.Series, outpath: str, title: str):
    plt.figure(figsize=(10, 4))
    plt.plot(serie)
    plt.title(title)
    plt.xlabel("Fecha")
    plt.ylabel("Precio")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_and_save_acf_pacf(serie: pd.Series, out_acf: str, out_pacf: str, lags: int = 48):
    plt.figure(figsize=(10, 3.5))
    plot_acf(serie, lags=lags, alpha=0.05)
    plt.title("ACF - Serie diferenciada (d=1)")
    plt.tight_layout()
    plt.savefig(out_acf, dpi=150)
    plt.close()

    plt.figure(figsize=(10, 3.5))
    plot_pacf(serie, lags=lags, alpha=0.05, method="ywm")
    plt.title("PACF - Serie diferenciada (d=1)")
    plt.tight_layout()
    plt.savefig(out_pacf, dpi=150)
    plt.close()


def safe_log(x: pd.Series, eps: float = 1e-6) -> pd.Series:
    """
    Log seguro por si hay valores 0 (no debería pasar en precios).
    """
    x = x.copy()
    if (x <= 0).any():
        shift = abs(float(x.min())) + eps
        x = x + shift
    else:
        shift = 0.0
    return np.log(x), shift


# =========================
# MAIN
# =========================
def main():
    ensure_dirs()

    # 1) Serie original mensual
    serie = load_monthly_series_from_raw(CSV_PATH)

    # Guardar la serie original (por reproducibilidad)
    serie.to_csv(f"{OUT_DIR_PROC}/serie_original_mensual.csv")

    # Plot serie original
    plot_and_save_series(
        serie,
        IMG_SERIE_ORIG,
        title=f"Precio medio mensual - {PROPERTY_TYPE.title()} {BEDROOMS} bedrooms (Serie original)"
    )

    # 2) Split train/test (últimos 12 meses)
    train = serie.iloc[:-TEST_SIZE]
    test = serie.iloc[-TEST_SIZE:]

    train.to_csv(f"{OUT_DIR_PROC}/train_original.csv")
    test.to_csv(f"{OUT_DIR_PROC}/test_original.csv")

    # 3) ACF/PACF sobre serie diferenciada (d=1) en escala original (solo diagnóstico Box-Jenkins)
    train_diff = train.diff().dropna()
    plot_and_save_acf_pacf(train_diff, IMG_ACF_DIFF, IMG_PACF_DIFF, lags=48)

    # 4) Ajuste ARIMA(1,0,1)
    #    Opción recomendada: ajustar en log y evaluar en escala original.
    if USE_LOG:
        y_train, log_shift = safe_log(train)
        # Para poder revertir forecast a escala original: exp(y) - shift
        y_name = "log_price"
    else:
        y_train = train.copy()
        log_shift = 0.0
        y_name = "price"

    model = SARIMAX(
        y_train,
        order=(1, 0, 1),
        seasonal_order=(1, 0, 1, 12),
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
        freq=FREQ
    )

    res = model.fit(disp=False)

    # Guardar summary
    with open(TXT_SUMMARY, "w", encoding="utf-8") as f:
        f.write(res.summary().as_text())

    # Diagnósticos de residuos
    fig = res.plot_diagnostics(figsize=(10, 8))
    fig.tight_layout()
    fig.savefig(IMG_DIAGNOSTICS, dpi=150)
    plt.close(fig)

    # 5) Forecast sobre el horizonte del test
    forecast_res = res.get_forecast(steps=len(test))
    pred_mean = forecast_res.predicted_mean

    # Alinear índices
    pred_mean.index = test.index

    # Revertir a escala original si se usó log
    if USE_LOG:
        pred = np.exp(pred_mean) - log_shift
        # (Opcional) Evitar negativos por redondeos numéricos
        pred = pred.clip(lower=0)
    else:
        pred = pred_mean

    # 6) Métricas en escala ORIGINAL
    mae = mean_absolute_error(test, pred)
    rmse = np.sqrt(mean_squared_error(test, pred))

    metrics = {
        "model": "ARIMA(1,0,1)",
        "use_log": bool(USE_LOG),
        "log_shift": float(log_shift),
        "mae_original_scale": float(mae),
        "rmse_original_scale": float(rmse),
        "aic": float(res.aic),
        "bic": float(res.bic),
        "hqic": float(res.hqic),
        "nobs_train": int(len(train)),
        "nobs_test": int(len(test)),
    }

    with open(JSON_METRICS, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    # Guardar predicciones
    out_pred = pd.DataFrame({
        "test": test,
        "pred": pred
    })
    out_pred.to_csv(CSV_PRED)

    # 7) Plot forecast
    plt.figure(figsize=(10, 5))
    plt.plot(train, label="Train")
    plt.plot(test, label="Test")
    plt.plot(pred, label="Predicción ARIMA(0,1,1)")
    plt.title("Forecast - ARIMA(0,1,1) (escala original)")
    plt.xlabel("Fecha")
    plt.ylabel("Precio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMG_FORECAST, dpi=150)
    plt.close()

    # 8) Log en consola
    print("✅ ARIMA(0,1,1) ajustado y evaluado.")
    print(f"MAE (escala original):  {mae:.3f}")
    print(f"RMSE (escala original): {rmse:.3f}")
    print(f"Summary guardado en: {TXT_SUMMARY}")
    print(f"Métricas guardadas en: {JSON_METRICS}")
    print("Imágenes:")
    print(f" - {IMG_SERIE_ORIG}")
    print(f" - {IMG_ACF_DIFF}")
    print(f" - {IMG_PACF_DIFF}")
    print(f" - {IMG_DIAGNOSTICS}")
    print(f" - {IMG_FORECAST}")


if __name__ == "__main__":
    main()