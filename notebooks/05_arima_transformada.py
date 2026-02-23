import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX


# -----------------------------
# Utils
# -----------------------------
def ensure_dirs():
    os.makedirs("imagenes", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)


def load_series(path_csv: str) -> pd.Series:
    """
    Carga una serie desde CSV guardado con índice temporal en la primera columna.
    Acepta:
      - CSV con 1 columna de valores (Series)
      - CSV con 1 columna llamada 'price' u otra
    Devuelve pd.Series con índice datetime.
    """
    df = pd.read_csv(path_csv, index_col=0, parse_dates=True)

    # Si viene como DataFrame 1-columna, lo convertimos a Series
    if isinstance(df, pd.DataFrame) and df.shape[1] == 1:
        s = df.iloc[:, 0]
    elif isinstance(df, pd.DataFrame) and "price" in df.columns:
        s = df["price"]
    else:
        # Caso raro: intentar convertir todo a Series si es posible
        if hasattr(df, "squeeze"):
            s = df.squeeze()
        else:
            raise ValueError(f"No se pudo interpretar la serie en {path_csv}")

    # Intentar fijar frecuencia mensual si procede (no rompe si no se puede)
    try:
        s = s.asfreq("MS")
    except Exception:
        pass

    # Evitar nulos
    s = s.dropna()
    return s


def save_plot(fig, outpath: str):
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def write_text(path: str, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# -----------------------------
# 1) Cargar train/test
# -----------------------------
def main():
    ensure_dirs()

    train = load_series("data/processed/train.csv")
    test = load_series("data/processed/test.csv")

    s = 12  # estacionalidad mensual

    # -----------------------------
    # 2) Plot serie train (transformada)
    # -----------------------------
    fig = plt.figure(figsize=(10, 4))
    plt.plot(train, label="Train")
    plt.title("Serie transformada (train)")
    plt.legend()
    save_plot(fig, "imagenes/serie_transformada_train.png")

    # -----------------------------
    # 3) ACF / PACF (train)
    # -----------------------------
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    plot_acf(train, lags=48, alpha=0.05, ax=ax[0])
    ax[0].set_title("ACF - Serie transformada (train)")
    plot_pacf(train, lags=48, alpha=0.05, ax=ax[1], method="ywm")
    ax[1].set_title("PACF - Serie transformada (train)")
    save_plot(fig, "imagenes/acf_pacf_transformada.png")

    # También guardamos por separado, por si lo quieres en LaTeX como 2 figuras:
    fig = plt.figure(figsize=(10, 3.5))
    plot_acf(train, lags=48, alpha=0.05)
    plt.title("ACF - Serie transformada (train)")
    save_plot(fig, "imagenes/acf_transformada.png")

    fig = plt.figure(figsize=(10, 3.5))
    plot_pacf(train, lags=48, alpha=0.05, method="ywm")
    plt.title("PACF - Serie transformada (train)")
    save_plot(fig, "imagenes/pacf_transformada.png")

    # -----------------------------
    # 4) SARIMA candidato (baseline)
    # -----------------------------
    # IMPORTANTE:
    # Como tu serie ya está transformada (log+diff+shift) normalmente d=0.
    # Probamos un candidato razonable (como en el ejemplo de Córdoba de Clase 2).
    # Si tu ACF muestra pico en lag 12, un punto de partida común es Q=1 con D=0 o D=1
    #
    # Aquí dejamos un candidato "seguro" para empezar:
    #   order=(1,0,1), seasonal_order=(0,0,1,12)
    #
    # Si quieres, luego hacemos un grid corto con 6-12 modelos.
    order = (1, 0, 1)
    seasonal_order = (0, 0, 1, s)

    warnings.filterwarnings("ignore")
    model = SARIMAX(
        endog=train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    res = model.fit(disp=False)
    warnings.filterwarnings("default")

    # Guardar summary como texto
    summary_txt = res.summary().as_text()
    write_text("data/processed/sarima_summary_transformada.txt", summary_txt)

    # -----------------------------
    # 5) Diagnóstico de residuos (figura)
    # -----------------------------
    # statsmodels trae un plot_diagnostics muy útil
    fig = res.plot_diagnostics(figsize=(10, 8))
    save_plot(fig, "imagenes/residuos_transformado.png")

    # -----------------------------
    # 6) Predicción sobre test
    # -----------------------------
    pred = res.get_forecast(steps=len(test)).predicted_mean
    pred.index = test.index  # alinear

    mae = mean_absolute_error(test, pred)
    rmse = np.sqrt(mean_squared_error(test, pred))

    metrics = (
        f"SARIMA (transformada)\n"
        f"order={order}, seasonal_order={seasonal_order}\n"
        f"MAE={mae:.6f}\n"
        f"RMSE={rmse:.6f}\n"
    )
    print(metrics)
    write_text("data/processed/metrics_sarima_transformada.txt", metrics)

    # -----------------------------
    # 7) Plot predicción vs test
    # -----------------------------
    fig = plt.figure(figsize=(10, 4.5))
    plt.plot(train, label="Train")
    plt.plot(test, label="Test")
    plt.plot(pred, label="Predicción SARIMA")
    plt.title("SARIMA - Predicción sobre serie transformada")
    plt.legend()
    save_plot(fig, "imagenes/forecast_sarima_transformada.png")


if __name__ == "__main__":
    main()