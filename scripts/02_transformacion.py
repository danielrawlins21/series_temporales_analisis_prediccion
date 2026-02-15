# scripts/03_transformacion_y_comparacion_decompose.py

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# =========================
# RUTAS (según tu estructura)
# =========================
CSV_PATH = "data/processed/serie_mensual.csv"

OUT_SERIE_TRANSF = "imagenes/serie_transformada.png"
OUT_DECOMP_TRANSF = "imagenes/decomposition_transformada.png"

# =========================
# FUNCIONES (las que pediste)
# =========================
def transform_log_diff_shift(serie: pd.Series, eps: float = 1e-6):
    """
    log -> diff(1) -> shift para no-negatividad.
    Devuelve: (serie_transformada, params)
    """
    s = serie.copy()

    # Asegurar positividad antes del log (en precios normalmente no hace falta, pero lo dejamos robusto)
    if (s <= 0).any():
        adj = abs(float(s.min())) + eps
        s = s + adj
    else:
        adj = 0.0

    s_log = np.log(s)
    s_log_diff = s_log.diff(1).dropna()

    # Shift para evitar negativos (como pediste)
    min_diff = float(s_log_diff.min())
    shift = (-min_diff) + eps if min_diff <= 0 else 0.0
    s_ready = s_log_diff + shift

    params = {
        "eps": eps,
        "add_to_make_positive_before_log": float(adj),
        "shift_after_diff": float(shift),
    }
    return s_ready, params


def plot_decomposition(result, outpath: str, title: str):
    plt.figure(figsize=(12, 8))

    plt.subplot(4, 1, 1)
    plt.plot(result.observed, label="Observado")
    plt.legend(loc="upper left")

    plt.subplot(4, 1, 2)
    plt.plot(result.trend, label="Tendencia")
    plt.legend(loc="upper left")

    plt.subplot(4, 1, 3)
    plt.plot(result.seasonal, label="Estacionalidad")
    plt.legend(loc="upper left")

    plt.subplot(4, 1, 4)
    plt.plot(result.resid, label="Residual")
    plt.legend(loc="upper left")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# =========================
# 1) CARGAR Y CREAR SERIE ORIGINAL
# =========================
df = pd.read_csv(CSV_PATH)

# Ajusta estos nombres si tu CSV usa otros
df['datesold'] = pd.to_datetime(df['datesold'], errors="coerce")

serie = (df
          .set_index('datesold')['price']
)


# =========================
# 3) TRANSFORMACIÓN (log -> diff -> shift)
# =========================
serie_transf, params = transform_log_diff_shift(serie)

# Guardar plot simple de la serie transformada
plt.figure(figsize=(10, 4))
plt.plot(serie_transf)
plt.title("Serie transformada: log + diff(1) + shift")
plt.xlabel("Fecha")
plt.ylabel("Valor transformado")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(OUT_SERIE_TRANSF, dpi=200)
plt.close()

# =========================
# 4) DECOMPOSE TRANSFORMADA (additive)
# =========================
# Nota: tras log+diff, suele tener más sentido aditivo (oscilaciones alrededor de un nivel)
decomp_transf = seasonal_decompose(serie_transf, model="additive", period=12)
plot_decomposition(
    decomp_transf,
    OUT_DECOMP_TRANSF,
    "Descomposición - Serie transformada (aditiva)"
)

# =========================
# 5) GUARDAR SERIE TRANSFORMADA
# =========================

os.makedirs("data/processed", exist_ok=True)

# Guardar serie transformada
serie_transf.to_csv("data/processed/serie_transformada.csv")

# Guardar parámetros de transformación
with open("data/processed/transform_params.json", "w") as f:
    json.dump(params, f, indent=4)

print("📁 Archivos adicionales guardados:")
print(" - data/processed/serie_transformada.csv")
print(" - data/processed/transform_params.json")

print("✅ Listo. Figuras guardadas:")
print(f" - {OUT_SERIE_TRANSF}")
print(f" - {OUT_DECOMP_TRANSF}")
print("Parámetros de transformación:", params)
