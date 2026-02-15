# scripts/02_descomposicion.py

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# =========================
# RUTAS (según tu estructura)
# =========================
CSV_PATH = "data/raw/raw_sales.csv"
OUT_SERIE = "imagenes/serie_original.png"
OUT_DECOMP = "imagenes/decomposition.png"

# =========================
# 1) CARGAR Y PREPARAR DATA
# =========================
df = pd.read_csv(CSV_PATH)

# Ajusta estos nombres si tu CSV usa otros
df["datesold"] = pd.to_datetime(df["datesold"], errors="coerce")

# Filtrar: houses + 3 bedrooms
df = df[(df["propertyType"].str.lower() == "house") & (df["bedrooms"] == 3)].copy()
df = df.dropna(subset=["datesold", "price"]).sort_values("datesold")

# =========================
# 2) SERIE MENSUAL (mean)
# =========================
serie = (
    df.set_index("datesold")["price"]
      .resample("MS")   # Monthly Start
      .mean()
      .asfreq("MS")
)

# Si hay meses sin ventas, interpolamos para no romper la descomposición
serie = serie.interpolate(limit_direction="both")

# =========================
# 3) PLOT SERIE ORIGINAL
# =========================
plt.figure(figsize=(10, 4))
plt.plot(serie)
plt.title("Precio medio mensual - Houses 3 bedrooms")
plt.xlabel("Fecha")
plt.ylabel("Precio medio")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(OUT_SERIE, dpi=200)
plt.close()

# =========================
# 4) DESCOMPOSICIÓN
# =========================
result = seasonal_decompose(serie, model="multiplicative", period=12)

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

plt.tight_layout()
plt.savefig(OUT_DECOMP, dpi=200)
plt.close()

print("✅ Listo:")
print(f" - {OUT_SERIE}")
print(f" - {OUT_DECOMP}")
