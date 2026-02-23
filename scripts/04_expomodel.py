import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing

# =========================
# 1) Serie original (mensual)
# =========================
# serie: pd.Series con índice datetime (MS) y valores en escala original
serie = pd.read_csv(
    "data/processed/serie_mensual.csv",
    index_col=0,
    parse_dates=True
).squeeze()

serie = serie.asfreq("MS")
serie = serie.interpolate(limit_direction="both")  # por si hay meses vacíos

# =========================
# 2) Train/Test split (últimos 12 meses)
# =========================
s = 12
train = serie.iloc[:-s]
test  = serie.iloc[-s:]

# =========================
# 3) SES
# =========================
ses = SimpleExpSmoothing(train, initialization_method="estimated").fit()
pred_ses = ses.forecast(s)

mae_ses = mean_absolute_error(test, pred_ses)
rmse_ses = np.sqrt(mean_squared_error(test, pred_ses))

print("Rendimiento del modelo SES - Dataset Original:")
print("MAE:", mae_ses)
print("RMSE:", rmse_ses)

plt.figure(figsize=(10, 5))
plt.plot(train, label="Train")
plt.plot(test, label="Test")
plt.plot(pred_ses, label="Predicción SES")
plt.title("Suavizado Exponencial Simple - Serie Original")
plt.legend()
plt.tight_layout()
plt.savefig("imagenes/SES_original.png")


# =========================
# 4) Holt-Winters (elige un esquema razonable)
# =========================
# Recomendación inicial: trend=None y seasonal='add' (o 'mul' si la estacionalidad crece con el nivel)
hw = ExponentialSmoothing(
    train,
    trend=None,
    seasonal="add",
    seasonal_periods=12,
    initialization_method="estimated"
).fit()

pred_hw = hw.forecast(s)

mae_hw = mean_absolute_error(test, pred_hw)
rmse_hw = np.sqrt(mean_squared_error(test, pred_hw))

print("\nRendimiento del modelo Holt-Winters - Dataset Original:")
print("MAE:", mae_hw)
print("RMSE:", rmse_hw)

plt.figure(figsize=(10, 5))
plt.plot(train, label="Train")
plt.plot(test, label="Test")
plt.plot(pred_hw, label="Holt-Winters")
plt.title("Holt-Winters - Serie Original")
plt.legend()
plt.tight_layout()
plt.savefig('imagenes/HW_original.png')