import pandas as pd
import os

def train_test_split_serie(serie, test_size=12):
    """
    Divide la serie temporal en train y test.
    Por defecto reserva los últimos 12 meses (serie mensual estacional).
    """
    train = serie.iloc[:-test_size]
    test = serie.iloc[-test_size:]
    return train, test


if __name__ == "__main__":
    
    # Cargar serie transformada
    serie_modelo = pd.read_csv(
        "data/processed/serie_transformada.csv",
        index_col=0,
        parse_dates=True
    ).squeeze()

    # División
    train, test = train_test_split_serie(serie_modelo, test_size=12)

    # Guardar resultados
    train.to_csv("data/processed/train.csv")
    test.to_csv("data/processed/test.csv")

    print("Train size:", len(train))
    print("Test size:", len(test))
