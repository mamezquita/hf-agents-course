"""Entrena un modelo para estimar si el precio subirá en el próximo día."""

import os
import time
from pathlib import Path

import joblib
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier

API_URL = "https://rest.coincap.io/v3/assets"
MODEL_PATH = Path(__file__).resolve().parent / "model" / "modelo_ml.pkl"
FEATURES = ["retorno_1d", "retorno_3d", "retorno_7d", "volatilidad_7d", "distancia_sma_7d"]
NUMERO_ACTIVOS = 4
DIAS_HISTORICOS = 120
INTERVALO = "d1"
PAUSA_ENTRE_PETICIONES = 16
ULTIMA_PETICION = 0.0


def obtener_api_key() -> str:
    api_key = os.getenv("COINCAP_API_KEY")
    if not api_key:
        raise RuntimeError("Falta COINCAP_API_KEY. Configúrala antes de ejecutar.")
    return api_key


def consultar(url: str, params: dict, api_key: str) -> dict:
    global ULTIMA_PETICION
    try:
        espera = PAUSA_ENTRE_PETICIONES - (time.monotonic() - ULTIMA_PETICION)
        if espera > 0:
            time.sleep(espera)
        for intento in range(3):
            respuesta = requests.get(url, params=params, headers={"Authorization": f"Bearer {api_key}"}, timeout=20)
            ULTIMA_PETICION = time.monotonic()
            if respuesta.status_code != 429:
                respuesta.raise_for_status()
                return respuesta.json()
            reintento = max(int(respuesta.headers.get("Retry-After", "60")), 60)
            print(f"Límite de API alcanzado; esperando {reintento} segundos...")
            time.sleep(reintento)
        respuesta.raise_for_status()
    except (requests.RequestException, ValueError) as error:
        raise RuntimeError(f"No fue posible consultar CoinCap: {error}") from error


def obtener_historial(asset_id: str, api_key: str) -> pd.DataFrame:
    fin = int(time.time() * 1000)
    inicio = fin - DIAS_HISTORICOS * 86_400_000
    datos = consultar(f"{API_URL}/{asset_id}/history", {"interval": INTERVALO, "start": inicio, "end": fin}, api_key)["data"]
    df = pd.DataFrame(datos)
    if df.empty:
        return df
    df["priceUsd"] = pd.to_numeric(df["priceUsd"], errors="coerce")
    return df.dropna(subset=["priceUsd"]).sort_values("time").reset_index(drop=True)


def crear_muestras(historial: pd.DataFrame) -> pd.DataFrame:
    """Las señales usan solo precios conocidos en cada día; no hay fuga de datos."""
    df = historial[["time", "priceUsd"]].copy()
    df["retorno_1d"] = df["priceUsd"].pct_change(1)
    df["retorno_3d"] = df["priceUsd"].pct_change(3)
    df["retorno_7d"] = df["priceUsd"].pct_change(7)
    df["volatilidad_7d"] = df["retorno_1d"].rolling(7).std()
    df["distancia_sma_7d"] = df["priceUsd"] / df["priceUsd"].rolling(7).mean() - 1
    df["precio_siguiente"] = df["priceUsd"].shift(-1)
    df = df.dropna(subset=FEATURES + ["precio_siguiente"])
    df["objetivo"] = (df["precio_siguiente"] > df["priceUsd"]).astype(int)
    return df


def main() -> None:
    api_key = obtener_api_key()
    activos = consultar(API_URL, {"limit": NUMERO_ACTIVOS}, api_key)["data"]
    muestras = []
    print("Descargando historial diario para entrenar (aprox. un minuto)...")
    for activo in activos:
        datos = crear_muestras(obtener_historial(activo["id"], api_key))
        if not datos.empty:
            muestras.append(datos.assign(asset_id=activo["id"]))

    if not muestras:
        raise RuntimeError("CoinCap no devolvió suficiente historial para entrenar.")
    df = pd.concat(muestras, ignore_index=True).sort_values("time")
    corte = df["time"].quantile(0.80)
    entrenamiento, prueba = df[df["time"] <= corte], df[df["time"] > corte]
    if entrenamiento["objetivo"].nunique() < 2 or prueba.empty:
        raise RuntimeError("No hay suficientes datos temporales para evaluar el modelo.")

    modelo = RandomForestClassifier(n_estimators=300, min_samples_leaf=4, class_weight="balanced", random_state=42, n_jobs=-1)
    modelo.fit(entrenamiento[FEATURES], entrenamiento["objetivo"])
    precision = modelo.score(prueba[FEATURES], prueba["objetivo"])
    referencia = max(prueba["objetivo"].mean(), 1 - prueba["objetivo"].mean())
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": modelo, "features": FEATURES}, MODEL_PATH)
    print(f"Modelo guardado en: {MODEL_PATH}")
    print(f"Muestras: {len(df)} | Entrenamiento: {len(entrenamiento)} | Prueba futura: {len(prueba)}")
    print(f"Exactitud temporal: {precision:.2%} | Referencia mayoritaria: {referencia:.2%}")


if __name__ == "__main__":
    main()
