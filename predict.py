"""Predice la dirección del próximo día usando datos históricos recientes."""

import os
import time
from pathlib import Path

import joblib
import pandas as pd
import requests

API_URL = "https://rest.coincap.io/v3/assets"
MODEL_PATH = Path(__file__).resolve().parent / "model" / "modelo_ml.pkl"
DIAS_HISTORICOS = 30
INTERVALO = "d1"
NUMERO_ACTIVOS = 4
PAUSA_ENTRE_PETICIONES = 16
ULTIMA_PETICION = 0.0


def obtener_api_key() -> str:
    api_key = os.getenv("COINCAP_API_KEY")
    if not api_key:
        raise RuntimeError("Falta COINCAP_API_KEY en esta sesión de PowerShell.")
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


def crear_senales(historial: pd.DataFrame) -> pd.DataFrame:
    df = historial[["time", "priceUsd"]].copy()
    df["retorno_1d"] = df["priceUsd"].pct_change(1)
    df["retorno_3d"] = df["priceUsd"].pct_change(3)
    df["retorno_7d"] = df["priceUsd"].pct_change(7)
    df["volatilidad_7d"] = df["retorno_1d"].rolling(7).std()
    df["distancia_sma_7d"] = df["priceUsd"] / df["priceUsd"].rolling(7).mean() - 1
    return df.dropna().tail(1)


def obtener_senales(asset_id: str, api_key: str) -> pd.DataFrame:
    fin = int(time.time() * 1000)
    inicio = fin - DIAS_HISTORICOS * 86_400_000
    datos = consultar(f"{API_URL}/{asset_id}/history", {"interval": INTERVALO, "start": inicio, "end": fin}, api_key)["data"]
    historial = pd.DataFrame(datos)
    if historial.empty:
        return historial
    historial["priceUsd"] = pd.to_numeric(historial["priceUsd"], errors="coerce")
    historial = historial.dropna(subset=["priceUsd"]).sort_values("time")
    return crear_senales(historial)


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No existe el modelo. Ejecuta primero train.py.")
    paquete = joblib.load(MODEL_PATH)
    if not isinstance(paquete, dict) or "model" not in paquete:
        raise RuntimeError("El modelo es antiguo. Vuelve a entrenarlo con python train.py.")

    api_key = obtener_api_key()
    modelo, features = paquete["model"], paquete["features"]
    activos = consultar(API_URL, {"limit": NUMERO_ACTIVOS}, api_key)["data"]
    resultados = []
    print("Consultando datos recientes y calculando señales históricas...")
    for activo in activos:
        senales = obtener_senales(activo["id"], api_key)
        if senales.empty:
            continue
        probabilidad = modelo.predict_proba(senales[features])[0, 1]
        resultados.append((activo["name"], float(activo["priceUsd"]), probabilidad))

    print("\nPredicción para el próximo día (no es una recomendación financiera)")
    print("-" * 76)
    for nombre, precio, probabilidad in resultados:
        tendencia = "Sube" if probabilidad >= 0.5 else "Baja"
        print(f"{nombre:<22} Precio: USD {precio:>14,.6f}  {tendencia:<4}  Prob.: {probabilidad:.1%}")


if __name__ == "__main__":
    main()
