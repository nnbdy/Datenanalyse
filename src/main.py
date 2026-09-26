from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm

#Pfade

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = (BASE_DIR / "data" /"processed" /"CO2_Aktualisierte_Datenbasis_Aufgabe_A.xlsx")
OUTPUT_DIR = BASE_DIR / "output"
FIG_DIR = OUTPUT_DIR / "abbildungen"
TABLE_DIR = OUTPUT_DIR / "tabellen"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Die Datei {DATA_FILE} wurde nicht gefunden. Bitte überprüfen Sie den Pfad.")

#Daten einlesen
master = pd.read_excel(
    DATA_FILE,
    sheet_name="02_Master_Land_Jahr",
    header=2
)

co2_temp = pd.read_excel(
    DATA_FILE,
    sheet_name="03_CO2_Temp_1895_2024",
    header=2
)

usa = pd.read_excel(
    DATA_FILE,
    sheet_name="04_USA_Zeitreihe",
    header=2
)

disasters = pd.read_excel(
    DATA_FILE,
    sheet_name="05_Disaster_US",
    header=2
)


print("\nMasterdatensatz:")
print(master.shape)

print("\nCO2-Temperatur-Datensatz:")
print(co2_temp.shape)

print("\nUSA-Datensatz:")
print(usa.shape)

print("\nDisaster-Datensatz:")
print(disasters.shape)

