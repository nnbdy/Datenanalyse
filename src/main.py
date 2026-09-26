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

# Grundlegende Datenprüfung
required_columns = [
    "Jahr",
    "US_Temp_F",
    "USA_CO2_kt",
    "China_CO2_kt",
    "Deutschland_CO2_kt",
    "Indien_CO2_kt",
    "UK_CO2_kt"
]

missing_columns = [
    col
    for col in required_columns
    if col not in co2_temp.columns
]

if missing_columns:
    raise ValueError(
        f"Folgende Spalten fehlen: {missing_columns}"
    )

print("\nAlle benötigten Spalten vorhanden.")

# Fehlende Werte
analysis_columns = [
    "US_Temp_F",
    "USA_CO2_kt",
    "China_CO2_kt",
    "Deutschland_CO2_kt",
    "Indien_CO2_kt",
    "UK_CO2_kt"
]

missing_values = pd.DataFrame({
    "Fehlende_Werte":
        co2_temp[analysis_columns].isna().sum(),

    "Anteil_Prozent":
        co2_temp[analysis_columns].isna().mean() * 100
})

missing_values["Anteil_Prozent"] = (
    missing_values["Anteil_Prozent"].round(2)
)

print("\nFehlende Werte:")
print(missing_values)

missing_values.to_csv(
    TABLE_DIR / "01_fehlende_werte.csv",
    sep=";",
    decimal=","
)

# Duplikate prüfen
duplicates = co2_temp.duplicated(
    subset=["Jahr"],
    keep=False
)

duplicate_rows = co2_temp[duplicates]

print("\nDoppelte Jahre:")
print(len(duplicate_rows))

if len(duplicate_rows) > 0:
    print(duplicate_rows[["Jahr"]])

duplicate_rows.to_csv(
    TABLE_DIR / "02_duplikate.csv",
    index=False,
    sep=";"
)

# Plausibilitätsprüfung
co2_columns = [
    "USA_CO2_kt",
    "China_CO2_kt",
    "Deutschland_CO2_kt",
    "Indien_CO2_kt",
    "UK_CO2_kt"
]

negative_co2 = (
    co2_temp[co2_columns] < 0
).sum()

print("\nNegative CO2-Werte:")
print(negative_co2)

print("\nTemperatur Minimum / Maximum:")
print(co2_temp["US_Temp_F"].min())
print(co2_temp["US_Temp_F"].max())