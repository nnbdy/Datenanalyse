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

# CO2-Daten in Long-Format
country_mapping = {
    "USA_CO2_kt": "USA",
    "China_CO2_kt": "China",
    "Deutschland_CO2_kt": "Deutschland",
    "Indien_CO2_kt": "Indien",
    "UK_CO2_kt": "Vereinigtes Königreich"
}

co2_long = co2_temp[
    ["Jahr"] + list(country_mapping.keys())
].melt(
    id_vars="Jahr",
    var_name="Variable",
    value_name="CO2_kt"
)

co2_long["Land"] = (
    co2_long["Variable"]
    .map(country_mapping)
)

co2_long["CO2_Mt"] = (
    co2_long["CO2_kt"] / 1000
)

print("\nCO2 Long-Format:")
print(co2_long.head())

#---------------------------------------------------------------------------
# Abbildung 1:
# CO2-Emissionen nach Ländern

plt.figure(figsize=(12, 7))

for land in country_mapping.values():

    df_land = co2_long[
        co2_long["Land"] == land
    ].dropna(subset=["CO2_Mt"])

    plt.plot(
        df_land["Jahr"],
        df_land["CO2_Mt"],
        label=land
    )

plt.xlabel("Jahr")
plt.ylabel("CO₂-Emissionen in Mio. Tonnen")
plt.title(
    "Entwicklung der CO₂-Emissionen "
    "ausgewählter Länder"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIG_DIR / "01_co2_entwicklung_laender.png",
    dpi=300
)

plt.close()

# Deskriptive Kennzahlen CO2
country_results = []

for land in country_mapping.values():

    df_land = co2_long[
        co2_long["Land"] == land
    ].dropna(subset=["CO2_Mt"])

    if df_land.empty:
        continue

    maximum_index = df_land["CO2_Mt"].idxmax()

    country_results.append({
        "Land": land,

        "Erstes_Jahr":
            int(df_land["Jahr"].min()),

        "Letztes_Jahr":
            int(df_land["Jahr"].max()),

        "Minimum_CO2_Mt":
            df_land["CO2_Mt"].min(),

        "Maximum_CO2_Mt":
            df_land["CO2_Mt"].max(),

        "Jahr_Maximum":
            int(
                df_land.loc[
                    maximum_index,
                    "Jahr"
                ]
            ),

        "Letzter_Wert_CO2_Mt":
            df_land.sort_values("Jahr")
            .iloc[-1]["CO2_Mt"]
    })


country_summary = pd.DataFrame(
    country_results
)

print("\nCO2-Zusammenfassung:")
print(country_summary)

country_summary.to_csv(
    TABLE_DIR / "03_co2_laender_kennzahlen.csv",
    index=False,
    sep=";",
    decimal=","
)

# Veränderung 1990–2024
change_results = []

for land in country_mapping.values():

    df_land = co2_long[
        co2_long["Land"] == land
    ]

    value_1990 = df_land.loc[
        df_land["Jahr"] == 1990,
        "CO2_Mt"
    ]

    value_2024 = df_land.loc[
        df_land["Jahr"] == 2024,
        "CO2_Mt"
    ]

    if (
        not value_1990.empty
        and not value_2024.empty
        and pd.notna(value_1990.iloc[0])
        and pd.notna(value_2024.iloc[0])
    ):

        start = value_1990.iloc[0]
        end = value_2024.iloc[0]

        change_percent = (
            (end - start)
            / start
            * 100
        )

        change_results.append({
            "Land": land,
            "CO2_1990_Mt": start,
            "CO2_2024_Mt": end,
            "Veraenderung_Prozent":
                change_percent
        })


change_df = pd.DataFrame(change_results)

change_df["Veraenderung_Prozent"] = (
    change_df["Veraenderung_Prozent"]
    .round(2)
)

print("\nVeränderung 1990–2024:")
print(change_df)

change_df.to_csv(
    TABLE_DIR
    / "04_co2_veraenderung_1990_2024.csv",
    index=False,
    sep=";",
    decimal=","
)

# USA Hauptanalyse
analysis_usa = co2_temp[
    [
        "Jahr",
        "USA_CO2_kt",
        "US_Temp_F"
    ]
].copy()

analysis_usa = analysis_usa.dropna()

analysis_usa["USA_CO2_Mt"] = (
    analysis_usa["USA_CO2_kt"]
    / 1000
)

print("\nUSA Hauptanalyse:")
print(
    f"Zeitraum: "
    f"{analysis_usa['Jahr'].min()}–"
    f"{analysis_usa['Jahr'].max()}"
)

print(
    f"Beobachtungen: "
    f"{len(analysis_usa)}"
)

# Pearson-Korrelation
r_usa, p_usa = pearsonr(
    analysis_usa["USA_CO2_Mt"],
    analysis_usa["US_Temp_F"]
)

print("\nPearson-Korrelation USA:")
print(f"r = {r_usa:.4f}")
print(f"p = {p_usa:.6f}")

# Korrelationen aller Länder
correlation_results = []

for column, land in country_mapping.items():

    temp_df = co2_temp[
        [
            "Jahr",
            column,
            "US_Temp_F"
        ]
    ].dropna()

    r, p = pearsonr(
        temp_df[column],
        temp_df["US_Temp_F"]
    )

    correlation_results.append({
        "Land": land,
        "Von": int(temp_df["Jahr"].min()),
        "Bis": int(temp_df["Jahr"].max()),
        "N": len(temp_df),
        "Pearson_r": r,
        "p_Wert": p
    })


correlation_df = pd.DataFrame(
    correlation_results
)

print("\nKorrelationen:")
print(correlation_df)

correlation_df.to_csv(
    TABLE_DIR
    / "05_korrelationen_co2_temperatur.csv",
    index=False,
    sep=";",
    decimal=","
)

# Scatterplot
plt.figure(figsize=(9, 6))

plt.scatter(
    analysis_usa["USA_CO2_Mt"],
    analysis_usa["US_Temp_F"]
)

plt.xlabel(
    "US-CO₂-Emissionen in Mio. Tonnen"
)

plt.ylabel(
    "US-Jahrestemperatur in °F"
)

plt.title(
    "Zusammenhang zwischen "
    "US-CO₂-Emissionen und US-Temperatur"
)

plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIG_DIR / "02_co2_temperatur_scatter.png",
    dpi=300
)

plt.close()

# Lineare Regression USA
X = analysis_usa[
    ["USA_CO2_Mt"]
]

y = analysis_usa[
    "US_Temp_F"
]

model = LinearRegression()

model.fit(
    X,
    y
)

prediction = model.predict(X)

r2 = r2_score(
    y,
    prediction
)

rmse = np.sqrt(
    mean_squared_error(
        y,
        prediction
    )
)

print("\nLineare Regression:")
print(
    f"Intercept = "
    f"{model.intercept_:.4f}"
)

print(
    f"Steigung = "
    f"{model.coef_[0]:.6f}"
)

print(
    f"R² = {r2:.4f}"
)

print(
    f"RMSE = {rmse:.4f} °F"
)

# Regressionsdiagramm
plt.figure(figsize=(9, 6))

plt.scatter(
    analysis_usa["USA_CO2_Mt"],
    analysis_usa["US_Temp_F"],
    label="Beobachtungen"
)

sort_index = np.argsort(
    analysis_usa["USA_CO2_Mt"]
)

plt.plot(
    analysis_usa["USA_CO2_Mt"]
    .iloc[sort_index],

    prediction[sort_index],

    label="Lineare Regression"
)

plt.xlabel(
    "US-CO₂-Emissionen in Mio. Tonnen"
)

plt.ylabel(
    "US-Jahrestemperatur in °F"
)

plt.title(
    f"Lineare Regression "
    f"(R² = {r2:.3f})"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIG_DIR
    / "03_regression_co2_temperatur.png",
    dpi=300
)

plt.close()

# Zeittrend
r_year_temp, p_year_temp = pearsonr(
    analysis_usa["Jahr"],
    analysis_usa["US_Temp_F"]
)

r_year_co2, p_year_co2 = pearsonr(
    analysis_usa["Jahr"],
    analysis_usa["USA_CO2_Mt"]
)

print("\nZeittrend:")

print(
    "Jahr ↔ Temperatur:",
    f"r = {r_year_temp:.4f}",
    f"p = {p_year_temp:.6f}"
)

print(
    "Jahr ↔ CO2:",
    f"r = {r_year_co2:.4f}",
    f"p = {p_year_co2:.6f}"
)

# Erste Differenzen
analysis_usa["Delta_CO2"] = (
    analysis_usa["USA_CO2_Mt"]
    .diff()
)

analysis_usa["Delta_Temp"] = (
    analysis_usa["US_Temp_F"]
    .diff()
)

difference_data = (
    analysis_usa
    .dropna(
        subset=[
            "Delta_CO2",
            "Delta_Temp"
        ]
    )
)

r_diff, p_diff = pearsonr(
    difference_data["Delta_CO2"],
    difference_data["Delta_Temp"]
)

print(
    "\nKorrelation der "
    "jährlichen Veränderungen:"
)

print(f"r = {r_diff:.4f}")
print(f"p = {p_diff:.6f}")

# OLS + Durbin-Watson
# DW ≈ 2                → wenig Hinweis auf Autokorrelation
# DW deutlich < 20      → Hinweis auf positive Autokorrelation
# DW deutlich > 2       → Hinweis auf negative Autokorrelation

X_ols = sm.add_constant(
    analysis_usa["USA_CO2_Mt"]
)

ols_model = sm.OLS(
    analysis_usa["US_Temp_F"],
    X_ols
).fit()

dw = durbin_watson(
    ols_model.resid
)

print("\nDurbin-Watson-Test:")
print(f"DW = {dw:.4f}")

# Zeitlicher Train-Test-Split
split_index = int(
    len(analysis_usa) * 0.8
)

train = analysis_usa.iloc[
    :split_index
]

test = analysis_usa.iloc[
    split_index:
]


X_train = train[
    ["USA_CO2_Mt"]
]

y_train = train[
    "US_Temp_F"
]

X_test = test[
    ["USA_CO2_Mt"]
]

y_test = test[
    "US_Temp_F"
]


test_model = LinearRegression()

test_model.fit(
    X_train,
    y_train
)


train_prediction = (
    test_model.predict(X_train)
)

test_prediction = (
    test_model.predict(X_test)
)


train_r2 = r2_score(
    y_train,
    train_prediction
)

test_r2 = r2_score(
    y_test,
    test_prediction
)


train_rmse = np.sqrt(
    mean_squared_error(
        y_train,
        train_prediction
    )
)

test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        test_prediction
    )
)


print("\nTrain/Test:")

print(
    f"Training: "
    f"{train['Jahr'].min()}–"
    f"{train['Jahr'].max()}"
)

print(
    f"Test: "
    f"{test['Jahr'].min()}–"
    f"{test['Jahr'].max()}"
)

print(
    f"Train R² = {train_r2:.4f}"
)

print(
    f"Train RMSE = {train_rmse:.4f}"
)

print(
    f"Test R² = {test_r2:.4f}"
)

print(
    f"Test RMSE = {test_rmse:.4f}"
)

# Ausreißer
def iqr_outliers(df, column):

    values = df[column].dropna()

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = df[
        (df[column] < lower)
        | (df[column] > upper)
    ].copy()

    return outliers, lower, upper


temp_outliers, lower_temp, upper_temp = (
    iqr_outliers(
        analysis_usa,
        "US_Temp_F"
    )
)

print("\nTemperatur-Ausreißer:")
print(
    temp_outliers[
        ["Jahr", "US_Temp_F"]
    ]
)

print(
    f"IQR-Grenzen: "
    f"{lower_temp:.2f} bis "
    f"{upper_temp:.2f}"
)

temp_outliers.to_csv(
    TABLE_DIR
    / "06_temperatur_ausreisser.csv",
    index=False,
    sep=";",
    decimal=","
)

# ----------------------------------------------------------------------
# Modellkennzahlen

metrics = pd.DataFrame([
    {
        "Analyse":
            "USA CO2 vs. US-Temperatur",

        "Zeitraum_von":
            int(
                analysis_usa["Jahr"].min()
            ),

        "Zeitraum_bis":
            int(
                analysis_usa["Jahr"].max()
            ),

        "N":
            len(analysis_usa),

        "Pearson_r":
            r_usa,

        "Pearson_p":
            p_usa,

        "R2":
            r2,

        "RMSE":
            rmse,

        "Durbin_Watson":
            dw,

        "Delta_Pearson_r":
            r_diff,

        "Delta_Pearson_p":
            p_diff,

        "Train_R2":
            train_r2,

        "Train_RMSE":
            train_rmse,

        "Test_R2":
            test_r2,

        "Test_RMSE":
            test_rmse
    }
])

metrics.to_csv(
    TABLE_DIR
    / "07_modellkennzahlen.csv",
    index=False,
    sep=";",
    decimal=","
)

print("\nModellkennzahlen:")
print(metrics.T)

# ----------------------------------------------------------------------
# Testprotokoll

test_protocol = pd.DataFrame([
    {
        "Test":
            "Fehlende Werte",

        "Problem":
            "Nicht alle Länder verfügen "
            "für jedes historische Jahr "
            "über CO2-Werte.",

        "Loesung":
            "Keine künstliche Interpolation. "
            "Analysen verwenden nur "
            "vollständige Wertepaarungen.",

        "Status":
            "dokumentiert"
    },

    {
        "Test":
            "Doppelte Jahre",

        "Problem":
            "Mehrfachbeobachtungen könnten "
            "statistische Ergebnisse verzerren.",

        "Loesung":
            "Jahresvariable auf Duplikate "
            "geprüft.",

        "Status":
            (
                "bestanden"
                if len(duplicate_rows) == 0
                else "prüfen"
            )
    },

    {
        "Test":
            "Ausreißer",

        "Problem":
            "Extreme Temperaturwerte "
            "können die Regression beeinflussen.",

        "Loesung":
            "IQR-Methode zur Identifikation; "
            "keine automatische Entfernung "
            "realer Beobachtungen.",

        "Status":
            "dokumentiert"
    },

    {
        "Test":
            "Autokorrelation",

        "Problem":
            "Zeitlich aufeinanderfolgende "
            "Beobachtungen sind möglicherweise "
            "nicht unabhängig.",

        "Loesung":
            "Durbin-Watson-Test und "
            "Analyse erster Differenzen.",

        "Status":
            "geprüft"
    }
])


test_protocol.to_csv(
    TABLE_DIR
    / "08_testprotokoll.csv",
    index=False,
    sep=";"
)