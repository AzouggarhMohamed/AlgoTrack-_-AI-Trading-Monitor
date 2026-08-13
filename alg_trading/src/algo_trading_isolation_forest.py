import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from RSI import calcul_rsi
from sklearn.preprocessing import MinMaxScaler

all_file = {
    "aapl": r"C:\Users\HP 850  G5\Desktop\alg_trading\AAPL - Données Historiques.csv",
    "nvda": r"C:\Users\HP 850  G5\Desktop\alg_trading\NVDA - Données Historiques.csv",
    "msft": r"C:\Users\HP 850  G5\Desktop\alg_trading\MSFT - Données Historiques.csv",
    "iam": r"C:\Users\HP 850  G5\Desktop\alg_trading\IAM - Données Historiques (1).csv",
    "boa": r"C:\Users\HP 850  G5\Desktop\alg_trading\BOA - Données Historiques.csv",
    "atw": r"C:\Users\HP 850  G5\Desktop\alg_trading\ATW - Données Historiques.csv",
}
all_df = []

for ticker, path in all_file.items():
    # Lire le fichier
    df = pd.read_csv(path)
    # Garder uniquement Date et Dernier (le prix de clôture)
    df = df[["Date", "Dernier"]].copy()
    # Ajouter le nom de l'entreprise pour ne pas les mélanger
    df["Ticker"] = ticker
    # Nettoyer le prix (enlever les virgules si elles existent et convertir en nombre)
    if df["Dernier"].dtype == "object":
        df["Dernier"] = df["Dernier"].str.replace(",", ".").astype(str)
        df["Dernier"] = df["Dernier"].str.replace("\xa0", "", regex=False)
        df["Dernier"] = df["Dernier"].str.replace(" ", "", regex=False)
    all_df.append(df)
final_df = pd.concat(all_df, ignore_index=True)
final_df["Dernier"] = pd.to_numeric(final_df["Dernier"], errors="coerce")
final_df["Date"] = pd.to_datetime(final_df["Date"], dayfirst=True)
final_df = final_df.sort_values(["Ticker", "Date"])
final_df["Rend_J"] = final_df.groupby("Ticker")["Dernier"].pct_change()
final_df['Prix_Normalise'] = final_df.groupby('Ticker')['Dernier'].transform(
    lambda x: (x / x.iloc[0]) * 100
)

# colonne momentum et signal
final_df["Momentum_20"] = (
    final_df.groupby("Ticker")["Dernier"].pct_change(periods=20) * 100
)
final_df["Momentum_Signal"] = np.where(
    final_df["Momentum_20"] > 5,
    "Buy",
    np.where(final_df["Momentum_20"] < -5, "Sell", "Hold"),
)

# colonne rsi et signal
final_df["Rsi_14"] = final_df.groupby("Ticker")["Dernier"].transform(
    lambda x: calcul_rsi(x)
)
final_df["Rsi_Signal"] = np.where(
    final_df["Rsi_14"] < 30, "Buy", np.where(final_df["Rsi_14"] > 70, "Sell", "Neutre")
)

final_df["BB_Milieu"] = final_df.groupby("Ticker")["Dernier"].transform(
    lambda x: x.rolling(20).mean()
)
final_df["BB_Haut"] = final_df.groupby("Ticker")["Dernier"].transform(
    lambda x: x.rolling(20).mean() + 2 * x.rolling(20).std()
)
final_df["BB_Bas"] = final_df.groupby("Ticker")["Dernier"].transform(
    lambda x: x.rolling(20).mean() - 2 * x.rolling(20).std()
)

final_df["Z_score"] = final_df.groupby("Ticker")["Dernier"].transform(
    lambda x: (x - x.rolling(20).mean()) / x.rolling(20).std()
)

final_df["Bollinger_Signal"] = np.where(
    final_df["Z_score"] > 2, "Sell", np.where(final_df["Z_score"] < -2, "Buy", "Neutre")
)
features = final_df[["Momentum_20", "Rsi_14", "Z_score"]].dropna()
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(features)
prediction = model.predict(features)
Anomalie_serie = pd.Series(
    np.where(prediction == -1, "Anomalie", "Normal"), index=features.index
)
final_df["Anomalie"] = Anomalie_serie.reindex(final_df.index).fillna("Normal")

final_df['Rang'] = final_df.groupby('Ticker').cumcount()
final_df.loc[final_df['Rang'] < 30, 'Anomalie'] = np.nan
final_df.drop(columns=['Rang'], inplace=True)
print(final_df['Anomalie'].value_counts())

performance = []
for ticker in final_df["Ticker"].unique():
    df_ticker = final_df[final_df["Ticker"] == ticker].copy()
    df_result = df_ticker.dropna(subset=["Momentum_20", "Rsi_14", "Z_score"])
    rendements_valides = df_result["Rend_J"].dropna()
    perf = {
        "Ticker": ticker,
        "Rend_T": round(
            (df_result["Dernier"].iloc[-1] / df_result["Dernier"].iloc[0] - 1) * 100, 2
        ),
        "Vol_%": round((rendements_valides.std() * np.sqrt(252) * 100), 2),
        "Sharpe_Ratio": round(
            (rendements_valides.mean() * 252)
            / (rendements_valides.std() * np.sqrt(252)),
            3,
        ),
        "Max_Drawdown": round(
            ((df_ticker["Dernier"] / df_ticker["Dernier"].cummax()) - 1).min() * 100, 2
        ),
        "Nbr_Anomalie": round(
            (df_ticker["Anomalie"] == "Anomalie").sum(),
        ),
    }
    performance.append(perf)
df_perf = pd.DataFrame(performance)
final_df.to_csv("trading_data.csv", index=False, encoding='utf-8-sig')
df_perf.to_csv("trading_performance.csv", index=False, encoding='utf-8-sig')

cols = ['Rend_T', 'Sharpe_Ratio', 'Vol_%', 
        'Max_Drawdown', 'Nbr_Anomalie']

scaler = MinMaxScaler(feature_range=(0, 100))
perf_df_scaled = df_perf.copy()
perf_df_scaled[cols] = scaler.fit_transform(df_perf[cols])
perf_df_scaled.to_csv("trading_radar.csv", 
                       index=False, 
                       encoding='utf-8-sig')

print(perf_df_scaled)