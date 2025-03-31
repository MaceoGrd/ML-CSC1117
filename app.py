import pandas as pd
import streamlit as st
import numpy as np

# === CHARGEMENT DES DONNÉES PRÉTRAITÉES ===
@st.cache_data
def load_data():
    race_data = pd.read_csv("data/merged_race_data.csv")
    qualif_score = pd.read_csv("data/qualif_score_2025.csv", index_col=0).squeeze()
    return race_data, qualif_score

df_race_all, qualif_score = load_data()

# === PARAMÈTRES DE BASE ===
pilotes_2025 = sorted(df_race_all['driver'].unique())
teams_2025 = sorted(df_race_all[df_race_all['season'] == 2025]['team'].dropna().unique())

# === INTERFACE STREAMLIT ===
st.title("🏎️ Simulation Classement F1 2025")
st.markdown("Modifie l’écurie de chaque pilote pour voir l’impact sur le classement !")

st.sidebar.header("🔧 Écurie des pilotes")
# Récupère la vraie team 2025 de chaque pilote
df_teams_2025 = df_race_all[df_race_all['season'] == 2025][['driver', 'team']].dropna().drop_duplicates()
default_teams = df_teams_2025.set_index('driver')['team'].to_dict()

# Affiche avec la vraie team par défaut dans l'interface
selected_teams = {}
for pilote in pilotes_2025:
    default_team = default_teams.get(pilote, teams_2025[0])  # fallback = premier team si inconnu
    selected_teams[pilote] = st.sidebar.selectbox(f"{pilote.title()}", teams_2025, index=teams_2025.index(default_team), key=pilote)

# === SCORE CALCULÉ ===
def calc_perf_score(pos):
    if pd.isna(pos): return -5
    if pos == 1: return 10
    if pos <= 3: return 8
    if pos <= 5: return 6
    if pos <= 10: return 4
    if pos <= 15: return 1
    return -5

df_race_all['perf_score'] = df_race_all['Position'].apply(calc_perf_score)

# Moyennes & tendances
df_avg_score = df_race_all.groupby(['driver', 'season'])['perf_score'].mean().reset_index()
df_avg = df_avg_score.groupby('driver')['perf_score'].mean()
df_trend = df_avg_score.groupby('driver').apply(lambda g: (g['perf_score'].iloc[-1] - g['perf_score'].iloc[0]) / (len(g) - 1) if len(g) > 1 else 0)

# Bonus 2025
df_2025 = df_race_all[df_race_all['season'] == 2025].copy()
df_2025['perf_score'] = df_2025['Position'].apply(calc_perf_score)
df_bonus_2025 = df_2025.groupby('driver')['perf_score'].mean()

# Team score basé sur 2024
df_2024 = df_race_all[df_race_all['season'] == 2024].copy()
df_2024['team_score'] = df_2024['Position'].apply(calc_perf_score)
team_avg_score_2024 = df_2024.groupby('team')['team_score'].mean()

# Teams personnalisées
driver_team_2025 = pd.Series(selected_teams)
team_score_per_driver = driver_team_2025.map(team_avg_score_2024).fillna(0)

# === ASSEMBLAGE FINAL ===
df_final = pd.DataFrame(index=pilotes_2025)
df_final['avg_score'] = df_final.index.map(df_avg).fillna(0)
df_final['trend'] = df_final.index.map(df_trend).fillna(0)
df_final['bonus_2025'] = df_final.index.map(df_bonus_2025).fillna(0)
df_final['qualif_score'] = df_final.index.map(qualif_score).fillna(0)
df_final['team_score'] = df_final.index.map(team_score_per_driver).fillna(0)

weights = {
    'avg_score': 0.20,
    'trend': 0.10,
    'bonus_2025': 0.25,
    'qualif_score': 0.30,
    'team_score': 0.15
}
df_final['final_score'] = (
    df_final['avg_score'] * weights['avg_score'] +
    df_final['trend'] * weights['trend'] +
    df_final['bonus_2025'] * weights['bonus_2025'] +
    df_final['qualif_score'] * weights['qualif_score'] +
    df_final['team_score'] * weights['team_score']
)

df_final_sorted = df_final.sort_values(by='final_score', ascending=False).reset_index()
df_final_sorted.rename(columns={'index': 'driver'}, inplace=True)

# === AFFICHAGE ===
st.subheader("📊 Classement Prédit")
st.dataframe(df_final_sorted[['driver', 'final_score']])

st.subheader("📈 Visualisation du classement")
st.bar_chart(df_final_sorted.set_index('driver')['final_score'])
