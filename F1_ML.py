# F1 2025 Performance Prediction Model
import pandas as pd
import numpy as np

# === CHARGEMENT DES DONNEES ===
race_files = [
    "data/Formula1_2022season_raceResults.csv",
    "data/Formula1_2023season_raceResults.csv",
    "data/Formula1_2024season_raceResults.csv",
    "data/Formula1_2025Season_RaceResults.csv",
    "data/Formula1_2025Season_SprintResults.csv"
]
race_dfs = [pd.read_csv(f) for f in race_files]

for f, year in zip(race_dfs, [2022, 2023, 2024, 2025, 2025]):
    f['season'] = year

df_race_all = pd.concat(race_dfs, ignore_index=True)

# Résultats qualif 2025
qualif_2025 = pd.read_csv("data/Formula1_2025Season_QualifyingResults.csv")
sprint_2025 = pd.read_csv("data/Formula1_2025Season_SprintQualifyingResults.csv")

# === LISTE PILOTES 2025 ===
pilotes_2025 = [
    'alexander albon', 'carlos sainz', 'charles leclerc', 'esteban ocon', 'fernando alonso',
    'gabriel bortoleto', 'george russell', 'isack hadjar', 'jack doohan', 'kimi antonelli',
    'lance stroll', 'lando norris', 'lewis hamilton', 'liam lawson', 'max verstappen',
    'nico hulkenberg', 'oliver bearman', 'oscar piastri', 'pierre gasly', 'yuki tsunoda'
]

df_race_all['driver'] = df_race_all['Driver'].str.lower().str.strip()
df_race_all['team'] = df_race_all['Team'].str.lower().str.strip()
df_race_all = df_race_all[df_race_all['driver'].isin(pilotes_2025)]
df_race_all['Position'] = pd.to_numeric(df_race_all['Position'], errors='coerce')

# === SCORE DE PERFORMANCE ===
def calc_perf_score(pos):
    if pd.isna(pos): return -5
    if pos == 1: return 10
    if pos <= 3: return 8
    if pos <= 5: return 6
    if pos <= 10: return 4
    if pos <= 15: return 1
    return -5

df_race_all['perf_score'] = df_race_all['Position'].apply(calc_perf_score)
df_avg_score = df_race_all.groupby(['driver', 'season'])['perf_score'].mean().reset_index()
df_avg = df_avg_score.groupby('driver')['perf_score'].mean()

# === TENDANCE DE PROGRESSION ===
def compute_trend(group):
    scores = group['perf_score'].values
    return (scores[-1] - scores[0]) / (len(scores) - 1) if len(scores) > 1 else 0
df_trend = df_avg_score.groupby('driver').apply(compute_trend)

# === TEMPS QUALIF 2025 ===
def time_to_sec(t):
    try:
        if pd.isnull(t): return None
        m, s = t.strip().split(':')
        return float(m)*60 + float(s)
    except: return None

for df in [qualif_2025, sprint_2025]:
    for q in ['Q1', 'Q2', 'Q3']:
        df[q + '_sec'] = df[q].apply(time_to_sec)
    df['best_qualif_time'] = df[['Q1_sec', 'Q2_sec', 'Q3_sec']].min(axis=1)
    df['driver'] = df['Driver'].str.lower().str.strip()

df_q = pd.concat([
    qualif_2025[['driver', 'best_qualif_time']],
    sprint_2025[['driver', 'best_qualif_time']]
])
avg_q_time = df_q.groupby('driver')['best_qualif_time'].mean()
min_time, max_time = avg_q_time.min(), avg_q_time.max()
qualif_score = 1 - ((avg_q_time - min_time) / (max_time - min_time))

# === Export pour Streamlit ===
df_race_all.to_csv("data/merged_race_data.csv", index=False)
qualif_score.to_csv("data/qualif_score_2025.csv")
print("✅ Données exportées pour Streamlit !")
