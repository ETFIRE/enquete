import pandas as pd
from sklearn.ensemble import IsolationForest

# 1. Chargement du dataset
dataset = pd.read_csv('ml_dataset_final.csv', index_col=0)

# 2. Préparation pour l'Isolation Forest
if 'LEFEVRE' in dataset.index:
    df_suspects = dataset.drop('LEFEVRE')
else:
    df_suspects = dataset

# 3. Initialisation et exécution du modèle
model = IsolationForest(n_estimators=100, contamination=0.3, random_state=42)
df_suspects['anomaly_score'] = model.fit_predict(df_suspects)

# 4. Identification des suspects 
df_suspects['is_anomaly'] = df_suspects['anomaly_score'] == -1

# 5. Affichage des résultats
print("Résultats de l'analyse ML :")
print(df_suspects[['anomaly_score', 'is_anomaly']].sort_values('anomaly_score'))