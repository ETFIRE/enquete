import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('ml_dataset_final.csv', index_col=0)
if 'LEFEVRE' in df.index:
    df = df.drop('LEFEVRE')

# Standardization
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# PCA
pca = PCA(n_components=2)
components = pca.fit_transform(df_scaled)
df['pca1'] = components[:, 0]
df['pca2'] = components[:, 1]

# Plot
plt.figure(figsize=(10, 6))
# Plot normal
normal = df[df.index.isin(['S03', 'S06', 'S07', 'S08', 'S09', 'S10'])]
plt.scatter(normal['pca1'], normal['pca2'], c='blue', label='Employés normaux', alpha=0.6)

# Plot suspects
suspects = df[df.index.isin(['S01', 'S04', 'S05'])]
plt.scatter(suspects['pca1'], suspects['pca2'], c='red', label='Suspects (S01, S04, S05)', s=100)

for i, txt in enumerate(suspects.index):
    plt.annotate(txt, (suspects.iloc[i]['pca1'], suspects.iloc[i]['pca2']), xytext=(5, 5), textcoords='offset points')

plt.title('Visualisation PCA des suspects par rapport au personnel')
plt.xlabel('Composante principale 1')
plt.ylabel('Composante principale 2')
plt.legend()
plt.grid(True)
plt.savefig('pca_suspects_plot.png')
print("Graphique PCA généré et sauvegardé dans pca_suspects_plot.png")