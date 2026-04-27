import pandas as pd

# Fusion des logs et données d'interrogatoire
df_interro = pd.read_csv('interrogatoires.csv')
stress = df_interro.groupby('id_suspect').apply(lambda x: (x['rythme_cardiaque_bpm'] * x['temps_reponse_sec']).mean())

# Agrégation des preuves matérielles
df_preuves = pd.read_csv('preuves_materielles.csv')
evidence_cols = ['presence_bureau_victime', 'empreintes_sur_verre', 'ADN_sur_porte', 'fibres_textile_compatibles', 'telephone_dans_zone']
df_preuves['evidence_score'] = df_preuves[evidence_cols].apply(lambda x: (x == 'oui').sum(), axis=1)

# Création du dataset ML
dataset = pd.concat([stress.rename('stress'), df_preuves.set_index('id_suspect')['evidence_score']], axis=1).fillna(0)