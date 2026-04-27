import pandas as pd

# Extraction des anciennetés
df_emp = pd.read_csv('employes.csv')
complices = ['S01', 'S04', 'S05']
anciennetes = df_emp[df_emp['id_suspect'].isin(complices)]['anciennete_annees'].values

# Le code est la concaténation du premier chiffre des anciennetés (12, 10, 4)
code_coffre = "".join([str(a)[0] for a in anciennetes]) 
print(f"Code du coffre : {code_coffre}")