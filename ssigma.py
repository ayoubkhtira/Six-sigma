# Code de test pour vérifier la conversion
test_data = {
    'A': ['N° de la pièce', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    'B': ['OPERATEUR 1', '45.08', '44.94', '45.08', '44.97', '44.92', '45.03', '45.24', '45.16', '45.2', '45.07'],
    'C': ['OPERATEUR 1', '45.1', '44.97', '45.11', '45.02', '44.95', '45.06', '45.36', '45.16', '45.2', '45.1'],
    'D': ['OPERATEUR 1', '45.09', '44.93', '45.13', '45.01', '45.01', '45.05', '45.31', '45.1', '45.24', '45.05'],
    'E': ['OPERATEUR 2', '45.01', '44.92', '44.95', '45.01', '44.95', '45.04', '45.12', '45.12', '45.06', '45.02'],
    'F': ['OPERATEUR 2', '45.04', '44.95', '45.11', '44.93', '44.99', '45.06', '45.21', '45.13', '45.19', '45.05'],
    'G': ['OPERATEUR 2', '45.02', '44.95', '45.11', '45.0', '44.97', '44.99', '45.32', '45.13', '45.24', '45.09'],
    'H': ['OPERATEUR 3', '45.04', '44.92', '45.1', '45.02', '44.99', '45.07', '45.26', '45.13', '45.19', '45.1'],
    'I': ['OPERATEUR 3', '45.07', '44.97', '45.12', '44.99', '44.97', '45.03', '45.29', '45.13', '45.17', '45.1'],
    'J': ['OPERATEUR 3', '45.07', '44.91', '45.11', '44.99', '44.95', '45.02', '45.28', '45.14', '45.16', '45.09']
}

df_test = pd.DataFrame(test_data)
print("DataFrame d'origine:")
print(df_test.head(3))

# Tester la conversion
df_converted = convert_wide_to_long_specific(df_test)
print("\nDataFrame converti (premières lignes):")
print(df_converted.head(15))
print(f"\nDimensions: {df_converted.shape}")
print(f"Pièces uniques: {df_converted['Pièce'].nunique()}")
print(f"Opérateurs uniques: {df_converted['Opérateur'].nunique()}")
print(f"Essais uniques: {df_converted['Essai'].nunique()}")

# Vérifier l'équilibre
counts = df_converted.groupby(['Pièce', 'Opérateur'])['Mesure'].count()
print(f"\nPlan équilibré: {counts.nunique() == 1}")
print(f"Mesures par couple: {counts.iloc[0] if not counts.empty else 'N/A'}")
