from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Charger le modèle
try:
    model = joblib.load(r'C:\Users\kaloui\Desktop\Credit score\final\model.pkl')
    print("Modèle chargé avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")
    model = None

# Charger les données
try:
    data_test = pd.read_csv(r'C:/Users/kaloui/application_test.csv')
    print("Données chargées avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement des données : {e}")
    data_test = pd.DataFrame()

# Liste des colonnes à supprimer (à définir selon votre logique de nettoyage)
columns_to_drop = []  # Remplacez par les colonnes que vous souhaitez supprimer

def feature_engineering(df):
    """Applique toutes les transformations d'ingénierie des fonctionnalités."""
    df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    
    df['EXT_SOURCE_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['EXT_SOURCE_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    
    return df

def encode_categorical_features(df):
    """Encode les caractéristiques catégorielles en valeurs numériques."""
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))  # Convertir en string si nécessaire
    return df

# Fonction pour prétraiter les données
def preprocess_data(df):
    """Applique toutes les transformations nécessaires aux données."""
    df = feature_engineering(df)  # Appliquez votre fonction d'ingénierie des fonctionnalités
    
    # Gérer les valeurs manquantes
    df.fillna(0, inplace=True)
    
    # Encoder les caractéristiques catégorielles
    df = encode_categorical_features(df)
    
    # Supprimer les colonnes non nécessaires
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    # Sélectionner aléatoirement 50 caractéristiques si le DataFrame contient suffisamment de colonnes
    if df.shape[1] > 50:
        random_features = df.sample(n=50, axis=1, random_state=42)  # Sélection aléatoire de 50 colonnes
        return random_features.values.flatten()  # Retourne les valeurs à plat
    else:
        return df.values.flatten()  # Retourne toutes les valeurs si moins de 50 colonnes

def get_client_data(client_id):
    """Récupère et prétraite les données du client en fonction de son ID."""
    client_data = data_test[data_test['SK_ID_CURR'] == client_id]
    if not client_data.empty:
        client_data = preprocess_data(client_data)  # Appliquez le prétraitement ici
        return client_data
    else:
        return None

@app.route('/predict/<int:client_id>', methods=['GET'])
def predict(client_id):
    """Prédire le résultat pour un client donné par son ID."""
    input_features = get_client_data(client_id)
    
    if input_features is None:
        return jsonify({"error": "Client not found"}), 404
    
    input_features = np.array(input_features).reshape(1, -1)
    
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        prediction = model.predict(input_features)
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        print(f"Erreur lors de la prédiction : {e}")
        return jsonify({"error": "Prediction error"}), 500

if __name__ == '__main__':
    app.run(debug=True)
