# Import des librairies
import streamlit as st
from PIL import Image
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Chargement des datasets
data_train = pd.read_csv("C:/Users/kaloui/application_train.csv")  # Remplacez par le bon chemin
data_test = pd.read_csv("C:/Users/kaloui/application_test.csv")    # Remplacez par le bon chemin

# Configuration de l'API locale
API_URL = "http://127.0.0.1:5000/"  # Assurez-vous que c'est le bon URL de votre API

# Fonction pour récupérer la prédiction via l'API
def get_prediction(client_id):
    url_get_pred = f"{API_URL}predict/{client_id}"
    response = requests.get(url_get_pred)
    
    if response.status_code == 200:
        response_data = response.json()  # Parse la réponse JSON directement
        proba_default = round(float(response_data['prediction']), 3)  # Assurez-vous que 'prediction' est bien dans la réponse
        best_threshold = 0.54
        decision = "Refusé" if proba_default >= best_threshold else "Accordé"
        return proba_default, decision
    else:
        st.error(f"Erreur lors de la récupération de la prédiction : {response.status_code} - {response.text}")
        return None, None

def df_voisins(client_id):
    """Récupère les clients similaires à celui dont l'ID est passé en paramètre."""
    url_get_df_voisins = f"{API_URL}clients_similaires/{client_id}"
    response = requests.get(url_get_df_voisins)
    
    if response.status_code == 200:
        data_voisins = pd.read_json(response.content)  # Assurez-vous que le format de réponse est correct
        return data_voisins
    else:
        st.error("Erreur lors de la récupération des clients similaires.")
        return pd.DataFrame()  # Retourne un DataFrame vide en cas d'erreur

# Fonction pour afficher une jauge de score
def jauge_score(proba):
    fig = go.Figure(go.Indicator(
        domain={'x': [0, 1], 'y': [0, 1]},
        value=proba * 100,
        mode="gauge+number+delta",
        title={'text': "Jauge de score"},
        delta={'reference': 54},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "black"},
            'bar': {'color': "MidnightBlue"},
            'steps': [
                {'range': [0, 20], 'color': "Green"},
                {'range': [20, 45], 'color': "LimeGreen"},
                {'range': [45, 54], 'color': "Orange"},
                {'range': [54, 100], 'color': "Red"}
            ],
            'threshold': {'line': {'color': "brown", 'width': 4}, 'thickness': 1, 'value': 54}
        }
    ))
    st.plotly_chart(fig)

# Afficher les caractéristiques des clients avec une probabilité de défaut élevée
def analyze_high_default_probability(data, threshold=0.9):
    high_default_clients = data[data['probability'] >= threshold]
    st.write("Clients avec une probabilité de défaut élevée :")
    st.write(high_default_clients)

# Titre de la page
st.set_page_config(page_title="Dashboard Credit Time", layout="wide")

# Sidebar
with st.sidebar:
    logo_path = r'C:\Users\kaloui\Desktop\Credit score\final\credit_time.png'  # Chemin complet vers l'image
    
    try:
        logo = Image.open(logo_path)
        st.image(logo, width=200)
    except FileNotFoundError:
        st.error(f"Le fichier image n'a pas été trouvé à l'emplacement : {logo_path}")

    page = st.selectbox('Navigation', ["Home", "Information du client"])

    list_id_client = list(data_test['SK_ID_CURR'])
    list_id_client.insert(0, '<Select>')
    
    id_client_dash = st.selectbox("ID Client", list_id_client)

if id_client_dash != '<Select>':
    st.write('Vous avez choisi le client ID : ' + str(id_client_dash))

if page == "Home":
    st.title("Dashboard Credit Time - Home Page")
    st.markdown("Ce site contient un dashboard interactif permettant d'expliquer aux clients les raisons\n"
                "d'approbation ou refus de leur demande de crédit.")

if page == "Information du client":
    st.title("Dashboard Credit Time - Page Information du client")

    if id_client_dash != '<Select>':
        if id_client_dash in data_test['SK_ID_CURR'].values:
            probability, decision = get_prediction(id_client_dash)

            if probability is not None and decision is not None:
                st.write(f"Probabilité de défaut : {probability}, Décision : {decision}")

                if decision == 'Accordé':
                    st.success("Crédit accordé")
                else:
                    st.error("Crédit refusé")

                jauge_score(probability)

                # Ajoutez une colonne "probability" à votre DataFrame data_test pour analyser les clients similaires
                data_test['probability'] = np.nan  # Initialiser la colonne avec NaN
                data_test.loc[data_test['SK_ID_CURR'] == id_client_dash, 'probability'] = probability
                
                analyze_high_default_probability(data_test)

                with st.expander("Afficher les informations du client", expanded=False):
                    st.info("Voici les informations du client:")
                    st.write(data_test[data_test['SK_ID_CURR'] == id_client_dash])
            else:
                st.error("Impossible d'obtenir la prédiction.")
        else:
            st.error("L'ID client n'existe pas dans les données.")
