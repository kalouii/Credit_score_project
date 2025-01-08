# Import des librairies
import streamlit as st
from PIL import Image
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt

# Chargement des datasets
data_train = pd.read_csv('train_data.csv')  # Remplacez par le bon chemin
data_test = pd.read_csv('test_data.csv')    # Remplacez par le bon chemin

# Configuration de l'API locale
API_URL = "http://127.0.0.1:5000/"  

# Fonction pour récupérer la prédiction via l'API
def get_prediction(client_id):
    url_get_pred = API_URL + "predict/" + str(client_id)
    response = requests.get(url_get_pred)
    proba_default = round(float(response.content), 3)
    best_threshold = 0.54
    decision = "Refusé" if proba_default >= best_threshold else "Accordé"
    return proba_default, decision


def df_voisins(client_id):
    """Récupère les clients similaires à celui dont l'ID est passé en paramètre.
    :param: client_id (int)
    :return: data_voisins
    """
    url_get_df_voisins = API_URL + "clients_similaires/" + str(client_id)
    response = requests.get(url_get_df_voisins)
    data_voisins = pd.read_json(response.content)  # Assurez-vous que le format de réponse est correct
    return data_voisins


# Fonction pour afficher une jauge de score
def jauge_score(proba):
    fig = go.Figure(go.Indicator(
        domain={'x': [0, 1], 'y': [0, 1]},
        value=proba * 100,
        mode="gauge+number+delta",
        title={'text': "Jauge de score"},
        delta={'reference': 54},
        gauge={'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "black"},
               'bar': {'color': "MidnightBlue"},
               'steps': [
                   {'range': [0, 20], 'color': "Green"},
                   {'range': [20, 45], 'color': "LimeGreen"},
                   {'range': [45, 54], 'color': "Orange"},
                   {'range': [54, 100], 'color': "Red"}],
               'threshold': {'line': {'color': "brown", 'width': 4}, 'thickness': 1, 'value': 54}}))
    st.plotly_chart(fig)

# Fonction pour récupérer les valeurs SHAP locales via l'API
def get_shap_val_local(client_id):
    url_get_shap_local = API_URL + "shaplocal/" + str(client_id)
    response = requests.get(url_get_shap_local)
    res = response.json()
    shap_val_local = res['shap_values']
    base_value = res['base_value']
    feat_values = res['data']
    feat_names = res['feature_names']

    explanation = shap.Explanation(np.reshape(np.array(shap_val_local, dtype='float'), (1, -1)),
                                   base_value,
                                   data=np.reshape(np.array(feat_values, dtype='float'), (1, -1)),
                                   feature_names=feat_names)

    return explanation[0]



def get_shap_val():
    """Récupère les shap values globales du jeu de données.
    :return: shap_global
    """
    url_get_shap = API_URL + "shap/"
    response = requests.get(url_get_shap)
    content = response.json()
    shap_val_glob_0 = content['shap_values_0']
    shap_val_glob_1 = content['shap_values_1']
    shap_globales = np.array([shap_val_glob_0, shap_val_glob_1])
    
    return shap_globales

# Titre de la page
st.set_page_config(page_title="Dashboard Credit Time", layout="wide")

# Sidebar
with st.sidebar:
    logo = Image.open('img/credit_time.png')  # Assurez-vous que ce fichier existe
    st.image(logo, width=200)
    
    # Page selection
    page = st.selectbox('Navigation', ["Home", "Information du client", "Interprétation locale", "Interprétation globale"])

    # ID Selection
    st.markdown("""---""")
    
    list_id_client = list(data_test['SK_ID_CURR'])
    list_id_client.insert(0, '<Select>')
    
    id_client_dash = st.selectbox("ID Client", list_id_client)
    
    st.write('Vous avez choisi le client ID : '+str(id_client_dash))

if page == "Home":
    st.title("Dashboard Credit Time - Home Page")
    st.markdown("Ce site contient un dashboard interactif permettant d'expliquer aux clients les raisons\n"
                "d'approbation ou refus de leur demande de crédit.\n"
                "\nLes prédictions sont calculées à partir d'un algorithme d'apprentissage automatique, "
                "préalablement entraîné. Il s'agit d'un modèle *Light GBM* (Light Gradient Boosting Machine). "
                "Les données utilisées sont disponibles [ici](https://www.kaggle.com/c/home-credit-default-risk/data). "
                "Lors du déploiement, un échantillon de ces données a été utilisé.\n"
                "\nLe dashboard est composé de plusieurs pages :\n"
                "- **Information du client**: Vous pouvez y retrouver toutes les informations relatives au client "
                "selectionné dans la colonne de gauche, ainsi que le résultat de sa demande de crédit. "
                "- **Interprétation locale**: Vous pouvez y retrouver quelles caractéristiques du client ont le plus "
                "influencé le choix d'approbation ou refus de la demande de crédit.\n"
                "- **Interprétation globale**: Vous pouvez y retrouver notamment des comparaisons du client avec "
                "les autres clients de la base de données ainsi qu'avec des clients similaires.")

if page == "Information du client":
    st.title("Dashboard Credit Time - Page Information du client")

    if id_client_dash != '<Select>':
        probability, decision = get_prediction(id_client_dash)

        if decision == 'Accordé':
            st.success("Crédit accordé")
        else:
            st.error("Crédit refusé")

        jauge_score(probability)

        # Affichage des informations client
        with st.expander("Afficher les informations du client", expanded=False):
            st.info("Voici les informations du client:")
            st.write(pd.DataFrame(data_test.loc[data_test['SK_ID_CURR'] == id_client_dash]))

if page == "Interprétation locale":
    st.title("Dashboard Credit Time - Page Interprétation locale")

    if id_client_dash != '<Select>':
        shap_val_local = get_shap_val_local(id_client_dash)
        nb_features = st.slider('Nombre de variables à visualiser', 0, 20, 10)
        
        fig = shap.waterfall_plot(shap_val_local, max_display=nb_features, show=False)
        st.pyplot(fig)

if page == "Interprétation globale":
    st.title("Dashboard Credit Time - Page Interprétation globale")
    
    if id_client_dash != '<Select>':
        data_voisins = df_voisins(id_client_dash)

        globale = st.checkbox("Importance globale")
        if globale:
            shap_values_globales = get_shap_val()  # Récupérer les valeurs SHAP globales si nécessaire
            
            # Affichage des valeurs SHAP globales ici si besoin
            
            nb_features = st.slider('Nombre de variables à visualiser', 0, 20, 10)
            
            fig, ax = plt.subplots()
            ax = shap.summary_plot(shap_values_globales[1], data_train.drop('TARGET', axis=1), plot_type='bar', max_display=nb_features)
            st.pyplot(fig)

            with st.expander("Explication du graphique", expanded=False):
                st.caption("Ici sont affichées les caractéristiques influençant de manière globale la décision.")