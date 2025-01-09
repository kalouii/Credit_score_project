# Importation des bibliothèques nécessaires
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 
from sklearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from imblearn.over_sampling import SMOTE
import shap
from scipy import stats
from scipy.stats import randint, uniform
import gc
import mlflow
import mlflow.sklearn
from evidently.report import Report
from evidently.metrics import DataDriftTable, DatasetDriftMetric
import mlflow.pyfunc
from shap import TreeExplainer
from IPython.display import display
import warnings
from sklearn.metrics import roc_curve




# Configuration de MLflow pour le suivi des expériences
mlflow.set_tracking_uri('http://127.0.0.1:5000/')

# Configuration pour afficher toutes les colonnes
pd.set_option('display.max_columns', None)

# Fonction pour réduire l'utilisation de la mémoire
def reduce_mem_usage(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    return df

# Chargement des données
print("Étape 1: Chargement des données")

train_data = pd.read_csv("C:/Users/kaloui/application_train.csv")
test_data = pd.read_csv("C:/Users/kaloui/application_test.csv")
train_data = reduce_mem_usage(train_data)
test_data = reduce_mem_usage(test_data)

print("Shape of train_data:", train_data.shape)
print("Shape of test_data:", test_data.shape)

# Analyse exploratoire des données
print("\nÉtape 2: Analyse exploratoire des données")

# Distribution de la variable cible
print("Distribution de la variable cible:")
target_counts = train_data['TARGET'].value_counts()
print(target_counts)
print(target_counts / len(train_data))

plt.figure(figsize=(8, 6))
sns.countplot(x='TARGET', data=train_data)
plt.title('Distribution de la variable cible')
plt.show()

# Ajout d'un graphique en camembert pour visualiser la distribution de la variable cible
plt.figure(figsize=(8, 6))
plt.pie(target_counts, labels=['Non Approuvé (0)', 'Approuvé (1)'], autopct='%1.1f%%', startangle=90)
plt.title('Répartition de la Variable Cible')
plt.axis('equal')  # Pour faire un cercle parfait
plt.show()


# Configuration pour afficher toutes les colonnes et toutes les lignes
pd.set_option('display.max_columns', None)  # Afficher toutes les colonnes
pd.set_option('display.max_rows', None)  
# Fonction pour calculer le pourcentage de valeurs manquantes
def missing_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1, keys=['Missing Values', '% of Total Values'])
    mis_val_table_ren_columns = mis_val_table.rename(
        columns = {'Missing Values': 'Missing Values', '% of Total Values': '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
           "There are " + str(mis_val_table_ren_columns.shape[0]) +
           " columns that have missing values.")
    return mis_val_table_ren_columns

print("\nAnalyse des valeurs manquantes:")
missing_values = missing_values_table(train_data)
print(missing_values.head(20))


print("\nAnalyse des valeurs manquantes:")
missing_values = missing_values_table(train_data)

# Utiliser display() pour montrer le DataFrame
display(missing_values)  


# Visualisation des valeurs manquantes avec un graphique à barres
plt.figure(figsize=(12, 8))
sns.barplot(x='Missing Values', y=missing_values.index, data=missing_values)
plt.title('Nombre de Valeurs Manquantes par Colonne')
plt.xlabel('Nombre de Valeurs Manquantes')
plt.ylabel('Colonnes')
plt.show()


# Afficher les types de données des colonnes dans train_data
sensitive_info = train_data.dtypes.reset_index()
sensitive_info.columns = ['Column Name', 'Data Type']
display(sensitive_info)

# Liste des colonnes sensibles à analyser
sensitive_columns = [
    'CODE_GENDER', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 
    'AMT_GOODS_PRICE', 'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 
    'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 
    'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 
    'FLAG_PHONE', 'FLAG_EMAIL', 'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS', 
    'ORGANIZATION_TYPE', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 
    'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE', 
    'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE', 
    'DAYS_LAST_PHONE_CHANGE', 'AMT_REQ_CREDIT_BUREAU_HOUR', 
    'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK', 
    'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT', 
    'AMT_REQ_CREDIT_BUREAU_YEAR'
]

# Filtrer les colonnes numériques parmi les colonnes sensibles
numeric_sensitive_columns = [col for col in sensitive_columns if col in train_data.columns and train_data[col].dtype != 'object']

print("Colonnes numériques sensibles :")
for col in numeric_sensitive_columns:
    print(f"{col}: {train_data[col].dtype}")


# 2. Créer un DataFrame avec des statistiques descriptives

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    stats_df = train_data[numeric_sensitive_columns].describe()
    display(stats_df)


# 3. Visualiser la distribution de chaque variable avec des histogrammes
fig, axes = plt.subplots(4, 4, figsize=(20, 20))
fig.suptitle("Distribution des variables numériques sensibles", fontsize=16)

for i, col in enumerate(numeric_sensitive_columns[:16]):
    ax = axes[i//4, i%4]
    sns.histplot(train_data[col].dropna(), ax=ax, kde=True)
    ax.set_title(col)
    ax.set_xlabel('')
    
plt.tight_layout()
plt.show()



# Suppression des avertissements
warnings.filterwarnings('ignore')

# Liste des nouvelles features
new_features = ['CREDIT_INCOME_PERCENT', 'ANNUITY_INCOME_PERCENT', 'CREDIT_TERM', 
                'DAYS_EMPLOYED_PERCENT', 'EXT_SOURCE_MEAN', 'EXT_SOURCE_STD']

# Création de la figure
fig, axes = plt.subplots(3, 2, figsize=(20, 20))
fig.suptitle("Distribution des Nouvelles Features par TARGET", fontsize=16)

# Boucle pour créer les graphiques
for i, feature in enumerate(new_features):
    ax = axes[i // 2, i % 2]
    
    # Création du violin plot
    sns.violinplot(data=train_data, x='TARGET', y=feature, ax=ax)
    
    ax.set_title(feature)
    ax.set_xlabel('TARGET')
    ax.set_ylabel(feature)

plt.tight_layout()
plt.show()





# Analyse Bivariée 

def analyze_variable(df, var):
    # Création de l'histogramme
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x=var, hue='TARGET', multiple='stack', stat='count')
    plt.title(f'Distribution de {var} par TARGET')
    plt.xlabel(var)
    plt.ylabel('Nombre')
    plt.legend(title='TARGET', labels=['0 (Non défaut)', '1 (Défaut)'])
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Calcul des statistiques
    stats = df.groupby(var)['TARGET'].agg(['count', 'sum'])
    stats['non_default'] = stats['count'] - stats['sum']
    stats['default_rate'] = (stats['sum'] / stats['count']) * 100
    stats = stats.rename(columns={'sum': 'default', 'count': 'total'})
    stats = stats[['total', 'non_default', 'default', 'default_rate']]
    
    print(f"\nStatistiques pour {var}:")
    print(stats.round(2))
    print("\n")

# Préparation des données pour AMT_CREDIT
train_data['AMT_CREDIT_RANGE'] = pd.cut(train_data['AMT_CREDIT'], bins=10)
train_data['AMT_CREDIT_RANGE'] = train_data['AMT_CREDIT_RANGE'].astype(str)

# Liste des variables à analyser
variables = ['CODE_GENDER', 'AMT_CREDIT_RANGE', 'OCCUPATION_TYPE', 'NAME_EDUCATION_TYPE']

# Analyse de chaque variable
for var in variables:
    analyze_variable(train_data, var)




# Feature Engineering
print("\nÉtape 3: Feature Engineering")

def feature_engineering(df):
    df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    
    df['EXT_SOURCE_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['EXT_SOURCE_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    
    return df
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    train_data = feature_engineering(train_data)
    test_data = feature_engineering(test_data)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    print("Nouvelles features créées:")
    print(train_data[['CREDIT_INCOME_PERCENT', 'ANNUITY_INCOME_PERCENT', 'CREDIT_TERM', 'DAYS_EMPLOYED_PERCENT', 'EXT_SOURCE_MEAN', 'EXT_SOURCE_STD']].describe())




# Calculer les pourcentages de valeurs manquantes
missing_stats = missing_values_table(train_data)

# Identifier les colonnes à supprimer (>40% ou entre 0% et 1% de valeurs manquantes)
columns_to_drop = missing_stats[
    (missing_stats['% of Total Values'] > 40) | 
    ((missing_stats['% of Total Values'] > 0) & (missing_stats['% of Total Values'] < 1))
].index.tolist()

# Supprimer les colonnes identifiées
train_data = train_data.drop(columns=columns_to_drop)
test_data = test_data.drop(columns=columns_to_drop)





# Séparation en ensembles d'entraînement et de validation
df, df_val = train_test_split(train_data, test_size=0.2, random_state=42, stratify=train_data['TARGET'])


# Identifier les colonnes avec des NaN après nettoyage
missing_columns = df.columns[df.isnull().any()].tolist()
print("Colonnes avec des valeurs manquantes dans df :", missing_columns)



# Remplacer NaN par 'Unknown' pour les variables catégorielles spécifiques
categorical_columns = ['OCCUPATION_TYPE']
for col in categorical_columns:
    if col in df.columns:
        df[col].fillna('Unknown', inplace=True)
    if col in df.columns:
        df_val[col].fillna('Unknown', inplace=True)

# Supprimer les lignes où TARGET = 0 et où il y a des NaN dans les colonnes identifiées
for col in missing_columns:
    df = df[~((df['TARGET'] == 0) & (df[col].isnull()))]

# Supprimer toutes les lignes où TARGET est NaN
df = df[df['TARGET'].notnull()]

# Réinitialiser l'index après suppression
df.reset_index(drop=True, inplace=True)

# Imputation des NaN restants dans les colonnes numériques avec KNNImputer
knn_imputer = KNNImputer(n_neighbors=5)  # k le nombre de voisins

# Appliquer KNNImputer sur les colonnes numériques restantes
numeric_columns = df[missing_columns].select_dtypes(include=['float16', 'float32', 'float64', 'int16', 'int32', 'int64']).columns # Sélectionner uniquement les colonnes numériques

# Imputer les valeurs manquantes
df[numeric_columns] = knn_imputer.fit_transform(df[numeric_columns])

# Vérifiez à nouveau les valeurs manquantes après imputation avec KNN
print("Valeurs manquantes dans df après imputation avec KNN :")
missing_values_after_knn_imputation = df.isnull().sum().sort_values(ascending=False)
print(missing_values_after_knn_imputation)








# Préparation des données pour la modélisation
print("\nÉtape 4: Préparation des données pour la modélisation")

# Séparation des features et de la target
X = df.drop('TARGET', axis=1)  # Retirer TARGET des caractéristiques
y = df['TARGET']  # Conserver TARGET comme variable cible

# Séparation en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Encodage des variables catégorielles sur l'ensemble d'entraînement et de test.
cat_columns_final = X.select_dtypes(include=['object']).columns.tolist()
onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_train_encoded = onehot_encoder.fit_transform(X_train[cat_columns_final])
X_test_encoded = onehot_encoder.transform(X_test[cat_columns_final])

# Normalisation des variables numériques (sans enlever TARGET)
num_columns_final = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
scaler_final = StandardScaler()

X_train_scaled = scaler_final.fit_transform(X_train[num_columns_final])
X_test_scaled = scaler_final.transform(X_test[num_columns_final])

# Combiner les DataFrames tout en conservant TARGET dans le DataFrame original 
final_X_train = np.hstack((X_train_scaled, X_train_encoded))
final_X_test = np.hstack((X_test_scaled, X_test_encoded))

print("Forme finale du DataFrame d'entraînement :", final_X_train.shape)

# Sélection des features avec SelectKBest sur les données d'entraînement combinées avant SMOTE.
selector_final = SelectKBest(f_classif, k=50)
X_selected_train_final = selector_final.fit_transform(final_X_train, y_train)

selected_features_final = selector_final.get_feature_names_out().tolist()
print("Selected features after initial selection:", selected_features_final)

# Modélisation sans SMOTE : Entraînement initial des modèles
models_initial  = {
     'Dummy': DummyClassifier(strategy='stratified'),
      'Logistic Regression': LogisticRegression(random_state=42),
      'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
      'XGBoost': XGBClassifier(n_estimators=50), 
      'LightGBM': LGBMClassifier(n_estimators=50)  
}

# Fonction de coût personnalisée pour évaluation du modèle
def custom_cost(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return 10 * fn + fp  # Pénalise 10 fois plus les faux négatifs que les faux positifs

# Fonction pour évaluer les modèles
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"AUC-ROC Score: {roc_auc_score(y_test, y_pred_proba)}")
    print(f"Custom Cost: {custom_cost(y_test, y_pred)}")

# Entraînement et évaluation des modèles sans SMOTE
for name, model in models_initial.items():
    print(f"\nEvaluating {name}")
    try:
        model.fit(X_selected_train_final, y_train)  # Utiliser l'ensemble d'entraînement sélectionné
        
        # Évaluation sur l'ensemble de test (appliquer le même prétraitement que pour l'entraînement).
        final_X_test_selected_voting_clf = selector_final.transform(final_X_test)
        evaluate_model(model, final_X_test_selected_voting_clf, y_test)  
        
    except Exception as e:
        print(f"An error occurred while evaluating {name}: {e}")


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    results = {
        'Précision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'Score F1': f1_score(y_test, y_pred),
        'AUC-ROC': roc_auc_score(y_test, y_pred_proba),
        'Coût personnalisé': 10 * fn + fp
    }
    
    return pd.DataFrame([results])

# Tableau récapitulatif pour tous les modèles
all_results_after_smote = pd.DataFrame()

for name, models_initial in models_initial.items():
    model.fit(X_selected_train_final, y_train)
    model_results = evaluate_model(model, final_X_test_selected_voting_clf, y_test)
    model_results.index = [name]
    all_results_after_smote = pd.concat([all_results_after_smote, model_results])

print("Tableau récapitulatif des performances des modèles :")
display(all_results_after_smote)



# Appliquer SMOTE pour gérer le déséquilibre des classes sur l'ensemble d'entraînement sélectionné.
smote_final = SMOTE(random_state=42)
X_train_resampled_final, y_train_resampled_final = smote_final.fit_resample(X_selected_train_final, y_train)

# Vérifiez la distribution des classes après suréchantillonnage.
print("Distribution des classes après suréchantillonnage :")
print(pd.Series(y_train_resampled_final).value_counts())

# Évaluation des modèles individuels après SMOTE

# Tableau récapitulatif pour tous les modèles
all_results_after_smote = pd.DataFrame()

for name, models_initial in models_initial.items():
    model.fit(X_selected_train_final, y_train)
    model_results = evaluate_model(model, final_X_test_selected_voting_clf, y_test)
    model_results.index = [name]
    all_results_after_smote = pd.concat([all_results_after_smote, model_results])

print("Tableau récapitulatif des performances des modèles :")
display(all_results_after_smote)



# Définir le Voting Classifier après SMOTE.
voting_clf = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100)),
        ('xgb', XGBClassifier()),
        ('lgb', LGBMClassifier())
    ],
    voting='soft'
)





# Démarrer une nouvelle expérience MLflow pour Voting Classifier.
mlflow.start_run()
param_value = 0.01  # Exemple de valeur pour un paramètre
metric_value = 0.85  # Exemple de valeur pour une métrique

# Optimisation des Hyperparamètres avec GridSearchCV pour Voting Classifier.
param_grid_voting = {
    'rf__n_estimators': [50, 100],
    'xgb__learning_rate': [0.01, 0.1],
    'lgb__num_leaves': [31, 50]
}

grid_search_voting = GridSearchCV(estimator=voting_clf, param_grid=param_grid_voting, cv=5)
grid_search_voting.fit(X_train_resampled_final, y_train_resampled_final)

print(f"Meilleurs hyperparamètres pour Voting Classifier : {grid_search_voting.best_params_}")

# Entraînement du modèle optimisé du Voting Classifier 
best_voting_clf = grid_search_voting.best_estimator_
best_voting_clf.fit(X_train_resampled_final, y_train_resampled_final)

final_X_test_selected_voting_clf = selector_final.transform(final_X_test)
y_pred_voting_clf_optimized = best_voting_clf.predict(final_X_test_selected_voting_clf)

print("\nÉvaluation du Voting Classifier optimisé :")
evaluate_model(best_voting_clf, final_X_test_selected_voting_clf , y_test)


# Calculer les métriques d'évaluation
y_pred = best_voting_clf.predict(final_X_test_selected_voting_clf)
y_pred_proba = best_voting_clf.predict_proba(final_X_test_selected_voting_clf)[:, 1]

# Calcul des métriques
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)


# Calcul du coût métier
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
cost_misclassification = (10 * fn) + fp 


# Journaliser les métriques dans MLflow
mlflow.log_metric("val_precision", precision)
mlflow.log_metric("val_recall", recall)
mlflow.log_metric("val_f1_score", f1)
mlflow.log_metric("val_rocauc", roc_auc)


# Créer un exemple d'entrée à partir de X_train_resampled_final
input_example = X_train_resampled_final[0].reshape(1, -1)  # Prendre la première ligne comme exemple


# Loguer le modèle dans MLflow avec l'exemple d'entrée
mlflow.sklearn.log_model(best_voting_clf, "model", input_example=input_example)

# Optionnel : Inscrire le modèle dans le registre des modèles
model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
mlflow.register_model(model_uri, "Voting_Classifier_Model")

mlflow.log_param("model_name", "Optimized Voting Classifier")
mlflow.log_metric("roc_auc", roc_auc_score(y_test , best_voting_clf.predict_proba(final_X_test_selected_voting_clf)[:, 1]))
mlflow.end_run()  # Terminer l'exécution MLflow après toutes les expériences.


# Charger le modèle enregistré
model_uri = "models:/Voting_Classifier_Model/1"  # Remplacez '1' par la version souhaitée
loaded_model = mlflow.pyfunc.load_model(model_uri)

# Préparez vos nouvelles données d'entrée (assurez-vous qu'elles ont la même structure que celles utilisées lors de l'entraînement)
new_data = final_X_test_selected_voting_clf  # Remplacez par vos nouvelles données

# Faire des prédictions avec le modèle chargé
predictions = loaded_model.predict(new_data)

print("Prédictions :", predictions)

# Valeurs Prédites :Les valeurs 0 et 1 correspondent typiquement à deux classes dans un problème de classification binaire. Par exemple, dans un problème de scoring de crédit, 0 pourrait représenter "non approuvé" et 1 pourrait représenter "approuvé".
# La sortie indique que le modèle a prédit que la majorité des instances dans votre ensemble de test appartiennent à la classe 0, avec quelques instances classées comme 1.


# Évaluer les performances du modèle
print("\nRapport de classification :")
print(classification_report(y_test, predictions))  # y_test est la vérité terrain

# Matrice de confusion
cm = confusion_matrix(y_test, predictions)
print("\nMatrice de confusion :")
print(cm)


# Visualiser la matrice de confusion
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non Approuvé', 'Approuvé'], yticklabels=['Non Approuvé', 'Approuvé'])
plt.ylabel('Vérité Terrain')
plt.xlabel('Prédictions')
plt.title('Matrice de Confusion')
plt.show()


# Interprétation
#Déséquilibre des Classes : Les résultats montrent un déséquilibre entre les classes, ce qui est souvent un problème dans les problèmes de classification binaire. Votre modèle est très performant pour prédire la classe majoritaire (classe 0) mais a du mal avec la classe minoritaire (classe 1).
#Faible Précision et Rappel pour la Classe 1 : Cela indique que le modèle ne parvient pas à identifier correctement les cas positifs, ce qui peut être problématique dans des applications où il est crucial d'identifier ces cas.

# Charger le modèle depuis MLflow
model_uri = "models:/Voting_Classifier_Model/1"  # Remplacez par votre URI
loaded_model = mlflow.pyfunc.load_model(model_uri)

# Sélectionner les colonnes catégorielles
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()

# Encoder les variables catégorielles
onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_encoded = onehot_encoder.fit_transform(X[categorical_columns])

# Créer un DataFrame avec les colonnes encodées
X_encoded_df = pd.DataFrame(X_encoded, columns=onehot_encoder.get_feature_names_out(categorical_columns))

# Combiner avec les colonnes numériques
numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
X_final = pd.concat([X_encoded_df, X[numeric_columns].reset_index(drop=True)], axis=1)


# Sélection des features avec SelectKBest sur le DataFrame final
selector_final = SelectKBest(f_classif, k=50)
X_selected_train_final = selector_final.fit_transform(X_final, y)

# Obtenir les indices des caractéristiques sélectionnées
selected_indices = selector_final.get_support(indices=True)

# Obtenir les noms des caractéristiques sélectionnées à partir du DataFrame d'origine
selected_features_final = X_final.columns[selected_indices].tolist()
print("Selected features after initial selection:", selected_features_final)
# Étape 8: Analyse SHAP
print("\nÉtape 8: Analyse SHAP")

# Créer un échantillon aléatoire de 100 lignes à partir des données de test pour SHAP
X_sample = final_X_test_selected_voting_clf[np.random.choice(final_X_test_selected_voting_clf.shape[0], size=100, replace=False)]

# Convertir X_sample en DataFrame avec les bons noms de colonnes
X_sample_df = pd.DataFrame(X_sample, columns=selected_features_final)

# Créer une fonction de prédiction
def predict_fn(data):
    return best_voting_clf.predict_proba(data)[:, 1]  # Assurez-vous d'utiliser predict_proba

# Initialiser KernelExplainer avec la fonction de prédiction
explainer = shap.KernelExplainer(predict_fn, X_sample_df)

# Calculer les valeurs SHAP pour l'échantillon sélectionné
shap_values = explainer.shap_values(X_sample_df)

# Visualisation des valeurs SHAP
shap.summary_plot(shap_values, X_sample_df)



# Détection de Dérive des Données (Data Drift)
# Vérifier si 'TARGET' existe dans train_data avant de le supprimer

# Vérification et gestion des NaN
if train_data.isnull().values.any():
    print("Des NaN ont été trouvés dans train_data.")
    train_data.fillna(0, inplace=True)  # Remplacer par 0 ou une autre méthode

if test_data.isnull().values.any():
    print("Des NaN ont été trouvés dans test_data.")
    test_data.fillna(0, inplace=True)  # Remplacer par 0 ou une autre méthode

# Détection de Dérive des Données (Data Drift)
# Vérifier si 'TARGET' existe dans train_data avant de le supprimer
if 'TARGET' in train_data.columns:
    ref_df = train_data.drop(columns=['TARGET'])  # Référence sans la cible
else:
    raise ValueError("La colonne 'TARGET' n'existe pas dans train_data.")

# Vérifier si 'TARGET' existe dans test_data avant de le supprimer
if 'TARGET' in test_data.columns:
    cur_df = test_data.drop(columns=['TARGET'])  # Données actuelles sans la cible
else:
    cur_df = test_data.copy()  # Utiliser test_data tel quel si TARGET n'existe pas

# Suppression des avertissements d'overflow lors de l'exécution du rapport
with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
    data_drift_dataset_report = Report(metrics=[DataDriftTable()])
    data_drift_dataset_report.run(reference_data=ref_df, current_data=cur_df)

# Afficher le rapport
data_drift_dataset_report.show(mode='inline')
