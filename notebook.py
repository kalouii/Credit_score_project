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
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
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

# Analyse des variables numériques
numeric_features = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'DAYS_BIRTH']

for feature in numeric_features:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=train_data, x=feature, hue='TARGET', kde=True)
    plt.title(f'Distribution of {feature}')
    plt.show()




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

train_data = feature_engineering(train_data)
test_data = feature_engineering(test_data)

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

# Appliquer SMOTE pour gérer le déséquilibre des classes sur l'ensemble d'entraînement sélectionné.
smote_final = SMOTE(random_state=42)
X_train_resampled_final, y_train_resampled_final = smote_final.fit_resample(X_selected_train_final, y_train)

# Vérifiez la distribution des classes après suréchantillonnage.
print("Distribution des classes après suréchantillonnage :")
print(pd.Series(y_train_resampled_final).value_counts())


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


# Loguer le modèle dans MLflow
mlflow.sklearn.log_model(best_voting_clf, "model")

# Optionnel : Inscrire le modèle dans le registre des modèles
model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
mlflow.register_model(model_uri, "Voting_Classifier_Model")

mlflow.log_param("model_name", "Optimized Voting Classifier")
mlflow.log_metric("roc_auc", roc_auc_score(y_test , best_voting_clf.predict_proba(final_X_test_selected_voting_clf)[:, 1]))

mlflow.end_run()  # Terminer l'exécution MLflow après toutes les expériences.

# Analyse SHAP pour l'interprétabilité locale
print("\nÉtape 8: Analyse SHAP")

explainer = shap.TreeExplainer(best_voting_clf)
shap_values = explainer.shap_values(final_X_test_selected_voting_clf)  # Utilisez les données testées

shap.summary_plot(shap_values, final_X_test_selected_voting_clf, plot_type="bar")

# Détection de Dérive des Données (Data Drift)
ref_df = train_data.drop(['TARGET'], axis=1)  # Référence sans la cible
cur_df = test_data.drop(['TARGET'], axis=1)   # Données actuelles sans la cible

data_drift_dataset_report = Report(metrics=[DataDriftTable()])
data_drift_dataset_report.run(reference_data=ref_df, current_data=cur_df)

data_drift_dataset_report.show(mode='inline')