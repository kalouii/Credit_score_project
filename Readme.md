# Projet de Scoring de Crédit

Ce projet utilise des techniques d'apprentissage automatique pour prédire la probabilité de défaut d'un client sur un crédit. Il inclut une API pour faire des prédictions et un tableau de bord interactif pour visualiser les résultats et les analyses.

## Objectifs du Projet

- **Construire un modèle de scoring** qui prédit automatiquement la probabilité de faillite d'un client.
- **Analyser les caractéristiques** qui influencent le modèle, tant au niveau global qu'au niveau individuel.
- **Mettre en production le modèle** via une API et réaliser une interface utilisateur pour tester cette API.
- **Implémenter une approche MLOps** pour gérer le cycle de vie du modèle, y compris le suivi des expérimentations et l'analyse en production du data drift.

## Technologies Utilisées

- **Python** : Langage principal utilisé pour le développement.
- **Flask** : Pour créer l'API RESTful.
- **Streamlit** : Pour le tableau de bord interactif.
- **MLflow** : Pour le suivi des expérimentations et la gestion des modèles.
- **Scikit-learn**, **XGBoost**, **LightGBM** : Bibliothèques pour l'entraînement des modèles.
- **Shap** : Pour l'interprétation des modèles.
- **Evidently** : Pour la détection de dérive des données.

## Installation

1. Clonez le dépôt :
    ```
    git clone https://github.com/votre_utilisateur/votre_projet.git
    cd votre_projet
    ```

2. Installez les dépendances :
    ```
    pip install -r requirements.txt
    ```

## Exécution

### API

Pour exécuter l'API :

L'API sera disponible sur `http://127.0.0.1:5000/`.

### Tableau de Bord

Pour exécuter le tableau de bord :

$streamlit run dashboard.py$



## Fonctionnalités

1. **Prédictions Automatiques** : L'API permet d'obtenir la probabilité de défaut d'un client ainsi que sa classe (accepté ou refusé) en fonction d'un seuil optimisé.
2. **Analyse SHAP** : Le tableau de bord affiche les valeurs SHAP pour expliquer les décisions du modèle à l'échelle locale et globale.
3. **Distribution et Comparaison** : Visualisation des distributions des caractéristiques pour les clients similaires et analyse bivariée.
4. **Dérive des Données** : Utilisation d'Evidently pour surveiller la dérive des données en production.

## MLOps

Le projet a été conçu avec une approche MLOps, incluant :

- Suivi des expérimentations avec MLflow.
- Stockage centralisé des modèles dans un "model registry".
- Tests automatisés avec Pytest intégrés dans un pipeline CI/CD via GitHub Actions.

## Avertissements

- Assurez-vous que l'API est en cours d'exécution avant d'accéder au tableau de bord.
- Les performances du modèle peuvent varier ; vérifiez toujours les métriques pertinentes telles que AUC et accuracy.

## Conclusion

Ce projet vise à fournir une solution complète pour le scoring de crédit, intégrant à la fois l'apprentissage automatique et les meilleures pratiques en matière d'ingénierie logicielle. Si vous avez des questions ou souhaitez contribuer, n'hésitez pas à ouvrir une issue ou à soumettre une pull request.