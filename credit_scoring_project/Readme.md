🏦 Credit Scoring Project – Fintech / Banking
📌 Contexte

Dans un contexte de banque digitale et de fintech, l’octroi rapide de crédit nécessite des outils fiables permettant d’évaluer le risque de défaut des clients.
Ce projet vise à développer un modèle de scoring crédit capable d’estimer la probabilité de défaut à partir de données financières et comportementales.

🎯 Objectif

Construire un modèle de machine learning permettant de :

Prédire la probabilité de défaut de paiement d’un client

Aider à la prise de décision d’octroi de crédit

Fournir un score interprétable conforme aux exigences réglementaires

📊 Données

Source : UCI Machine Learning Repository

Taille : 30 000 clients

Variables :

Informations démographiques

Historique de paiement

Montants facturés et montants payés

Variable cible :

1 : défaut de paiement

0 : client solvable

🔍 Méthodologie
1. Analyse exploratoire (EDA)

Analyse du déséquilibre de la variable cible (22 % de défaut)

Identification des variables clés du risque (retards de paiement, limite de crédit)

Analyse des corrélations et redondances

2. Feature Engineering

Création de variables agrégées (montants moyens facturés et payés)

Synthèse du comportement de paiement

Suppression des variables non pertinentes

3. Préprocessing

Séparation train / test avec stratification

Standardisation des variables

Gestion du déséquilibre via pondération des classes

4. Modélisation

Régression logistique (modèle de référence en scoring crédit)

Modèle interprétable et robuste

5. Évaluation

ROC AUC : 0.71

Recall défaut : jusqu’à 79 % après optimisation du seuil

Analyse du compromis risque / acceptation

6. Optimisation du seuil

Détermination d’un seuil optimal (0.55)

Traduction du score en décision métier :

Accord

Analyse manuelle

Refus

🧠 Résultats clés

Le statut de paiement récent est le facteur le plus prédictif du défaut

Le modèle permet de détecter efficacement les clients à risque

Le seuil de décision peut être ajusté selon la politique de risque

Conclusion

Ce projet démontre la capacité à transformer un modèle de machine learning en outil opérationnel d’aide à la décision crédit, intégrant à la fois performance, interprétabilité et logique métier.
