# Credit Scoring — Prédiction du risque de défaut de paiement

Modèle de scoring crédit qui estime la probabilité qu'un client fasse défaut sur son paiement, à partir de son profil et de son historique de paiement.

## Contexte

Dans une banque ou une fintech, l'octroi de crédit repose souvent sur une analyse manuelle du dossier client — un processus lent et difficile à standardiser d'une agence à l'autre. Ce projet part d'un besoin concret : disposer d'un outil qui donne un score de risque objectif, rapide à calculer, et surtout explicable — ce dernier point compte autant que la performance quand le résultat sert à justifier une décision de crédit.

## Problématique

À partir des données démographiques et de l'historique de paiement d'un client, peut-on prédire s'il fera défaut sur son prochain paiement — et avec quel niveau de confiance ?

## Données

- Source : UCI Machine Learning Repository
- 30 000 clients
- Variables : informations démographiques, historique de paiement, montants facturés, montants payés
- Variable cible : défaut de paiement (1) / client solvable (0), avec un déséquilibre marqué — 22 % de défauts seulement

## Démarche

**1. Analyse exploratoire (EDA)**
Premier constat : le déséquilibre de la cible (22 % de défauts) allait poser problème pour l'entraînement — un modèle naïf pourrait atteindre 78 % de précision juste en prédisant "pas de défaut" à chaque fois, ce qui ne sert à rien en pratique. L'exploration a aussi permis d'identifier les variables les plus liées au risque : les retards de paiement récents et la limite de crédit ressortent nettement, ainsi que des corrélations/redondances entre certaines variables à traiter avant la modélisation.

**2. Feature engineering**
Création de variables agrégées (montant moyen facturé, montant moyen payé) pour synthétiser le comportement de paiement sur plusieurs mois plutôt que de garder des colonnes mois par mois peu exploitables telles quelles. Suppression des variables redondantes ou non pertinentes identifiées à l'étape précédente.

**3. Préprocessing**
Séparation train/test avec stratification (indispensable vu le déséquilibre de la cible, pour garder la même proportion de défauts dans les deux jeux). Standardisation des variables numériques. Gestion du déséquilibre par pondération des classes plutôt que par sur-échantillonnage.

**4. Modélisation**
Choix d'une régression logistique — le modèle de référence en scoring crédit, précisément parce qu'il reste interprétable : chaque variable a un poids qu'on peut expliquer à un client ou à un régulateur, contrairement à un modèle plus complexe type random forest ou XGBoost.

**5. Évaluation et optimisation du seuil**
ROC AUC de 0.71. Recall sur la classe défaut jusqu'à 79 % après optimisation du seuil de décision (fixé à 0.55 plutôt que le seuil par défaut de 0.5). Ce choix de seuil traduit un arbitrage métier : en scoring crédit, rater un client à risque coûte généralement plus cher que refuser par excès de prudence un client solvable — le seuil a été ajusté dans cette logique plutôt que pour maximiser une métrique globale.

## Résultats clés

- Le statut de paiement récent est le facteur le plus prédictif du défaut — plus déterminant que les variables purement démographiques
- Le score produit se traduit directement en trois catégories de décision : accord / analyse manuelle / refus, ce qui le rend directement exploitable dans un processus métier
- Le seuil de décision est un paramètre ajustable, pas une valeur figée — il peut être resserré ou desserré selon la politique de risque de l'établissement

## Limites

- Dataset public (UCI), pas des données bancaires réelles — les variables et leur pouvoir prédictif ne se transposent pas forcément tels quels à un contexte togolais ou UEMOA
- ROC AUC de 0.71 : correct mais pas exceptionnel — un modèle plus complexe (gradient boosting) donnerait probablement une meilleure performance brute, au prix d'une interprétabilité moindre, ce qui n'était pas l'objectif ici
- Le seuil de 0.55 a été choisi de façon raisonnée mais pas calibré sur un vrai coût métier (coût d'un défaut non détecté vs coût d'un refus injustifié) — dans un cas réel, ce chiffre se déciderait avec l'équipe risque

## Comment lancer le projet

```bash
git clone <lien-du-repo>
cd credit-scoring-model
pip install -r requirements.txt
jupyter notebook notebooks/credit_scoring.ipynb
```

## Structure du repo

```
├── data/            # dataset UCI (ou lien de téléchargement si trop volumineux)
├── notebooks/        # EDA, feature engineering, modélisation
├── src/               # fonctions réutilisables (preprocessing, évaluation)
├── visuals/          # courbe ROC, matrice de confusion, importance des variables
├── requirements.txt
└── README.md
```

## Stack technique
`Python` · `pandas` · `scikit-learn` · `matplotlib` / `seaborn`

## Auteur
ALI Essonani (Credus) — [LinkedIn](https://linkedin.com/in/essonaniali) — aliessonani@gmail.com
