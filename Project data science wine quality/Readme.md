Wine Quality Prediction — Classification de la qualité du vin

Projet de classification qui prédit la qualité d'un vin rouge à partir de ses caractéristiques physico-chimiques, avec un focus particulier sur la gestion du déséquilibre des classes et la comparaison méthodique de plusieurs modèles.

Contexte

Ce projet n'est pas ancré dans le secteur bancaire — c'est volontaire. Il sert à démontrer une méthodologie complète de bout en bout (exploration, comparaison de modèles, optimisation, gestion du déséquilibre) sur un jeu de données différent de mes autres projets, pour montrer que la démarche analytique se transpose au-delà d'un seul domaine.

Problématique

Peut-on prédire la note de qualité d'un vin (de 3 à 8) à partir de ses caractéristiques chimiques (acidité, alcool, sulfates, etc.) ? Et si la classification fine à 6 niveaux s'avère difficile, une reformulation du problème permet-elle d'obtenir un modèle réellement exploitable ?

Données
Source : UCI Machine Learning Repository (Wine Quality — Red Wine)
1599 échantillons, 11 variables physico-chimiques + une note de qualité (3 à 8)
Distribution très déséquilibrée : la majorité des vins sont notés 5 ou 6, les notes extrêmes (3, 4, 8) sont rares
Démarche

1. Analyse exploratoire et corrélations La matrice de corrélation fait ressortir trois relations utiles : l'alcool est corrélé positivement à la qualité, l'acidité volatile négativement, et la densité négativement aussi. Le sucre résiduel et les chlorures, eux, n'ont quasiment aucun lien avec la note. Ce constat a directement guidé la sélection des variables pour la première itération du modèle (alcohol, volatile acidity, density, sulphates, citric acid) plutôt que de garder les 11 variables sans discernement.

2. Premier modèle multi-classes (6 niveaux de qualité) Un Random Forest de référence donne une accuracy de 0.60 — mais le rapport de classification révèle le vrai problème : les classes rares (3, 4, 8) ne sont jamais prédites, faute d'exemples suffisants pour apprendre un pattern fiable. L'accuracy globale masque cette faiblesse, parce qu'elle est tirée vers le haut par les classes majoritaires (5 et 6).

3. Comparaison de plusieurs modèles Quatre modèles testés sur les mêmes variables : Random Forest, régression logistique, SVC, Gradient Boosting. Le Random Forest l'emporte en accuracy brute (0.634), mais la régression logistique produit des probabilités mieux calibrées (log loss plus bas, 0.97 contre 0.99). Ce résultat illustre un point souvent oublié : le modèle le plus précis n'est pas toujours celui qui donne les probabilités les plus fiables — un choix qui compte si le score sert ensuite à une décision, pas juste à un classement.

4. Optimisation par GridSearchCV Recherche sur grille (n_estimators, max_depth, min_samples_split, min_samples_leaf) avec validation croisée à 5 plis. Meilleurs paramètres : max_depth=10, n_estimators=300. Résultat notable : l'accuracy baisse légèrement (0.634 → 0.631) mais le log loss s'améliore nettement (0.99 → 0.85) — le modèle optimisé est donc moins précis au sens strict, mais plus fiable dans ses probabilités et moins sujet au surapprentissage.

5. Analyse des erreurs et limite du multi-classe La matrice de confusion confirme le diagnostic initial : confusions marquées entre classes adjacentes (5↔6, 6↔7), classes 3/4/8 quasiment jamais prédites. Le weighted F1-score (0.61) est nettement plus représentatif de la réalité que le macro average (0.30), justement parce que ce dernier pénalise à parts égales des classes qui n'ont presque pas d'exemples.

6. Reformulation en classification binaire Face à ces limites, le problème a été reformulé : qualité > 5 (bon vin) vs qualité ≤ 5 (moyen/mauvais). Cette reformulation rééquilibre naturellement la distribution des classes (environ 850 vs 750 observations) et rend le problème beaucoup plus exploitable en pratique — la question "ce vin est-il bon ou pas" a plus de sens opérationnel que "quelle note exacte sur 8" de toute façon.

Résultats clés

Le modèle binaire final (Random Forest, mêmes hyperparamètres optimisés, mais entraîné sur l'ensemble des variables disponibles) atteint :

Accuracy : 0.80
ROC AUC : 0.889
Precision et recall équilibrés sur les deux classes (~0.79-0.82)

La reformulation binaire est le vrai levier de performance ici — pas un modèle plus complexe, mais une meilleure formulation du problème par rapport à ce que les données permettent réellement d'apprendre.

Limites
Le modèle multi-classes final utilisait 5 variables sélectionnées par corrélation, mais le modèle binaire final a été entraîné sur l'ensemble des variables (le code repasse par df.drop(['quality', 'quality_binary'])) — un choix qui mériterait d'être documenté plus explicitement, et testé aussi avec le sous-ensemble de 5 variables pour vérifier si ça change vraiment la performance.
Les classes rares (3, 4, 8) restent, quoi qu'il arrive, hors de portée d'un modèle entraîné sur si peu d'exemples — un rééquilibrage plus poussé (SMOTE, sur-échantillonnage) n'a pas été testé.
Dataset de taille modeste (1599 lignes) — la marge d'erreur sur les métriques rapportées n'est pas négligeable, une validation croisée sur le modèle binaire final renforcerait la confiance dans les chiffres.
Comment lancer le projet
bash
git clone <lien-du-repo>
cd wine-quality-classification
pip install -r requirements.txt
jupyter notebook notebooks/wine_quality.ipynb
Structure du repo
├── data/                 # winequality-red.csv
├── notebooks/             # EDA, comparaison de modèles, optimisation, version binaire
├── src/                    # fonctions réutilisables (evaluation, preprocessing)
├── visuals/               # matrice de corrélation, matrices de confusion, distributions
├── requirements.txt
└── README.md
Stack technique

Python · pandas · scikit-learn (RandomForest, LogisticRegression, SVC, GradientBoosting, GridSearchCV) · matplotlib / seaborn

Auteur

ALI Essonani (Credusco) GIT — LinkedIn — aliessonani@gmail.com
