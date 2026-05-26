# OJS Submission Analytics Dashboard — Project Roadmap & Technical Specifications

This repository contains the source code, data pipelines, and interactive analytics dashboard developed during my research internship at Ritsumeikan University. The project focuses on exploring large-scale code submission datasets (Project CodeNet: AtCoder & AIZU) to model programmer proficiency, error resolution characteristics, and code evolution.

---

## 1. Préambule : Environnement de Développement

Pour garantir la reproductibilité, l'isolation vis-à-vis des autres projets du laboratoire, et l'efficacité face à un volume massif de données (plus de 12 millions de lignes), la pile technique suivante a été rigoureusement sélectionnée :

### 1.1. L'IDE (Environnement de Travail)
* **Visual Studio Code (VS Code)** : Choisi comme éditeur principal pour sa légèreté, son écosystème d'extensions robuste, et son intégration native des **Jupyter Notebooks** (`.ipynb`). Les notebooks seront exclusivement utilisés pour la phase d'exploration R&D et de prototypage rapide avant la refactorisation en scripts Python purs (`.py`).

### 1.2. Le Framework Applicatif (Squelette du Projet)
* **Streamlit** : Ce framework open-source est utilisé comme structure principale pour l'application web. Il permet de convertir des scripts d'analyse de données en applications web interactives et dynamiques de manière native en Python, éliminant ainsi le besoin de développer une couche front-end complexe (HTML/CSS/JavaScript).
* **PyTorch / PyTorch Geometric (Prévu V1/V2)** : Utilisé comme framework de Deep Learning pour la manipulation des tenseurs, des plongements vectoriels (embeddings) et l'entraînement des réseaux de neurones.

### 1.3. Les Librairies Écosystémiques (Boîtes à Outils)
* **Polars / Pandas** : *Pandas* sera utilisé pour le prototypage rapide. Cependant, face aux ~14 millions de soumissions de CodeNet, **Polars** sera privilégié en production pour ses performances hautement optimisées (exécution multi-threadée en Rust) permettant de manipuler des gigaoctets de données en mémoire sans saturation.
* **Scikit-Learn** : La librairie de référence pour le Machine Learning classique. Elle sera le pilier de la V0 pour l'implémentation des algorithmes de réduction de dimensionnalité et de partitionnement de données.
* **Matplotlib & Seaborn** : Utilisées pour la génération de visualisations scientifiques statiques avancées (boxplots logarithmiques, matrices de corrélation, diagrammes de distribution), qui seront ensuite intégrées de manière dynamique dans l'interface Streamlit.

### 1.4. Outillage & Gestion de Projet
* **Conda** : Utilisation obligatoire d'un gestionnaire d'environnements virtuels pour isoler les dépendances du projet et éviter tout conflit sur le serveur du laboratoire.
* **Git & GitHub** : Versionnage du code source et suivi des fonctionnalités à travers le dépôt `OJS-submission-analytics`.
* **GitHub Copilot** : Utilisé comme assistant de programmation IA pour accélérer l'écriture du code boilerplate (notamment la configuration des layouts Streamlit et des paramètres complexes de Matplotlib).

---

## 2. Architecture Itérative du Projet (Roadmap)

Le projet est découpé en trois phases incrémentales (V0, V1, V2) afin de sécuriser des livrables fonctionnels et validés scientifiquement avant d'augmenter la complexité algorithmique.

```
[ Données CSV Brut ] ➔  V0: ML Classique & Métadonnées  ➔ Dashboard V0 (Livrable 1)
                                   ↓
[ Codes Sources Java/Py ] ➔  V1: Embeddings & NLP de Code ➔ Dashboard V1 (Livrable 2)
                                   ↓
[ Graphes de Code / DL ] ➔  V2: Deep Learning Avancé     ➔ Modèle Prédictif Fini
```

### 2.1. Version 0 : MVP — Statistiques Descriptives & Machine Learning Classique
* **Périmètre des données** : Analyse exclusive des **métadonnées tabulaires** (fichiers `problem_list.csv` pour les métriques de structures et fichiers `pXXXXX.csv` contenant l'historique des soumissions : ID utilisateur anonymisé, timestamp, langage utilisé, verdict du juge, temps CPU, mémoire consommée). Le code source brut n'est pas lu à ce stade.
* **Objectifs Scientifiques** : 
    1.  Répliquer de manière dynamique et interactive les conclusions du papier de recherche du laboratoire (*An Empirical Study of the Error Characteristics in an Online Judge System*).
    2.  Vérifier empiriquement des phénomènes précis, comme la prédominance et la difficulté de résolution des erreurs de type *Time Limit Exceeded* (TLE), ou l'inversion des proportions d'erreurs de compilation (CE) vs logiques (TLE) selon la difficulté du problème.
* **Algorithmes de Machine Learning Appliqués** :
    * **ACP (Analyse en Composantes Principales)** : Pour réduire la dimensionnalité des profils d'utilisateurs (basés sur leurs taux d'erreurs, vitesse de soumission, diversité des langages) et projeter ces profils en 2D/3D afin d'identifier visuellement des groupements naturels.
    * **K-Means (Clustering)** : Algorithme non supervisé utilisé pour segmenter et classifier les programmeurs de manière automatique en l'absence d'étiquettes de niveau ("labels") dans le dataset brut. L'objectif est de voir si l'algorithme recrée naturellement les groupes de niveau de G1 à G6 identifiés manuellement par les chercheurs.
* **Livrable Interface** : Un tableau de bord Streamlit fonctionnel en local permettant de filtrer les distributions d'erreurs et les temps de résolution par plateforme (AtCoder / AIZU), par langage et par niveau de difficulté (A à F).

### 2.2. Version 1 : Analyse Textuelle & Introduction aux Embeddings de Code
* **Périmètre des données** : Extension de l'analyse aux **fichiers sources bruts** (`.cpp`, `.py`, `.java`, `.c`) contenus dans l'archive de CodeNet.
* **Objectifs Scientifiques** : Dépasser le simple constat de l'erreur (le verdict "WA" ou "TLE") pour analyser *ce que l'étudiant a écrit*. Le but est de cartographier la similarité sémantique des codes soumis pour un même problème.
* **Techniques Fondamentales** :
    * **Tokenization & Représentation Vectorielle** : Transformation du texte informatique en structures numériques intelligibles par une IA.
    * **Code Embeddings** : Apprentissage et utilisation d'espaces vectoriels denses. Exploration et manipulation de modèles de langage spécialisés pour le code (ex: *CodeBERT*, *GraphCodeBERT* ou tokenizers d'IBM) pour transformer un script entier en un vecteur numérique.
    * **Visualisation Topologique** : Utilisation de techniques de réduction de dimension non linéaires (comme **t-SNE** ou **UMAP**) pour afficher sur le dashboard des "nuages de points" représentant les codes des étudiants. Deux codes appliquant la même logique algorithmique se retrouveront géographiquement proches dans le nuage, permettant de visualiser les patterns de correction ou les erreurs récurrentes.

### 2.3. Version 2 : Deep Learning Avancé & Modèles Prédictifs
* **Périmètre des données** : Intégration complète des structures de données avancées d'IBM CodeNet (graphes de flux de contrôle, arbres de syntaxe abstraite - AST).
* **Objectifs Scientifiques** : Construire des architectures capables d'anticiper le comportement de l'apprenant ou la performance du programme avant même sa soumission à l'OJS.
* **Techniques de Deep Learning Appliquées** :
    * **Réseaux de Neurones Supervisés (Classification Fine)** : Entraînement d'un réseau multi-couches (MLP) ou récurrent sur les embeddings générés en V1 pour prédire à l'avance le type d'erreur exact (Classification parmi CE, WA, RE, TLE) qu'un étudiant va déclencher, en fonction de son profil historique et de son code actuel.
    * **Modélisation de la Trajectoire d'Apprentissage** : Tentative de prédiction du "dropout" (abandon) de l'utilisateur sur la plateforme en analysant la stagnation de ses vecteurs de code au fil du temps.
    * **Exploration GNN (Optionnelle)** : Étude de l'implémentation d'IBM basée sur les *Graph Neural Networks* (GNN) exploitant le mécanisme de *Message Passing* et de *Virtual Nodes* pour analyser l'AST du code source, ouvrant la voie à des systèmes d'aide à la localisation automatique de bugs.

---

## 3. Timeline & Planning Prévisionnel (17 semaines)

Le calendrier suivant est calé sur la durée standard du stage de recherche (mi-mai à mi-septembre), structuré de manière rigoureuse pour jalonner la progression.

* **Phase 1 : Setup & Data Engineering (Semaines 1 — 3)**
  * *Tâches Principales :* Décompression et déploiement de l'archive CodeNet (7.8Go compressés) sur le serveur du laboratoire. Extraction de métadonnées et création d'un pipeline d'échantillonnage de données (via *Polars*). Initialisation du dépôt Git, configuration des environnements Conda.
  * *Livrables :* Environnement de calcul opérationnel. Scripts de nettoyage de données validés sur un échantillon de 100 000 lignes.

* **Phase 2 : Développement V0 (Semaines 4 — 7)**
  * *Tâches Principales :* Calcul des features agrégées par utilisateur (taux de TLE, vitesse d'envoi, etc.). Implémentation de la PCA et du K-Means pour le profilage non supervisé. Codage du layout Streamlit et des filtres dynamiques.
  * *Livrables :* **Livrable V0** : Dashboard fonctionnel affichant les statistiques descriptives et les premiers clusters de profils utilisateurs. Rapport de validation des conclusions du papier initial.

* **Phase 3 : Recherche V1 (Semaines 8 — 11)**
  * *Tâches Principales :* Extraction des fichiers de codes sources bruts correspondants aux pannes majeures. Prise en main des modèles de NLP de code (*CodeBERT*) et extraction des premiers *Embeddings*. Cartographie vectorielle des codes via t-SNE.
  * *Livrables :* **Livrable V1** : Mise à jour du Dashboard avec un onglet "Analyse Sémantique du Code Source". Visualisation graphique des nuages de codes.

* **Phase 4 : Modélisation V2 (Semaines 12 — 14)**
  * *Tâches Principales :* Conception de l'architecture du réseau de neurones sous *PyTorch*. Entraînement et optimisation des hyperparamètres pour la prédiction de verdicts. Intégration optionnelle de données comparatives issues de *LeetCode*.
  * *Livrables :* **Livrable V2** : Modèle prédictif entraîné. Onglet "Prédiction d'Erreurs par IA" fonctionnel sur l'interface Streamlit.

* **Phase 5 : Clôture & Soutenance (Semaines 15 — 17)**
  * *Tâches Principales :* Phase intensive de tests, refactorisation et documentation du code source. Rédaction finale du rapport de stage de recherche en anglais/français. Préparation des supports visuels pour la soutenance devant les enseignants-chercheurs.
  * *Livrables :* **Rapport de recherche final**. Code source documenté sur GitHub. Support de présentation prêt.