# Mémoire du projet : annulations de deux hôtels (TD2 ML, M1 CGAO)

Ce fichier sert de mémoire à l'assistant d'IA : à relire au début de chaque session de travail sur ce dépôt.

## Contexte
- Étudiant : Stevan, Master 1 CGAO (parcours SICG), IGR-IAE Rennes. Projet en groupe de 2 ou 3, noté.
- Rendu : notebook `Projet_COURTES_DIAWARA_LEROUX.ipynb`, lien Streamlit en `.streamlit.app` (ouvert sans compte), fichier `app.py`, lien GitHub envoyé par mail au professeur.
- Données : `reservations_hotels.csv`, 119 390 réservations (Lisbonne et Algarve, arrivées du 01/07/2015 au 31/08/2017), 19 colonnes, une ligne = une réservation. Source : Antonio, de Almeida et Nunes, Hotel booking demand datasets, Data in Brief, 2019, CC BY 4.0.

## Choix faits (étape 1)
- Enfants manquants (4) : remplacés par 0.
- Lignes retirées (827 au total) : 715 réservations de 0 nuit, 180 sans voyageur, 1 prix négatif, 1 prix à 5 400 € la nuit.
- Lignes gardées : prix à 0 € avec au moins une nuit (1 244, séjours offerts), délais de plus d'un an (3 148), lignes identiques (35 652, pas des doublons faute d'identifiant client).
- Fichier propre : 118 563 réservations. CA d'une réservation = nuits x prix_nuit.

## Indicateurs (étape 2)
- Taux d'annulation : annulee moyenne sur toutes les réservations propres.
- CA réalisé, nuits vendues, prix moyen par nuit (CA / nuits) : sur les séjours effectués (annulee = 0).
- CA annulé : somme de ca sur les réservations annulées (annulee = 1).
- Résultats : taux 37,3 % (Lisbonne 41,9 %, Algarve 28,0 %), CA réalisé 25 986 976 €, 254 282 nuits, 102,20 € la nuit, CA annulé 16 721 837 €.

## Modèle (étape 3)
- Régression logistique (`max_iter=3000`), entraînement sur les arrivées avant le 01/01/2017 (78 054 lignes), test sur 2017 (40 509 lignes).
- Variables : hotel, mois, delai_reservation, nuits, adultes, enfants, segment, type_client, client_habitue, annulations_precedentes, depot, prix_nuit.
- Écartées : statut_final et annulee (la cible), chambre_changee (connue à l'arrivée), demandes_speciales (peut changer après la réservation), reservation, date_arrivee, annee.
- Référence "personne n'annule" : 61,2 % d'exactitude. Modèle au seuil 0,5 : exactitude 72,2 %, précision 84,3 %, rappel 34,8 %.
- Seuils : 0,6 pour la surréservation (précision 96,8 %, rappel 27,3 %), 0,3 pour les relances (précision 53,4 %, rappel 80,4 %).
- Avec statut_final (fuite de données) : 100 %, inutilisable. Sans la variable depot : 69,5 %.
- Points à vérifier avec l'hôtel : 99,4 % des dépôts non remboursables sont annulés, 94,5 % des clients avec une annulation précédente.

## Règles de travail
- Chaque chiffre affiché dans `app.py` doit être celui du notebook.
- Écrire sans tics d'IA, guillemets droits, un tiret par sous-question dans les réponses de TD.
- Dans la section "Usage de l'IA", mettre les prompts réels tels qu'envoyés.

## État avant le rendu
- Fait : notebook renommé avec les noms du groupe, application publiée sur Streamlit Community Cloud (https://projet-courtes-diawara-leroux.streamlit.app/), lien collé dans le notebook, section "Usage de l'IA" complétée avec les vrais prompts.
- Versions fixées dans requirements.txt (pandas, numpy, scikit-learn, matplotlib) ; streamlit non fixé car la version 1.51.0 ne s'installe pas sur Python 3.14 (pyarrow). Un écart d'environ 0,1 point de précision est possible entre le notebook (84,6 %) et l'application (84,7 %).
- Reste à faire : tester le lien en navigation privée, puis envoyer le lien GitHub par mail au professeur.
