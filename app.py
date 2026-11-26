# Tableau de bord Streamlit : les annulations de deux hôtels portugais (Lisbonne et Algarve)
# Projet TD2 Machine Learning, Master 1 CGAO. Toutes les règles sont celles du notebook.
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

st.set_page_config(page_title="Annulations hôtelières", layout="wide")

# Aspect commun des graphiques : mêmes couleurs pour les deux hôtels partout, texte lisible
COULEURS = {"Lisbonne": "#1F5FA8", "Algarve": "#E8833A"}
COULEUR_BARRES = "#5B7DB1"
plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})


def fr(nombre, decimales=0):
    # écrit un nombre à la française : 25 950 266 ou 0,76
    return f"{nombre:,.{decimales}f}".replace(",", " ").replace(".", ",")


def pc(proportion):
    # écrit une proportion en pourcentage : 37,3 %
    return f"{100 * proportion:.1f}".replace(".", ",") + " %"


# ---------- 1. Données propres (étape 1 du notebook) ----------
@st.cache_data(show_spinner="Chargement et nettoyage des données...")
def charger():
    d = pd.read_csv("reservations_hotels.csv")
    d["enfants"] = d["enfants"].fillna(0)                       # 4 valeurs manquantes -> 0
    anomalies = ((d["nuits"] == 0)                              # séjour de 0 nuit
                 | ((d["adultes"] + d["enfants"]) == 0)         # aucun voyageur
                 | (d["prix_nuit"] < 0) | (d["prix_nuit"] > 1000))  # prix absurde
    p = d[~anomalies].copy()
    p["ca"] = p["nuits"] * p["prix_nuit"]                        # CA = nuits x prix par nuit
    p["periode"] = pd.to_datetime(p["date_arrivee"]).dt.to_period("M").dt.to_timestamp()
    return p, int(anomalies.sum())


propres, nb_retirees = charger()


# ---------- 2. Indicateurs (étape 2 du notebook) ----------
def indicateurs(t):
    realisees, annulees = t[t["annulee"] == 0], t[t["annulee"] == 1]
    nuits = realisees["nuits"].sum()
    return {
        "réservations": len(t),
        "taux": t["annulee"].mean() if len(t) else 0,          # annulées / toutes les réservations
        "ca": realisees["ca"].sum(),                            # séjours effectués
        "nuits": nuits,                                         # séjours effectués
        "prix": realisees["ca"].sum() / nuits if nuits else 0,  # CA réalisé / nuits vendues
        "ca_annule": annulees["ca"].sum(),                      # réservations annulées
    }


# Indicateur proposé pour le graphique mensuel : (clé, unité de l'axe, facteur d'affichage)
CHOIX_MENSUEL = {
    "CA réalisé": ("ca", "milliers d'euros", 1 / 1000),
    "Taux d'annulation": ("taux", "%", 100),
    "Nuits vendues": ("nuits", "nuits", 1),
    "Prix moyen par nuit": ("prix", "euros par nuit", 1),
    "CA annulé": ("ca_annule", "milliers d'euros", 1 / 1000),
}


def par_mois(t, cle, facteur):
    # un indicateur calculé pour chaque mois d'arrivée et chaque hôtel
    lignes = {}
    for (periode, hotel), g in t.groupby(["periode", "hotel"]):
        lignes[(periode, hotel)] = indicateurs(g)[cle] * facteur
    return pd.Series(lignes).unstack()


# ---------- 3. Modèle (étape 3 du notebook) ----------
COLONNES = ["hotel", "mois", "delai_reservation", "nuits", "adultes", "enfants", "segment",
            "type_client", "client_habitue", "annulations_precedentes", "depot", "prix_nuit"]
QUALITATIVES = ["hotel", "mois", "segment", "type_client", "depot"]


@st.cache_resource(show_spinner="Entraînement du modèle de prévision en cours, environ une minute au premier affichage...")
def entrainer():
    X = pd.get_dummies(propres[COLONNES], columns=QUALITATIVES, dtype=int)
    y = propres["annulee"]
    train = propres["date_arrivee"] < "2017-01-01"              # entraînement : avant 2017
    # solveur exact : le résultat est le même que dans le notebook, sur toutes les machines
    modele = LogisticRegression(solver="newton-cholesky", max_iter=3000).fit(X[train], y[train])
    return modele, list(X.columns), modele.predict_proba(X[~train])[:, 1], y[~train]


modele, colonnes_modele, proba_test, y_test = entrainer()

# ---------- 4. Page ----------
st.title("Les annulations de deux hôtels portugais")
st.caption(f"{fr(len(propres))} réservations propres, arrivées du 1er juillet 2015 au 31 août 2017 "
           f"({fr(nb_retirees)} lignes retirées). Source : Antonio, de Almeida et Nunes, "
           "Hotel booking demand datasets, Data in Brief, 2019 (CC BY 4.0).")

onglet1, onglet2 = st.tabs(["Indicateurs de gestion", "Prévoir une annulation"])

with onglet1:
    with st.sidebar:
        st.header("Filtres")
        hotels = st.multiselect("Hôtel", ["Lisbonne", "Algarve"], default=["Lisbonne", "Algarve"])
        annees = st.multiselect("Année d'arrivée", [2015, 2016, 2017], default=[2015, 2016, 2017])
        st.caption("Les filtres s'appliquent à l'onglet « Indicateurs de gestion ».")
    t = propres[propres["hotel"].isin(hotels) & propres["annee"].isin(annees)]

    if t.empty:
        st.warning("Aucune réservation avec ces filtres.")
    else:
        k = indicateurs(t)
        cartes = [("Taux d'annulation", pc(k["taux"]), "Réservations annulées ou non présentées / toutes les réservations"),
                  ("CA réalisé", fr(k["ca"]) + " €", "Nuits x prix des séjours effectués"),
                  ("Nuits vendues", fr(k["nuits"]), "Nuits des séjours effectués"),
                  ("Prix moyen par nuit", fr(k["prix"], 2) + " €", "CA réalisé / nuits vendues"),
                  ("CA annulé", fr(k["ca_annule"]) + " €", "Nuits x prix des réservations annulées")]
        for colonne, (nom, valeur, aide) in zip(st.columns(5), cartes):
            with colonne.container(border=True):
                st.metric(nom, valeur, help=aide)
        st.caption("Taux d'annulation : annulées ou non présentées / toutes les réservations. "
                   "CA réalisé, nuits et prix moyen : séjours effectués (prix moyen = CA / nuits). "
                   "CA annulé : nuits x prix des réservations annulées.")

        # Graphique mensuel : la direction choisit l'indicateur, par hôtel et par mois
        choix = st.selectbox("Indicateur à suivre mois par mois", list(CHOIX_MENSUEL))
        cle, unite, facteur = CHOIX_MENSUEL[choix]
        mensuel = par_mois(t, cle, facteur)
        gauche, droite = st.columns([2, 1])
        with gauche:
            fig, ax = plt.subplots(figsize=(10, 3.6))
            for h in mensuel.columns:
                ax.plot(mensuel.index, mensuel[h], marker="o", markersize=4, linewidth=2,
                        label=h, color=COULEURS[h])
            ax.set_title(choix + " par mois d'arrivée"); ax.set_ylabel(unite)
            ax.legend(frameon=False); fig.autofmt_xdate()
            st.pyplot(fig)
        with droite:
            st.subheader("Par hôtel")
            lignes = [("Réservations", "réservations", fr), ("Taux d'annulation", "taux", pc),
                      ("CA réalisé (€)", "ca", fr), ("Nuits vendues", "nuits", fr),
                      ("Prix moyen (€)", "prix", lambda v: fr(v, 2)), ("CA annulé (€)", "ca_annule", fr)]
            par_hotel = {h: indicateurs(t[t["hotel"] == h]) for h in hotels}
            tab = pd.DataFrame({h: {nom: mise_en_forme(par_hotel[h][cle_h]) for nom, cle_h, mise_en_forme in lignes}
                                for h in hotels})
            st.dataframe(tab)

        g1, g2 = st.columns(2)
        with g1:
            classes = pd.cut(t["delai_reservation"], [-1, 7, 30, 90, 180, 365, 800],
                             labels=["0-7 j", "8-30 j", "31-90 j", "91-180 j", "181-365 j", "plus d'un an"])
            par_delai = t.groupby(classes, observed=True)["annulee"].mean() * 100
            fig, ax = plt.subplots(figsize=(8, 3.2))
            ax.bar(par_delai.index.astype(str), par_delai.values, color=COULEUR_BARRES)
            ax.set_title("Taux d'annulation selon le délai de réservation"); ax.set_ylabel("%")
            ax.tick_params(axis="x", rotation=20)
            st.pyplot(fig)
        with g2:
            par_segment = (t.groupby("segment")["annulee"].mean().sort_values() * 100)
            fig, ax = plt.subplots(figsize=(8, 3.2))
            ax.barh(par_segment.index, par_segment.values, color=COULEUR_BARRES)
            ax.set_title("Taux d'annulation selon le segment"); ax.set_xlabel("%")
            st.pyplot(fig)

with onglet2:
    st.write("Régression logistique entraînée sur les arrivées de 2015 et 2016, évaluée sur 2017. "
             "Elle n'utilise que des informations connues le jour de la réservation. "
             "Elle donne un risque d'annulation, pas une certitude.")

    # Deux seuils recommandés, plus un réglage libre
    if "seuil" not in st.session_state:
        st.session_state["seuil"] = 0.5

    def regler(valeur):
        st.session_state["seuil"] = valeur

    b1, b2, _ = st.columns([1, 1, 3])
    b1.button("Seuil 0,3 : relances", on_click=regler, args=(0.3,))
    b2.button("Seuil 0,6 : surréservation", on_click=regler, args=(0.6,))
    seuil = st.slider("Seuil d'alerte (on signale la réservation si le risque dépasse ce seuil)",
                      0.1, 0.9, step=0.05, key="seuil")
    st.caption("Seuil haut (0,6 et plus) : peu de fausses alertes, adapté à la surréservation. "
               "Seuil bas (0,3) : on repère la plupart des annulations, adapté aux relances.")

    prevu = (proba_test >= seuil).astype(int)
    reference = 1 - y_test.mean()
    m = st.columns(4)
    with m[0].container(border=True):
        st.metric("Exactitude", pc(accuracy_score(y_test, prevu)), f"référence {pc(reference)}", delta_color="off",
                  help="Part des prévisions justes")
    with m[1].container(border=True):
        st.metric("Précision", pc(precision_score(y_test, prevu, zero_division=0)),
                  help="Parmi les réservations signalées, part qui s'annule vraiment")
    with m[2].container(border=True):
        st.metric("Rappel", pc(recall_score(y_test, prevu)),
                  help="Parmi les annulations réelles, part repérée par le modèle")
    with m[3].container(border=True):
        st.metric("Réservations signalées", fr(prevu.sum()), help="Réservations dont le risque dépasse le seuil")
    st.caption("Mesures sur les " + fr(len(y_test)) + " réservations de 2017. Référence : prévoir que personne n'annule.")
    cm = confusion_matrix(y_test, prevu)
    st.dataframe(pd.DataFrame(cm, index=["réel : séjour", "réel : annulée"],
                              columns=["prévu : séjour", "prévu : annulée"]))
    st.info("Point de vigilance : 99,4 % des réservations à dépôt non remboursable sont annulées, ce qui est "
            "surprenant. Le dépôt est à vérifier avec l'hôtel avant de s'y fier. Sans cette variable, "
            "le modèle reste meilleur que la référence, mais moins précis.")

    st.subheader("Tester une réservation")
    a, b, c3 = st.columns(3)
    hotel = a.selectbox("Hôtel", ["Lisbonne", "Algarve"])
    mois = a.selectbox("Mois d'arrivée", list(range(1, 13)), index=6)
    delai = a.number_input("Délai de réservation (jours)", 0, 800, 90)
    nuits = b.number_input("Nuits", 1, 70, 3)
    adultes = b.number_input("Adultes", 1, 10, 2)
    enfants = b.number_input("Enfants", 0, 10, 0)
    prix = b.number_input("Prix par nuit (€)", 0.0, 1000.0, 100.0)
    segment = c3.selectbox("Segment", sorted(propres["segment"].unique()), index=1)
    type_client = c3.selectbox("Type de client", sorted(propres["type_client"].unique()))
    depots = sorted(propres["depot"].unique())
    depot = c3.selectbox("Dépôt", depots, index=depots.index("Sans dépôt"))
    habitue = c3.selectbox("Client habitué", [0, 1])
    precedentes = c3.number_input("Annulations précédentes", 0, 30, 0)

    ligne = pd.DataFrame([{"hotel": hotel, "mois": mois, "delai_reservation": delai, "nuits": nuits,
                           "adultes": adultes, "enfants": enfants, "segment": segment,
                           "type_client": type_client, "client_habitue": habitue,
                           "annulations_precedentes": precedentes, "depot": depot, "prix_nuit": prix}])
    base = pd.concat([propres[COLONNES].head(1), ligne], ignore_index=True)
    X_ligne = pd.get_dummies(base, columns=QUALITATIVES, dtype=int).iloc[[1]].reindex(columns=colonnes_modele, fill_value=0)
    risque = modele.predict_proba(X_ligne)[0, 1]
    st.metric("Risque d'annulation", pc(risque))
    if risque >= seuil:
        st.error("Réservation à risque : au-dessus du seuil choisi.")
    else:
        st.success("Réservation sous le seuil choisi.")
