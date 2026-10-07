# Credit Scoring – Régression logistique vs LightGBM

Comparaison de l'approche traditionnelle des banques (régression logistique sur variables discrétisées, WoE, grille de score) et d'une approche machine learning (LightGBM), sous l'angle de la **performance** et de l'**interprétabilité**.

Données : [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) (Kaggle), environ 150 000 emprunteurs. Cible : `SeriousDlqin2yrs`, un retard de 90 jours ou plus dans les deux ans.

## 1. Le problème métier

**Probabilité de défaut (PD).** C'est la probabilité qu'un emprunteur ne rembourse pas son crédit (ici, un retard de 90 jours ou plus) sur un horizon donné (ici, 2 ans). C'est une estimation chiffrée du risque porté par chaque client.

**Pourquoi une banque en a besoin.**
- *Octroi* : accepter ou refuser une demande, et fixer le taux d'intérêt en fonction du risque.
- *Provisionnement (IFRS 9)* : la perte attendue se calcule comme `EL = PD × LGD × EAD`.
- *Fonds propres (Bâle III)* : en approche IRB, la PD entre directement dans le calcul du capital réglementaire.
- *Conformité* : le modèle doit pouvoir être expliqué au régulateur et au client, qui a droit à une explication en cas de refus.

**Les deux erreurs n'ont pas le même coût.**
- *Faux négatif* (on accorde le crédit à un client qui fera défaut) : la banque perd une partie du capital prêté, souvent plusieurs milliers d'euros.
- *Faux positif* (on refuse un bon client) : la banque perd seulement la marge d'intérêt qu'elle aurait gagnée, donc un montant bien plus faible.

C'est pour cette raison qu'on ne choisit pas le seuil de décision à 0,5 et qu'on ne juge pas les modèles sur l'*accuracy*. Avec environ 7 % de défauts, un modèle qui accepte tout le monde atteint déjà 93 % d'accuracy. On utilise plutôt des métriques de classement (AUC, Gini, KS), puis on choisit un seuil qui correspond au taux d'acceptation visé par la banque.

## 2. Données
_À compléter après l'exploration._

## 3. Méthode
_À compléter._

## 4. Résultats

| Modèle | AUC | Gini | KS |
|---|---|---|---|
| Régression logistique (WoE) | | | |
| LightGBM | | | |

## 5. Limites et pistes d'amélioration
_À compléter._

## Reproduire

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Placer cs-training.csv (Kaggle) dans data/
jupyter notebook notebooks/
```
