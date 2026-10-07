# Credit Scoring – Régression logistique vs LightGBM

Ce projet cherche à prédire si un client risque de ne pas rembourser son crédit. On compare deux façons de faire :

- **La méthode classique des banques** : une régression logistique, c'est-à-dire un modèle simple où chaque caractéristique du client ajoute ou retire des points. On obtient une **grille de score**, comme un barème (par exemple : « moins de 25 ans : −10 points », « aucun retard de paiement passé : +30 points »).
- **Une méthode de machine learning plus récente** : LightGBM. Ce modèle combine des centaines de petits arbres de décision. Il est souvent plus précis, mais plus difficile à expliquer.

On compare les deux sur deux critères : la **performance** (lequel repère le mieux les clients à risque ?) et l'**interprétabilité** (peut-on expliquer facilement pourquoi un client a été refusé ?).

**Données** : le jeu de données [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) (Kaggle), qui décrit environ 150 000 emprunteurs. On cherche à prédire la colonne `SeriousDlqin2yrs`, qui vaut 1 si le client a eu un retard de paiement de 90 jours ou plus au cours des deux années suivantes, et 0 sinon.

## 1. Le problème métier

### Qu'est-ce que la probabilité de défaut ?

On dit qu'un client fait **défaut** quand il ne rembourse plus son crédit normalement. Ici, cela veut dire un retard de paiement d'au moins 90 jours.

La **probabilité de défaut (PD)** est la probabilité que cela arrive pour un client donné sur une période fixée (ici, 2 ans). Par exemple, une PD de 5 % signifie que, sur 100 clients qui ressemblent à celui-ci, on s'attend à ce qu'environ 5 fassent défaut.

### Pourquoi une banque en a besoin

- **Accepter ou refuser un crédit** : la banque accorde le crédit si le risque est faible. Elle peut aussi adapter le taux d'intérêt : plus le client est risqué, plus le taux est élevé.
- **Mettre de l'argent de côté pour les pertes à venir** : les règles comptables (norme IFRS 9) obligent les banques à prévoir à l'avance l'argent qu'elles risquent de perdre. Ce montant s'appelle la **perte attendue** et se calcule ainsi :

  **EL = PD × LGD × EAD**

  - **EL** (*Expected Loss*, perte attendue) : le montant que la banque s'attend à perdre en moyenne sur ce crédit.
  - **PD** (*Probability of Default*, probabilité de défaut) : la probabilité que le client ne rembourse pas (voir plus haut).
  - **LGD** (*Loss Given Default*, perte en cas de défaut) : la part de l'argent que la banque ne récupérera pas si le client fait défaut. Elle est souvent inférieure à 100 %, car la banque récupère une partie de la somme (garantie, vente d'un bien, recouvrement…).
  - **EAD** (*Exposure At Default*, montant exposé) : la somme que le client doit encore à la banque au moment où il arrête de payer.

  *Exemple* : un client doit encore 10 000 € (EAD), a 5 % de chances de faire défaut (PD) et, s'il fait défaut, la banque ne récupère que 60 % de la somme, donc elle en perd 40 % (LGD). La perte attendue vaut alors EL = 0,05 × 0,40 × 10 000 = **200 €**.

- **Garder assez d'argent pour faire face aux risques** : les règles bancaires internationales (accords de Bâle III) imposent aux banques de conserver un minimum de fonds propres, c'est-à-dire leur propre argent, pour pouvoir encaisser des pertes. Plus les PD des clients sont élevées, plus la banque doit garder d'argent en réserve.
- **Pouvoir expliquer ses décisions** : le régulateur (l'autorité qui contrôle les banques) doit pouvoir comprendre le modèle, et un client refusé a le droit de savoir pourquoi. Un modèle trop opaque pose donc problème.

### Les deux erreurs possibles n'ont pas le même coût

Un modèle peut se tromper de deux façons :

- **Accorder un crédit à un client qui ne remboursera pas** (on appelle cela un *faux négatif*) : la banque perd une partie de l'argent prêté, souvent plusieurs milliers d'euros.
- **Refuser un client qui aurait remboursé** (un *faux positif*) : la banque perd seulement les intérêts qu'elle aurait gagnés, ce qui coûte beaucoup moins cher.

### Pourquoi on ne mesure pas la performance avec le « taux de bonnes réponses »

On pourrait juger un modèle sur son **taux de bonnes réponses** (en anglais *accuracy*), c'est-à-dire le pourcentage de clients correctement classés. Mais ici cette mesure est trompeuse : seuls environ 7 % des clients font défaut. Un modèle qui accepterait tout le monde, sans rien analyser, aurait donc déjà 93 % de bonnes réponses, tout en étant inutile.

On utilise plutôt des indicateurs qui vérifient si le modèle **range correctement les clients du moins risqué au plus risqué** :

- **AUC** : la probabilité que le modèle donne un score de risque plus élevé à un client qui fera défaut qu'à un client qui remboursera, quand on tire un client de chaque groupe au hasard. 0,5 correspond à un modèle qui répond au hasard, 1 à un modèle parfait.
- **Gini** : la même information exprimée sur une autre échelle, très utilisée dans les banques : **Gini = 2 × AUC − 1**. 0 correspond au hasard, 1 à un modèle parfait.
- **KS** (Kolmogorov-Smirnov) : mesure à quel point le modèle sépare bien les bons et les mauvais clients. On compare la répartition des scores des deux groupes et on retient le plus grand écart entre elles. Plus le KS est élevé, plus le modèle distingue clairement les deux groupes.

Enfin, on ne fixe pas automatiquement le seuil de décision à 50 % (« refuser si la PD dépasse 0,5 »). On choisit le seuil en fonction de la politique de la banque, par exemple « accepter 80 % des demandes », et en tenant compte du fait qu'un mauvais client coûte bien plus cher qu'un bon client refusé.

## 2. Données
_À compléter après l'exploration._

## 3. Méthode
_À compléter._

## 4. Résultats

| Modèle | AUC | Gini | KS |
|---|---|---|---|
| Régression logistique (grille de score) | | | |
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
