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

Le fichier `cs-training.csv` contient **150 000 emprunteurs** et 11 colonnes : la cible et 10 variables qui décrivent le client.

| Variable | Signification |
|---|---|
| `SeriousDlqin2yrs` | **Cible** : 1 si le client a eu un retard de 90 jours ou plus dans les 2 ans, 0 sinon |
| `RevolvingUtilizationOfUnsecuredLines` | Part utilisée des cartes de crédit et découverts (solde ÷ plafond autorisé) |
| `age` | Âge de l'emprunteur |
| `NumberOfTime30-59DaysPastDueNotWorse` | Nombre de retards de 30 à 59 jours |
| `NumberOfTime60-89DaysPastDueNotWorse` | Nombre de retards de 60 à 89 jours |
| `NumberOfTimes90DaysLate` | Nombre de retards de 90 jours ou plus |
| `DebtRatio` | Taux d'endettement : dépenses mensuelles ÷ revenu mensuel |
| `MonthlyIncome` | Revenu mensuel |
| `NumberOfOpenCreditLinesAndLoans` | Nombre de crédits en cours (prêts et cartes) |
| `NumberRealEstateLoansOrLines` | Nombre de prêts immobiliers |
| `NumberOfDependents` | Nombre de personnes à charge |

**Ce que l'exploration a montré** (notebook `01_exploration`) :

- **Peu de défauts** : 6,7 % des clients seulement, soit environ 14 bons clients pour 1 mauvais.
- **Valeurs manquantes** : le revenu manque pour 20 % des clients et le nombre de personnes à charge pour 2,6 %. Les clients sans revenu renseigné font *moins* défaut que les autres : le fait que l'information manque est donc lui-même une information.
- **Les variables les plus utiles** sont les retards de paiement passés, la part utilisée des cartes de crédit et l'âge. Par exemple, un client sans aucun retard de 90 jours fait défaut 4,6 % du temps, contre 34 % avec un seul retard.
- **Plusieurs valeurs bizarres**, à corriger avant de construire un modèle :
  - **les codes 96 et 98** dans les colonnes de retards (269 clients). Avoir 98 retards en 2 ans est impossible : ce sont des codes, pas de vrais nombres. Ces clients font pourtant défaut à 55 % ;
  - **un client de 0 an** ;
  - **une utilisation des cartes jusqu'à 50 708** au lieu d'une valeur entre 0 et 1 environ ;
  - **un taux d'endettement qui vaut un montant en dollars** quand le revenu manque, au lieu d'un rapport ;
  - **des revenus de 0 ou 1 $** par mois ;
  - **609 lignes en double**.

## 3. Méthode

### Préparation des données (notebook `02_preparation`)

1. **Suppression des doublons**, pour qu'un même client ne se retrouve pas à la fois dans les données d'apprentissage et dans celles de test.
2. **Séparation en trois jeux, avant toute correction** :

   | Jeu | Part | Rôle |
   |---|---|---|
   | Entraînement (*train*) | 60 % | Le modèle apprend sur ces clients |
   | Validation | 20 % | Comparer les modèles et faire les réglages |
   | Test | 20 % | Mesurer la performance finale, une seule fois, à la fin |

   La séparation est **stratifiée** : chaque jeu garde 6,7 % de défauts.

   On sépare *avant* de corriger pour éviter la **fuite de données**. Par exemple, si on remplaçait les revenus manquants par la médiane calculée sur tout le fichier, le modèle aurait indirectement « vu » les clients du test, et ses résultats seraient trop optimistes. Toutes les valeurs de correction (médianes, plafonds) sont donc calculées sur le jeu d'entraînement seulement.
3. **Correction des valeurs bizarres** : chaque valeur impossible est remplacée par la médiane. Quand l'anomalie porte une information (codes 96/98, revenu manquant, revenu de 0 ou 1), on ajoute une colonne qui la signale. Les valeurs extrêmes sont plafonnées.

### Modèle traditionnel (notebook `03_modelisation`)

1. **Découpage de chaque variable en classes**, par exemple l'âge en « moins de 30 ans », « 30 à 40 ans »… Cela permet de capter des relations qui ne sont pas en ligne droite (le risque peut monter puis redescendre).
2. **Calcul du WoE et de l'IV**, sur le jeu d'entraînement :

   **WoE = ln( % des bons clients dans la classe ÷ % des mauvais clients dans la classe )**

   - **WoE** (*Weight of Evidence*) : mesure le risque d'une classe. Positif = classe moins risquée que la moyenne, négatif = plus risquée.
   - **% des bons clients dans la classe** : parmi tous les clients qui ont remboursé, la part qui se trouve dans cette classe.
   - **% des mauvais clients dans la classe** : parmi tous les clients en défaut, la part qui se trouve dans cette classe.

   **IV = Σ ( % des bons − % des mauvais ) × WoE**, sur toutes les classes d'une variable

   - **IV** (*Information Value*) : mesure à quel point une variable sépare bien les bons et les mauvais clients. En dessous de 0,02, la variable est inutile ; au-dessus de 0,3, elle est forte.
3. **Régression logistique** sur les WoE. Le revenu a été retiré, car il n'apportait plus rien une fois les autres variables connues. Il reste 9 variables.
4. **Grille de score par points**, comme dans une vraie banque : chaque classe de chaque variable rapporte des points, et on additionne pour obtenir le score du client. Plus le score est haut, moins le client est risqué. Les réglages suivent les conventions bancaires :
   - **600 points** correspondent à 50 bons clients pour 1 mauvais ;
   - **+20 points** correspondent à un risque divisé par 2.

   La grille complète est dans [`reports/grille_score.csv`](reports/grille_score.csv).

## 4. Résultats

Résultats sur le jeu de **validation**.

| Modèle | AUC | Gini | KS |
|---|---|---|---|
| Régression logistique (grille de score) | 0,859 | 0,719 | 0,557 |

- **AUC** : si l'on prend au hasard un client en défaut et un client qui rembourse, c'est la probabilité que le modèle juge le client en défaut plus risqué. 0,5 = hasard, 1 = parfait.
- **Gini = 2 × AUC − 1** : la même information sur une échelle de 0 (hasard) à 1 (parfait), très utilisée dans les banques. Au-dessus de 0,6, un modèle de scoring est considéré comme bon.
- **KS** : le plus grand écart entre la répartition des scores des bons clients et celle des mauvais clients. Plus il est grand, mieux le modèle sépare les deux groupes.

**Ce qu'il faut retenir du modèle traditionnel :**

- **Il ne surapprend pas** : il obtient les mêmes résultats sur le jeu d'entraînement et sur la validation, donc il fonctionne aussi bien sur des clients qu'il n'a jamais vus.
- **Il classe bien les clients** :
  - les 10 % de clients avec les scores les plus bas font défaut à 36 % et regroupent plus de la moitié des défauts ;
  - les 10 % avec les scores les plus hauts ne font défaut qu'à 0,5 %.
- **Il reste facile à expliquer** : un client refusé peut savoir exactement quelles caractéristiques lui ont fait perdre des points.

## 5. Limites et pistes d'amélioration

- **Les données sont anonymes et anciennes** : on ne connaît ni le pays, ni la date, ni le type de crédit. On ne sait donc pas si le modèle fonctionnerait aujourd'hui sur d'autres clients.
- **Certaines anomalies ont été interprétées sans pouvoir les vérifier** : on ne sait pas avec certitude ce que signifient les codes 96/98 ni le taux d'endettement en dollars. Les choix faits sont les plus raisonnables au vu des données, mais restent des hypothèses.
- **Les petits groupes sont fragiles** : par exemple, les codes 96/98 ne concernent que 123 clients dans le jeu d'entraînement. Leur poids dans la grille dépend donc de peu d'exemples.
- **Les classes ont été choisies à la main** : un découpage automatique (arbre de décision, regroupement optimal des classes) pourrait améliorer un peu le résultat.
- **Le modèle n'est pas suivi dans le temps** : une vraie banque vérifie régulièrement que la population de clients ne change pas, avec des indicateurs de stabilité.

## Reproduire

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Placer cs-training.csv (Kaggle) dans data/
jupyter notebook notebooks/
```
