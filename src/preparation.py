"""Préparation des données Give Me Some Credit.

Chaque traitement correspond à une anomalie repérée dans notebooks/01_exploration.ipynb.
Toutes les valeurs « apprises » (médianes, plafonds) sont calculées avec `fit` sur le
jeu d'entraînement uniquement, puis appliquées telles quelles à la validation et au test.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

TARGET = 'SeriousDlqin2yrs'

LATE = [
    'NumberOfTime30-59DaysPastDueNotWorse',
    'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfTimes90DaysLate',
]
COUNTS = LATE + [
    'NumberOfOpenCreditLinesAndLoans',
    'NumberRealEstateLoansOrLines',
    'NumberOfDependents',
]
CONTINUOUS = ['RevolvingUtilizationOfUnsecuredLines', 'age', 'DebtRatio', 'MonthlyIncome']

# Règles fixes, issues de l'exploration (aucune n'est apprise sur les données)
LATE_CODES = [96, 98]   # codes spéciaux dans les colonnes de retards
AGE_MIN = 18            # en dessous : âge impossible pour un emprunteur
RU_ERROR = 10           # utilisation du crédit au-delà : erreur de saisie
RU_CAP = 2              # utilisation du crédit plafonnée à 2 fois le plafond autorisé
INCOME_MIN = 1          # revenu de 0 ou 1 : valeur saisie par défaut
CAP_QUANTILE = 0.99     # plafond des autres variables : 99e percentile du train


def split_stratified(df, val_size=0.2, test_size=0.2, random_state=42):
    """Découpe df en train / validation / test en gardant la même proportion de défauts."""
    from sklearn.model_selection import train_test_split

    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[TARGET], random_state=random_state)
    train, val = train_test_split(
        train_val, test_size=val_size / (1 - test_size),
        stratify=train_val[TARGET], random_state=random_state)
    return train, val, test


class Preparateur(BaseEstimator, TransformerMixin):
    """Corrige les anomalies, ajoute les indicateurs, impute les manquants et plafonne.

    Étapes de `transform` :
      1. corrections fixes (règles ci-dessus) : les valeurs fausses deviennent NaN ;
      2. imputation des NaN par la médiane du train ;
      3. plafonnement des grandes valeurs (RU_CAP, ou 99e percentile du train).
    """

    def fit(self, X, y=None):
        X = self._corriger(X)
        features = CONTINUOUS + COUNTS
        self.medianes_ = X[features].median()
        self.plafonds_ = X[features].quantile(CAP_QUANTILE)
        self.plafonds_['RevolvingUtilizationOfUnsecuredLines'] = RU_CAP
        self.plafonds_['age'] = np.inf  # l'âge n'a pas besoin de plafond
        return self

    def transform(self, X):
        X = self._corriger(X)
        features = CONTINUOUS + COUNTS
        X[features] = X[features].fillna(self.medianes_)
        X[features] = X[features].clip(upper=self.plafonds_, axis=1)
        return X

    @staticmethod
    def _corriger(X):
        """Règles fixes : repère les valeurs fausses, les remplace par NaN, crée les indicateurs."""
        X = X.copy()

        # Codes 96/98 : ce ne sont pas des nombres de retards
        code = X[LATE].isin(LATE_CODES).any(axis=1)
        X['retards_code_special'] = code.astype(int)
        X.loc[code, LATE] = np.nan

        # Âge impossible
        X.loc[X['age'] < AGE_MIN, 'age'] = np.nan

        # Utilisation du crédit aberrante
        ru = 'RevolvingUtilizationOfUnsecuredLines'
        X.loc[X[ru] > RU_ERROR, ru] = np.nan

        # Revenu manquant ou saisi par défaut (0 ou 1)
        income = X['MonthlyIncome']
        X['revenu_manquant'] = income.isna().astype(int)
        X['revenu_0_1'] = (income <= INCOME_MIN).astype(int)
        X.loc[income <= INCOME_MIN, 'MonthlyIncome'] = np.nan

        # Sans revenu fiable, DebtRatio est un montant en $ et non un rapport
        sans_revenu = (X['revenu_manquant'] == 1) | (X['revenu_0_1'] == 1)
        X.loc[sans_revenu, 'DebtRatio'] = np.nan

        return X
