from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import normalize, LabelEncoder, StandardScaler
import pandas as pd
import numpy as np

DATE_TYPES = [
    {"name": "year", "period": "Y"},
    {"name": "month", "period": "-m"},
    {"name": "day", "period": "d"},
    {"name": "weekday", "period": "w"}
]


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select only specified columns."""
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.columns]


class Normalizer(TransformerMixin):
    def __init__(self):
        pass

    def fit(self):
        pass

    def transform(self, X, y=None):
        return normalize(X)

    def fit_transform(self, X, y=None):
        return self.transform(X)


class CustomLabelEncoder(TransformerMixin):
    def __init__(self):
        pass

    def fit(self):
        pass

    def transform(self, X, y=None):
        return self.fit_transform(X)

    def fit_transform(self, X, y=None):
        for col in X.columns:
            X[col] = LabelEncoder().fit_transform(X[col])
        return X


class AgeReplacer(TransformerMixin):
    def __init__(self, col_name='age', min_tresh=7, max_tresh=120):
        self.col_name = col_name
        self.min_tresh = min_tresh
        self.max_tresh = max_tresh
        pass

    def fit(self):
        pass

    def transform(self, X, y=None):
        return self.fit_transform(X)

    def fit_transform(self, X, y=None):
        X[self.col_name] = X[[self.col_name]].apply(lambda x: np.where(x <= self.min_tresh, self.min_tresh, x))
        X[self.col_name] = X[[self.col_name]].apply(lambda x: np.where(x >= self.max_tresh, 2021 - x, x))
        return X


class TimeFeatureCreator(TransformerMixin):
    def __init__(self):
        pass

    def fit(self):
        pass

    def transform(self, X, y=None):
        datetimes = pd.to_datetime(X['date_account_created'])
        timestamp_first_active = pd.to_datetime(X['timestamp_first_active'], format='%Y%m%d%H%M%S').dt.date
        date_account_created = pd.to_datetime(X['date_account_created']).dt.date
        X['first_active_on_creation_date'] = timestamp_first_active == date_account_created
        X['first_active_on_creation_date'] = X['first_active_on_creation_date'].astype(int)

        for col in DATE_TYPES:
            col_name = col['name']
            col_period = col['period']
            X[f"register_{col_name}"] = pd.to_numeric(datetimes.dt.strftime(f"%{col_period}"))
    
        return X.drop('date_account_created', axis=1)

    def fit_transform(self, X, y=None):
        return self.transform(X)


class FeatureNamer(TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        pass

    def fit(self):
        pass

    def transform(self, X, y=None):
        return pd.DataFrame(X, columns=self.columns)

    def fit_transform(self, X, y=None):
        return self.transform(X)


class OutlierRemover(TransformerMixin):
    def __init__(self, limites, columns):
        self.limites = limites
        self.columns = columns

    def fit(self):
        pass

    def transform(self, X, y=None):
        if (len(self.limites) != len(self.columns)):
            print('ERRO: Número de limites diferente do número de colunas')
            return X

        for i in range(0, len(self.limites)):
            df = pd.DataFrame(X[self.columns[i]])
            numerics = pd.DataFrame(StandardScaler().fit_transform(df), columns = [self.columns[i]])
            X[self.columns[i]] = df[numerics < self.limites[i]]

        return X

    def fit_transform(self, X, y=None):
        return self.transform(X)


class FeatureRemover(BaseEstimator, TransformerMixin):
    "Remove atributos a partir de uma lista de nomes."

    def __init__(self, removidos):
        "Inicialização do objeto."

        # recupera lista de atributos
        self.removidos = removidos

    def fit(self, X, y=None, **fit_params):
        pass

    def transform(self, X, y=None):
        "Efetua a alteração dos dados."

        novo = X.copy()
        atributos = [atributo for atributo in novo.columns if atributo not in self.removidos]

        return novo[atributos]

    def fit_transform(self, X, y=None):
        return self.transform(X)
