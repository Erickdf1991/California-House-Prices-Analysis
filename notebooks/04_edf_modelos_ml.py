# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: machine_learning
#     language: python
#     name: python3
# ---

# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OrdinalEncoder,
    OneHotEncoder,
    PolynomialFeatures,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
    QuantileTransformer,
)

from joblib import dump

from notebooks.src.config import DADOS_LIMPOS, MODELO_FINAL
from notebooks.src.auxiliares import dataframe_coeficientes
from notebooks.src.graficos import (
    plot_coeficientes,
    plot_comparar_metricas_modelos,
    plot_residuos_estimador,
)

from notebooks.src.models import (
    organiza_resultados,
    treinar_e_validar_modelo_regressao,
    grid_search_cv_regressor,
)

# %%
df = pd.read_parquet(DADOS_LIMPOS)

df.head()

# %%
df.info()

# %%
df['ocean_proximity'] = df['ocean_proximity'].astype('category')

# %%
df.info()

# %% [markdown]
# # Separando as colunas por tipo

# %%
coluna_target = ['median_house_value']

coluna_one_hot_encoder = ['ocean_proximity']

coluna_ordinal_encoder = ['median_income_cat']

# %% [markdown]
# # Criando o preprocessamento

# %%
colunas_robust_scaler = df.columns.difference(coluna_target + coluna_one_hot_encoder + coluna_ordinal_encoder)

colunas_robust_scaler

# %%
pipeline_robust = Pipeline(steps=[
    ("robust_scaler", RobustScaler()),
    ("poly", PolynomialFeatures(degree=1, include_bias=False))
])

preprocessamento = ColumnTransformer(
    transformers=[
        ('ordinal_encoder', OrdinalEncoder(categories="auto"), coluna_ordinal_encoder),
        ('one_hot', OneHotEncoder(drop="first"), coluna_one_hot_encoder),
        ('robust_scaler_poly', pipeline_robust, colunas_robust_scaler),
    ],
)

# %% [markdown]
# # Separando X e y

# %%
X = df.drop(columns=coluna_target)
y = df[coluna_target]

# %% [markdown]
# # Introdozindo o Grid Search

# %%
param_grid = {
    "regressor__preprocessor__robust_scaler_poly__poly__degree": [1, 2, 3],
    "regressor__reg__alpha": [1E-2, 5E-2, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0],
}

# %%
grid_search = grid_search_cv_regressor(
    regressor = Ridge(),
    preprocessor = preprocessamento,
    target_transformer = QuantileTransformer(output_distribution="normal"),
    param_grid = param_grid
)

grid_search

# %%
grid_search.fit(X, y)

# %%
grid_search.best_params_

# %%
grid_search.best_score_

# %%
coefs = dataframe_coeficientes(
    grid_search.best_estimator_.regressor_["reg"].coef_,
    grid_search.best_estimator_.regressor_["preprocessor"].get_feature_names_out(),
)

coefs

# %%
coefs[coefs["coeficiente"] == 0]

# %%
coefs[coefs["coeficiente"].between(-0.2, 0.2) & (coefs["coeficiente"] != 0)]

# %%
plot_coeficientes(coefs[~coefs["coeficiente"].between(-0.2, 0.2)])

# %%
regressors = {
    "DummyRegressor": {
        "preprocessor": None,
        "regressor": DummyRegressor(strategy="mean"),
        'target_transformer': None,
    },

    "LinearRegression": {
        "preprocessor": preprocessamento,
        "regressor": LinearRegression(),
        'target_transformer': None,
    },

    "LinearRegression_target": {
        "preprocessor": preprocessamento,
        "regressor": LinearRegression(),
        'target_transformer': QuantileTransformer(output_distribution="normal"),
    },

    "Ridge_grid_search": {
        "preprocessor": grid_search.best_estimator_.regressor_["preprocessor"],
        "regressor": grid_search.best_estimator_.regressor_["reg"],
        'target_transformer': grid_search.best_estimator_.transformer_,
    },
}

resultados = {
    nome_modelo: treinar_e_validar_modelo_regressao(X, y, **regressor)
    for nome_modelo, regressor in regressors.items()
}

df_resultados = organiza_resultados(resultados)

df_resultados

# %%
df_resultados.groupby("model").mean()

# %%
df_resultados.groupby("model").mean().sort_values(by="test_neg_root_mean_squared_error")

# %%
plot_comparar_metricas_modelos(df_resultados)

# %%
plot_residuos_estimador(grid_search.best_estimator_, X, y, fracao_amostra=0.1, eng_formatter=True)

# %%
dump(grid_search.best_estimator_, MODELO_FINAL)
