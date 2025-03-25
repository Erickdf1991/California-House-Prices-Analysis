# -*- coding: utf-8 -*-
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

# %% [markdown]
# ## Importação das bibliotecas e modelos

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from notebooks.src.config import DADOS_ORIGINAIS, DADOS_LIMPOS
from notebooks.src.graficos import PALETTE, SCATTER_ALPHA

sns.set_theme(palette="bright")

# %% [markdown]
# ## Análise Exploratória

# %%
df = pd.read_csv(DADOS_ORIGINAIS, compression="zip")

df.head()

# %%
df.tail()

# %%
df.info()

# %%
df.describe()

# %%
df.describe(exclude="number")

# %%
sns.pairplot(df, diag_kind="kde", plot_kws={"alpha": SCATTER_ALPHA})

# %% [markdown]
# You can observe that for median_income and median_house_value it had seeling cut, so it had establish a maximum and minimum value.

# %%
df.select_dtypes(include="number").skew()

# %% [markdown]
# Existem valores de assimetria positiva (total_rooms, total_bedrooms, population, households), ou seja, cauda longa para direita.

# %%
df.select_dtypes(include="number").kurtosis()

# %% [markdown]
# O mesmo se observe para a Kurtosis (total_rooms, total_bedrooms, population, households), ou seja, um grande pico e uma queda abrupta.

# %%
df[df.duplicated()]

# %% [markdown]
# Nao há registros duplicados em nosso dataframe

# %%
df[df.isnull().any(axis=1)]

# %% [markdown]
# Filtrando os valores nulos do dataframe. Aparentemente apenas nessa coluna existem colunas nulas.

# %%
df[df.isnull().any(axis=1)].describe()

# %% [markdown]
# Não são linhas continuas, vide primeiro filtro de nulos. Exibe uma aleatoriedade

# %%
df["ocean_proximity"].value_counts()

# %% [markdown]
# Nota-se que ilha possui poucos registros e talvez durante a etapa de separação entre treino e teste, o modelo não conseguirá interpretar esses dados. Futuramente avaliar retirar estes 5 registros. Em um caso onde o cliente solicitasse uma verificação de preço em todas as regiões, haveria a necessidade de retornar ao cliente e dizer que não há registros suficientes para esta avaliação.

# %%
fig, axs = plt.subplots(3, 3, figsize=(12, 6))

for ax, coluna in zip(axs.flatten(), df.columns):
    sns.boxplot(data=df, x=coluna, ax=ax, showmeans=True)

plt.tight_layout()
plt.show()

# %% [markdown]
# O boxplot de housing_median_age mostra uma maior assimetria, e existem muitos outliers nos demais boxplots. E na target (median_house_value) notamos uma calda longa para direita e outliers para valores altos.

# %%
matriz = np.triu(df.select_dtypes(include="number").corr())

fig, ax = plt.subplots(figsize=(12, 6))

sns.heatmap(
    df.select_dtypes(include="number").corr(), 
    mask=matriz,
    annot=True,
    fmt=".2f", 
    ax=ax,
    cmap=PALETTE
)

plt.show()

# %% [markdown]
# A maior correlação com o nosso target é a renda (median_income). Existem colunas que podem ser redundantes devida sua alta correlação, preciso das duas colunas?

# %% [markdown]
# Algumas variáveis novas (novas colunas para ver a correlação com o target):
#
#  - Criar Classes em 'median_income'
#  - Cômodos por domicilio
#  - Pessoas por domicilio
#  - Quartos por cômodos

# %% [markdown]
# ## Criando as novas colunas

# %%
df["median_income_cat"] = pd.cut(
    df["median_income"],
    bins=[0, 1.5, 3, 4.5, 6, np.inf],
    labels=[1, 2, 3, 4, 5]
)

df.info()

# %% [markdown]
# Avaliando o describe podemos determinar os "bins" dessa nova medida categórica

# %%
df["median_income_cat"].value_counts().sort_index()

# %%
df["median_income_cat"].value_counts().sort_index().plot(kind="bar")

# %% [markdown]
# Com esta categorização, podemos notar uma concentração em salários mais altos.

# %%
df.columns

# %%
df["rooms_per_household"] = df["total_rooms"] / df["households"]
df["population_per_room"] = df["population"] / df["households"]
df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]

df.info()

# %%
df.describe()

# %% [markdown]
# ## Refazendo os gráficos com as novas colunas

# %%
fig, axs = plt.subplots(4, 3, figsize=(12, 10))

for ax, coluna in zip(axs.flatten(), df.select_dtypes("number").columns):
    sns.boxplot(data=df, x=coluna, ax=ax, showmeans=True)

plt.tight_layout()
plt.show()

# %%
matriz = np.triu(df.select_dtypes(include="number").corr())

fig, ax = plt.subplots(figsize=(8, 8))

sns.heatmap(
    df.select_dtypes(include="number").corr(), 
    mask=matriz,
    annot=True,
    fmt=".2f", 
    ax=ax,
    cmap=PALETTE
)

plt.show()

# %% [markdown]
# ## Tratando e ajustando os valores das colunas

# %%
df[
    df["median_house_value"] == df["median_house_value"].max()
]

# %%
965 / df.shape[0]

# %% [markdown]
# Podemos aplicar o conceito de "quantile"

# %%
df["median_house_value"].quantile(0.95)

# %%
df_clean = df.copy()

df_clean.info()

# %%
QUANTIL = 0.99

df_clean = df_clean[
    (df_clean["housing_median_age"] < df_clean["housing_median_age"].quantile(QUANTIL)) &
    (df_clean["total_rooms"] < df_clean["total_rooms"].quantile(QUANTIL)) &
    (df_clean["total_bedrooms"] < df_clean["total_bedrooms"].quantile(QUANTIL)) &
    (df_clean["population"] < df_clean["population"].quantile(QUANTIL)) &
    (df_clean["households"] < df_clean["households"].quantile(QUANTIL)) &
    (df_clean["median_income"] < df_clean["median_income"].quantile(QUANTIL)) &
    (df_clean["median_house_value"] < df_clean["median_house_value"].quantile(QUANTIL)) &
    (df_clean["rooms_per_household"] < df_clean["rooms_per_household"].quantile(QUANTIL)) &
    (df_clean["population_per_room"] < df_clean["population_per_room"].quantile(QUANTIL)) &
    (df_clean["bedrooms_per_room"] < df_clean["bedrooms_per_room"].quantile(QUANTIL))
]

df_clean.info()

# %%
1 - df_clean.shape[0] / df.shape[0]

# %%
df_clean.describe()

# %% [markdown]
# Com essa "limpeza" os valores tornam-se mais razoáveis e também eliminamos alguns outliers. Porém abaixo veremos novamente os gráficos.

# %%
sns.pairplot(df_clean, diag_kind="kde", plot_kws={"alpha": SCATTER_ALPHA})

# %%
fig, axs = plt.subplots(4, 3, figsize=(12, 10))

for ax, coluna in zip(axs.flatten(), df_clean.select_dtypes("number").columns):
    sns.boxplot(data=df_clean, x=coluna, ax=ax, showmeans=True)

plt.tight_layout()
plt.show()

# %%
matriz = np.triu(df_clean.select_dtypes(include="number").corr())

fig, ax = plt.subplots(figsize=(8, 8))

sns.heatmap(
    df_clean.select_dtypes(include="number").corr(), 
    mask=matriz,
    annot=True,
    fmt=".2f", 
    ax=ax,
    cmap=PALETTE
)

plt.show()

# %%
df_clean["ocean_proximity"].value_counts()

# %%
df_clean = df_clean.loc[df_clean["ocean_proximity"] != "ISLAND"]

df_clean["ocean_proximity"].value_counts()

# %% [markdown]
# Retirando os dois registros de ISLAND da coluna "ocean_proximity"

# %%
df_clean["ocean_proximity"] = df_clean["ocean_proximity"].astype("category")

df_clean.info()

# %%
colunas_valores_inteiros = []

for coluna in df_clean.select_dtypes(include="number").columns:
    if df_clean[coluna].apply(float.is_integer).all():
        colunas_valores_inteiros.append(coluna)

colunas_valores_inteiros

# %%
colunas_valores_float = df_clean.select_dtypes(include="number").columns.difference(colunas_valores_inteiros)

colunas_valores_float

# %%
df_clean[colunas_valores_inteiros] = df_clean[colunas_valores_inteiros].apply(
    pd.to_numeric, downcast="integer"
)

df_clean[colunas_valores_float] = df_clean[colunas_valores_float].apply(
    pd.to_numeric, downcast="float"
)

df_clean.info()

# %%
df_clean.describe()

# %% [markdown]
# ## Exportando para parquet o dataframe limpo/ aprimorado.

# %%
df_clean.to_parquet(DADOS_LIMPOS, index=False)

# %% [markdown]
# Exportamos para parquet, pois neste formato ele mantém as alterações de tipo das colunas que realizamos.
