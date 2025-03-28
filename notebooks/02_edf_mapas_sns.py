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

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from notebooks.src.config import DADOS_LIMPOS
from notebooks.src.graficos import PALETTE, SCATTER_ALPHA

sns.set_theme(palette="bright", style="white")

# %%
df = pd.read_parquet(DADOS_LIMPOS)

df.head()

# %%
df.tail()

# %%
df.info()

# %%
df.describe()

# %%
fig, ax = plt.subplots(figsize=(10, 6))

sns.scatterplot(data=df, x="longitude", y="latitude", ax=ax, alpha=SCATTER_ALPHA)

plt.show()

# %%
sns.jointplot(data=df, x="longitude", y="latitude", alpha=SCATTER_ALPHA)

# %%
fig, ax = plt.subplots(figsize=(10, 6))

sns.scatterplot(data=df, x="longitude", y="latitude", ax=ax, alpha=SCATTER_ALPHA, hue="ocean_proximity")

plt.show()

# %%
fig, ax = plt.subplots(figsize=(10, 6))

sns.scatterplot(data=df, x="longitude", y="latitude", ax=ax, alpha=1, hue="median_income_cat")

plt.show()

# %%
fig, ax = plt.subplots(figsize=(10, 6))

norm_median_house_value = plt.Normalize(df["median_house_value"].min(), df["median_house_value"].max())
sm_median_house_value = plt.cm.ScalarMappable(cmap=PALETTE, norm=norm_median_house_value)

sns.scatterplot(data=df, x="longitude", y="latitude", ax=ax, alpha=1, hue="median_house_value", palette=PALETTE)

ax.get_legend().remove()

fig.colorbar(sm_median_house_value, ax=ax)

plt.show()

# %% [markdown]
# Vericando a concetração das casas, pelas colunas categóricas.
