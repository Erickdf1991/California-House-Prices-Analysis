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

from notebooks.src.config import DADOS_ORIGINAIS

sns.set_theme(palette="bright")

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
