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
import folium
import geopandas as gpd
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from folium import plugins
from shapely.geometry import Point

from notebooks.src.config import DADOS_LIMPOS, DADOS_GEO_MEDIAN, DADOS_GEO_ORIGINAIS
from notebooks.src.graficos import PALETTE, SCATTER_ALPHA

sns.set_theme(palette="bright", style="white")

# %%
df = pd.read_parquet(DADOS_LIMPOS)

df.head()

# %%
df.info()

# %%
gdf_counties = gpd.read_file(DADOS_GEO_ORIGINAIS)

gdf_counties.head()

# %%
pontos = [Point(long, lat) for long, lat in zip(df["longitude"], df["latitude"])]

pontos[:5]

# %%
gdf = gpd.GeoDataFrame(df, geometry=pontos)

gdf.head()

# %%
gdf.info()

# %%
gdf = gdf.set_crs(epsg=4326)

gdf_counties = gdf_counties.to_crs(epsg=4326)

# %%
gdf_joined = gpd.sjoin(gdf, gdf_counties, how="left", predicate="within")

gdf_joined.head()

# %%
gdf_joined.tail()

# %%
gdf_counties.iloc[12]

# %%
gdf_joined = gdf_joined.drop(
    columns=[
        "index_right",
        "fullname",
        "abcode",
        "ansi"
    ]
)

gdf_joined.head()

# %%
gdf_joined.info()

# %%
gdf_joined.isnull().sum()

# %%
gdf_joined[gdf_joined.isnull().any(axis=1)]

# %%
linhas_faltantes = gdf_joined[gdf_joined.isnull().any(axis=1)].index

linhas_faltantes

# %% [markdown]
# Utilizando o centroide para avaliar em quais condados estas linhas faltantes pertencem.

# %%
gdf_counties["centroid"] = gdf_counties.centroid

gdf_counties.head()

# %%
gdf_counties.shape

# %%
gdf_joined.loc[1507]

# %%
print(gdf_joined.loc[1507, "geometry"])

# %%
gdf_counties["centroid"].distance(gdf_joined.loc[1507, "geometry"])

# %%
gdf_counties["centroid"].distance(gdf_joined.loc[1507, "geometry"]).idxmin()

# %%
gdf_counties.loc[1]


# %%
def condado_mais_proximo(linha):
    ponto = linha["geometry"]
    distancias = gdf_counties["centroid"].distance(ponto)
    idx_condado_mais_proximo = distancias.idxmin()
    condado_mais_proximo = gdf_counties.loc[idx_condado_mais_proximo]
    return condado_mais_proximo[["name", "abbrev"]]


# %%
condado_mais_proximo(gdf_joined.loc[1507])

# %%
gdf_joined.loc[linhas_faltantes, ["name", "abbrev"]] = gdf_joined.loc[linhas_faltantes].apply(condado_mais_proximo, axis=1)

# %%
gdf_joined.isnull().sum()

# %%
gdf_joined.loc[linhas_faltantes, ["name", "abbrev"]]

# %%
gdf_joined.loc[linhas_faltantes, ["name", "abbrev"]].value_counts()

# %%
gdf_joined["name"].value_counts()

# %%
gdf_counties.plot()

# %%
fig, ax = plt.subplots(figsize=(10, 10))

gdf_counties.plot(
    ax=ax,
    edgecolor="black",
    color="lightgray"
)

ax.scatter(
    gdf_joined["longitude"],
    gdf_joined["latitude"],
    color="red",
    s=1,
    alpha=SCATTER_ALPHA
)

for x, y, abbrev in zip(gdf_counties["centroid"].x, gdf_counties["centroid"].y, gdf_counties["abbrev"]):
    ax.text(x, y, abbrev, ha="center", va="center", fontsize=8)


plt.show()

# %%
gdf_joined.groupby("name").median(numeric_only=True)

# %%
gdf_counties = gdf_counties.merge(
    gdf_joined.groupby("name").median(numeric_only=True),
    left_on="name",
    right_index=True
)

# %%
gdf_counties.head()

# %%
gdf_joined[["name", "ocean_proximity"]].groupby("name").describe()

# %%
county_ocean_prox = gdf_joined[["name", "ocean_proximity"]].groupby("name").agg(pd.Series.mode)

county_ocean_prox

# %%
gdf_counties = gdf_counties.merge(
    county_ocean_prox, 
    left_on="name", 
    right_index=True
)

# %%
gdf_counties.head()

# %%
gdf_counties.info()

# %%
fig, ax = plt.subplots(figsize=(10, 10))

gdf_counties.plot(
    ax=ax,
    edgecolor="black",
    column="median_house_value",
    cmap=PALETTE,
    legend=True,
    legend_kwds={
        "label": "Median House Value",
        "orientation": "vertical"
    }
)

for x, y, abbrev in zip(gdf_counties["centroid"].x, gdf_counties["centroid"].y, gdf_counties["abbrev"]):
    ax.text(x, y, abbrev, ha="center", va="center", fontsize=8)


plt.show()

# %%
fig, ax = plt.subplots(figsize=(10, 10))

gdf_counties.plot(
    ax=ax,
    edgecolor="black",
    column="median_income",
    cmap=PALETTE,
    legend=True,
    legend_kwds={
        "label": "Median Indome",
        "orientation": "vertical"
    }
)

for x, y, abbrev in zip(gdf_counties["centroid"].x, gdf_counties["centroid"].y, gdf_counties["abbrev"]):
    ax.text(x, y, abbrev, ha="center", va="center", fontsize=8)


plt.show()

# %%
fig, ax = plt.subplots(figsize=(10, 10))

gdf_counties.plot(
    ax=ax,
    edgecolor="black",
    column="housing_median_age",
    cmap=PALETTE,
    legend=True,
    legend_kwds={
        "label": "Housing Median Age",
        "orientation": "vertical"
    }
)

for x, y, abbrev in zip(gdf_counties["centroid"].x, gdf_counties["centroid"].y, gdf_counties["abbrev"]):
    ax.text(x, y, abbrev, ha="center", va="center", fontsize=8)


plt.show()

# %%
fig, axs = plt.subplots(1, 2, figsize=(20, 7))

gdf_counties.plot(
    ax=axs[0],
    edgecolor="black",
    column="median_house_value",
    cmap=PALETTE,
    legend=True,
    legend_kwds={
        "label": "Median House Value",
        "orientation": "vertical"
    }
)

gdf_counties.plot(
    ax=axs[1],
    edgecolor="black",
    column="median_income",
    cmap="YlOrRd",
    legend=True,
    legend_kwds={
        "label": "Median Indome",
        "orientation": "vertical"
    }
)

for x, y, abbrev in zip(gdf_counties["centroid"].x, gdf_counties["centroid"].y, gdf_counties["abbrev"]):
    axs[0].text(x, y, abbrev, ha="center", va="center", fontsize=8)
    axs[1].text(x, y, abbrev, ha="center", va="center", fontsize=8)


plt.show()

# %%
gdf_counties.to_parquet(DADOS_GEO_MEDIAN)

# %% [markdown]
# ## Biblioteca Folium

# %%
centro_mapa = [df["latitude"].mean(), df["longitude"].mean()]

centro_mapa

# %%
with open(DADOS_GEO_ORIGINAIS, "r") as arquivo:
    geojson = json.load(arquivo)

# %%
tamanho_mapa_folium = {"width": 500, "height": 500}

fig = folium.Figure(**tamanho_mapa_folium)

mapa = folium.Map(
    location=centro_mapa, 
    zoom_start=5,
    tiles="cartodb voyager",
    control_scale=True
).add_to(fig)

folium.Choropleth(
    geo_data=geojson,
    name="choropleth",
    data=gdf_counties,
    columns=["abbrev", "median_income"],
    key_on="feature.properties.abbrev",
    fill_color="YlGn",
    legend_name="Median Income",
    fill_opacity=0.7,
    line_opacity=0.3,
).add_to(mapa)

folium.LayerControl().add_to(mapa)

folium.LatLngPopup().add_to(mapa)

plugins.MousePosition().add_to(mapa)

mapa

# %%
tamanho_mapa_folium = {"width": 500, "height": 500}

fig = folium.Figure(**tamanho_mapa_folium)

mapa = folium.Map(
    location=centro_mapa, 
    zoom_start=5,
    tiles="cartodb voyager",
    control_scale=True
).add_to(fig)

folium.Choropleth(
    geo_data=geojson,
    name="choropleth",
    data=gdf_counties,
    columns=["abbrev", "median_house_value"],
    key_on="feature.properties.abbrev",
    fill_color="YlOrRd",
    legend_name="Median House Value",
    fill_opacity=0.7,
    line_opacity=0.3,
).add_to(mapa)

folium.LayerControl().add_to(mapa)

folium.LatLngPopup().add_to(mapa)

plugins.MousePosition().add_to(mapa)

mapa
