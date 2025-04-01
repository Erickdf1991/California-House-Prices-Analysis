# Previsão de Preços de Imóveis na Califórnia

## Visão Geral
Este projeto utiliza técnicas de análise exploratória de dados, visualização geográfica e modelos de machine learning para prever os preços de imóveis na Califórnia. O objetivo é desenvolver um sistema interativo que permite aos usuários selecionar um condado e obter previsões de preço com base em variáveis socioeconômicas e geográficas.

## Estrutura do Projeto

O projeto é dividido em quatro principais etapas:

1. **Análise Exploratória dos Dados** (`01_edf_eda_projetoregressao_casascalifornia.py`):
   - Carregamento e limpeza dos dados.
   - Exploração estatística das variáveis relevantes.
   - Identificação de correlações entre os atributos.

2. **Mapas e Visualização Geográfica** (`02_edf_mapas_sns.py`):
   - Utilização de `seaborn` para visualização de distribuição geográfica.
   - Geração de mapas interativos para análise espacial.

3. **Análise Geográfica** (`03_edf_analise_geografica.py`):
   - Manipulação de dados geoespaciais utilizando `geopandas`.
   - Processamento de informações geográficas para integração com modelos preditivos.

4. **Modelos de Machine Learning** (`04_edf_modelos_ml.py`):
   - Implementa modelos de regressão para previsão de preços.
   - Avalia o desempenho dos modelos com métricas apropriadas.

## Interface Interativa

A interface do projeto foi desenvolvida utilizando `Streamlit`, permitindo que os usuários selecionem um condado e insiram informações sobre um imóvel para prever seu preço. As principais funcionalidades incluem:

- **Carregamento e processamento dos dados**:
  - `carregar_dados_limpos()`: Carrega os dados processados e limpos.
  - `carregar_dados_geo()`: Processa dados geográficos e ajusta geometrias.
  - `carregar_modelo()`: Carrega o modelo treinado para previsões.

- **Entrada de dados**:
  - Usuário seleciona o condado e insere informações como idade do imóvel e renda média.

- **Visualização no mapa**:
  - Um mapa interativo exibe os condados da Califórnia e destaca o condado selecionado.

- **Previsão de preço**:
  - Com base nos dados inseridos, o modelo retorna um preço previsto para o imóvel.

## Link para a página do App: https://california-house-prices-erick-fernandes.streamlit.app/

## Como Executar

1. Clone este repositório:
   ```bash
   git clone https://github.com/Erickdf1991/California-House-Prices-Analysis.git
   cd seu_projeto
   ```

2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

3. Execute a aplicação Streamlit:
   ```bash
   streamlit run home.py
   ```

## Tecnologias Utilizadas
- Python
- Pandas, NumPy, Scikit-Learn
- Geopandas, Shapely, Pydeck
- Streamlit
- Seaborn, Matplotlib
- Joblib

## Autor
Este projeto foi desenvolvido por **Erick Duarte Fernandes**. GitHub https://github.com/Erickdf1991

## Licença
Este projeto é distribuído sob a licença MIT. Consulte o arquivo `LICENSE` para mais detalhes.
