# Machine Leaning - Regressão Linear e XGBoost
Como desenvolver modelos de Regressão e fazer deploy com Flask.

### Propósito
Melhorar a estimativa básica de Previsão de tarifa de táxi. 

Neste caso não considero nenhum tratamento de inclusão de tarifas de pedágios, datas de feriados; apenas é considerado os dados entregues pelo dataset.

Métrica RMSE do modelo baseline do Kaggle é dado entre $5-$8, dependendo do algoritmo usado.

#### Dataset: https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction/data .
O dataset disponível contém 55 milhões de instância, das quais foi usado 1 milhão (sem tratamento de aleatoriedade).

### Resultado
Para construção do modelo baseline foi usado um algoritmo de Regressão Linear, sem normalização.

Foram construídas 5 versões baseadas em variações de Feature Selection, sendo uma desta seleções gerada pelo algoritmo de Regularização LASSO.

Foi usados as métricas RMSE e r2, e as 5 versões apresentaram resultados bem próximos, resultado em valores médios do RMSE e r2, respectivamente $4.89 e 0.76.

Após modelo baseline definido, foi usado o algortimo XGBoost com 3 versões:
 - v1: uso de 10 estimadores.
 - v2: uso da estruta DMatrix para otimização de memória e velocidade de treinamento.
 - v3: uso do GridSelection para seleção de Hyperparâmetros.

A etapa de Feature Selection foi realizada pelo XGBoost.

A versão com uso do GridSelection apresentou o melhor resultado sobre o tendo RMSE = $3.36, e r2 = 0.88.

## Estrutura de Diretórios

#### /api: 
 Código-fonte da API.
 - main.py - Arquivo de inicialização do servidor Flask e predições batch/online.

#### /app:  
 Código-fonte da consrtução dos modelos.
- AnaliseExporatoriaDados.ipynb - Análise Explorária dos Dados.
- CriandoModeloBaseline.ipynb - Algoritmo de Regressão Linear para construção do modelo Baseline.
- ModeloXGBoost.ipynb - Algoritmo XGBoost para escolha do melhor modelo de Regressão.
- TestePredicao_OnlineBatch.ipynb - Simulação de inferências online e batch no Jupyter Notebook.
- tools.py - biblioteca com funções gerais.

#### /data:
 Arquivos de entrada e gerado durante os processo de construção e execução do modelo.	
 - data_clean_1000000_20220320.csv - dataset gerado após a operação de Saneamento dos dados. Gerado pelo processo AnaliseExporatoriaDados.ipynb.
 - data_fe_10000000_20220320.csv - arquivo gerado após o processo de feature engineering. Usado para treinamento do modelo. Gerado pelo processo AnaliseExporatoriaDados.ipynb.
 - modelo_baseline_1000000_float64_20220320 - arquivo contendo as métricas RMSE e R2 geradas por cada modelo baseline com tipo de dados float64.
 - modelo_baseline_1000000_float32_20220320 - arquivo contendo as métricas RMSE e R2 geradas por cada modelo baseline com tipo de dados float32.
 - train.csv - dataset (55 milhões de instâncias). Usado no processo AnaliseExporatoriaDados.ipynb. Descompactado do arquivo train.rar
 - test.csv - dataset de Teste. Usado para realizar o teste da inferência batch. Usado em TestePredicao_OnlineBatch.ipynb e main.py (API Flask)
 - test_pred_batch.csv - arquivo de retorno do processo de predição batch. Resultado da predição batch API Flask (main.py)

#### /modelo:
 Modelo serializado.
 - modelo_20220321.sav - Gerado pelo processo ModeloXGBoost.ipynb.

## Instruções de execução da API Flask:
 - certifique de ter o flask instalado (pip install flask),
 - entre na pasta /api e execute o comando >> python main.py,
 - após execução da api, copie o endereço http e use as rotas /pred_batch e /pred_online, para as predições.

 - Exemplo predição batch: 
   http://127.0.0.1:5000/pred_batch/d:/serasa/data/test.csv
 - Exemplo predição online: 
   http://127.0.0.1:5000/pred_online/2015-01-27 13:08:24.0000002,2015-01-27 13:08:24 UTC,-73.973320007324219,40.7638053894043,-73.981430053710938,40.74383544921875,1
   
### Keys:
    - Regularizacao
    - LASSO
    - Métricas Algoritmos de Regressão RMSE, r2
    - Algoritmos Regressao Linear & XGBoost 
    - Feature Engineering
    - Feature Selection   

# Deploy com Google App Engine

App Engine é uma plataforma de computação para desenvolver e hospedar aplicações web na infraestrutura da GCP.

### Requisitos
 - SDK da GCP (gcloud)
 - requiriments.txt - todas dependências de pacotes necessários para a app.
 - app.yaml - arquivo de configuração para o App Engine executar.
 - main.py -  código da app.

### Sobre gunicorn
 	O flask é um framework de desenvolvimento para trabalhar com os request http, mas nao é recomendado em ambiente de produção, e sim o uso do gunicorn que após o App Engine subir o ambiente, colocará o web server na frente que vai tratar as chamadas http de uma forma melhor que o flask.

### Instruções do deploy 
1 - Instale o gcloud (SDK da GCP) 		
 - https://cloud.google.com/sdk/docs/install

2 - Entre no path /api, ou onde estejam os arquivos app.yaml, main.py (renomeio o main_gcloud.py), requirements.txt.

3 - Realize o deploy
Execute: gcloud app deploy

Vai aparecer as seguintes informações:
 - arquivo de configuração identidicado, app.yaml.
 - local de onde executará os arquivos.
 - nome do Projeto criando na GCP.
 - url onde ficará a app.     

Demora certa de 30 min.... 

4 - Testando
Execute: coloque_aqui_url_gerada_pelo_passo_acima. Rotas '/' (home), /pred_online/ (predição online), /pred_batch/' (predição batch).

